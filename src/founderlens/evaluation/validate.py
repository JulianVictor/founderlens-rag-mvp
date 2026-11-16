"""Structural validation of the evaluation question set.

Usage::

    python -m founderlens.evaluation.validate

Checks JSON structure, required fields, categories, exact split sizes and
category distribution, question-id uniqueness across splits, evidence
references against the corpus manifest (document ids, one-based pages within
page counts), fact/evidence coverage, answerable/unanswerable consistency,
cross-document requirements, normalized duplicate questions, and that both
files still match the SHA-256 recorded in ``evaluation/dataset.json``.

This proves the dataset is *well-formed*. It does not prove that any reference
answer or evidence label is semantically correct.

Output contains only identifiers and counts; question text and reference
answers are never printed, so held-out content is not exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from founderlens.config import ConfigError, load_settings
from founderlens.corpus.manifest import ManifestError, load_manifest
from founderlens.evaluation.schema import Category, EvalQuestion, parse_jsonl

DATASET_METADATA_FILENAME = "dataset.json"
SPLIT_FILES: Mapping[str, str] = {"dev": "questions.dev.jsonl", "test": "questions.test.jsonl"}

EXPECTED_DISTRIBUTION: Mapping[str, Mapping[Category, int]] = {
    "dev": {
        Category.SINGLE_DOC_FACTUAL: 7,
        Category.SINGLE_DOC_CONCEPTUAL: 5,
        Category.CROSS_DOC_COMPARISON: 5,
        Category.TEMPORAL_MULTI_EVIDENCE: 2,
        Category.UNANSWERABLE: 5,
    },
    "test": {
        Category.SINGLE_DOC_FACTUAL: 5,
        Category.SINGLE_DOC_CONCEPTUAL: 3,
        Category.CROSS_DOC_COMPARISON: 3,
        Category.TEMPORAL_MULTI_EVIDENCE: 2,
        Category.UNANSWERABLE: 3,
    },
}

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_question(text: str) -> str:
    """Lowercase and collapse all non-alphanumerics, for duplicate detection."""
    return _NORMALIZE_RE.sub(" ", text.lower()).strip()


@dataclass(slots=True)
class DatasetReport:
    problems: list[str] = field(default_factory=list)
    counts: dict[str, Counter[Category]] = field(default_factory=dict)
    dataset_version: str | None = None
    annotation_status: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


def check_question(q: EvalQuestion, page_counts: Mapping[str, int]) -> list[str]:
    """Return consistency problems for one parsed question."""
    p: list[str] = []
    qid = q.question_id
    is_unanswerable_category = q.category is Category.UNANSWERABLE

    if q.answerable == is_unanswerable_category:
        p.append(f"{qid}: answerable={q.answerable} is inconsistent with category {q.category}")

    if not q.answerable:
        if q.reference_answer is not None:
            p.append(f"{qid}: unanswerable question must have reference_answer null")
        if q.required_facts:
            p.append(f"{qid}: unanswerable question must have no required_facts")
        if q.accepted_evidence:
            p.append(f"{qid}: unanswerable question must have no accepted_evidence")
        if q.expected_abstention_reason is None:
            p.append(f"{qid}: unanswerable question needs expected_abstention_reason")
        return p

    if q.reference_answer is None:
        p.append(f"{qid}: answerable question needs a non-empty reference_answer")
    if q.expected_abstention_reason is not None:
        p.append(f"{qid}: answerable question must have expected_abstention_reason null")
    if not q.required_facts:
        p.append(f"{qid}: answerable question needs at least one required fact")

    fact_ids = [f.fact_id for f in q.required_facts]
    for fid in sorted({f for f in fact_ids if fact_ids.count(f) > 1}):
        p.append(f"{qid}: duplicate fact_id {fid}")
    for fid in fact_ids:
        if not fid.startswith(f"{qid}-f"):
            p.append(f"{qid}: fact_id {fid} must be prefixed with '{qid}-f'")

    seen_refs: set[tuple[str, str, int]] = set()
    for ref in q.accepted_evidence:
        key = (ref.fact_id, ref.document_id, ref.page_number)
        if key in seen_refs:
            p.append(f"{qid}: duplicate evidence {key}")
        seen_refs.add(key)
        if ref.fact_id not in fact_ids:
            p.append(f"{qid}: evidence references unknown fact_id {ref.fact_id}")
        pages = page_counts.get(ref.document_id)
        if pages is None:
            p.append(f"{qid}: evidence references unknown document {ref.document_id!r}")
        elif not 1 <= ref.page_number <= pages:
            p.append(
                f"{qid}: page {ref.page_number} out of range for {ref.document_id} (1..{pages})"
            )

    for fid in fact_ids:
        if not q.evidence_for(fid):
            p.append(f"{qid}: required fact {fid} has no accepted evidence")

    if q.category is Category.CROSS_DOC_COMPARISON:
        docs_per_fact = [{e.document_id for e in q.evidence_for(fid)} for fid in fact_ids]
        all_docs = set().union(*docs_per_fact) if docs_per_fact else set()
        if len(all_docs) < 2:
            p.append(f"{qid}: cross-document question must cite at least two documents")
        covering = [d for d in sorted(all_docs) if all(d in docs for docs in docs_per_fact)]
        if covering:
            p.append(
                f"{qid}: cross-document question is answerable from a single document "
                f"({covering[0]}) according to its own evidence"
            )
    if q.category is Category.TEMPORAL_MULTI_EVIDENCE and len(fact_ids) < 2:
        p.append(f"{qid}: temporal/multi-evidence question needs at least two required facts")
    return p


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_metadata(evaluation_dir: Path, report: DatasetReport) -> None:
    path = evaluation_dir / DATASET_METADATA_FILENAME
    try:
        meta: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.problems.append(f"{DATASET_METADATA_FILENAME}: cannot load ({exc})")
        return
    if not isinstance(meta, dict):
        report.problems.append(f"{DATASET_METADATA_FILENAME}: root must be an object")
        return
    version = meta.get("dataset_version")
    status = meta.get("annotation_status")
    report.dataset_version = version if isinstance(version, str) else None
    report.annotation_status = status if isinstance(status, str) else None
    if report.dataset_version is None:
        report.problems.append(f"{DATASET_METADATA_FILENAME}: 'dataset_version' missing")
    if report.annotation_status is None:
        report.problems.append(f"{DATASET_METADATA_FILENAME}: 'annotation_status' missing")
    files = meta.get("files")
    if not isinstance(files, dict):
        report.problems.append(f"{DATASET_METADATA_FILENAME}: 'files' must be an object")
        return
    for filename in SPLIT_FILES.values():
        entry = files.get(filename)
        expected = entry.get("sha256") if isinstance(entry, dict) else None
        file_path = evaluation_dir / filename
        if not isinstance(expected, str):
            report.problems.append(f"{DATASET_METADATA_FILENAME}: no sha256 for {filename}")
        elif file_path.is_file() and _sha256(file_path) != expected:
            report.problems.append(
                f"{filename}: SHA-256 differs from {DATASET_METADATA_FILENAME}. Dataset changes "
                "must be deliberate: bump dataset_version, record the change, update the hash."
            )


def validate_dataset(evaluation_dir: Path, manifest_path: Path) -> DatasetReport:
    report = DatasetReport()
    try:
        page_counts = load_manifest(manifest_path).page_counts
    except ManifestError as exc:
        report.problems.extend(f"manifest: {p}" for p in exc.problems)
        return report

    all_questions: list[EvalQuestion] = []
    for split, filename in SPLIT_FILES.items():
        path = evaluation_dir / filename
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            report.problems.append(f"{filename}: cannot read ({exc})")
            continue
        records, parse_problems = parse_jsonl(text, source=filename)
        report.problems.extend(parse_problems)
        for q in records:
            if q.split != split:
                report.problems.append(f"{q.question_id}: belongs in the {q.split} file")
            report.problems.extend(check_question(q, page_counts))
        counts: Counter[Category] = Counter(q.category for q in records)
        report.counts[split] = counts
        expected = EXPECTED_DISTRIBUTION[split]
        for category in Category:
            if counts[category] != expected[category]:
                report.problems.append(
                    f"{filename}: expected {expected[category]} {category} questions, "
                    f"found {counts[category]}"
                )
        all_questions.extend(records)

    ids = Counter(q.question_id for q in all_questions)
    report.problems.extend(f"duplicate question_id {i}" for i, n in sorted(ids.items()) if n > 1)
    by_text: dict[str, list[str]] = {}
    for q in all_questions:
        by_text.setdefault(normalize_question(q.question), []).append(q.question_id)
    report.problems.extend(
        f"duplicate question text: {', '.join(qids)}" for qids in by_text.values() if len(qids) > 1
    )
    _check_metadata(evaluation_dir, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.validate",
        description="Validate evaluation/questions.*.jsonl structure (prints no answers).",
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = validate_dataset(settings.evaluation_dir, settings.corpus_manifest_path)
    print(
        f"Evaluation dataset v{report.dataset_version or '?'} "
        f"(annotation status: {report.annotation_status or '?'})"
    )
    for split, counts in report.counts.items():
        detail = ", ".join(f"{c.value}={counts[c]}" for c in Category)
        print(f"  {split}: {sum(counts.values())} questions ({detail})")
    if report.ok:
        print("Dataset structurally valid. (Semantic correctness of labels is not verified here.)")
        return 0
    print(f"Dataset INVALID ({len(report.problems)} problem(s)):", file=sys.stderr)
    for problem in report.problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
