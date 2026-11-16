"""Evaluation dataset parsing and structural validation.

Synthetic datasets are used for negative cases. The committed dataset is only
schema-validated here; tests never assert on held-out question content.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.evaluation import validate as ev
from founderlens.evaluation.schema import Category, RecordError, parse_jsonl, parse_record
from founderlens.evaluation.validate import (
    EXPECTED_DISTRIBUTION,
    check_question,
    normalize_question,
    validate_dataset,
)
from tests.conftest import DOC_A, DOC_B

Record = dict[str, Any]
PAGES = {DOC_A: 2, DOC_B: 3}


def _record(qid: str, category: Category) -> Record:
    rec: Record = {
        "question_id": qid,
        "question": f"Synthetic question {qid} ({category.value})?",
        "category": category.value,
        "answerable": category is not Category.UNANSWERABLE,
        "reference_answer": f"Synthetic answer for {qid}.",
        "required_facts": [{"fact_id": f"{qid}-f1", "statement": "Fact one."}],
        "accepted_evidence": [{"fact_id": f"{qid}-f1", "document_id": DOC_A, "page_number": 1}],
        "expected_abstention_reason": None,
    }
    if category in (Category.CROSS_DOC_COMPARISON, Category.TEMPORAL_MULTI_EVIDENCE):
        rec["required_facts"].append({"fact_id": f"{qid}-f2", "statement": "Fact two."})
        second_doc = DOC_B if category is Category.CROSS_DOC_COMPARISON else DOC_A
        rec["accepted_evidence"].append(
            {"fact_id": f"{qid}-f2", "document_id": second_doc, "page_number": 2}
        )
    if category is Category.UNANSWERABLE:
        rec.update(
            reference_answer=None,
            required_facts=[],
            accepted_evidence=[],
            expected_abstention_reason="The corpus does not report this figure.",
        )
    return rec


def _split(split: str) -> list[Record]:
    records: list[Record] = []
    for category, count in EXPECTED_DISTRIBUTION[split].items():
        for _ in range(count):
            records.append(_record(f"{split}-{len(records) + 1:03d}", category))
    return records


Writer = Callable[..., Path]


@pytest.fixture
def write_dataset(make_project: Callable[..., Path]) -> Writer:
    """Write a synthetic dataset into a synthetic project; returns the project root."""

    def writer(
        dev: list[Record] | None = None,
        test: list[Record] | None = None,
        *,
        raw_lines: dict[str, str] | None = None,
        rehash: bool = True,
    ) -> Path:
        root = make_project(write_files=False)
        eval_dir = root / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        contents = {
            "questions.dev.jsonl": "".join(json.dumps(r) + "\n" for r in (dev or _split("dev"))),
            "questions.test.jsonl": "".join(json.dumps(r) + "\n" for r in (test or _split("test"))),
        }
        contents.update(raw_lines or {})
        files: dict[str, Any] = {}
        for name, text in contents.items():
            (eval_dir / name).write_text(text, encoding="utf-8", newline="\n")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if rehash else "0" * 64
            files[name] = {"sha256": digest}
        meta = {"dataset_version": "0.0.1", "annotation_status": "synthetic", "files": files}
        (eval_dir / "dataset.json").write_text(json.dumps(meta), encoding="utf-8")
        return root

    return writer


def _problems(root: Path) -> list[str]:
    return validate_dataset(root / "evaluation", root / "data" / "manifest.json").problems


def _check(rec: Record) -> list[str]:
    return check_question(parse_record(rec), PAGES)


# --- record parsing -------------------------------------------------------------------


def test_valid_records_parse_and_check_clean() -> None:
    for category in Category:
        rec = _record("dev-001", category)
        question = parse_record(rec)
        assert question.category is category
        assert question.split == "dev"
        assert _check(rec) == []


def test_numeric_fact_parses() -> None:
    rec = _record("dev-001", Category.SINGLE_DOC_FACTUAL)
    rec["required_facts"][0]["numeric"] = {
        "company": "Example",
        "fiscal_period": "FY2024 Q1",
        "metric": "revenue",
        "value": 123,
        "unit": "USD millions",
        "basis": "GAAP",
    }
    numeric = parse_record(rec).required_facts[0].numeric
    assert numeric is not None
    assert numeric.value == 123.0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("question"), "missing ['question']"),
        (lambda r: r.update(extra=1), "unknown ['extra']"),
        (lambda r: r.update(category="trivia"), "invalid category"),
        (lambda r: r.update(answerable="yes"), "'answerable' must be true or false"),
        (lambda r: r.update(question_id="q1"), "malformed question_id"),
        (lambda r: r["accepted_evidence"][0].update(page_number="1"), "must be an integer"),
        (lambda r: r["required_facts"][0].update(fact_id="f1"), "malformed fact_id"),
    ],
)
def test_structural_record_errors(mutate: Callable[[Record], object], message: str) -> None:
    rec = _record("dev-001", Category.SINGLE_DOC_FACTUAL)
    mutate(rec)
    with pytest.raises(RecordError, match=message.replace("[", r"\[").replace("]", r"\]")):
        parse_record(rec)


def test_numeric_unit_and_basis_validated() -> None:
    rec = _record("dev-001", Category.SINGLE_DOC_FACTUAL)
    rec["required_facts"][0]["numeric"] = {
        "company": "E",
        "fiscal_period": "Q1",
        "metric": "m",
        "value": 1,
        "unit": "furlongs",
        "basis": "GAAP",
    }
    with pytest.raises(RecordError, match="unit"):
        parse_record(rec)


def test_invalid_jsonl_lines_reported_with_line_numbers() -> None:
    good = json.dumps(_record("dev-001", Category.SINGLE_DOC_FACTUAL))
    records, problems = parse_jsonl(f"{good}\n{{broken\n\n", source="x.jsonl")
    assert len(records) == 1
    assert problems == [
        "x.jsonl:2: invalid JSON (Expecting property name enclosed in double quotes)",
        "x.jsonl:3: blank line",
    ]


# --- per-question consistency ---------------------------------------------------------


def test_unknown_document_reference() -> None:
    rec = _record("dev-001", Category.SINGLE_DOC_FACTUAL)
    rec["accepted_evidence"][0]["document_id"] = "zzz-2099-q9"
    assert any("unknown document" in p for p in _check(rec))


@pytest.mark.parametrize("page", [0, -1, 3])
def test_out_of_range_page(page: int) -> None:
    rec = _record("dev-001", Category.SINGLE_DOC_FACTUAL)  # DOC_A has 2 pages
    rec["accepted_evidence"][0]["page_number"] = page
    assert any("out of range" in p for p in _check(rec))


def test_required_fact_without_evidence() -> None:
    rec = _record("dev-001", Category.SINGLE_DOC_FACTUAL)
    rec["required_facts"].append({"fact_id": "dev-001-f2", "statement": "Uncovered."})
    assert "dev-001: required fact dev-001-f2 has no accepted evidence" in _check(rec)


def test_evidence_for_unknown_fact() -> None:
    rec = _record("dev-001", Category.SINGLE_DOC_FACTUAL)
    rec["accepted_evidence"].append(
        {"fact_id": "dev-001-f9", "document_id": DOC_A, "page_number": 1}
    )
    assert any("unknown fact_id dev-001-f9" in p for p in _check(rec))


def test_duplicate_evidence_reference() -> None:
    rec = _record("dev-001", Category.SINGLE_DOC_FACTUAL)
    rec["accepted_evidence"].append(dict(rec["accepted_evidence"][0]))
    assert any("duplicate evidence" in p for p in _check(rec))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_answer", "Some answer", "reference_answer null"),
        ("required_facts", [{"fact_id": "dev-001-f1", "statement": "x"}], "no required_facts"),
        (
            "accepted_evidence",
            [{"fact_id": "dev-001-f1", "document_id": DOC_A, "page_number": 1}],
            "no accepted_evidence",
        ),
        ("expected_abstention_reason", None, "needs expected_abstention_reason"),
        ("answerable", True, "inconsistent with category"),
    ],
)
def test_invalid_unanswerable_annotations(field: str, value: object, message: str) -> None:
    rec = _record("dev-001", Category.UNANSWERABLE)
    rec[field] = value
    assert any(message in p for p in _check(rec))


def test_answerable_question_rules() -> None:
    rec = _record("dev-001", Category.SINGLE_DOC_FACTUAL)
    rec.update(reference_answer=None, expected_abstention_reason="should not be set")
    problems = _check(rec)
    assert any("needs a non-empty reference_answer" in p for p in problems)
    assert any("expected_abstention_reason null" in p for p in problems)


def test_cross_document_question_needs_two_documents() -> None:
    rec = _record("dev-001", Category.CROSS_DOC_COMPARISON)
    rec["accepted_evidence"][1]["document_id"] = DOC_A
    assert any("at least two documents" in p for p in _check(rec))


def test_cross_document_question_rejected_when_one_document_covers_all_facts() -> None:
    rec = _record("dev-001", Category.CROSS_DOC_COMPARISON)
    # Fact 1 is also supported by DOC_B, so DOC_B alone answers the question.
    rec["accepted_evidence"].append(
        {"fact_id": "dev-001-f1", "document_id": DOC_B, "page_number": 1}
    )
    assert any("answerable from a single document (bbb-2024-q1)" in p for p in _check(rec))


def test_temporal_question_needs_multiple_facts() -> None:
    rec = _record("dev-001", Category.TEMPORAL_MULTI_EVIDENCE)
    rec["required_facts"].pop()
    rec["accepted_evidence"].pop()
    assert any("at least two required facts" in p for p in _check(rec))


def test_normalized_duplicate_detection_ignores_case_and_punctuation() -> None:
    assert normalize_question("What was NVIDIA's revenue?") == normalize_question(
        "what was nvidia s   REVENUE"
    )


# --- whole-dataset validation ---------------------------------------------------------


def test_valid_synthetic_dataset_passes(write_dataset: Writer) -> None:
    root = write_dataset()
    report = validate_dataset(root / "evaluation", root / "data" / "manifest.json")
    assert report.problems == []
    assert sum(report.counts["dev"].values()) == 24
    assert sum(report.counts["test"].values()) == 16


def test_incorrect_distribution_rejected(write_dataset: Writer) -> None:
    dev = _split("dev")
    dev[-1] = _record("dev-024", Category.SINGLE_DOC_FACTUAL)  # was unanswerable
    problems = _problems(write_dataset(dev=dev))
    assert "questions.dev.jsonl: expected 7 single_doc_factual questions, found 8" in problems
    assert "questions.dev.jsonl: expected 5 unanswerable questions, found 4" in problems


def test_duplicate_question_ids_across_splits_rejected(write_dataset: Writer) -> None:
    test = _split("test")
    test[0]["question_id"] = "dev-001"
    for fact in test[0]["required_facts"]:
        fact["fact_id"] = fact["fact_id"].replace("test-001", "dev-001")
    for ref in test[0]["accepted_evidence"]:
        ref["fact_id"] = ref["fact_id"].replace("test-001", "dev-001")
    problems = _problems(write_dataset(test=test))
    assert "duplicate question_id dev-001" in problems
    assert "dev-001: belongs in the dev file" in problems


def test_duplicate_question_text_across_splits_rejected(write_dataset: Writer) -> None:
    dev, test = _split("dev"), _split("test")
    test[0]["question"] = dev[0]["question"].upper() + "!!"
    assert "duplicate question text: dev-001, test-001" in _problems(write_dataset(dev, test))


def test_invalid_jsonl_rejected(write_dataset: Writer) -> None:
    root = write_dataset(raw_lines={"questions.test.jsonl": "{oops\n"})
    problems = _problems(root)
    assert any(p.startswith("questions.test.jsonl:1: invalid JSON") for p in problems)


def test_silent_modification_detected_by_hash(write_dataset: Writer) -> None:
    root = write_dataset(rehash=False)
    assert any("SHA-256 differs" in p for p in _problems(root))


def test_cli_output_never_contains_question_or_answer_text(
    write_dataset: Writer, capsys: pytest.CaptureFixture[str]
) -> None:
    root = write_dataset()
    assert ev.main(["--project-root", str(root)]) == 0
    captured = capsys.readouterr()
    assert "Synthetic question" not in captured.out + captured.err
    assert "Synthetic answer" not in captured.out + captured.err
    assert "dev: 24 questions" in captured.out


def test_cli_fails_on_invalid_dataset(
    write_dataset: Writer, capsys: pytest.CaptureFixture[str]
) -> None:
    root = write_dataset(rehash=False)
    assert ev.main(["--project-root", str(root)]) == 1
    assert "Dataset INVALID" in capsys.readouterr().err


# --- committed dataset (schema validation only) ---------------------------------------


def test_committed_dataset_is_structurally_valid() -> None:
    settings = load_settings()
    report = validate_dataset(settings.evaluation_dir, settings.corpus_manifest_path)
    assert report.problems == []
    assert {split: sum(c.values()) for split, c in report.counts.items()} == {"dev": 24, "test": 16}
    for split, expected in EXPECTED_DISTRIBUTION.items():
        assert dict(report.counts[split]) == dict(expected)


def test_committed_dataset_metadata_marks_candidate_status() -> None:
    meta = json.loads((load_settings().evaluation_dir / "dataset.json").read_text(encoding="utf-8"))
    assert meta["annotation_status"].startswith("candidate")
    assert meta["corpus_version"] == "1.0.0"
