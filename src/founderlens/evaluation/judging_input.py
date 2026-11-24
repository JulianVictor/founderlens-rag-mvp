"""Verified, read-only input for formal judging (Stage 6, Commit 7.1; protocol §15).

Usage::

    python -m founderlens.evaluation.judging_input --run live-20261124T045739024562Z-8665d379

Loads one frozen live generation run (git-ignored records, report and run log),
the hash-checked development split and the verified V2 chunk collection, and
checks every link a judgment will rely on. Nothing is regenerated and nothing
is retrieved again: selected evidence is resolved by chunk identity against the
verified collection, answers are revalidated with the existing response
invariants, and citations are checked against the selected evidence.

The result keeps each question's execution outcome. Valid answers and local
abstentions are eligible for semantic judgment. Model-output failures stay
reliability outcomes: they are ineligible, get no claims, and are never treated
as abstentions. The input records keep their own generation protocol (1.3).

The verified corpus holds successful answer text in memory for later private
judging; :meth:`VerifiedRun.public_summary` and the CLI print only identifiers,
codes and counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import ValidationError

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.evaluation import completion as cp
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation.benchmark import DevSplit, EvaluationError, load_dev_split
from founderlens.evaluation.judgments import (
    ClaimAspectUnit,
    ClaimCitationUnit,
    ClaimDecomposition,
    ClaimUnit,
    ClaimValueUnit,
    Dimension,
    EvidenceKind,
    EvidenceReference,
    Judgment,
    RequiredFactUnit,
    SubclaimRef,
    SubclaimUnit,
    rubric_id,
)
from founderlens.generation.answer import GroundedAnswer, answer_problem_details
from founderlens.generation.context import EvidenceRecord, EvidenceRegistry
from founderlens.ingestion.chunk import ChunkRecord
from founderlens.retrieval.schema import ChunkCollection, RetrievalError

GENERATION_PROTOCOL: Final = cp.PROTOCOL_V13
ACCEPTED_STATES: Final = frozenset(
    {cp.CompletionState.ANSWER_COMPLETE.value, cp.CompletionState.EXECUTION_COMPLETE.value}
)
ELIGIBLE: Final = frozenset(
    {cp.ExecutionOutcome.VALID_ANSWER.value, cp.ExecutionOutcome.LOCAL_ABSTENTION.value}
)
ANSWER_LEVEL: Final = frozenset(
    {
        Dimension.ANSWER_COMPLETENESS,
        Dimension.ABSTENTION_APPROPRIATENESS,
        Dimension.ANSWER_RELEVANCE,
        Dimension.FACTUAL_CORRECTNESS,
    }
)


class JudgingInputError(RuntimeError):
    """The run cannot be used as judging input. ``problems`` never quotes content."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems[:8]))


@dataclass(frozen=True, slots=True)
class VerifiedEvidence:
    evidence_id: str
    rank: int
    chunk_id: str
    document_id: str
    page_number: int


@dataclass(frozen=True, slots=True)
class VerifiedCitation:
    number: int
    evidence_id: str
    chunk_id: str
    document_id: str
    page_number: int


@dataclass(frozen=True, slots=True)
class VerifiedClaim:
    position: int
    claim_type: str
    evidence_ids: tuple[str, ...]
    citation_numbers: tuple[int, ...]
    text_sha256: str
    text: str = field(repr=False)  # private: never in public output


@dataclass(frozen=True, slots=True)
class VerifiedQuestion:
    question_id: str
    category: str
    answerable: bool
    record_sha256: str
    execution_outcome: str
    status: str | None
    eligible: bool
    ineligibility_reason: str | None
    failure_codes: tuple[str, ...]
    required_fact_ids: tuple[str, ...]
    claims: tuple[VerifiedClaim, ...]
    citations: tuple[VerifiedCitation, ...]
    selected_evidence: tuple[VerifiedEvidence, ...]


@dataclass(frozen=True, slots=True)
class VerifiedRun:
    run_id: str
    generation_commit: str
    generation_protocol: str
    completion_state: str
    identities: Mapping[str, str]
    dataset_sha256: str
    split_sha256: str
    collection_artifact_sha256: str
    questions: tuple[VerifiedQuestion, ...]

    def question(self, question_id: str) -> VerifiedQuestion | None:
        return next((q for q in self.questions if q.question_id == question_id), None)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for q in self.questions:
            out[q.execution_outcome] = out.get(q.execution_outcome, 0) + 1
        return dict(sorted(out.items()))

    def public_summary(self) -> dict[str, Any]:
        """Identifiers, codes and counts only."""
        return {
            "run_id": self.run_id,
            "generation_commit": self.generation_commit,
            "generation_protocol": self.generation_protocol,
            "completion_state": self.completion_state,
            "rubric_id": rubric_id(),
            "scheduled_questions": len(self.questions),
            "eligible_for_semantic_judgment": sum(q.eligible for q in self.questions),
            "outcomes": self.counts(),
            "ineligible": [
                {
                    "question_id": q.question_id,
                    "category": q.category,
                    "reason": q.ineligibility_reason,
                    "failure_codes": list(q.failure_codes),
                }
                for q in self.questions
                if not q.eligible
            ],
            "claims": sum(len(q.claims) for q in self.questions),
            "citations": sum(len(q.citations) for q in self.questions),
            "collection_artifact_sha256": self.collection_artifact_sha256,
            "identities": dict(self.identities),
        }


# --------------------------------------------------------------------------- #
# Run verification
# --------------------------------------------------------------------------- #


def _verify_report(
    report: Mapping[str, Any], run_id: str, split: DevSplit, expected: Mapping[str, str]
) -> list[str]:
    problems = []
    checks = {
        "run_id": report.get("run_id") == run_id,
        "live mode": report.get("mode") == "live",
        "live origin": report.get("origin_mode") == "live" and report.get("synthetic") is False,
        "origin run": report.get("origin_run_id") == run_id,
        "generation protocol 1.3": report.get("protocol_version") == GENERATION_PROTOCOL,
        "accepted completion state": report.get("completion_state") in ACCEPTED_STATES,
        "reportable": (report.get("reportability") or {}).get(
            "reportable_as_hosted_model_diagnostic"
        )
        is True,
        "frozen identities": report.get("identities") == dict(expected),
        "dataset hash": (report.get("dataset") or {}).get("dataset_sha256") == split.dataset_sha256,
        "split hash": (report.get("dataset") or {}).get("split_sha256") == split.split_sha256,
    }
    problems += [f"report check failed: {name}" for name, ok in checks.items() if not ok]
    git = (report.get("operational") or {}).get("git") or {}
    if not git.get("commit") or git.get("working_tree_clean") is not True:
        problems.append("report check failed: generation commit on a clean tree")
    return problems


def _verify_run_log(
    entries: Sequence[Mapping[str, Any]], run_id: str, report: Mapping[str, Any]
) -> list[str]:
    mine = [e for e in entries if e.get("run_id") == run_id]
    pre = [e for e in mine if e.get("event") == "preregistered"]
    fin = [e for e in mine if e.get("event") == "finished"]
    if len(pre) != 1 or len(fin) != 1:
        return ["run log must hold exactly one preregistered and one finished entry"]
    problems = []
    git = (report.get("operational") or {}).get("git")
    if pre[0].get("git") != git:
        problems.append("run log preregistration names a different commit or tree state")
    if pre[0].get("identities") != report.get("identities"):
        problems.append("run log preregistration names different identities")
    if fin[0].get("completion_state") != report.get("completion_state"):
        problems.append("run log finish entry disagrees with the report")
    return problems


def _resolve_chunk(
    item: Mapping[str, Any], chunks: Mapping[str, ChunkRecord], where: str, problems: list[str]
) -> ChunkRecord | None:
    chunk = chunks.get(str(item.get("chunk_id")))
    if chunk is None:
        problems.append(f"{where}: chunk not in the verified collection")
        return None
    if (chunk.document_id, chunk.page_number) != (item.get("document_id"), item.get("page_number")):
        problems.append(f"{where}: document or page differs from the verified chunk")
        return None
    return chunk


def _verify_valid_answer(
    record: Mapping[str, Any], chunks: Mapping[str, ChunkRecord], problems: list[str]
) -> tuple[tuple[VerifiedClaim, ...], tuple[VerifiedCitation, ...], tuple[VerifiedEvidence, ...]]:
    qid = record["question_id"]
    evidence: list[VerifiedEvidence] = []
    registry_records: list[EvidenceRecord] = []
    for item in record.get("selected_evidence") or []:
        chunk = _resolve_chunk(item, chunks, f"{qid} {item.get('evidence_id')}", problems)
        if chunk is None:
            continue
        evidence.append(
            VerifiedEvidence(
                str(item["evidence_id"]),
                int(item["rank"]),
                chunk.chunk_id,
                chunk.document_id,
                chunk.page_number,
            )
        )
        # score is not part of any invariant check; the frozen record does not store it
        registry_records.append(
            EvidenceRecord(str(item["evidence_id"]), int(item["rank"]), 0.0, chunk)
        )
    if record.get("selected_evidence_source") != "prepared_input":
        problems.append(f"{qid}: selected evidence was not recorded from the prepared input")
    try:
        registry = EvidenceRegistry(registry_records)
        answer = GroundedAnswer.model_validate(record.get("answer"))
    except (ValueError, ValidationError):
        problems.append(f"{qid}: stored answer or evidence order fails validation")
        return (), (), tuple(evidence)
    if answer_problem_details(answer, registry):
        problems.append(f"{qid}: stored answer breaks the response invariants")
    if answer.status.value != record.get("status"):
        problems.append(f"{qid}: answer status differs from the record status")
    by_id = {e.evidence_id: e for e in evidence}
    citations: list[VerifiedCitation] = []
    for c in record.get("citations") or []:
        ev = by_id.get(str(c.get("evidence_id")))
        if ev is None or (ev.chunk_id, ev.document_id, ev.page_number) != (
            c.get("chunk_id"),
            c.get("document_id"),
            c.get("page_number"),
        ):
            problems.append(f"{qid}: citation {c.get('number')} does not match selected evidence")
            continue
        citations.append(
            VerifiedCitation(
                int(c["number"]), ev.evidence_id, ev.chunk_id, ev.document_id, ev.page_number
            )
        )
    if [c.number for c in citations] != list(range(1, len(citations) + 1)):
        problems.append(f"{qid}: citation numbers are not 1..n")
    by_number = {c.number: c for c in citations}
    claims: list[VerifiedClaim] = []
    stored = record.get("claims") or []
    if len(stored) != len(answer.claims):
        problems.append(f"{qid}: stored claims differ from the stored answer")
    for position, (claim, meta) in enumerate(zip(answer.claims, stored, strict=False), start=1):
        numbers = tuple(int(n) for n in meta.get("citation_numbers", []))
        cited = tuple(by_number[n].evidence_id for n in numbers if n in by_number)
        if (
            meta.get("position") != position
            or meta.get("claim_type") != claim.claim_type.value
            or tuple(meta.get("evidence_ids", [])) != claim.evidence_ids
            or cited != claim.evidence_ids
        ):
            problems.append(f"{qid}: claim {position} association differs from its citations")
        claims.append(
            VerifiedClaim(
                position,
                claim.claim_type.value,
                claim.evidence_ids,
                numbers,
                hashlib.sha256(claim.text.encode("utf-8")).hexdigest(),
                claim.text,
            )
        )
    return tuple(claims), tuple(citations), tuple(evidence)


def verify_generation_run(
    *,
    run_id: str,
    records: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    run_log: Sequence[Mapping[str, Any]],
    split: DevSplit,
    collection: ChunkCollection,
    expected_identities: Mapping[str, str] = ge.EXPECTED_IDENTITIES,
) -> VerifiedRun:
    """Verify every link a judgment relies on, or raise :class:`JudgingInputError`."""
    problems = _verify_report(report, run_id, split, expected_identities)
    problems += _verify_run_log(run_log, run_id, report)
    if collection.artifact_sha256 != expected_identities.get("collection_artifact_sha256"):
        problems.append("source collection is not the frozen V2 artifact")
    questions = {q.question_id: q for q in split.questions}
    ids = [str(r.get("question_id")) for r in records]
    if sorted(ids) != sorted(questions) or len(set(ids)) != len(ids):
        problems.append("records do not account for every scheduled question exactly once")
    chunks = {c.chunk_id: c for c in collection.chunks}
    verified: list[VerifiedQuestion] = []
    for record in records:
        qid = str(record.get("question_id"))
        where = f"record {qid}"
        record_checks = {
            "record schema 3": record.get("record_schema_version") == 3,
            "record digest": ge.is_sealed(record),
            "run id": record.get("run_id") == run_id and record.get("origin_run_id") == run_id,
            "live origin": record.get("mode") == "live"
            and record.get("origin_mode") == "live"
            and record.get("synthetic") is False,
            "generation protocol 1.3": record.get("protocol_version") == GENERATION_PROTOCOL,
            "frozen identities": record.get("identities") == dict(expected_identities),
            "execution outcome": record.get("execution_outcome")
            == cp.execution_outcome(record).value,
        }
        problems += [
            f"{where}: {name} check failed" for name, ok in record_checks.items() if not ok
        ]
        question = questions.get(qid)
        if question is None:
            continue
        if (
            record.get("category") != question.category.value
            or bool(record.get("answerable")) is not question.answerable
        ):
            problems.append(f"{where}: category or answerability differs from the frozen split")
        outcome = str(record.get("execution_outcome"))
        claims: tuple[VerifiedClaim, ...] = ()
        citations: tuple[VerifiedCitation, ...] = ()
        evidence: tuple[VerifiedEvidence, ...] = ()
        failure_codes: tuple[str, ...] = ()
        reason = None
        if outcome == cp.ExecutionOutcome.VALID_ANSWER.value:
            claims, citations, evidence = _verify_valid_answer(record, chunks, problems)
        elif outcome == cp.ExecutionOutcome.LOCAL_ABSTENTION.value:
            if record.get("claims") or record.get("citations"):
                problems.append(f"{where}: a local abstention has claims or citations")
        elif outcome == cp.ExecutionOutcome.MODEL_OUTPUT_FAILURE.value:
            reason = "model_output_failure"
            failure = record.get("failure") or {}
            failure_codes = tuple(str(p.get("code")) for p in failure.get("problems", []))
            if record.get("answer") is not None or record.get("claims"):
                problems.append(f"{where}: a failed record carries an answer")
            if record.get("provider_attempted") is not True:
                problems.append(f"{where}: a model-output failure without a provider attempt")
        else:
            problems.append(f"{where}: outcome {outcome} is not admissible for judging")
        verified.append(
            VerifiedQuestion(
                question_id=qid,
                category=question.category.value,
                answerable=question.answerable,
                record_sha256=str(record.get("record_sha256")),
                execution_outcome=outcome,
                status=record.get("status"),
                eligible=outcome in ELIGIBLE,
                ineligibility_reason=reason,
                failure_codes=failure_codes,
                required_fact_ids=tuple(f.fact_id for f in question.required_facts),
                claims=claims,
                citations=citations,
                selected_evidence=evidence,
            )
        )
    if problems:
        raise JudgingInputError(problems)
    git = report["operational"]["git"]
    return VerifiedRun(
        run_id=run_id,
        generation_commit=str(git["commit"]),
        generation_protocol=GENERATION_PROTOCOL,
        completion_state=str(report["completion_state"]),
        identities=dict(expected_identities),
        dataset_sha256=split.dataset_sha256,
        split_sha256=split.split_sha256,
        collection_artifact_sha256=collection.artifact_sha256,
        questions=tuple(sorted(verified, key=lambda q: q.question_id)),
    )


# --------------------------------------------------------------------------- #
# Evidence references and judgment binding
# --------------------------------------------------------------------------- #


def span_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_reference(
    collection: ChunkCollection,
    chunk_id: str,
    start: int,
    end: int,
    kind: EvidenceKind,
    evidence_id: str | None = None,
) -> EvidenceReference:
    """A reference to ``[start, end)`` of a frozen chunk, pinned by the span hash."""
    chunk = next((c for c in collection.chunks if c.chunk_id == chunk_id), None)
    if chunk is None:
        raise JudgingInputError(["chunk not in the verified collection"])
    return EvidenceReference(
        kind=kind,
        evidence_id=evidence_id,
        document_id=chunk.document_id,
        page_number=chunk.page_number,
        chunk_id=chunk.chunk_id,
        collection_artifact_sha256=collection.artifact_sha256,
        start=start,
        end=end,
        span_sha256=span_sha256(chunk.text[start:end]),
    )


def _on_base_character(text: str, index: int) -> bool:
    return index == len(text) or unicodedata.combining(text[index]) == 0


def reference_problems(
    ref: EvidenceReference,
    question: VerifiedQuestion,
    collection: ChunkCollection,
    chunks: Mapping[str, ChunkRecord] | None = None,
) -> list[str]:
    """Problems resolving one reference against the frozen chunk text (no text quoted)."""
    lookup = chunks if chunks is not None else {c.chunk_id: c for c in collection.chunks}
    problems = []
    if ref.collection_artifact_sha256 != collection.artifact_sha256:
        problems.append("reference names a different source artifact")
    chunk = lookup.get(ref.chunk_id)
    if chunk is None:
        return [*problems, "reference chunk not in the verified collection"]
    if (chunk.document_id, chunk.page_number) != (ref.document_id, ref.page_number):
        problems.append("reference document or page differs from its chunk")
    if ref.kind is EvidenceKind.SELECTED_CONTEXT:
        selected = {e.evidence_id: e for e in question.selected_evidence}
        ev = selected.get(str(ref.evidence_id))
        if ev is None or ev.chunk_id != ref.chunk_id:
            problems.append("reference evidence ID does not name this chunk in the request")
    elif ref.kind is EvidenceKind.CITED:
        cited = {c.evidence_id: c for c in question.citations}.get(str(ref.evidence_id))
        if cited is None or cited.chunk_id != ref.chunk_id:
            problems.append("cited reference does not name a chunk this answer cites")
    # reference_source: any frozen chunk, but it never counts as retrieved or cited
    if ref.end > len(chunk.text):
        problems.append("reference offsets outside the chunk text")
    elif not (
        _on_base_character(chunk.text, ref.start) and _on_base_character(chunk.text, ref.end)
    ):
        problems.append("reference offsets split a combining character sequence")
    elif span_sha256(chunk.text[ref.start : ref.end]) != ref.span_sha256:
        problems.append("reference span hash does not match the chunk text")
    return problems


def check_judgment(
    judgment: Judgment,
    run: VerifiedRun,
    collection: ChunkCollection,
    decompositions: Mapping[str, ClaimDecomposition] | None = None,
) -> list[str]:
    """Bind a schema-valid judgment to the verified run; empty when it binds cleanly.

    Subclaim units bind only through their registered decomposition, which must belong
    to the same question, record and claim, and pin that claim's exact text hash.
    """
    question = run.question(judgment.question_id)
    if question is None:
        return ["judgment names a question outside the verified run"]
    problems = []
    if judgment.generation_run_id != run.run_id:
        problems.append("judgment names a different generation run")
    if judgment.generation_record_sha256 != question.record_sha256:
        problems.append("judgment names a different generation record")
    if not question.eligible:
        problems.append(
            f"question is ineligible for semantic judgment: {question.ineligibility_reason}"
        )
        return problems
    unit = judgment.unit
    claims = {c.position: c for c in question.claims}
    if (
        isinstance(unit, ClaimUnit | SubclaimUnit | ClaimAspectUnit | ClaimValueUnit)
        and unit.claim_position not in claims
    ):
        problems.append("judgment names a claim that does not exist")
    if isinstance(unit, SubclaimUnit | ClaimCitationUnit) and unit.subclaim is not None:
        problems += _subclaim_problems(
            unit.subclaim, judgment, claims.get(unit.claim_position), decompositions or {}
        )
    if isinstance(unit, ClaimCitationUnit):
        claim = claims.get(unit.claim_position)
        cited = {c.number: c for c in question.citations}.get(unit.citation_number)
        if claim is None or cited is None or unit.citation_number not in claim.citation_numbers:
            problems.append("judgment names a citation the claim does not carry")
        elif cited.evidence_id != unit.evidence_id:
            problems.append("judgment citation evidence ID differs from the cited record")
    if isinstance(unit, RequiredFactUnit | ClaimValueUnit):
        fact = unit.fact_id
        if fact is not None and fact not in question.required_fact_ids:
            problems.append("judgment names a required fact of another question")
    if not question.claims and judgment.dimension not in ANSWER_LEVEL:
        problems.append("an answer without claims takes answer-level dimensions only")
    chunks = {c.chunk_id: c for c in collection.chunks}
    for ref in judgment.evidence:
        problems += reference_problems(ref, question, collection, chunks)
    if judgment.dimension is Dimension.CITATION_SUPPORT and isinstance(unit, ClaimCitationUnit):
        cited = {c.number: c for c in question.citations}.get(unit.citation_number)
        if cited is not None and any(r.chunk_id != cited.chunk_id for r in judgment.evidence):
            problems.append("citation_support evidence must be spans of the cited chunk")
    return problems


def _subclaim_problems(
    sub: SubclaimRef,
    judgment: Judgment,
    claim: VerifiedClaim | None,
    decompositions: Mapping[str, ClaimDecomposition],
) -> list[str]:
    decomposition = decompositions.get(sub.decomposition_id)
    if decomposition is None or decomposition.decomposition_id != sub.decomposition_id:
        return ["subclaim judgment needs its registered decomposition"]
    problems = []
    if (decomposition.question_id, decomposition.generation_record_sha256) != (
        judgment.question_id,
        judgment.generation_record_sha256,
    ) or (claim is not None and decomposition.claim_position != claim.position):
        problems.append("decomposition belongs to another question, record or claim")
    if claim is not None and decomposition.claim_text_sha256 != claim.text_sha256:
        problems.append("decomposition does not pin the original claim text")
    elif claim is not None:
        try:
            decomposition.check_against(claim.text)
        except ValueError:
            problems.append("decomposition spans do not fit the original claim text")
    if sub.index > len(decomposition.subclaims):
        problems.append("subclaim index outside the decomposition")
    return problems


# --------------------------------------------------------------------------- #
# Loading and CLI
# --------------------------------------------------------------------------- #


def load_verified_run(settings: Settings, run_id: str) -> tuple[VerifiedRun, ChunkCollection]:
    from founderlens.retrieval.collections import load_named_collection

    directory = settings.reports_dir / ge.REPORT_SUBDIR
    try:
        report = json.loads((directory / f"{run_id}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise JudgingInputError(["cannot read the run report"]) from None
    records = ge.load_records(directory / f"{run_id}.records.jsonl")
    entries, skipped = cp.read_run_log(directory / cp.RUN_LOG_FILENAME)
    split = load_dev_split(settings.evaluation_dir, settings.corpus_manifest_path)
    collection = load_named_collection(settings, "v2")
    if skipped:
        raise JudgingInputError(["the run log has unreadable lines"])
    run = verify_generation_run(
        run_id=run_id,
        records=records,
        report=report,
        run_log=entries,
        split=split,
        collection=collection,
    )
    return run, collection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.judging_input",
        description="Verify a frozen live generation run as formal-judging input (offline).",
    )
    parser.add_argument("--run", required=True, help="live run ID under reports/generation_eval")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    try:
        run, _ = load_verified_run(load_settings(args.project_root), args.run)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (JudgingInputError, ge.EvaluationRunError, EvaluationError, RetrievalError) as exc:
        print(f"Judging input verification FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(run.public_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
