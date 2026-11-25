"""Private, offline human annotation (Stage 6, Commit 7.3; protocol §17).

Usage (every command is offline; nothing calls a model or the network)::

    python -m founderlens.evaluation.annotation packets   --run <run_id>
    python -m founderlens.evaluation.annotation worksheet --run <run_id> --question dev-001
        --annotator ann-a1   (one command; pseudonymous annotator ID)
    python -m founderlens.evaluation.annotation validate  --run <run_id> --worksheet <path>
    python -m founderlens.evaluation.annotation import    --run <run_id> --worksheet <path>
    python -m founderlens.evaluation.annotation status    --run <run_id>
    python -m founderlens.evaluation.annotation history   --run <run_id> --question dev-001

The verified judging-input gate (:mod:`founderlens.evaluation.judging_input`)
supplies the frozen run. For every *eligible* answer a private **packet**
holds what a reviewer needs: the question, its frozen category and
answerability, the response status, claims, limitations, the evidence the
generator received (``selected_context``), the evidence each claim cites
(``cited``), the trusted pages of the dataset's required facts
(``reference_source``) with their chunk text, and the existing deterministic
scope and financial diagnostics. Model-output failures get no packet.

A reviewer copies nothing by hand: ``worksheet`` writes a per-annotator JSON
**worksheet** in which every unit is listed with ``review_action: "pending"``
and no label. Nothing is pre-labelled: a citation does not make a claim
supported, and ``insufficient_evidence`` does not make an abstention correct.
The reviewer fills entries and sets ``review_action`` to ``confirmed`` (or
``unresolved`` for a reference fact that cannot be verified), saving as often
as they like. ``import`` validates the whole worksheet and appends only
confirmed, new or revised entries to the append-only **ledger**; pending
entries are skipped, so partial work can be imported repeatedly.

Every entry is bound to the run, record digest, packet digest, rubric,
extractor and checker identities and a pseudonymous annotator ID
(``ann-…``). A changed label must name the entry it supersedes and a reason;
history is never overwritten. Local digests detect accidental edits; they do
not make the files tamper-proof.

All files live under ``reports/annotation/<run_id>/`` (git-ignored) and are
private. CLI output and :func:`ensure_public` summaries hold only counts,
identifiers and status codes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.evaluation import financial_checks as fc
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation import judging_input as ji
from founderlens.evaluation.benchmark import DevSplit, EvaluationError, load_dev_split
from founderlens.evaluation.judgments import (
    AttributionAspect,
    ClaimDecomposition,
    ClaimValueUnit,
    Dimension,
    EvidenceKind,
    EvidenceReference,
    EvidenceSufficiency,
    Judgment,
    NotAssessableReason,
    ReferenceBasis,
    Role,
    SubclaimSpan,
    rubric_id,
)
from founderlens.evaluation.judgments import Unit as JudgmentUnit
from founderlens.evaluation.schema import EvalQuestion
from founderlens.guardrails.scope import CorpusVocabulary
from founderlens.ingestion.chunk import ChunkRecord
from founderlens.retrieval.schema import ChunkCollection, RetrievalError

if TYPE_CHECKING:
    from founderlens.evaluation.partition import PartitionManifest

ANNOTATION_VERSION: Final = 1
PACKET_VERSION: Final = 1
WORKSHEET_VERSION: Final = 1
LEDGER_SCHEMA_VERSION: Final = 1
SUBDIR: Final = "annotation"
LEDGER_FILENAME: Final = "ledger.jsonl"

ANNOTATOR_PATTERN: Final = r"^ann-[a-z0-9]{2,16}$"  # pseudonyms only; never names or emails
RUN_PATTERN: Final = r"^live-[0-9A-Za-z-]{1,90}$"
QUESTION_PATTERN: Final = r"^dev-\d{3}$"  # held-out IDs are rejected by construction
FACT_PATTERN: Final = r"^dev-\d{3}-f\d+$"
ENTRY_ID_PATTERN: Final = r"^[0-9a-f]{24}$"
VARIANT_PATTERN: Final = r"^[0-9a-f]{16}$"
_HEX64: Final = r"^[0-9a-f]{64}$"
_HEX12: Final = r"^[0-9a-f]{12}$"
_SAFE: Final = r"^[A-Za-z0-9._:-]{1,64}$"

# Keys that hold (or could hold) private text; public summaries must never carry them.
PRIVATE_KEYS: Final = frozenset(
    {
        "text",
        "question",
        "statement",
        "rationale",
        "limitations",
        "variant_text",
        "passage",
        "span_text",
        "claim_text",
        "reference_answer",
        "expected_abstention_reason",
    }
)
PUBLIC_STRING_LIMIT: Final = 96
VALUE_KINDS: Final = frozenset(
    {fc.OccurrenceKind.FINANCIAL_QUANTITY, fc.OccurrenceKind.UNITLESS_NUMBER}
)


class AnnotationError(RuntimeError):
    """The annotation step must not proceed. ``problems`` never quotes content."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems[:12]))


class ReviewState(StrEnum):
    """Review state of one unit across reviewers (see :meth:`Ledger.resolution`)."""

    UNLABELED = "unlabeled"
    SINGLE_REVIEWED = "single_reviewed"
    INDEPENDENTLY_AGREED = "independently_agreed"
    DISPUTED = "disputed"
    ADJUDICATED = "adjudicated"
    WITHDRAWN = "withdrawn"


class RevisionReason(StrEnum):
    CORRECTION = "correction"
    RECONSIDERED_EVIDENCE = "reconsidered_evidence"
    RUBRIC_CLARIFICATION = "rubric_clarification"
    ENTRY_ERROR = "entry_error"


class ErrorType(StrEnum):
    """Optional human tag for *why* a negative label applies (calibration categories)."""

    WRONG_VALUE = "wrong_value"
    WRONG_SCALE = "wrong_scale"
    WRONG_CURRENCY = "wrong_currency"
    WRONG_COMPANY = "wrong_company"
    WRONG_PERIOD = "wrong_period"
    WRONG_METRIC = "wrong_metric"
    WRONG_BASIS = "wrong_basis"
    INCORRECT_CALCULATION = "incorrect_calculation"
    UNSUPPORTED_BY_EVIDENCE = "unsupported_by_evidence"
    IRRELEVANT_CITATION = "irrelevant_citation"


NEGATIVE_LABELS: Final = frozenset(
    {
        "unsupported",
        "contradicted",
        "does_not_support",
        "partially_supports",
        "incorrect",
        "missing",
        "false_abstention",
        "missed_abstention",
        "partial",
        "none",
        "off_topic",
        "partially_relevant",
    }
)


def digest(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def annotation_identity() -> dict[str, str]:
    return {
        "annotation_version": str(ANNOTATION_VERSION),
        "rubric_id": rubric_id(),
        "extractor_id": fc.extractor_id(),
        "checker_id": fc.checker_id(),
    }


# --------------------------------------------------------------------------- #
# Verified context
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AnnotationContext:
    """A verified run plus the frozen inputs a packet needs, bound to each other."""

    run: ji.VerifiedRun
    collection: ChunkCollection = field(repr=False)  # holds passage text
    split: DevSplit = field(repr=False)  # holds question text
    records: Mapping[str, Mapping[str, Any]] = field(repr=False)  # holds answer text
    vocabulary: CorpusVocabulary = field(repr=False)
    chunks: Mapping[str, ChunkRecord] = field(repr=False)

    def question(self, question_id: str) -> ji.VerifiedQuestion:
        q = self.run.question(question_id)
        if q is None:
            raise AnnotationError(["question outside the verified run"])
        return q

    def eval_question(self, question_id: str) -> EvalQuestion:
        return next(q for q in self.split.questions if q.question_id == question_id)

    def page_chunks(self, document_id: str, page_number: int) -> list[ChunkRecord]:
        return [
            c
            for c in self.collection.chunks
            if (c.document_id, c.page_number) == (document_id, page_number)
        ]


def make_context(
    run: ji.VerifiedRun,
    collection: ChunkCollection,
    split: DevSplit,
    records: Iterable[Mapping[str, Any]],
    vocabulary: CorpusVocabulary,
) -> AnnotationContext:
    """Bind the loaded records, split and collection to the verified run, or refuse."""
    by_id = {str(r.get("question_id")): r for r in records}
    problems = []
    if set(by_id) != {q.question_id for q in run.questions}:
        problems.append("records do not match the verified run's questions")
    for q in run.questions:
        record = by_id.get(q.question_id)
        if record is None or not ge.is_sealed(record):
            problems.append(f"{q.question_id}: record missing or not sealed")
        elif record.get("record_sha256") != q.record_sha256:
            problems.append(f"{q.question_id}: record differs from the verified run (stale)")
    if collection.artifact_sha256 != run.collection_artifact_sha256:
        problems.append("collection is not the verified run's source artifact")
    if (split.dataset_sha256, split.split_sha256) != (run.dataset_sha256, run.split_sha256):
        problems.append("development split differs from the verified run's split")
    if problems:
        raise AnnotationError(problems)
    return AnnotationContext(
        run=run,
        collection=collection,
        split=split,
        records=by_id,
        vocabulary=vocabulary,
        chunks={c.chunk_id: c for c in collection.chunks},
    )


def load_context(settings: Settings, run_id: str) -> AnnotationContext:
    """Load a frozen run through the verified judging-input gate. Read-only."""
    from founderlens.corpus.manifest import load_manifest

    if not re.fullmatch(RUN_PATTERN, run_id):
        raise AnnotationError(["run ID must name a live development run"])
    run, collection = ji.load_verified_run(settings, run_id)
    directory = settings.reports_dir / ge.REPORT_SUBDIR
    records = ge.load_records(directory / f"{run_id}.records.jsonl")
    split = load_dev_split(settings.evaluation_dir, settings.corpus_manifest_path)
    vocabulary = CorpusVocabulary.from_manifest(load_manifest(settings.corpus_manifest_path))
    return make_context(run, collection, split, records, vocabulary)


@dataclass(frozen=True, slots=True)
class AnnotationPaths:
    root: Path

    @property
    def packets(self) -> Path:
        return self.root / "packets"

    @property
    def worksheets(self) -> Path:
        return self.root / "worksheets"

    @property
    def ledger(self) -> Path:
        return self.root / LEDGER_FILENAME

    @property
    def calibration(self) -> Path:
        return self.root / "calibration"


def annotation_paths(settings: Settings, run_id: str) -> AnnotationPaths:
    if not re.fullmatch(RUN_PATTERN, run_id):
        raise AnnotationError(["run ID must name a live development run"])
    return AnnotationPaths(settings.reports_dir / SUBDIR / run_id)


# --------------------------------------------------------------------------- #
# Packets (private)
# --------------------------------------------------------------------------- #


def run_counts(ctx: AnnotationContext) -> dict[str, Any]:
    """Structural counts of the verified run (no text)."""
    eligible = [q for q in ctx.run.questions if q.eligible]
    financial = 0
    for q in eligible:
        for claim in q.claims:
            financial += sum(
                o.kind is fc.OccurrenceKind.FINANCIAL_QUANTITY
                for o in fc.extract_occurrences(claim.text, ctx.vocabulary)
            )
    return {
        "scheduled_outcomes": len(ctx.run.questions),
        "eligible_answers": len(eligible),
        "answers_with_claims": sum(bool(q.claims) for q in eligible),
        "answers_without_claims": sum(not q.claims for q in eligible),
        "claims": sum(len(q.claims) for q in eligible),
        "claim_evidence_associations": sum(len(c.evidence_ids) for q in eligible for c in q.claims),
        "distinct_citations": sum(len(q.citations) for q in eligible),
        "financial_quantities": financial,
        "ineligible": sorted(q.question_id for q in ctx.run.questions if not q.eligible),
    }


def _chunk_view(chunk: ChunkRecord, **extra: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "page_number": chunk.page_number,
        **extra,
        "text": chunk.text,
    }


def build_packet(ctx: AnnotationContext, question_id: str) -> dict[str, Any]:
    """The private review packet of one eligible answer. Deterministic; no timestamps."""
    q = ctx.question(question_id)
    if not q.eligible:
        raise AnnotationError(
            [f"{question_id}: ineligible for semantic judgment ({q.ineligibility_reason})"]
        )
    eq = ctx.eval_question(question_id)
    record = ctx.records[question_id]
    answer = record.get("answer") or {}
    cited_ids = {c.evidence_id for c in q.citations}
    claims = []
    for claim in q.claims:
        occurrences = fc.extract_occurrences(claim.text, ctx.vocabulary)
        values = []
        for occ in occurrences:
            checks = fc.check_value(
                question_id=question_id,
                generation_record_sha256=q.record_sha256,
                claim_position=claim.position,
                claim_text=claim.text,
                occurrences=occurrences,
                value_index=occ.value_index,
                association=None,  # associations are human work; none is assumed
                facts={},
                vocabulary=ctx.vocabulary,
            )
            values.append(
                {
                    **occ.public(),
                    "span_text": claim.text[occ.start : occ.end],
                    "deterministic_checks": [
                        {
                            "aspect": c.aspect,
                            "outcome": c.outcome.value,
                            "reasons": [r.value for r in c.reasons],
                        }
                        for c in checks
                    ],
                }
            )
        claims.append(
            {
                "position": claim.position,
                "claim_type": claim.claim_type,
                "citation_numbers": list(claim.citation_numbers),
                "evidence_ids": list(claim.evidence_ids),
                "text_sha256": claim.text_sha256,
                "numeric_values": values,
                "text": claim.text,
            }
        )
    selected = [
        _chunk_view(
            ctx.chunks[e.chunk_id],
            evidence_id=e.evidence_id,
            rank=e.rank,
            cited_by_answer=e.evidence_id in cited_ids,
        )
        for e in q.selected_evidence
    ]
    cited = [
        _chunk_view(
            ctx.chunks[c.chunk_id],
            citation_number=c.number,
            evidence_id=c.evidence_id,
            cited_by_claims=[x.position for x in q.claims if c.number in x.citation_numbers],
        )
        for c in q.citations
    ]
    selected_chunks = {e.chunk_id for e in q.selected_evidence}
    cited_chunks = {c.chunk_id for c in q.citations}
    reference = []
    for fact in eq.required_facts:
        numeric = fact.numeric
        reference.append(
            {
                "fact_id": fact.fact_id,
                "statement": fact.statement,
                "dataset_numeric": None
                if numeric is None
                else {
                    "company": numeric.company,
                    "fiscal_period": numeric.fiscal_period,
                    "metric": numeric.metric,
                    "value": numeric.value,
                    "unit": numeric.unit,
                    "basis": numeric.basis.value,
                },
                "trusted_locations": [
                    {
                        "document_id": loc.document_id,
                        "page_number": loc.page_number,
                        "chunks": [
                            _chunk_view(
                                c,
                                in_selected_context=c.chunk_id in selected_chunks,
                                cited_by_answer=c.chunk_id in cited_chunks,
                            )
                            for c in ctx.page_chunks(loc.document_id, loc.page_number)
                        ],
                    }
                    for loc in eq.evidence_for(fact.fact_id)
                ],
            }
        )
    guardrails = record.get("guardrails") or {}
    packet: dict[str, Any] = {
        "packet_version": PACKET_VERSION,
        "run_id": ctx.run.run_id,
        "generation_commit": ctx.run.generation_commit,
        "generation_protocol": ctx.run.generation_protocol,
        "question_id": question_id,
        "generation_record_sha256": q.record_sha256,
        "collection_artifact_sha256": ctx.run.collection_artifact_sha256,
        "dataset_sha256": ctx.run.dataset_sha256,
        "split_sha256": ctx.run.split_sha256,
        **annotation_identity(),
        "question": {
            "question_id": question_id,
            "category": q.category,
            "answerable": q.answerable,
            "required_fact_ids": list(q.required_fact_ids),
            "expected_abstention_reason": eq.expected_abstention_reason,
            "question": eq.question,
        },
        "response": {
            "execution_outcome": q.execution_outcome,
            "status": q.status,
            "claims": claims,
            "limitations": list(answer.get("limitations") or []),
        },
        "evidence": {
            "selected_context": selected,
            "cited": cited,
            "reference_source": reference,
        },
        "diagnostics": {
            "guardrails_status": guardrails.get("status"),
            "scope": guardrails.get("report"),
            "financial_checks": "per claim value under response.claims[].numeric_values; "
            "no verified association is assumed",
        },
        "trust": {
            "selected_context": "what the generator received (claim faithfulness)",
            "cited": "what each claim cites (citation support, one citation at a time)",
            "reference_source": "the dataset's trusted pages for each required fact "
            "(factual correctness and reference-fact verification); not retrieved or cited "
            "unless flagged",
        },
    }
    packet["packet_sha256"] = digest(packet)
    return packet


def worksheet_template(packet: Mapping[str, Any], annotator_id: str, role: Role) -> dict[str, Any]:
    """Every judgeable unit, pending and unlabelled. No label is pre-filled."""
    if not re.fullmatch(ANNOTATOR_PATTERN, annotator_id):
        raise AnnotationError(["annotator ID must be a pseudonym like ann-a1"])
    entries: list[dict[str, Any]] = []

    def judgment(dimension: Dimension, unit: dict[str, Any]) -> None:
        entries.append(
            {
                "entry_type": "judgment",
                "review_action": "pending",
                "dimension": dimension.value,
                "unit": unit,
                "label": None,
            }
        )

    for claim in packet["response"]["claims"]:
        p = claim["position"]
        judgment(Dimension.CLAIM_FAITHFULNESS, {"kind": "claim", "claim_position": p})
        for number, evidence_id in zip(
            claim["citation_numbers"], claim["evidence_ids"], strict=True
        ):
            judgment(
                Dimension.CITATION_SUPPORT,
                {
                    "kind": "claim_citation",
                    "claim_position": p,
                    "citation_number": number,
                    "evidence_id": evidence_id,
                },
            )
        for aspect in AttributionAspect:
            judgment(
                Dimension.FINANCIAL_ATTRIBUTION,
                {"kind": "claim_aspect", "claim_position": p, "aspect": aspect.value},
            )
        for value in claim["numeric_values"]:
            if value["kind"] == fc.OccurrenceKind.FINANCIAL_QUANTITY.value:
                judgment(
                    Dimension.NUMERICAL_CORRECTNESS,
                    {
                        "kind": "claim_value",
                        "claim_position": p,
                        "value_index": value["value_index"],
                        "aspect": "value",
                    },
                )
    for fact in packet["evidence"]["reference_source"]:
        judgment(
            Dimension.FACTUAL_CORRECTNESS, {"kind": "required_fact", "fact_id": fact["fact_id"]}
        )
        entries.append(
            {"entry_type": "reference_fact", "review_action": "pending", "fact_id": fact["fact_id"]}
        )
    for dimension in (
        Dimension.ANSWER_COMPLETENESS,
        Dimension.ABSTENTION_APPROPRIATENESS,
        Dimension.ANSWER_RELEVANCE,
    ):
        judgment(dimension, {"kind": "answer"})
    return {
        "worksheet_version": WORKSHEET_VERSION,
        "run_id": packet["run_id"],
        "question_id": packet["question_id"],
        "generation_record_sha256": packet["generation_record_sha256"],
        "packet_sha256": packet["packet_sha256"],
        "rubric_id": packet["rubric_id"],
        "collection_artifact_sha256": packet["collection_artifact_sha256"],
        "annotator_id": annotator_id,
        "role": role.value,
        "entries": entries,
    }


# --------------------------------------------------------------------------- #
# Worksheet entries (human input)
# --------------------------------------------------------------------------- #

_MODEL = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class SpanSpec(BaseModel):
    """A span of a frozen chunk; the importer pins it with the span hash."""

    model_config = _MODEL
    kind: EvidenceKind
    evidence_id: str | None = Field(default=None, pattern=r"^E\d{2}$")
    chunk_id: str = Field(min_length=1, max_length=128)
    start: int = Field(ge=0)
    end: int = Field(ge=1)


class _Revisable(BaseModel):
    model_config = _MODEL
    rationale: str | None = Field(default=None, max_length=2000, repr=False)  # private
    supersedes: str | None = Field(default=None, pattern=ENTRY_ID_PATTERN)
    revision_reason: RevisionReason | None = None
    # A third reviewer's decision on a unit two other reviewers dispute.
    adjudication: bool = False

    @model_validator(mode="after")
    def _revision(self) -> _Revisable:
        if (self.supersedes is None) != (self.revision_reason is None):
            raise ValueError("a revision names the entry it supersedes and a reason")
        return self


class JudgmentEntry(_Revisable):
    entry_type: Literal["judgment"]
    review_action: Literal["pending", "confirmed", "withdrawn"]
    dimension: Dimension
    unit: JudgmentUnit
    label: str | None = None
    not_assessable_reason: NotAssessableReason | None = None
    evidence: tuple[SpanSpec, ...] = ()
    reference_basis: ReferenceBasis | None = None
    evidence_sufficiency: EvidenceSufficiency | None = None
    error_type: ErrorType | None = None

    @model_validator(mode="after")
    def _decided(self) -> JudgmentEntry:
        if self.review_action == "confirmed" and self.label is None:
            raise ValueError("a confirmed judgment needs a label")
        if self.error_type is not None and self.label not in NEGATIVE_LABELS:
            raise ValueError("an error type applies to negative labels only")
        return self


class QuantitySpec(BaseModel):
    model_config = _MODEL
    magnitude: str = Field(pattern=r"^\d{1,15}(\.\d{1,6})?$")
    negative: bool = False
    unit: Literal["money", "percent", "percentage_points", "ratio"]
    scale: Literal["thousand", "million", "billion", "trillion"] | None = None
    currency: Literal["USD", "EUR", "GBP"] | None = None

    @model_validator(mode="after")
    def _applicable(self) -> QuantitySpec:
        # never record an attribute that does not apply to the unit kind
        if self.unit != "money" and (self.currency is not None or self.scale is not None):
            raise ValueError("currency and scale apply to monetary values only")
        return self

    def quantity(self) -> fc.Quantity:
        decimals = len(self.magnitude.split(".")[1]) if "." in self.magnitude else 0
        return fc.Quantity(
            Decimal(self.magnitude),
            -1 if self.negative else 1,
            self.negative,
            fc.Unit(self.unit),
            self.scale,
            None if self.currency is None else self.currency.lower(),
            self.currency,
            decimals,
        )


class PeriodSpec(BaseModel):
    model_config = _MODEL
    label: str = Field(pattern=r"^(FY|CY)\d{4}( Q[1-4])?$")
    basis: Literal["fiscal", "calendar"]
    company: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _basis(self) -> PeriodSpec:
        if (self.label.startswith("FY")) != (self.basis == "fiscal"):
            raise ValueError("FY labels are fiscal and CY labels calendar")
        return self


class ReferenceFactEntry(_Revisable):
    entry_type: Literal["reference_fact"]
    review_action: Literal["pending", "confirmed", "unresolved", "withdrawn"]
    fact_id: str = Field(pattern=FACT_PATTERN)
    metric: str | None = Field(default=None, pattern=_SAFE)
    company: str | None = Field(default=None, max_length=64)
    segment: str | None = Field(default=None, max_length=64)
    period: PeriodSpec | None = None
    value: QuantitySpec | None = None
    # "unverified": the source does not establish GAAP/non-GAAP; never guessed
    basis: Literal["GAAP", "non-GAAP", "not_applicable", "unverified"] | None = None
    evidence: tuple[SpanSpec, ...] = ()
    unresolved_reason: NotAssessableReason | None = None

    @model_validator(mode="after")
    def _complete(self) -> ReferenceFactEntry:
        if self.review_action == "confirmed":
            missing = [
                name
                for name in ("metric", "company", "period", "value", "basis")
                if getattr(self, name) is None
            ]
            if missing or not self.evidence:
                raise ValueError("a confirmed fact states every field and its source evidence")
            if self.unresolved_reason is not None:
                raise ValueError("a confirmed fact has no unresolved reason")
        if (self.review_action == "unresolved") != (self.unresolved_reason is not None):
            raise ValueError("an unresolved fact needs a reason, and only it has one")
        return self


class AssociationEntry(_Revisable):
    entry_type: Literal["fact_association"]
    review_action: Literal["pending", "confirmed", "withdrawn"]
    claim_position: int = Field(ge=1)
    value_index: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    fact_id: str = Field(pattern=FACT_PATTERN)
    metric: str = Field(pattern=_SAFE)


class DecompositionEntry(_Revisable):
    entry_type: Literal["decomposition"]
    review_action: Literal["pending", "confirmed", "withdrawn"]
    claim_position: int = Field(ge=1)
    subclaims: tuple[tuple[int, int], ...] = Field(min_length=2)


class SyntheticReviewEntry(_Revisable):
    entry_type: Literal["synthetic_review"]
    review_action: Literal["pending", "confirmed", "withdrawn"]
    variant_id: str = Field(pattern=VARIANT_PATTERN)
    decision: Literal["accept_expected_label", "reject_expected_label"] | None = None

    @model_validator(mode="after")
    def _decided(self) -> SyntheticReviewEntry:
        if self.review_action == "confirmed" and self.decision is None:
            raise ValueError("a confirmed synthetic review needs a decision")
        return self


Entry = Annotated[
    JudgmentEntry
    | ReferenceFactEntry
    | AssociationEntry
    | DecompositionEntry
    | SyntheticReviewEntry,
    Field(discriminator="entry_type"),
]


class Worksheet(BaseModel):
    model_config = _MODEL
    worksheet_version: Literal[1]
    run_id: str = Field(pattern=RUN_PATTERN)
    question_id: str = Field(pattern=QUESTION_PATTERN)
    generation_record_sha256: str = Field(pattern=_HEX64)
    packet_sha256: str = Field(pattern=_HEX64)
    rubric_id: str = Field(pattern=_HEX12)
    collection_artifact_sha256: str = Field(pattern=_HEX64)
    annotator_id: str = Field(pattern=ANNOTATOR_PATTERN)
    role: Role
    entries: tuple[Entry, ...]


# --------------------------------------------------------------------------- #
# Ledger (append-only, private)
# --------------------------------------------------------------------------- #

EntryType = Literal[
    "judgment", "reference_fact", "fact_association", "decomposition", "synthetic_review"
]


class LedgerEntry(BaseModel):
    """One imported human decision. ``entry_id`` hashes everything but import metadata."""

    model_config = _MODEL
    ledger_schema_version: Literal[1] = LEDGER_SCHEMA_VERSION
    entry_id: str = Field(pattern=ENTRY_ID_PATTERN)
    entry_type: EntryType
    unit_key: str = Field(min_length=1, max_length=160)
    action: Literal["confirmed", "unresolved", "withdrawn"]
    run_id: str = Field(pattern=RUN_PATTERN)
    question_id: str = Field(pattern=QUESTION_PATTERN)
    generation_record_sha256: str = Field(pattern=_HEX64)
    packet_sha256: str = Field(pattern=_HEX64)
    rubric_id: str = Field(pattern=_HEX12)
    extractor_id: str = Field(pattern=_HEX12)
    checker_id: str = Field(pattern=_HEX12)
    collection_artifact_sha256: str = Field(pattern=_HEX64)
    annotator_id: str = Field(pattern=ANNOTATOR_PATTERN)
    role: Role
    revision: int = Field(ge=1)
    supersedes: str | None = Field(default=None, pattern=ENTRY_ID_PATTERN)
    revision_reason: RevisionReason | None = None
    # JSON of the judgment, fact, association, ... (private)
    payload: dict[str, Any] = Field(repr=False)
    error_type: ErrorType | None = None
    rationale: str | None = Field(default=None, max_length=2000, repr=False)
    adjudication: bool = False
    content_sha256: str = Field(pattern=_HEX64)
    worksheet_sha256: str = Field(pattern=_HEX64)
    imported_at: str = Field(min_length=1, max_length=40)


_IMPORT_METADATA: Final = frozenset({"entry_id", "worksheet_sha256", "imported_at"})


def _entry_id(fields: Mapping[str, Any]) -> str:
    return digest({k: v for k, v in fields.items() if k not in _IMPORT_METADATA})[:24]


def _agreement_key(entry: LedgerEntry) -> str:
    """What two annotators must share to agree on a unit (not who or when)."""
    p = entry.payload
    if entry.action != "confirmed":
        return entry.action
    if entry.entry_type == "judgment":
        return digest([p.get("label"), p.get("not_assessable_reason")])
    if entry.entry_type == "reference_fact":
        keep = ("metric", "company", "segment", "period", "value", "basis")
        return digest({k: p.get(k) for k in keep})
    if entry.entry_type == "fact_association":
        return digest([p.get("fact_id"), p.get("metric")])
    if entry.entry_type == "decomposition":
        return digest(p.get("subclaims"))
    return digest(p.get("decision"))


@dataclass(frozen=True, slots=True)
class Ledger:
    entries: tuple[LedgerEntry, ...] = ()

    def heads(self) -> dict[tuple[str, str], LedgerEntry]:
        """The current entry of each (annotator, unit) chain, withdrawn ones included."""
        out: dict[tuple[str, str], LedgerEntry] = {}
        for e in self.entries:
            key = (e.annotator_id, e.unit_key)
            if key not in out or e.revision > out[key].revision:
                out[key] = e
        return out

    def active(self, entry_type: str | None = None) -> list[LedgerEntry]:
        return sorted(
            (
                e
                for e in self.heads().values()
                if e.action != "withdrawn" and (entry_type is None or e.entry_type == entry_type)
            ),
            key=lambda e: (e.unit_key, e.annotator_id),
        )

    def by_unit(self, entry_type: str | None = None) -> dict[str, list[LedgerEntry]]:
        """Every annotator's current, non-withdrawn entry, per unit (disputes included)."""
        out: dict[str, list[LedgerEntry]] = {}
        for e in self.active(entry_type):
            out.setdefault(e.unit_key, []).append(e)
        return out

    def resolution(
        self, entry_type: str | None = None
    ) -> dict[str, tuple[ReviewState, list[LedgerEntry]]]:
        """Each recorded unit's review state and the entries that decide it.

        One reviewer is ``single_reviewed``, never agreement. Two or more distinct
        reviewers with the same decision are ``independently_agreed``, differing ones
        ``disputed`` (no decisive entry). A third reviewer's adjudication of a dispute
        makes it ``adjudicated``. A unit whose every entry was withdrawn is
        ``withdrawn``. Superseded revisions stay in the ledger as history.
        """
        units: dict[str, list[LedgerEntry]] = {}
        for e in self.heads().values():
            if entry_type is None or e.entry_type == entry_type:
                units.setdefault(e.unit_key, []).append(e)
        out: dict[str, tuple[ReviewState, list[LedgerEntry]]] = {}
        for unit, heads in sorted(units.items()):
            active = sorted(
                (e for e in heads if e.action != "withdrawn"), key=lambda e: e.annotator_id
            )
            adjudicated = [e for e in active if e.adjudication]
            if adjudicated:
                agreed = len({_agreement_key(e) for e in adjudicated}) == 1
                state = ReviewState.ADJUDICATED if agreed else ReviewState.DISPUTED
                out[unit] = (state, adjudicated if agreed else [])
            elif not active:
                out[unit] = (ReviewState.WITHDRAWN, [])
            elif len(active) == 1:
                out[unit] = (ReviewState.SINGLE_REVIEWED, active)
            elif len({_agreement_key(e) for e in active}) == 1:
                out[unit] = (ReviewState.INDEPENDENTLY_AGREED, active)
            else:
                out[unit] = (ReviewState.DISPUTED, [])
        return out

    def disagreements(self) -> list[str]:
        """Units whose reviewers' current entries differ and are not adjudicated."""
        return [u for u, (s, _) in self.resolution().items() if s is ReviewState.DISPUTED]

    def independently_reviewed_units(self, entry_type: str = "judgment") -> list[str]:
        """Units with at least two distinct reviewers' independent (non-adjudication)
        current entries: the only units an agreement statistic may use."""
        return sorted(
            unit
            for unit, entries in self.by_unit(entry_type).items()
            if len({e.annotator_id for e in entries if not e.adjudication}) >= 2
        )

    def superseded(self) -> int:
        """Entries replaced by a later revision of the same reviewer's chain."""
        return sum(1 for e in self.entries if e.supersedes is not None)

    def judgments(self) -> tuple[Judgment, ...]:
        """Every reviewer's current judgment (disputed ones included, one per reviewer)."""
        return tuple(Judgment.model_validate(e.payload) for e in self.active("judgment"))

    def decompositions(self) -> dict[str, ClaimDecomposition]:
        out = {}
        for e in self.active("decomposition"):
            d = ClaimDecomposition.model_validate(e.payload)
            out[d.decomposition_id] = d
        return out

    def verified_facts(self) -> dict[str, fc.VerifiedReferenceFact]:
        """Facts whose deciding entries confirm them identically. Unresolved, withdrawn or
        disputed (unadjudicated) facts are never verified references."""
        out = {}
        for _, decisive in self.resolution("reference_fact").values():
            if decisive and all(e.action == "confirmed" for e in decisive):
                fact = fact_from_payload(decisive[0].payload)
                out[fact.fact_id] = fact
        return out

    def unresolved_facts(self) -> list[str]:
        return sorted(
            str(e.payload.get("fact_id"))
            for e in self.active("reference_fact")
            if e.action == "unresolved"
        )

    def associations(self) -> tuple[fc.FactAssociation, ...]:
        """One association per undisputed unit, from its deciding entry."""
        return tuple(
            fc.FactAssociation(**decisive[0].payload)
            for _, decisive in self.resolution("fact_association").values()
            if decisive
        )

    def partitions(self) -> dict[str, Role]:
        """The calibration/evaluation partition of every question with any entry (history
        included): a question used for calibration never becomes evaluation data."""
        out: dict[str, Role] = {}
        for e in self.entries:
            out.setdefault(e.question_id, e.role)
        return out


def load_ledger(path: Path) -> Ledger:
    """Read and re-verify the ledger. Digests catch accidental edits; not tamper-proof."""
    if not path.is_file():
        return Ledger()
    problems = []
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = LedgerEntry.model_validate_json(line)
        except ValidationError:
            problems.append(f"ledger line {number}: not a valid ledger entry")
            continue
        if _entry_id(entry.model_dump(mode="json")) != entry.entry_id:
            problems.append(f"ledger line {number}: entry digest does not match its content")
        entries.append(entry)
    ids = [e.entry_id for e in entries]
    if len(ids) != len(set(ids)):
        problems.append("ledger holds a duplicated entry")
    by_id = {e.entry_id: e for e in entries}
    chains: dict[tuple[str, str], list[LedgerEntry]] = {}
    for e in entries:
        chains.setdefault((e.annotator_id, e.unit_key), []).append(e)
    for chain in chains.values():
        if [e.revision for e in chain] != list(range(1, len(chain) + 1)):
            problems.append(f"ledger revision chain broken for {chain[0].entry_id}")
            continue
        for prev, cur in itertools.pairwise(chain):
            if cur.supersedes != prev.entry_id or by_id.get(prev.entry_id) is None:
                problems.append(f"ledger entry {cur.entry_id} does not supersede its predecessor")
    if problems:
        raise AnnotationError(problems)
    return Ledger(tuple(entries))


# --------------------------------------------------------------------------- #
# Reference facts
# --------------------------------------------------------------------------- #


def fact_payload(fact: fc.VerifiedReferenceFact) -> dict[str, Any]:
    v = fact.value
    return {
        "fact_id": fact.fact_id,
        "metric": fact.metric,
        "company": fact.company,
        "segment": fact.segment,
        "period": {
            "label": fact.period.label,
            "basis": fact.period.basis,
            "company": fact.period.company,
        },
        "value": {
            "magnitude": str(v.magnitude),
            "sign": v.sign,
            "explicit_sign": v.explicit_sign,
            "unit": v.unit.value,
            "scale": v.scale,
            "currency_marker": v.currency_marker,
            "currency": v.currency,
            "decimals": v.decimals,
        },
        "basis": fact.basis,
        "evidence": [e.model_dump(mode="json") for e in fact.evidence],
        "verified_by": fact.verified_by,
        "verification_role": fact.verification_role.value,
    }


def fact_from_payload(p: Mapping[str, Any]) -> fc.VerifiedReferenceFact:
    v = p["value"]
    return fc.VerifiedReferenceFact(
        fact_id=p["fact_id"],
        metric=p["metric"],
        company=p["company"],
        segment=p["segment"],
        period=fc.ReportingPeriod(**p["period"]),
        value=fc.Quantity(
            Decimal(v["magnitude"]),
            int(v["sign"]),
            bool(v["explicit_sign"]),
            fc.Unit(v["unit"]),
            v["scale"],
            v["currency_marker"],
            v["currency"],
            int(v["decimals"]),
        ),
        basis=p["basis"],
        evidence=tuple(EvidenceReference.model_validate(e) for e in p["evidence"]),
        verified_by=p["verified_by"],
        verification_role=Role(p["verification_role"]),
    )


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


class _Problem(Exception):
    """One refused worksheet entry; the message names codes, never content."""


def _validation_codes(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False, include_input=False):
        loc = ".".join(str(x) for x in err["loc"])
        detail = err["msg"] if err["type"] == "value_error" else err["type"]
        parts.append(f"{loc}: {detail}")
    return "; ".join(parts[:6])


def _reference(ctx: AnnotationContext, q: ji.VerifiedQuestion, span: SpanSpec) -> EvidenceReference:
    if span.chunk_id not in ctx.chunks:
        raise _Problem("evidence chunk is not in the verified source collection")
    ref = ji.make_reference(
        ctx.collection, span.chunk_id, span.start, span.end, span.kind, span.evidence_id
    )
    problems = ji.reference_problems(ref, q, ctx.collection, ctx.chunks)
    if problems:
        raise _Problem("; ".join(problems))
    return ref


@dataclass(frozen=True, slots=True)
class _Material:
    unit_key: str
    action: Literal["confirmed", "unresolved", "withdrawn"]
    payload: dict[str, Any]
    error_type: ErrorType | None = None


def _claim(q: ji.VerifiedQuestion, position: int) -> ji.VerifiedClaim:
    claim = next((c for c in q.claims if c.position == position), None)
    if claim is None:
        raise _Problem("claim position does not exist in the generation record")
    return claim


def _judgment_material(
    ctx: AnnotationContext,
    q: ji.VerifiedQuestion,
    ws: Worksheet,
    e: JudgmentEntry,
    staged: Ledger,
) -> _Material:
    unit_key = f"judgment:{e.unit.unit_id(q.question_id)}#{e.dimension.value}"
    if e.review_action == "withdrawn":
        return _Material(unit_key, "withdrawn", {})
    evidence = [_reference(ctx, q, s) for s in e.evidence]
    try:
        judgment = Judgment.model_validate(
            {
                "question_id": q.question_id,
                "generation_run_id": ctx.run.run_id,
                "generation_record_sha256": q.record_sha256,
                "dimension": e.dimension,
                "unit": e.unit,
                "label": e.label,
                "not_assessable_reason": e.not_assessable_reason,
                "evidence": evidence,
                "reference_basis": e.reference_basis,
                "evidence_sufficiency": e.evidence_sufficiency,
                "provenance": {
                    "source": "human",
                    "role": ws.role,
                    "annotator_id": ws.annotator_id,
                    "rubric_id": rubric_id(),
                },
            }
        )
    except ValidationError as exc:
        raise _Problem(f"judgment breaks the rubric contract ({_validation_codes(exc)})") from None
    problems = ji.check_judgment(judgment, ctx.run, ctx.collection, staged.decompositions())
    unit = judgment.unit
    if isinstance(unit, ClaimValueUnit) and not problems:
        claim = _claim(q, unit.claim_position)
        occurrences = fc.extract_occurrences(claim.text, ctx.vocabulary)
        occ = next((o for o in occurrences if o.value_index == unit.value_index), None)
        if occ is None:
            problems.append("value index outside the claim's numeric values")
        elif occ.kind not in VALUE_KINDS:
            problems.append("value index names a period, page, year or range, not a value")
    if problems:
        raise _Problem("; ".join(problems))
    return _Material(unit_key, "confirmed", judgment.model_dump(mode="json"), e.error_type)


def _fact_material(
    ctx: AnnotationContext,
    q: ji.VerifiedQuestion,
    ws: Worksheet,
    e: ReferenceFactEntry,
) -> _Material:
    unit_key = f"fact:{e.fact_id}"
    if e.fact_id not in q.required_fact_ids:
        raise _Problem("fact ID is not a required fact of this question")
    if e.review_action == "withdrawn":
        return _Material(unit_key, "withdrawn", {})
    if e.review_action == "unresolved":
        reason = "" if e.unresolved_reason is None else e.unresolved_reason.value
        return _Material(unit_key, "unresolved", {"fact_id": e.fact_id, "reason": reason})
    if (
        e.company is None
        or e.period is None
        or e.value is None
        or e.metric is None
        or e.basis is None
    ):
        raise _Problem("a confirmed fact states every field")  # also enforced by the model
    companies = {c.company: c for c in ctx.vocabulary.companies}
    entry = companies.get(e.company)
    if entry is None:
        raise _Problem("company is not a corpus company")
    if e.segment is not None and e.segment not in entry.segments:
        raise _Problem("segment is not an approved segment of that company")
    if e.period.company not in (None, e.company):
        raise _Problem("period company differs from the fact company")
    if any(s.kind is not EvidenceKind.REFERENCE_SOURCE for s in e.evidence):
        raise _Problem("reference facts take reference_source evidence only")
    trusted = {
        (loc.document_id, loc.page_number)
        for loc in ctx.eval_question(q.question_id).evidence_for(e.fact_id)
    }
    refs = []
    for span in e.evidence:
        ref = _reference(ctx, q, span)
        if (ref.document_id, ref.page_number) not in trusted:
            raise _Problem("evidence is not on a trusted reference page of this fact")
        refs.append(ref)
    try:
        fact = fc.VerifiedReferenceFact(
            fact_id=e.fact_id,
            metric=e.metric,
            company=e.company,
            segment=e.segment,
            period=fc.ReportingPeriod(e.period.label, e.period.basis, e.period.company),
            value=e.value.quantity(),
            basis=e.basis,
            evidence=tuple(refs),
            verified_by=ws.annotator_id,
            verification_role=ws.role,
        )
    except fc.FinancialCheckError as exc:
        raise _Problem(str(exc)) from None
    return _Material(unit_key, "confirmed", fact_payload(fact))


def _association_material(
    ctx: AnnotationContext,
    q: ji.VerifiedQuestion,
    ws: Worksheet,
    e: AssociationEntry,
    staged: Ledger,
) -> _Material:
    unit_key = f"assoc:{q.question_id}/claim:{e.claim_position}/value:{e.value_index}"
    if e.review_action == "withdrawn":
        return _Material(unit_key, "withdrawn", {})
    claim = _claim(q, e.claim_position)
    occurrences = fc.extract_occurrences(claim.text, ctx.vocabulary)
    occ = next((o for o in occurrences if o.value_index == e.value_index), None)
    if occ is None:
        raise _Problem("value index outside the claim's numeric values")
    if (occ.start, occ.end) != (e.start, e.end):
        raise _Problem("span does not match the indexed value in the claim text")
    if occ.kind not in VALUE_KINDS:
        raise _Problem("value index names a period, page, year or range, not a value")
    if e.fact_id not in q.required_fact_ids:
        raise _Problem("fact belongs to another question")
    fact = staged.verified_facts().get(e.fact_id)
    if fact is None:
        raise _Problem("reference fact is unknown, unresolved or not agreed as verified")
    if e.metric != fact.metric:
        raise _Problem("associated metric differs from the verified fact's metric")
    try:
        association = fc.FactAssociation(
            question_id=q.question_id,
            generation_record_sha256=q.record_sha256,
            claim_position=e.claim_position,
            value_index=e.value_index,
            start=e.start,
            end=e.end,
            fact_id=e.fact_id,
            metric=e.metric,
            associated_by=ws.annotator_id,
        )
    except fc.FinancialCheckError as exc:
        raise _Problem(str(exc)) from None
    payload = {
        name: getattr(association, name)
        for name in (
            "question_id",
            "generation_record_sha256",
            "claim_position",
            "value_index",
            "start",
            "end",
            "fact_id",
            "metric",
            "associated_by",
        )
    }
    return _Material(unit_key, "confirmed", payload)


def _decomposition_material(q: ji.VerifiedQuestion, e: DecompositionEntry) -> _Material:
    unit_key = f"decomp:{q.question_id}/claim:{e.claim_position}"
    if e.review_action == "withdrawn":
        return _Material(unit_key, "withdrawn", {})
    claim = _claim(q, e.claim_position)
    try:
        decomposition = ClaimDecomposition(
            question_id=q.question_id,
            generation_record_sha256=q.record_sha256,
            claim_position=claim.position,
            claim_text_sha256=claim.text_sha256,
            method="human",
            subclaims=tuple(
                SubclaimSpan(index=i, start=s, end=t)
                for i, (s, t) in enumerate(e.subclaims, start=1)
            ),
        )
        decomposition.check_against(claim.text)
    except ValidationError as exc:
        raise _Problem(f"decomposition is malformed ({_validation_codes(exc)})") from None
    except ValueError:
        raise _Problem("decomposition spans do not fit the original claim text") from None
    return _Material(unit_key, "confirmed", decomposition.model_dump(mode="json"))


def _synthetic_material(
    ctx: AnnotationContext, q: ji.VerifiedQuestion, e: SyntheticReviewEntry, staged: Ledger
) -> _Material:
    from founderlens.evaluation import calibration

    unit_key = f"synthetic:{e.variant_id}"
    if e.review_action == "withdrawn":
        return _Material(unit_key, "withdrawn", {})
    variants = {
        v.variant_id: v
        for v in calibration.synthetic_variants(ctx, staged)
        if v.question_id == q.question_id
    }
    if e.variant_id not in variants:
        raise _Problem("synthetic variant does not exist for this question")
    return _Material(unit_key, "confirmed", {"variant_id": e.variant_id, "decision": e.decision})


_ORDER: Final = {
    "decomposition": 0,
    "reference_fact": 1,
    "judgment": 2,
    "fact_association": 3,
    "synthetic_review": 4,
}


def _adjudication_problem(ledger: Ledger, annotator_id: str, material: _Material) -> str | None:
    """An adjudication needs a live dispute between two other reviewers and an adjudicator
    who did not review the unit independently."""
    own = [
        e
        for e in ledger.entries
        if (e.annotator_id, e.unit_key) == (annotator_id, material.unit_key)
    ]
    if any(not e.adjudication for e in own):
        return "an adjudicator must not be one of the unit's independent reviewers"
    others = [
        e
        for e in ledger.by_unit().get(material.unit_key, [])
        if e.annotator_id != annotator_id and not e.adjudication
    ]
    if len({e.annotator_id for e in others}) < 2 or len({_agreement_key(e) for e in others}) < 2:
        return "adjudication needs a dispute between two other reviewers"
    return None


@dataclass(frozen=True, slots=True)
class ImportResult:
    question_id: str
    accepted: tuple[LedgerEntry, ...]
    duplicates: int
    pending: int

    def public(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for e in self.accepted:
            types[e.entry_type] = types.get(e.entry_type, 0) + 1
        return ensure_public(
            {
                "question_id": self.question_id,
                "accepted": len(self.accepted),
                "accepted_by_type": dict(sorted(types.items())),
                "revisions": sum(e.revision > 1 for e in self.accepted),
                "duplicates_skipped": self.duplicates,
                "pending_skipped": self.pending,
            }
        )


def plan_import(
    ctx: AnnotationContext,
    ledger: Ledger,
    worksheet: Mapping[str, Any],
    *,
    worksheet_sha256: str,
    now: datetime,
) -> ImportResult:
    """Validate one worksheet against the run and the ledger; nothing is written.

    All-or-nothing: any refused entry refuses the worksheet, listing every problem
    by entry number and code.
    """
    try:
        ws = Worksheet.model_validate(worksheet)
    except ValidationError as exc:
        raise AnnotationError([f"worksheet is malformed ({_validation_codes(exc)})"]) from None
    q = ctx.run.question(ws.question_id)
    if q is None:
        raise AnnotationError(["worksheet names a question outside the verified run"])
    binding = {
        "run": ws.run_id == ctx.run.run_id,
        "generation record digest (stale or foreign record)": ws.generation_record_sha256
        == q.record_sha256,
        "rubric": ws.rubric_id == rubric_id(),
        "source artifact": ws.collection_artifact_sha256 == ctx.run.collection_artifact_sha256,
    }
    problems = [f"worksheet binding failed: {name}" for name, ok in binding.items() if not ok]
    if not q.eligible:
        problems.append(
            f"{q.question_id} is ineligible for semantic judgment ({q.ineligibility_reason}); "
            "nothing may be annotated"
        )
    partition = ledger.partitions().get(q.question_id)
    if partition is not None and partition is not ws.role:
        problems.append(
            f"{q.question_id} belongs to the {partition.value} partition; calibration and "
            "evaluation labels must not share a question"
        )
    if problems:
        raise AnnotationError(problems)
    packet = build_packet(ctx, q.question_id)
    if ws.packet_sha256 != packet["packet_sha256"]:
        raise AnnotationError(["worksheet binding failed: packet digest (stale packet)"])
    identity = annotation_identity()
    staged = list(ledger.entries)
    heads = Ledger(tuple(staged)).heads()
    accepted: list[LedgerEntry] = []
    seen: set[str] = set()
    duplicates = pending = 0
    order = sorted(enumerate(ws.entries, start=1), key=lambda x: (_ORDER[x[1].entry_type], x[0]))
    for number, e in order:
        if e.review_action == "pending":
            pending += 1
            continue
        current = Ledger(tuple(staged))
        try:
            if isinstance(e, JudgmentEntry):
                material = _judgment_material(ctx, q, ws, e, current)
            elif isinstance(e, ReferenceFactEntry):
                material = _fact_material(ctx, q, ws, e)
            elif isinstance(e, AssociationEntry):
                material = _association_material(ctx, q, ws, e, current)
            elif isinstance(e, DecompositionEntry):
                material = _decomposition_material(q, e)
            else:
                material = _synthetic_material(ctx, q, e, current)
        except _Problem as exc:
            problems.append(f"entry {number} ({e.entry_type}): {exc}")
            continue
        if material.unit_key in seen:
            problems.append(f"entry {number}: duplicate entry for the same unit in one worksheet")
            continue
        seen.add(material.unit_key)
        content = digest(
            {
                "entry_type": e.entry_type,
                "unit_key": material.unit_key,
                "action": material.action,
                "payload": material.payload,
                "error_type": None if material.error_type is None else material.error_type.value,
                "rationale": e.rationale,
                "role": ws.role.value,
                "adjudication": e.adjudication,
            }
        )
        key = (ws.annotator_id, material.unit_key)
        head = heads.get(key)
        if e.adjudication and material.action != "withdrawn":
            problem = _adjudication_problem(Ledger(tuple(staged)), ws.annotator_id, material)
            if problem:
                problems.append(f"entry {number}: {problem}")
                continue
        if head is not None and head.content_sha256 == content:
            duplicates += 1  # re-import of an unchanged decision
            continue
        if head is None:
            if material.action == "withdrawn":
                problems.append(f"entry {number}: nothing to withdraw")
                continue
            if e.supersedes is not None:
                problems.append(f"entry {number}: supersedes an entry that is not current")
                continue
            revision = 1
        else:
            if e.supersedes != head.entry_id:
                problems.append(
                    f"entry {number}: conflicts with current entry {head.entry_id}; a change "
                    "must supersede it and give a revision reason"
                )
                continue
            revision = head.revision + 1
        fields: dict[str, Any] = {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "entry_type": e.entry_type,
            "unit_key": material.unit_key,
            "action": material.action,
            "run_id": ctx.run.run_id,
            "question_id": q.question_id,
            "generation_record_sha256": q.record_sha256,
            "packet_sha256": packet["packet_sha256"],
            "rubric_id": identity["rubric_id"],
            "extractor_id": identity["extractor_id"],
            "checker_id": identity["checker_id"],
            "collection_artifact_sha256": ctx.run.collection_artifact_sha256,
            "annotator_id": ws.annotator_id,
            "role": ws.role.value,
            "revision": revision,
            "supersedes": e.supersedes if head is not None else None,
            "revision_reason": None
            if head is None or e.revision_reason is None
            else e.revision_reason.value,
            "payload": material.payload,
            "error_type": None if material.error_type is None else material.error_type.value,
            "rationale": e.rationale,
            "adjudication": e.adjudication,
            "content_sha256": content,
            "worksheet_sha256": worksheet_sha256,
            "imported_at": now.astimezone(UTC).isoformat(),
        }
        fields["entry_id"] = _entry_id(fields)
        entry = LedgerEntry.model_validate(fields)
        staged.append(entry)
        heads[key] = entry
        accepted.append(entry)
    if problems:
        raise AnnotationError(problems)
    return ImportResult(q.question_id, tuple(accepted), duplicates, pending)


def import_worksheet(
    ctx: AnnotationContext,
    paths: AnnotationPaths,
    worksheet_path: Path,
    *,
    manifest: PartitionManifest,
    dry_run: bool = False,
    now: datetime | None = None,
) -> ImportResult:
    """Validate a worksheet and, unless ``dry_run``, append its entries atomically.

    The worksheet's question and role must match the frozen partition manifest, and
    the existing ledger history must agree with it; otherwise nothing is imported.
    Worksheets are read only from the private annotation directory. The new ledger
    is written to a temporary file and moved into place in one step, so a failed
    validation or an interrupted write leaves the previous ledger unchanged.
    """
    from founderlens.evaluation import partition as pt

    if not worksheet_path.resolve().is_relative_to(paths.root.resolve()):
        raise AnnotationError(["worksheets must live in the private annotation directory"])
    try:
        raw = worksheet_path.read_bytes()
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        raise AnnotationError(["cannot read the worksheet as JSON"]) from None
    if not isinstance(data, dict):
        raise AnnotationError(["worksheet must be a JSON object"])
    if manifest.source_run_id != ctx.run.run_id:
        raise AnnotationError(["the partition manifest belongs to another generation run"])
    try:
        question_id, role = str(data["question_id"]), Role(data["role"])
    except (KeyError, ValueError):
        raise AnnotationError(["worksheet names no valid question and role"]) from None
    problems = pt.check_worksheet(manifest, question_id, role)
    before = paths.ledger.read_bytes() if paths.ledger.is_file() else b""
    ledger = load_ledger(paths.ledger)
    problems += pt.ledger_conflicts(manifest, ledger)
    if problems:
        raise AnnotationError(["partition check failed: " + p for p in problems])
    result = plan_import(
        ctx,
        ledger,
        data,
        worksheet_sha256=hashlib.sha256(raw).hexdigest(),
        now=now or datetime.now(UTC),
    )
    if dry_run or not result.accepted:
        return result
    current = paths.ledger.read_bytes() if paths.ledger.is_file() else b""
    if current != before:
        raise AnnotationError(["the ledger changed during import; nothing written, retry"])
    added = "".join(e.model_dump_json() + "\n" for e in result.accepted).encode("utf-8")
    paths.root.mkdir(parents=True, exist_ok=True)
    temporary = paths.ledger.with_name(LEDGER_FILENAME + ".tmp")
    try:
        temporary.write_bytes(before + added)
        temporary.replace(paths.ledger)
    finally:
        temporary.unlink(missing_ok=True)
    return result


# --------------------------------------------------------------------------- #
# Files and public summaries
# --------------------------------------------------------------------------- #


def ensure_public(summary: dict[str, Any]) -> dict[str, Any]:
    """Refuse a summary that carries a private text field or a long free-text value."""

    def walk(value: object, where: str) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                if str(k) in PRIVATE_KEYS:
                    raise AnnotationError([f"public output carries a private field at {where}"])
                walk(v, f"{where}.{k}")
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item, where)
        elif isinstance(value, str) and len(value) > PUBLIC_STRING_LIMIT:
            raise AnnotationError([f"public output carries free text at {where}"])

    walk(summary, "summary")
    return summary


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_packets(ctx: AnnotationContext, paths: AnnotationPaths) -> dict[str, Any]:
    """Write one private packet per eligible answer. Packets are deterministic; an existing
    packet that differs means the inputs changed, and it is never overwritten."""
    index = []
    for q in ctx.run.questions:
        if not q.eligible:
            continue
        packet = build_packet(ctx, q.question_id)
        path = paths.packets / f"{q.question_id}.json"
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing != packet:
                raise AnnotationError(
                    [f"{q.question_id}: existing packet differs; inputs changed, not overwritten"]
                )
        else:
            write_json(path, packet)
        index.append(
            {
                "question_id": q.question_id,
                "packet_sha256": packet["packet_sha256"],
                "generation_record_sha256": q.record_sha256,
                "template_units": len(
                    worksheet_template(packet, "ann-template", Role.CALIBRATION)["entries"]
                ),
            }
        )
    summary = {
        "run_id": ctx.run.run_id,
        **annotation_identity(),
        **run_counts(ctx),
        "packets": len(index),
        "template_units": sum(i["template_units"] for i in index),
        "human_judgments_created": 0,
        "verified_reference_facts_created": 0,
    }
    write_json(paths.packets / "index.json", {**summary, "index": index})
    return ensure_public(summary)


def create_worksheet(
    ctx: AnnotationContext,
    paths: AnnotationPaths,
    question_id: str,
    annotator_id: str,
    role: Role | None,
    *,
    manifest: PartitionManifest,
) -> Path:
    """A pending worksheet whose role is the question's frozen partition.

    ``role=None`` takes the role from the manifest; an explicit role that differs
    from the manifest is refused.
    """
    from founderlens.evaluation import partition as pt

    chosen = manifest.role_of(question_id) if role is None else role
    problems = pt.check_worksheet(manifest, question_id, chosen or Role.CALIBRATION)
    if problems or chosen is None:
        raise AnnotationError(["partition check failed: " + p for p in problems])
    packet = build_packet(ctx, question_id)
    template = worksheet_template(packet, annotator_id, chosen)
    path = paths.worksheets / f"{question_id}.{annotator_id}.json"
    if path.exists():
        raise AnnotationError(["worksheet already exists; resume editing it"])
    write_json(path, template)
    return path


AGREEMENT_NOTE: Final = "not computed: needs two independent reviews per unit (Commit 7.4)"


def template_judgment_units(ctx: AnnotationContext) -> list[str]:
    """Ledger unit keys of every judgment unit the worksheet templates list."""
    keys = []
    for q in ctx.run.questions:
        if not q.eligible:
            continue
        template = worksheet_template(
            build_packet(ctx, q.question_id), "ann-template", Role.CALIBRATION
        )
        for raw in template["entries"]:
            if raw["entry_type"] == "judgment":
                e = JudgmentEntry.model_validate(raw)
                keys.append(f"judgment:{e.unit.unit_id(q.question_id)}#{e.dimension.value}")
    return keys


def review_state_counts(ctx: AnnotationContext, ledger: Ledger) -> dict[str, int]:
    """Judgment units by review state; template units without any entry are unlabeled."""
    resolution = ledger.resolution("judgment")
    counts = {state.value: 0 for state in ReviewState}
    for unit in sorted(set(template_judgment_units(ctx)) | set(resolution)):
        state = resolution[unit][0] if unit in resolution else ReviewState.UNLABELED
        counts[state.value] += 1
    return counts


def progress(ctx: AnnotationContext, ledger: Ledger) -> dict[str, Any]:
    dims: dict[str, int] = {}
    for j in ledger.judgments():
        dims[j.dimension.value] = dims.get(j.dimension.value, 0) + 1
    active = ledger.active()
    return ensure_public(
        {
            "run_id": ctx.run.run_id,
            **annotation_identity(),
            "eligible_answers": sum(q.eligible for q in ctx.run.questions),
            "ineligible": sorted(q.question_id for q in ctx.run.questions if not q.eligible),
            "ledger_entries": len(ledger.entries),
            "active_entries": len(active),
            "active_judgments_by_dimension": dict(sorted(dims.items())),
            "questions_with_entries": len({e.question_id for e in active}),
            "annotators": len({e.annotator_id for e in ledger.entries}),
            "revisions": sum(e.revision > 1 for e in ledger.entries),
            "withdrawn": sum(e.action == "withdrawn" for e in ledger.heads().values()),
            "disagreements": len(ledger.disagreements()),
            "verified_reference_facts": len(ledger.verified_facts()),
            "unresolved_reference_facts": len(ledger.unresolved_facts()),
            "fact_associations": len(ledger.associations()),
            "decompositions": len(ledger.decompositions()),
            "synthetic_reviews": len(ledger.active("synthetic_review")),
            "judgment_review_states": review_state_counts(ctx, ledger),
            "pending_citation_support_units": sum(
                1
                for unit in template_judgment_units(ctx)
                if unit.endswith("#citation_support") and unit not in ledger.resolution()
            ),
            "independently_reviewed_units": len(ledger.independently_reviewed_units()),
            "superseded_entries": ledger.superseded(),
            "question_partitions": {
                role.value: sum(r is role for r in ledger.partitions().values()) for role in Role
            },
            "agreement_statistics": AGREEMENT_NOTE,
        }
    )


def history(ledger: Ledger, question_id: str) -> list[dict[str, Any]]:
    """Entry IDs, revisions and label codes for one question (for writing revisions)."""
    rows = []
    for e in ledger.entries:
        if e.question_id != question_id:
            continue
        label = e.payload.get("label") or e.payload.get("decision") or e.payload.get("reason")
        rows.append(
            {
                "entry_id": e.entry_id,
                "unit_key": e.unit_key,
                "annotator_id": e.annotator_id,
                "revision": e.revision,
                "action": e.action,
                "label": label if isinstance(label, str) else None,
                "supersedes": e.supersedes,
            }
        )
    ensure_public({"rows": rows})
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.annotation",
        description="Private, offline human annotation of a frozen development run.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("packets", "worksheet", "validate", "import", "status", "history"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--project-root", default=None)
        if name in ("worksheet", "history"):
            p.add_argument("--question", required=True)
        if name == "worksheet":
            p.add_argument("--annotator", required=True)
            p.add_argument(
                "--role",
                choices=[r.value for r in Role],
                default=None,
                help="defaults to the question's frozen partition; a different role is refused",
            )
        if name in ("validate", "import"):
            p.add_argument("--worksheet", required=True)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        paths = annotation_paths(settings, args.run)
        ctx = load_context(settings, args.run)
        output: object
        if args.command == "packets":
            output = write_packets(ctx, paths)
        elif args.command == "worksheet":
            if not re.fullmatch(QUESTION_PATTERN, args.question):
                raise AnnotationError(["only development questions may be annotated"])
            from founderlens.evaluation import partition as pt

            role = None if args.role is None else Role(args.role)
            path = create_worksheet(
                ctx,
                paths,
                args.question,
                args.annotator,
                role,
                manifest=pt.load_manifest(settings, ctx),
            )
            output = {"worksheet": path.name, "question_id": args.question}
        elif args.command in ("validate", "import"):
            from founderlens.evaluation import partition as pt

            result = import_worksheet(
                ctx,
                paths,
                Path(args.worksheet),
                manifest=pt.load_manifest(settings, ctx),
                dry_run=args.command == "validate",
            )
            output = {**result.public(), "written": args.command == "import"}
        elif args.command == "status":
            output = progress(ctx, load_ledger(paths.ledger))
        else:
            output = history(load_ledger(paths.ledger), args.question)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (
        AnnotationError,
        ji.JudgingInputError,
        ge.EvaluationRunError,
        EvaluationError,
        RetrievalError,
    ) as exc:
        print(f"Annotation {args.command} FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
