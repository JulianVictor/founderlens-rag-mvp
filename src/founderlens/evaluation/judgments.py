"""Formal evaluation rubric v1 and typed judgment contracts (Stage 6, Commit 7.1).

Protocol v1.4 §15. This module defines *what* a judgment is; it creates no
labels. Human, judge and deterministic judgments all use the same contract,
and every one names its source. Schema validity is not semantic correctness:
a well-formed judgment can still be wrong. Binding a judgment to the actual
frozen generation record (question, claim, citation, fact and evidence
offsets) is done separately by
:func:`founderlens.evaluation.judging_input.check_judgment`.

**Units.** The generator's structured claims are the primary units, addressed
by their stable one-based ``claim_position``. A compound claim may be split
later by a versioned :class:`ClaimDecomposition` into character spans of the
original claim text (nothing is rewritten), each addressable as a subclaim
with parent provenance. No LLM segments claims.

**Compound claims.** A claim judged as a whole while it holds several
propositions must not be scored as fully supported unless every proposition
is: use ``not_assessable`` with ``claim_not_atomic``, or decompose it and
derive the claim label with :func:`derive_faithfulness`.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RUBRIC_VERSION: Final = 1
JUDGMENT_SCHEMA_VERSION: Final = 1
DECOMPOSITION_VERSION: Final = 1
NOT_ASSESSABLE: Final = "not_assessable"

_HEX64 = r"^[0-9a-f]{64}$"
_HEX12 = r"^[0-9a-f]{12}$"
_QUESTION_ID = r"^dev-\d{3}$"  # held-out IDs are rejected by construction
_FACT_ID = r"^dev-\d{3}-f\d+$"
_EVIDENCE_ID = r"^E\d{2}$"
_SAFE_TOKEN = r"^[A-Za-z0-9._:-]{1,64}$"


class Dimension(StrEnum):
    CLAIM_FAITHFULNESS = "claim_faithfulness"
    CITATION_SUPPORT = "citation_support"
    FACTUAL_CORRECTNESS = "factual_correctness"
    FINANCIAL_ATTRIBUTION = "financial_attribution"
    NUMERICAL_CORRECTNESS = "numerical_correctness"
    ANSWER_COMPLETENESS = "answer_completeness"
    ABSTENTION_APPROPRIATENESS = "abstention_appropriateness"
    ANSWER_RELEVANCE = "answer_relevance"


class UnitKind(StrEnum):
    CLAIM = "claim"
    SUBCLAIM = "subclaim"
    CLAIM_CITATION = "claim_citation"
    REQUIRED_FACT = "required_fact"
    CLAIM_ASPECT = "claim_aspect"
    CLAIM_VALUE = "claim_value"
    ANSWER = "answer"


class NotAssessableReason(StrEnum):
    CLAIM_NOT_ATOMIC = "claim_not_atomic"
    EVIDENCE_UNREADABLE = "evidence_unreadable"  # e.g. a table broken by PDF extraction
    EVIDENCE_MISSING = "evidence_missing"
    REFERENCE_AMBIGUOUS = "reference_ambiguous"
    REFERENCE_UNAVAILABLE = "reference_unavailable"
    OUTSIDE_RUBRIC_SCOPE = "outside_rubric_scope"
    ANNOTATOR_UNCERTAIN = "annotator_uncertain"


class Source(StrEnum):
    HUMAN = "human"
    JUDGE = "judge"
    DETERMINISTIC = "deterministic"


class Role(StrEnum):
    CALIBRATION = "calibration"  # used to measure or tune agreement; not independent evidence
    EVALUATION = "evaluation"


class AttributionAspect(StrEnum):
    COMPANY = "company"
    REPORTING_PERIOD = "reporting_period"
    FINANCIAL_BASIS = "financial_basis"  # GAAP / non-GAAP


class NumericAspect(StrEnum):
    VALUE = "value"
    UNIT_SCALE = "unit_scale"
    CURRENCY = "currency"
    REPORTING_PERIOD = "reporting_period"
    CALCULATED_COMPARISON = "calculated_comparison"
    FINANCIAL_BASIS = "financial_basis"


class EvidenceSufficiency(StrEnum):
    """Whether the context the generator saw held accepted evidence (abstention only).

    Corpus-level answerability comes from the frozen dataset label; this records
    whether retrieval actually supplied enough, so a false abstention caused by
    retrieval is distinguishable from one caused by generation.
    """

    SUFFICIENT = "retrieved_sufficient"
    INSUFFICIENT = "retrieved_insufficient"
    UNKNOWN = "unknown"


class ReferenceBasis(StrEnum):
    REFERENCE_FACT = "reference_fact"  # the dataset's structured required fact
    SOURCE_EVIDENCE = "source_evidence"  # human-verified spans of the frozen source chunks


# --------------------------------------------------------------------------- #
# Rubric v1
# --------------------------------------------------------------------------- #

RUBRIC: Final[dict[Dimension, dict[str, Any]]] = {
    Dimension.CLAIM_FAITHFULNESS: {
        "question": "Is the claim supported by the exact context supplied to the generator?",
        "units": [UnitKind.CLAIM, UnitKind.SUBCLAIM],
        "labels": ["supported", "unsupported", "contradicted", NOT_ASSESSABLE],
        "evidence": "spans of selected-evidence chunks the generator received",
        "evidence_required_for": ["supported", "contradicted"],
    },
    Dimension.CITATION_SUPPORT: {
        "question": "Does this particular cited evidence record support the claim?",
        "units": [UnitKind.CLAIM_CITATION],
        "labels": ["supports", "partially_supports", "does_not_support", NOT_ASSESSABLE],
        "evidence": "spans of the one cited chunk only; citations are judged individually",
        "evidence_required_for": ["supports", "partially_supports"],
    },
    Dimension.FACTUAL_CORRECTNESS: {
        "question": "Is the required financial fact stated correctly against verified reference?",
        "units": [UnitKind.REQUIRED_FACT],
        "labels": ["correct", "incorrect", "missing", NOT_ASSESSABLE],
        "evidence": "the structured reference fact, or human-verified source spans; never "
        "unsupported recollection. Citation support is not factual correctness.",
        "evidence_required_for": [],
        "reference_basis_required_for": ["correct", "incorrect"],
    },
    Dimension.FINANCIAL_ATTRIBUTION: {
        "question": "Is the claim's company, reporting period or financial basis right?",
        "units": [UnitKind.CLAIM_ASPECT, UnitKind.SUBCLAIM],
        "labels": ["correct", "incorrect", "unverified", "not_applicable", NOT_ASSESSABLE],
        "evidence": "source spans stating the attribution; a document's primary reporting "
        "period does not describe every value on its pages",
        "evidence_required_for": ["correct", "incorrect"],
    },
    Dimension.NUMERICAL_CORRECTNESS: {
        "question": "Is this stated number right in value, unit and scale, currency, period, "
        "calculation and basis?",
        "units": [UnitKind.CLAIM_VALUE],
        "labels": ["correct", "incorrect", "unverified", "not_applicable", NOT_ASSESSABLE],
        "evidence": "a reference fact or source span; deterministic checks arrive in 7.2",
        "evidence_required_for": [],
        "reference_basis_required_for": ["correct", "incorrect"],
    },
    Dimension.ANSWER_COMPLETENESS: {
        "question": "Does the answer state every required fact (not: is it long)?",
        "units": [UnitKind.ANSWER],
        "labels": ["complete", "partial", "none", NOT_ASSESSABLE],
        "evidence": "derived from factual_correctness of the question's required facts",
        "evidence_required_for": [],
    },
    Dimension.ABSTENTION_APPROPRIATENESS: {
        "question": "Did the system abstain exactly when the corpus cannot answer?",
        "units": [UnitKind.ANSWER],
        "labels": [
            "correct_abstention",
            "correct_attempt",
            "false_abstention",
            "missed_abstention",
            NOT_ASSESSABLE,
        ],
        "evidence": "corpus answerability from the frozen dataset label; evidence_sufficiency "
        "records whether retrieval supplied enough. insufficient_evidence is not "
        "automatically correct.",
        "evidence_required_for": [],
        "qualifier": "evidence_sufficiency",
    },
    Dimension.ANSWER_RELEVANCE: {
        "question": "Does the answer address the question asked (company, period, metric)?",
        "units": [UnitKind.ANSWER],
        "labels": ["directly_answers", "partially_relevant", "off_topic", NOT_ASSESSABLE],
        "ordinal": {"directly_answers": 2, "partially_relevant": 1, "off_topic": 0},
        "evidence": "question and answer only; diagnostic, never factual correctness",
        "evidence_required_for": [],
    },
}


def rubric_definition() -> dict[str, Any]:
    return {
        "rubric_version": RUBRIC_VERSION,
        "judgment_schema_version": JUDGMENT_SCHEMA_VERSION,
        "decomposition_version": DECOMPOSITION_VERSION,
        "dimensions": {
            d.value: {k: [str(x) for x in v] if isinstance(v, list) else v for k, v in spec.items()}
            for d, spec in RUBRIC.items()
        },
        "unit_kinds": [u.value for u in UnitKind],
        "not_assessable_reasons": [r.value for r in NotAssessableReason],
        "sources": [s.value for s in Source],
        "source_identity_fields": {
            "human": "annotator_id",
            "judge": "judge_config_id",
            "deterministic": "check_id",
        },
        "roles": [r.value for r in Role],
        "attribution_aspects": [a.value for a in AttributionAspect],
        "numeric_aspects": [a.value for a in NumericAspect],
        "evidence_sufficiency": [e.value for e in EvidenceSufficiency],
        "reference_bases": [b.value for b in ReferenceBasis],
        "evidence_kinds": {
            "selected_context": "a chunk the generator received; names its evidence ID",
            "cited": "a chunk the claim actually cites; names its evidence ID",
            "reference_source": "a human-verified frozen chunk span used as reference; no "
            "evidence ID; never presented as retrieved or cited",
        },
        "evidence_kinds_by_dimension": {d.value: list(k) for d, k in EVIDENCE_KINDS.items()},
        "question_id_pattern": _QUESTION_ID,
        "offset_semantics": (
            "Unicode code-point offsets [start, end) into the exact stored text of a frozen "
            "V2 chunk; no re-normalization; spans start and end on base characters"
        ),
        "compound_claims": (
            "generator claims are primary units; decomposition only into spans of the "
            "original claim text on base characters; a decomposed claim is supported only "
            "if every one of its subclaims is labelled supported; a missing or not_assessable "
            "subclaim makes the claim not_assessable, any unsupported one unsupported, any "
            "contradicted one contradicted"
        ),
    }


def rubric_id() -> str:
    canonical = json.dumps(rubric_definition(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def labels_for(dimension: Dimension) -> tuple[str, ...]:
    return tuple(str(label) for label in RUBRIC[dimension]["labels"])


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #

# Validation errors never echo inputs, which may hold private claim or passage text.
_FROZEN = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class SubclaimRef(BaseModel):
    model_config = _FROZEN
    decomposition_id: str = Field(pattern=_HEX12)
    index: int = Field(ge=1)


class ClaimUnit(BaseModel):
    model_config = _FROZEN
    kind: Literal[UnitKind.CLAIM] = UnitKind.CLAIM
    claim_position: int = Field(ge=1)

    def unit_id(self, question_id: str) -> str:
        return f"{question_id}/claim:{self.claim_position}"


class SubclaimUnit(BaseModel):
    model_config = _FROZEN
    kind: Literal[UnitKind.SUBCLAIM] = UnitKind.SUBCLAIM
    claim_position: int = Field(ge=1)
    subclaim: SubclaimRef
    aspect: AttributionAspect | None = None  # financial_attribution only

    def unit_id(self, question_id: str) -> str:
        base = (
            f"{question_id}/claim:{self.claim_position}"
            f"/sub:{self.subclaim.decomposition_id}:{self.subclaim.index}"
        )
        return base if self.aspect is None else f"{base}/aspect:{self.aspect.value}"


class ClaimCitationUnit(BaseModel):
    model_config = _FROZEN
    kind: Literal[UnitKind.CLAIM_CITATION] = UnitKind.CLAIM_CITATION
    claim_position: int = Field(ge=1)
    citation_number: int = Field(ge=1)
    evidence_id: str = Field(pattern=_EVIDENCE_ID)
    subclaim: SubclaimRef | None = None

    def unit_id(self, question_id: str) -> str:
        sub = (
            ""
            if self.subclaim is None
            else (f"/sub:{self.subclaim.decomposition_id}:{self.subclaim.index}")
        )
        return (
            f"{question_id}/claim:{self.claim_position}{sub}"
            f"/citation:{self.citation_number}:{self.evidence_id}"
        )


class RequiredFactUnit(BaseModel):
    model_config = _FROZEN
    kind: Literal[UnitKind.REQUIRED_FACT] = UnitKind.REQUIRED_FACT
    fact_id: str = Field(pattern=_FACT_ID)

    def unit_id(self, question_id: str) -> str:
        return f"{question_id}/fact:{self.fact_id}"


class ClaimAspectUnit(BaseModel):
    model_config = _FROZEN
    kind: Literal[UnitKind.CLAIM_ASPECT] = UnitKind.CLAIM_ASPECT
    claim_position: int = Field(ge=1)
    aspect: AttributionAspect

    def unit_id(self, question_id: str) -> str:
        return f"{question_id}/claim:{self.claim_position}/aspect:{self.aspect.value}"


class ClaimValueUnit(BaseModel):
    """One stated number in a claim (the n-th numeric value) and one aspect of it."""

    model_config = _FROZEN
    kind: Literal[UnitKind.CLAIM_VALUE] = UnitKind.CLAIM_VALUE
    claim_position: int = Field(ge=1)
    value_index: int = Field(ge=1)
    aspect: NumericAspect
    fact_id: str | None = Field(default=None, pattern=_FACT_ID)

    def unit_id(self, question_id: str) -> str:
        return (
            f"{question_id}/claim:{self.claim_position}/value:{self.value_index}"
            f"/aspect:{self.aspect.value}"
        )


class AnswerUnit(BaseModel):
    model_config = _FROZEN
    kind: Literal[UnitKind.ANSWER] = UnitKind.ANSWER

    def unit_id(self, question_id: str) -> str:
        return f"{question_id}/answer"


Unit = Annotated[
    ClaimUnit
    | SubclaimUnit
    | ClaimCitationUnit
    | RequiredFactUnit
    | ClaimAspectUnit
    | ClaimValueUnit
    | AnswerUnit,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- #
# Evidence references and provenance
# --------------------------------------------------------------------------- #


class EvidenceKind(StrEnum):
    """How a span entered the judgment; a reference span never poses as retrieved or cited."""

    SELECTED_CONTEXT = "selected_context"  # in the generator's context
    CITED = "cited"  # cited by the judged claim
    REFERENCE_SOURCE = "reference_source"  # human-verified source span (trust: see protocol §15)


# Which evidence kinds each dimension may rely on.
EVIDENCE_KINDS: Final[dict[Dimension, tuple[EvidenceKind, ...]]] = {
    Dimension.CLAIM_FAITHFULNESS: (EvidenceKind.SELECTED_CONTEXT, EvidenceKind.CITED),
    Dimension.CITATION_SUPPORT: (EvidenceKind.CITED,),
    Dimension.FACTUAL_CORRECTNESS: tuple(EvidenceKind),
    Dimension.FINANCIAL_ATTRIBUTION: tuple(EvidenceKind),
    Dimension.NUMERICAL_CORRECTNESS: tuple(EvidenceKind),
    Dimension.ANSWER_COMPLETENESS: (),
    Dimension.ABSTENTION_APPROPRIATENESS: (EvidenceKind.SELECTED_CONTEXT,),
    Dimension.ANSWER_RELEVANCE: (),
}


class EvidenceReference(BaseModel):
    """A span of one frozen V2 chunk. ``span_sha256`` pins the text without copying it.

    ``kind`` says how the span entered the judgment. Selected-context and cited spans
    name their request-scoped evidence ID; a ``reference_source`` span (human-verified
    evidence, possibly outside the generator's context) has none and is never
    presented as retrieved or cited.
    """

    model_config = _FROZEN
    kind: EvidenceKind
    evidence_id: str | None = Field(default=None, pattern=_EVIDENCE_ID)
    document_id: str = Field(min_length=1, max_length=64)
    page_number: int = Field(ge=1)
    chunk_id: str = Field(min_length=1, max_length=128)
    collection_artifact_sha256: str = Field(pattern=_HEX64)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    span_sha256: str = Field(pattern=_HEX64)

    @model_validator(mode="after")
    def _ordered(self) -> EvidenceReference:
        if self.end <= self.start:
            raise ValueError("evidence span must have end > start")
        if (self.kind is EvidenceKind.REFERENCE_SOURCE) != (self.evidence_id is None):
            raise ValueError(
                "selected_context and cited spans name an evidence ID; reference "
                "source spans do not"
            )
        return self


class Provenance(BaseModel):
    model_config = _FROZEN
    source: Source
    role: Role
    annotator_id: str | None = Field(default=None, pattern=_SAFE_TOKEN)  # pseudonymous
    judge_config_id: str | None = Field(default=None, pattern=_HEX12)
    check_id: str | None = Field(default=None, pattern=_SAFE_TOKEN)  # deterministic check
    rubric_id: str = Field(pattern=_HEX12)

    @model_validator(mode="after")
    def _source_fields(self) -> Provenance:
        expected = {
            Source.HUMAN: ("annotator_id",),
            Source.JUDGE: ("judge_config_id",),
            Source.DETERMINISTIC: ("check_id",),
        }[self.source]
        present = {
            name
            for name in ("annotator_id", "judge_config_id", "check_id")
            if getattr(self, name) is not None
        }
        if present != set(expected):
            raise ValueError(
                f"{self.source.value} provenance needs exactly {expected[0]}, got {sorted(present)}"
            )
        if self.rubric_id != rubric_id():
            raise ValueError("provenance names a different rubric")
        return self


# --------------------------------------------------------------------------- #
# Judgment
# --------------------------------------------------------------------------- #


class Judgment(BaseModel):
    """One label for one unit and dimension, with evidence and provenance.

    Validity here is structural: allowed unit kind and label for the dimension,
    ``not_assessable`` exactly when a reason is given, evidence and reference
    basis where the rubric requires them, and provenance consistent with the
    source. It says nothing about whether the label is right.
    """

    model_config = _FROZEN
    judgment_schema_version: Literal[1] = JUDGMENT_SCHEMA_VERSION
    rubric_version: Literal[1] = RUBRIC_VERSION
    question_id: str = Field(pattern=_QUESTION_ID)
    generation_run_id: str = Field(min_length=1, max_length=96, pattern=r"^live-[0-9A-Za-z-]+$")
    generation_record_sha256: str = Field(pattern=_HEX64)
    dimension: Dimension
    unit: Unit
    label: str
    not_assessable_reason: NotAssessableReason | None = None
    evidence: tuple[EvidenceReference, ...] = ()
    reference_basis: ReferenceBasis | None = None
    evidence_sufficiency: EvidenceSufficiency | None = None
    provenance: Provenance

    @property
    def unit_id(self) -> str:
        return self.unit.unit_id(self.question_id)

    @model_validator(mode="after")
    def _contract(self) -> Judgment:
        spec = RUBRIC[self.dimension]
        problems: list[str] = []
        if self.unit.kind not in spec["units"]:
            problems.append(f"{self.dimension.value} does not take {self.unit.kind.value} units")
        if isinstance(self.unit, SubclaimUnit) and (self.unit.aspect is None) != (
            self.dimension is not Dimension.FINANCIAL_ATTRIBUTION
        ):
            problems.append("a subclaim aspect is required exactly for financial_attribution")
        if self.label not in labels_for(self.dimension):
            problems.append(f"unknown label for {self.dimension.value}")
        if (self.label == NOT_ASSESSABLE) != (self.not_assessable_reason is not None):
            problems.append("not_assessable requires a reason, and only not_assessable has one")
        if self.label in spec.get("evidence_required_for", []) and not self.evidence:
            problems.append(f"label {self.label} requires evidence references")
        basis_required = self.label in spec.get("reference_basis_required_for", [])
        if basis_required and self.reference_basis is None:
            problems.append(f"label {self.label} requires a reference basis")
        if self.reference_basis is ReferenceBasis.SOURCE_EVIDENCE and not self.evidence:
            problems.append("a source_evidence basis requires evidence references")
        if self.reference_basis is not None and "reference_basis_required_for" not in spec:
            problems.append(f"{self.dimension.value} takes no reference basis")
        if (self.evidence_sufficiency is not None) != (
            self.dimension is Dimension.ABSTENTION_APPROPRIATENESS and self.label != NOT_ASSESSABLE
        ):
            problems.append("evidence_sufficiency is required exactly for assessed abstention")
        allowed_kinds = EVIDENCE_KINDS[self.dimension]
        if any(e.kind not in allowed_kinds for e in self.evidence):
            problems.append(f"{self.dimension.value} does not accept that evidence kind")
        unit = self.unit
        if isinstance(unit, ClaimCitationUnit) and any(
            e.evidence_id != unit.evidence_id for e in self.evidence
        ):
            problems.append("citation_support evidence must come from the cited record only")
        if problems:
            raise ValueError("; ".join(problems))
        return self


# --------------------------------------------------------------------------- #
# Compound claims
# --------------------------------------------------------------------------- #


class SubclaimSpan(BaseModel):
    model_config = _FROZEN
    index: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(ge=1)


class ClaimDecomposition(BaseModel):
    """Versioned split of one generated claim into spans of its original text.

    The parent claim text is pinned by ``claim_text_sha256``; subclaims are
    character spans, so nothing is paraphrased. ``method`` is ``human`` or a named
    deterministic rule; LLM segmentation is not a permitted method.
    """

    model_config = _FROZEN
    decomposition_version: Literal[1] = DECOMPOSITION_VERSION
    question_id: str = Field(pattern=_QUESTION_ID)
    generation_record_sha256: str = Field(pattern=_HEX64)
    claim_position: int = Field(ge=1)
    claim_text_sha256: str = Field(pattern=_HEX64)
    method: Literal["human", "deterministic_rule"]
    rule_id: str | None = Field(default=None, pattern=_SAFE_TOKEN)
    subclaims: tuple[SubclaimSpan, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _spans(self) -> ClaimDecomposition:
        if (self.method == "deterministic_rule") != (self.rule_id is not None):
            raise ValueError("a deterministic decomposition names its rule, a human one does not")
        indexes = [s.index for s in self.subclaims]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("subclaim indexes must be 1..n in order")
        for span in self.subclaims:
            if span.end <= span.start:
                raise ValueError("subclaim spans must have end > start")
        return self

    @property
    def decomposition_id(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def check_against(self, claim_text: str) -> None:
        """Bind the decomposition to the exact original claim text."""
        if hashlib.sha256(claim_text.encode("utf-8")).hexdigest() != self.claim_text_sha256:
            raise ValueError("decomposition does not match the original claim text")
        for span in self.subclaims:
            if span.end > len(claim_text):
                raise ValueError("subclaim span outside the claim text")
            if not (_base(claim_text, span.start) and _base(claim_text, span.end)):
                raise ValueError("subclaim span splits a combining character sequence")

    def subclaim_texts(self, claim_text: str) -> tuple[str, ...]:
        """The exact original substrings (private: for annotators, never for reports)."""
        self.check_against(claim_text)
        return tuple(claim_text[s.start : s.end] for s in self.subclaims)


def _base(text: str, index: int) -> bool:
    return index == len(text) or unicodedata.combining(text[index]) == 0


def derive_faithfulness(
    decomposition: ClaimDecomposition, subclaim_labels: Mapping[int, str]
) -> str:
    """Parent label from every subclaim's label: supported only if all are supported.

    A subclaim without a label counts as ``not_assessable``, so a missing judgment
    can never let a compound claim pass as supported.
    """
    allowed = set(labels_for(Dimension.CLAIM_FAITHFULNESS))
    indexes = {s.index for s in decomposition.subclaims}
    if not set(subclaim_labels) <= indexes:
        raise ValueError("labels name subclaims outside the decomposition")
    if any(label not in allowed for label in subclaim_labels.values()):
        raise ValueError("subclaim labels must be faithfulness labels")
    labels = [subclaim_labels.get(i, NOT_ASSESSABLE) for i in sorted(indexes)]
    if "contradicted" in labels:
        return "contradicted"
    if NOT_ASSESSABLE in labels:
        return NOT_ASSESSABLE
    if "unsupported" in labels:
        return "unsupported"
    return "supported"


_UNSAFE = re.compile(r"[^A-Za-z0-9._:/-]")


def public_view(judgment: Judgment) -> dict[str, Any]:
    """Publication-safe fields: identifiers, codes and offsets, never text."""
    return {
        "question_id": judgment.question_id,
        "unit_id": _UNSAFE.sub("?", judgment.unit_id),
        "dimension": judgment.dimension.value,
        "label": judgment.label,
        "not_assessable_reason": None
        if judgment.not_assessable_reason is None
        else judgment.not_assessable_reason.value,
        "source": judgment.provenance.source.value,
        "role": judgment.provenance.role.value,
        "evidence_count": len(judgment.evidence),
        "rubric_id": judgment.provenance.rubric_id,
    }
