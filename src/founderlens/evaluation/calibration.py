"""Versioned calibration-set builder (Stage 6, Commit 7.3; protocol §17).

Usage (offline; private output under ``reports/annotation/<run_id>/``)::

    python -m founderlens.evaluation.calibration build          --run <run_id>
    python -m founderlens.evaluation.calibration review-worksheets --run <run_id> --annotator ann-a1

**Natural examples** are units of the frozen development responses carrying
*human* labels (role ``calibration``) in the annotation ledger. A unit whose
annotators disagree, or which is ``not_assessable``, is excluded and counted.
Each example gets a calibration category (supported claim, wrong scale, wrong
company, false abstention, ...) from its label, unit and the annotator's
optional error type.

**Selection** is deterministic (:data:`SELECTION_POLICY`): per category, units
are interleaved across question category, answer state and claim type, in an
order fixed by a hash of the policy seed and the example ID, with a cap per
category so easy factual claims cannot dominate. It reads human labels and
frozen structure only; no judge output exists or is consulted, and the
held-out split is never opened. The development sample is small and is
disclosed as such.

**Synthetic variants** apply versioned deterministic rules
(:data:`TRANSFORMATIONS`) to selected natural examples: they are marked
``synthetic``, keep their source example identity, rule and transformation
identity, never modify the source record, and carry an *expected* label that is
a rule proposal, not a human judgment, until a human accepts it through a
``synthetic_review`` ledger entry. Natural and synthetic examples are kept and
reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final

from founderlens.config import ConfigError, load_settings
from founderlens.evaluation import annotation as an
from founderlens.evaluation import financial_checks as fc
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation import judging_input as ji
from founderlens.evaluation.benchmark import EvaluationError
from founderlens.evaluation.judgments import (
    NOT_ASSESSABLE,
    ClaimAspectUnit,
    ClaimCitationUnit,
    ClaimUnit,
    ClaimValueUnit,
    Dimension,
    Judgment,
    Role,
    SubclaimUnit,
    derive_faithfulness,
    rubric_id,
)
from founderlens.retrieval.schema import RetrievalError

CALIBRATION_VERSION: Final = 1


class Category(StrEnum):
    SUPPORTED_CLAIM = "supported_claim"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    COMPOUND_MIXED_SUPPORT = "compound_mixed_support"
    SUPPORTED_CITATION = "supported_citation"
    PARTIAL_CITATION = "partial_citation"
    INVALID_CITATION = "invalid_citation"
    CORRECT_ATTRIBUTION = "correct_attribution"
    WRONG_COMPANY = "wrong_company"
    WRONG_PERIOD = "wrong_period"
    WRONG_BASIS = "wrong_basis"
    WRONG_METRIC = "wrong_metric"
    CORRECT_VALUE = "correct_value"
    INCORRECT_VALUE = "incorrect_value"
    WRONG_SCALE = "wrong_scale"
    WRONG_CURRENCY = "wrong_currency"
    INCORRECT_CALCULATION = "incorrect_calculation"
    CORRECT_FACT = "correct_fact"
    INCORRECT_FACT = "incorrect_fact"
    MISSING_FACT = "missing_fact"
    CORRECT_ABSTENTION = "correct_abstention"
    CORRECT_ATTEMPT = "correct_attempt"
    FALSE_ABSTENTION = "false_abstention"
    MISSED_ABSTENTION = "missed_abstention"
    ANSWER_COMPLETENESS = "answer_completeness"
    ANSWER_RELEVANCE = "answer_relevance"


ERROR_CATEGORIES: Final = {
    an.ErrorType.WRONG_VALUE: Category.INCORRECT_VALUE,
    an.ErrorType.WRONG_SCALE: Category.WRONG_SCALE,
    an.ErrorType.WRONG_CURRENCY: Category.WRONG_CURRENCY,
    an.ErrorType.WRONG_COMPANY: Category.WRONG_COMPANY,
    an.ErrorType.WRONG_PERIOD: Category.WRONG_PERIOD,
    an.ErrorType.WRONG_METRIC: Category.WRONG_METRIC,
    an.ErrorType.WRONG_BASIS: Category.WRONG_BASIS,
    an.ErrorType.INCORRECT_CALCULATION: Category.INCORRECT_CALCULATION,
    an.ErrorType.UNSUPPORTED_BY_EVIDENCE: Category.UNSUPPORTED_CLAIM,
    an.ErrorType.IRRELEVANT_CITATION: Category.INVALID_CITATION,
}

_ASPECT_ERRORS: Final = {
    "company": Category.WRONG_COMPANY,
    "reporting_period": Category.WRONG_PERIOD,
    "financial_basis": Category.WRONG_BASIS,
    "unit_scale": Category.WRONG_SCALE,
    "currency": Category.WRONG_CURRENCY,
    "calculated_comparison": Category.INCORRECT_CALCULATION,
    "value": Category.INCORRECT_VALUE,
}


def categorize(judgment: Judgment, error_type: an.ErrorType | None) -> Category | None:
    """The calibration category of one human label, or ``None`` (no gold label)."""
    label, d = judgment.label, judgment.dimension
    if label == NOT_ASSESSABLE:
        return None
    if error_type is not None:
        return ERROR_CATEGORIES[error_type]
    aspect = getattr(judgment.unit, "aspect", None)
    aspect_value = None if aspect is None else str(aspect.value)
    if d is Dimension.CLAIM_FAITHFULNESS:
        return Category.SUPPORTED_CLAIM if label == "supported" else Category.UNSUPPORTED_CLAIM
    if d is Dimension.CITATION_SUPPORT:
        return {
            "supports": Category.SUPPORTED_CITATION,
            "partially_supports": Category.PARTIAL_CITATION,
        }.get(label, Category.INVALID_CITATION)
    if d in (Dimension.FINANCIAL_ATTRIBUTION, Dimension.NUMERICAL_CORRECTNESS):
        if label == "correct":
            return (
                Category.CORRECT_ATTRIBUTION
                if d is Dimension.FINANCIAL_ATTRIBUTION
                else Category.CORRECT_VALUE
            )
        if label == "incorrect" and aspect_value is not None:
            return _ASPECT_ERRORS.get(aspect_value)
        return None  # unverified / not_applicable carry no gold label
    if d is Dimension.FACTUAL_CORRECTNESS:
        return {
            "correct": Category.CORRECT_FACT,
            "incorrect": Category.INCORRECT_FACT,
            "missing": Category.MISSING_FACT,
        }[label]
    if d is Dimension.ABSTENTION_APPROPRIATENESS:
        return Category(label)
    if d is Dimension.ANSWER_COMPLETENESS:
        return Category.ANSWER_COMPLETENESS
    return Category.ANSWER_RELEVANCE


# --------------------------------------------------------------------------- #
# Natural examples and selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NaturalExample:
    example_id: str
    question_id: str
    question_category: str
    answer_status: str
    claim_type: str | None
    claim_position: int | None
    unit_key: str
    dimension: str
    label: str
    category: Category
    agreement: str  # review state: single_reviewed | independently_agreed | adjudicated | ...
    source_entry_ids: tuple[str, ...]

    def public(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "question_id": self.question_id,
            "question_category": self.question_category,
            "answer_status": self.answer_status,
            "claim_type": self.claim_type,
            "claim_position": self.claim_position,
            "dimension": self.dimension,
            "label": self.label,
            "category": self.category.value,
            "agreement": self.agreement,
            "source_entry_ids": list(self.source_entry_ids),
            "synthetic": False,
        }


def _example_id(run_id: str, unit_key: str) -> str:
    return hashlib.sha256(f"{run_id}|{unit_key}".encode()).hexdigest()[:16]


def _claim_position(judgment: Judgment) -> int | None:
    unit = judgment.unit
    if isinstance(
        unit, ClaimUnit | SubclaimUnit | ClaimCitationUnit | ClaimAspectUnit | ClaimValueUnit
    ):
        return unit.claim_position
    return None


EXCLUSION_REASONS: Final = (
    "disputed",
    "withdrawn",
    "not_assessable",
    "unverified_or_not_applicable",
    "evaluation_partition",
)
ADMITTED_STATES: Final = (
    an.ReviewState.SINGLE_REVIEWED,
    an.ReviewState.INDEPENDENTLY_AGREED,
    an.ReviewState.ADJUDICATED,
)


def classify_units(
    ctx: an.AnnotationContext, ledger: an.Ledger
) -> tuple[list[NaturalExample], list[dict[str, Any]]]:
    """Human-labelled judgment units as natural examples, and every excluded unit with
    its reason and review state (IDs and codes only; kept for diagnostic reporting)."""
    examples: list[NaturalExample] = []
    excluded: list[dict[str, Any]] = []
    decided: dict[str, Judgment] = {}
    partitions = ledger.partitions()

    def exclude(unit_key: str, question_id: str, reason: str, state: an.ReviewState) -> None:
        excluded.append(
            {
                "unit_key": unit_key,
                "question_id": question_id,
                "reason": reason,
                "review_state": state.value,
            }
        )

    for unit_key, (state, decisive) in ledger.resolution("judgment").items():
        question_id = unit_key.split(":", 1)[1].split("/", 1)[0]
        if partitions.get(question_id) is not Role.CALIBRATION:
            exclude(unit_key, question_id, "evaluation_partition", state)
            continue
        if state is an.ReviewState.WITHDRAWN:
            exclude(unit_key, question_id, "withdrawn", state)
            continue
        judgments = [Judgment.model_validate(e.payload) for e in decisive]
        categories = {categorize(j, e.error_type) for j, e in zip(judgments, decisive, strict=True)}
        if state is an.ReviewState.DISPUTED or len(categories) > 1:
            exclude(unit_key, question_id, "disputed", an.ReviewState.DISPUTED)
            continue
        judgment = judgments[0]
        decided[unit_key] = judgment
        category = categories.pop()
        if category is None:
            reason = (
                "not_assessable"
                if judgment.label == NOT_ASSESSABLE
                else "unverified_or_not_applicable"
            )
            exclude(unit_key, question_id, reason, state)
            continue
        examples.append(_example(ctx, unit_key, judgment, category, decisive, state))
    examples += _compound_examples(ctx, ledger, decided)
    return sorted(examples, key=lambda x: x.example_id), excluded


def natural_candidates(
    ctx: an.AnnotationContext, ledger: an.Ledger
) -> tuple[list[NaturalExample], dict[str, int]]:
    """Natural examples plus counts of excluded units by reason."""
    examples, excluded = classify_units(ctx, ledger)
    counts = dict.fromkeys(EXCLUSION_REASONS, 0)
    for item in excluded:
        counts[item["reason"]] += 1
    return examples, counts


def _example(
    ctx: an.AnnotationContext,
    unit_key: str,
    judgment: Judgment,
    category: Category,
    entries: Sequence[an.LedgerEntry],
    state: an.ReviewState,
) -> NaturalExample:
    q = ctx.question(judgment.question_id)
    position = _claim_position(judgment)
    claim = next((c for c in q.claims if c.position == position), None)
    return NaturalExample(
        example_id=_example_id(ctx.run.run_id, unit_key),
        question_id=q.question_id,
        question_category=q.category,
        answer_status=q.status or q.execution_outcome,
        claim_type=None if claim is None else claim.claim_type,
        claim_position=position,
        unit_key=unit_key,
        dimension=judgment.dimension.value,
        label=judgment.label,
        category=category,
        agreement=state.value,
        source_entry_ids=tuple(sorted(e.entry_id for e in entries)),
    )


def _compound_examples(
    ctx: an.AnnotationContext, ledger: an.Ledger, agreed: Mapping[str, Judgment]
) -> list[NaturalExample]:
    """A decomposed claim whose subclaims are partly supported and partly not."""
    out = []
    for decomposition_id, d in sorted(ledger.decompositions().items()):
        labels: dict[int, str] = {}
        sources: list[str] = []
        for unit_key, j in agreed.items():
            unit = j.unit
            if (
                j.dimension is Dimension.CLAIM_FAITHFULNESS
                and isinstance(unit, SubclaimUnit)
                and unit.subclaim.decomposition_id == decomposition_id
            ):
                labels[unit.subclaim.index] = j.label
                sources.append(unit_key)
        values = set(labels.values())
        if "supported" not in values or not values & {"unsupported", "contradicted"}:
            continue
        q = ctx.question(d.question_id)
        claim = next(c for c in q.claims if c.position == d.claim_position)
        unit_key = f"compound:{d.question_id}/claim:{d.claim_position}/{decomposition_id}"
        entry_ids = sorted(
            e.entry_id for k in sources for e in ledger.by_unit("judgment").get(k, [])
        )
        out.append(
            NaturalExample(
                example_id=_example_id(ctx.run.run_id, unit_key),
                question_id=q.question_id,
                question_category=q.category,
                answer_status=q.status or q.execution_outcome,
                claim_type=claim.claim_type,
                claim_position=claim.position,
                unit_key=unit_key,
                dimension=Dimension.CLAIM_FAITHFULNESS.value,
                label=derive_faithfulness(d, labels),
                category=Category.COMPOUND_MIXED_SUPPORT,
                agreement="derived_from_subclaims",
                source_entry_ids=tuple(entry_ids),
            )
        )
    return out


SELECTION_POLICY: Final[dict[str, Any]] = {
    "calibration_version": CALIBRATION_VERSION,
    "seed": "founderlens-calibration-v1",
    "inputs": "active human labels with role calibration on eligible development answers",
    "review_states_admitted": [s.value for s in ADMITTED_STATES],
    "single_reviewed": "admitted and labelled single_reviewed; never reported as agreement",
    "agreement": "independent agreement needs two or more distinct reviewers; agreement "
    "statistics are computed only on such units (Commit 7.4), never here",
    "partition": "a question belongs to the partition of its first ledger entry; calibration "
    "questions never supply evaluation examples, and synthetic variants inherit it",
    "excluded": [
        "disputed units (no adjudication)",
        "withdrawn units",
        "not_assessable labels",
        "unverified and not_applicable labels (no gold label)",
        "units of evaluation-partition questions",
        "model-output failures (no units exist)",
    ],
    "excluded_units_reported": "every excluded unit is listed with its reason and state",
    "never_uses": ["judge outputs", "held-out split", "synthetic variants"],
    "max_per_category": 6,
    "max_total": 80,
    "stratify_by": ["question_category", "answer_status", "claim_type"],
    "within_stratum_order": "sha256(seed|example_id) ascending",
    "within_category": "round robin over strata in sorted order",
    "across_categories": "round robin in category enum order until max_total",
}


def selection_policy_id() -> str:
    canonical = json.dumps(SELECTION_POLICY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _rank(example: NaturalExample) -> str:
    return hashlib.sha256(f"{SELECTION_POLICY['seed']}|{example.example_id}".encode()).hexdigest()


def _round_robin(groups: Sequence[Sequence[NaturalExample]]) -> list[NaturalExample]:
    out: list[NaturalExample] = []
    for i in range(max((len(g) for g in groups), default=0)):
        out += [g[i] for g in groups if i < len(g)]
    return out


def select(candidates: Sequence[NaturalExample]) -> list[NaturalExample]:
    """Deterministic, stratified selection; independent of input order."""
    per_category: list[list[NaturalExample]] = []
    for category in Category:
        strata: dict[tuple[str, str, str], list[NaturalExample]] = {}
        for x in candidates:
            if x.category is category:
                key = (x.question_category, x.answer_status, x.claim_type or "-")
                strata.setdefault(key, []).append(x)
        ordered = [sorted(strata[k], key=_rank) for k in sorted(strata)]
        per_category.append(_round_robin(ordered)[: SELECTION_POLICY["max_per_category"]])
    return _round_robin(per_category)[: SELECTION_POLICY["max_total"]]


# --------------------------------------------------------------------------- #
# Synthetic variants
# --------------------------------------------------------------------------- #

METRIC_SWAPS: Final = (
    ("net income", "operating income"),
    ("operating income", "net income"),
    ("gross margin", "operating margin"),
    ("operating margin", "gross margin"),
    ("revenue", "net income"),
)

TRANSFORMATIONS: Final[dict[str, dict[str, Any]]] = {
    "scale_shift_v1": {
        "source": "supported_claim",
        "rule": "first financial quantity with a spelled scale: billion<->million, bn<->mn",
        "expected": ["wrong_scale", "numerical_correctness", "unit_scale", "incorrect"],
    },
    "company_swap_v1": {
        "source": "supported_claim",
        "rule": "earliest exact-case whole-word corpus company alias -> the next manifest "
        "company's first alias",
        "expected": ["wrong_company", "financial_attribution", "company", "incorrect"],
    },
    "period_shift_v1": {
        "source": "supported_claim",
        "rule": "first whole-word quarter token Qn -> Q(n mod 4 + 1)",
        "expected": ["wrong_period", "financial_attribution", "reporting_period", "incorrect"],
    },
    "metric_swap_v1": {
        "source": "supported_claim",
        "rule": "earliest metric phrase of METRIC_SWAPS (case-insensitive, whole words) -> "
        "its fixed replacement",
        "table": [list(pair) for pair in METRIC_SWAPS],
        "expected": ["wrong_metric", "claim_faithfulness", None, "unsupported"],
    },
    "percent_shift_v1": {
        "source": "supported_claim of claim_type calculated",
        "rule": "first percent or percentage-point quantity + one unit of stated precision",
        "expected": [
            "incorrect_calculation",
            "numerical_correctness",
            "calculated_comparison",
            "incorrect",
        ],
    },
    "citation_swap_v1": {
        "source": "supported_citation",
        "rule": "replace the cited evidence ID by the best-ranked selected evidence ID the "
        "claim does not cite (text unchanged)",
        "expected": ["invalid_citation", "citation_support", None, "does_not_support"],
    },
    "abstention_flip_v1": {
        "source": "correct_attempt on an answerable question",
        "rule": "the response becomes insufficient_evidence with no claims (structure only)",
        "expected": ["false_abstention", "abstention_appropriateness", None, "false_abstention"],
    },
    "compound_join_v1": {
        "source": "lowest-position supported_claim and unsupported_claim of one question",
        "rule": "join the two claim texts with '; and ' into one claim",
        "expected": ["compound_mixed_support", "claim_faithfulness", None, "unsupported"],
    },
}


def transformation_id() -> str:
    canonical = json.dumps(
        {"calibration_version": CALIBRATION_VERSION, "rules": TRANSFORMATIONS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class SyntheticVariant:
    variant_id: str
    question_id: str
    rule_id: str
    source_example_ids: tuple[str, ...]
    claim_position: int | None
    source_text_sha256: str | None
    variant_text_sha256: str | None
    variant_evidence_ids: tuple[str, ...] | None
    variant_status: str | None
    expected: Mapping[str, str | None]
    review: str  # pending | accepted | rejected | disagreement
    variant_text: str | None = field(default=None, repr=False)  # private

    def record(self, *, private: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "variant_id": self.variant_id,
            "synthetic": True,
            "question_id": self.question_id,
            "rule_id": self.rule_id,
            "transformation_id": transformation_id(),
            "source_example_ids": list(self.source_example_ids),
            "claim_position": self.claim_position,
            "source_text_sha256": self.source_text_sha256,
            "variant_text_sha256": self.variant_text_sha256,
            "variant_evidence_ids": None
            if self.variant_evidence_ids is None
            else list(self.variant_evidence_ids),
            "variant_status": self.variant_status,
            "expected": dict(self.expected),
            "expected_label_source": "transformation_rule (not a human judgment)",
            "human_review": self.review,
            "partition": "calibration",  # inherited from the source question
            "source_question_id": self.question_id,
        }
        if private:
            out["variant_text"] = self.variant_text
        return out


def _scale_shift(text: str, occurrences: Sequence[fc.NumericOccurrence]) -> str | None:
    swap = {"billion": "million", "million": "billion", "bn": "mn", "mn": "bn"}
    for occ in occurrences:
        if occ.kind is fc.OccurrenceKind.FINANCIAL_QUANTITY and occ.quantity is not None:
            span = text[occ.start : occ.end]
            m = re.search(r"\b(billion|million|bn|mn)\b", span, re.IGNORECASE)
            if m:
                word = m.group(1)
                new = swap[word.lower()]
                new = new.capitalize() if word[0].isupper() else new
                at = occ.start + m.start(1)
                return text[:at] + new + text[at + len(word) :]
    return None


def _company_swap(text: str, ctx: an.AnnotationContext) -> str | None:
    companies = list(ctx.vocabulary.companies)
    best: tuple[int, int, str] | None = None
    for i, company in enumerate(companies):
        for alias, _ in company.aliases:
            m = re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", text)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), i, alias)
    if best is None or len(companies) < 2:
        return None
    start, i, alias = best
    target = companies[(i + 1) % len(companies)].aliases[0][0]
    return text[:start] + target + text[start + len(alias) :]


def _period_shift(text: str) -> str | None:
    m = re.search(r"(?<![A-Za-z0-9])Q([1-4])(?![A-Za-z0-9])", text)
    if m is None:
        return None
    return text[: m.start(1)] + str(int(m.group(1)) % 4 + 1) + text[m.end(1) :]


def _metric_swap(text: str) -> str | None:
    best: tuple[int, str, str] | None = None
    for phrase, replacement in METRIC_SWAPS:
        m = re.search(rf"(?<![A-Za-z]){re.escape(phrase)}(?![A-Za-z])", text, re.IGNORECASE)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.group(0), replacement)
    if best is None:
        return None
    start, found, replacement = best
    new = replacement[0].upper() + replacement[1:] if found[0].isupper() else replacement
    return text[:start] + new + text[start + len(found) :]


def _percent_shift(text: str, occurrences: Sequence[fc.NumericOccurrence]) -> str | None:
    units = (fc.Unit.PERCENT, fc.Unit.PERCENTAGE_POINTS)
    for occ in occurrences:
        if occ.quantity is not None and occ.quantity.unit in units:
            span = text[occ.start : occ.end]
            m = re.search(r"\d+(?:\.\d+)?", span)
            if m is None:
                continue
            decimals = occ.quantity.decimals
            shifted = Decimal(m.group(0)) + Decimal(1).scaleb(-decimals)
            at = occ.start + m.start()
            return text[:at] + f"{shifted:.{decimals}f}" + text[at + len(m.group(0)) :]
    return None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def synthetic_variants(ctx: an.AnnotationContext, ledger: an.Ledger) -> list[SyntheticVariant]:
    """Every rule applied to every eligible natural source example. Deterministic.

    Sources are read, never modified; a variant is a new private object.
    """
    candidates, _ = natural_candidates(ctx, ledger)
    reviews = ledger.by_unit("synthetic_review")
    tid = transformation_id()
    out: list[SyntheticVariant] = []

    def add(
        rule: str,
        sources: Sequence[NaturalExample],
        *,
        claim: ji.VerifiedClaim | None = None,
        text: str | None = None,
        evidence: tuple[str, ...] | None = None,
        status: str | None = None,
    ) -> None:
        category, dimension, aspect, label = TRANSFORMATIONS[rule]["expected"]
        body = {
            "rule": rule,
            "transformation_id": tid,
            "sources": [s.example_id for s in sources],
            "source_text": None if claim is None else claim.text_sha256,
            "variant_text": None if text is None else _sha(text),
            "evidence": evidence,
            "status": status,
        }
        variant_id = an.digest(body)[:16]
        decisions = {
            str(e.payload.get("decision")) for e in reviews.get(f"synthetic:{variant_id}", [])
        }
        review = (
            "pending"
            if not decisions
            else "disagreement"
            if len(decisions) > 1
            else "accepted"
            if decisions == {"accept_expected_label"}
            else "rejected"
        )
        out.append(
            SyntheticVariant(
                variant_id=variant_id,
                question_id=sources[0].question_id,
                rule_id=rule,
                source_example_ids=tuple(s.example_id for s in sources),
                claim_position=None if claim is None else claim.position,
                source_text_sha256=None if claim is None else claim.text_sha256,
                variant_text_sha256=None if text is None else _sha(text),
                variant_evidence_ids=evidence,
                variant_status=status,
                expected={
                    "category": category,
                    "dimension": dimension,
                    "aspect": aspect,
                    "label": label,
                },
                review=review,
                variant_text=text,
            )
        )

    def claim_of(x: NaturalExample) -> ji.VerifiedClaim | None:
        q = ctx.question(x.question_id)
        return next((c for c in q.claims if c.position == x.claim_position), None)

    supported = [
        x
        for x in candidates
        if x.category is Category.SUPPORTED_CLAIM
        and x.unit_key.startswith("judgment:")
        and "/sub:" not in x.unit_key
    ]
    for x in supported:
        claim = claim_of(x)
        if claim is None:
            continue
        occurrences = fc.extract_occurrences(claim.text, ctx.vocabulary)
        texts = {
            "scale_shift_v1": _scale_shift(claim.text, occurrences),
            "company_swap_v1": _company_swap(claim.text, ctx),
            "period_shift_v1": _period_shift(claim.text),
            "metric_swap_v1": _metric_swap(claim.text),
            "percent_shift_v1": _percent_shift(claim.text, occurrences)
            if claim.claim_type == "calculated"
            else None,
        }
        for rule, text in texts.items():
            if text is not None and text != claim.text:
                add(rule, [x], claim=claim, text=text)
    for x in candidates:
        if x.category is Category.SUPPORTED_CITATION:
            claim = claim_of(x)
            q = ctx.question(x.question_id)
            if claim is None:
                continue
            spare = [
                e.evidence_id
                for e in q.selected_evidence
                if e.evidence_id not in claim.evidence_ids
            ]
            if spare:
                cited = x.unit_key.rsplit(":", 1)[-1].split("#", 1)[0]
                swapped = tuple(spare[0] if e == cited else e for e in claim.evidence_ids)
                add("citation_swap_v1", [x], claim=claim, evidence=swapped)
        if x.category is Category.CORRECT_ATTEMPT and ctx.question(x.question_id).answerable:
            add("abstention_flip_v1", [x], status="insufficient_evidence")
    by_question: dict[str, dict[Category, list[NaturalExample]]] = {}
    for x in candidates:
        if x.dimension == Dimension.CLAIM_FAITHFULNESS.value and "/sub:" not in x.unit_key:
            by_question.setdefault(x.question_id, {}).setdefault(x.category, []).append(x)
    for groups in by_question.values():
        good = sorted(groups.get(Category.SUPPORTED_CLAIM, []), key=lambda x: x.claim_position or 0)
        bad = sorted(
            groups.get(Category.UNSUPPORTED_CLAIM, []), key=lambda x: x.claim_position or 0
        )
        if good and bad:
            a, b = claim_of(good[0]), claim_of(bad[0])
            if a is not None and b is not None:
                add("compound_join_v1", [good[0], bad[0]], claim=a, text=f"{a.text}; and {b.text}")
    return sorted(out, key=lambda v: (v.question_id, v.rule_id, v.variant_id))


# --------------------------------------------------------------------------- #
# Calibration set
# --------------------------------------------------------------------------- #

DISCLOSURE: Final = (
    "Small development sample from one frozen run of one model; agreement measured on it is "
    "a development diagnostic, not a held-out or population estimate."
)


def build_calibration_set(ctx: an.AnnotationContext, ledger: an.Ledger) -> dict[str, Any]:
    candidates, excluded_units = classify_units(ctx, ledger)
    _, excluded = natural_candidates(ctx, ledger)
    selected = select(candidates)
    by_state: dict[str, int] = {}
    for x in selected:
        by_state[x.agreement] = by_state.get(x.agreement, 0) + 1
    independent = ledger.independently_reviewed_units()
    chosen = {x.example_id for x in selected}
    variants = [v for v in synthetic_variants(ctx, ledger) if v.source_example_ids[0] in chosen]
    selection_id = an.digest(
        {
            "policy": selection_policy_id(),
            "run_id": ctx.run.run_id,
            "rubric_id": rubric_id(),
            "candidates": [(x.example_id, x.label, x.category.value) for x in candidates],
        }
    )[:12]
    return {
        "calibration_version": CALIBRATION_VERSION,
        "selection_id": selection_id,
        "selection_policy_id": selection_policy_id(),
        "selection_policy": SELECTION_POLICY,
        "transformation_id": transformation_id(),
        "transformations": TRANSFORMATIONS,
        "run_id": ctx.run.run_id,
        **an.annotation_identity(),
        "ledger_entries": sorted(e.entry_id for e in ledger.entries),
        "disclosure": {
            "note": DISCLOSURE,
            "scheduled_outcomes": len(ctx.run.questions),
            "eligible_answers": sum(q.eligible for q in ctx.run.questions),
            "candidate_examples": len(candidates),
            "excluded": excluded,
            "selected_natural": len(selected),
            "selected_by_review_state": dict(sorted(by_state.items())),
            "independently_reviewed_units": len(independent),
            "agreement_statistics": an.AGREEMENT_NOTE,
        },
        "partition": "calibration",
        "excluded_units": excluded_units,
        "reporting_rule": "natural and synthetic results are reported separately; synthetic "
        "expected labels are rule proposals until a human accepts them",
        "natural": [x.public() for x in selected],
        "synthetic": [v.record(private=True) for v in variants],
    }


def public_summary(calibration: Mapping[str, Any]) -> dict[str, Any]:
    natural: dict[str, int] = {}
    for x in calibration["natural"]:
        natural[x["category"]] = natural.get(x["category"], 0) + 1
    rules: dict[str, int] = {}
    reviews: dict[str, int] = {}
    for v in calibration["synthetic"]:
        rules[v["rule_id"]] = rules.get(v["rule_id"], 0) + 1
        reviews[v["human_review"]] = reviews.get(v["human_review"], 0) + 1
    disclosure = {k: v for k, v in calibration["disclosure"].items() if k != "note"}
    return an.ensure_public(
        {
            "selection_id": calibration["selection_id"],
            "selection_policy_id": calibration["selection_policy_id"],
            "transformation_id": calibration["transformation_id"],
            "run_id": calibration["run_id"],
            "rubric_id": calibration["rubric_id"],
            "disclosure": disclosure,
            "natural_by_category": dict(sorted(natural.items())),
            "synthetic_by_rule": dict(sorted(rules.items())),
            "synthetic_review": dict(sorted(reviews.items())),
        }
    )


def write_calibration_set(ctx: an.AnnotationContext, paths: an.AnnotationPaths) -> dict[str, Any]:
    calibration = build_calibration_set(ctx, an.load_ledger(paths.ledger))
    an.write_json(paths.calibration / f"{calibration['selection_id']}.json", calibration)
    return public_summary(calibration)


def write_review_worksheets(
    ctx: an.AnnotationContext, paths: an.AnnotationPaths, annotator_id: str, role: Role
) -> list[str]:
    """Pending ``synthetic_review`` worksheets for questions with unreviewed variants."""
    calibration = build_calibration_set(ctx, an.load_ledger(paths.ledger))
    by_question: dict[str, list[str]] = {}
    for v in calibration["synthetic"]:
        by_question.setdefault(v["question_id"], []).append(v["variant_id"])
    written = []
    for qid, ids in sorted(by_question.items()):
        template = an.worksheet_template(an.build_packet(ctx, qid), annotator_id, role)
        template["entries"] = [
            {"entry_type": "synthetic_review", "review_action": "pending", "variant_id": v}
            for v in ids
        ]
        path = paths.worksheets / f"{qid}.{annotator_id}.synthetic.json"
        if path.exists():
            continue  # never overwrite reviewer work
        an.write_json(path, template)
        written.append(path.name)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.calibration",
        description="Build the private, versioned calibration set (offline).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "review-worksheets"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--project-root", default=None)
        if name == "review-worksheets":
            p.add_argument("--annotator", required=True)
            p.add_argument("--role", choices=[r.value for r in Role], default="calibration")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        paths = an.annotation_paths(settings, args.run)
        ctx = an.load_context(settings, args.run)
        output: object
        if args.command == "build":
            output = write_calibration_set(ctx, paths)
        else:
            output = {
                "worksheets": write_review_worksheets(ctx, paths, args.annotator, Role(args.role))
            }
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (
        an.AnnotationError,
        ji.JudgingInputError,
        ge.EvaluationRunError,
        EvaluationError,
        RetrievalError,
    ) as exc:
        print(f"Calibration {args.command} FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
