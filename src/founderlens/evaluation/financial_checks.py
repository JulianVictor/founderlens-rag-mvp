"""Deterministic financial checks (Stage 6, Commit 7.2; protocol §15).

Usage::

    python -m founderlens.evaluation.financial_checks --run live-20261124T045739024562Z-8665d379

What this establishes, and what it does not:

- **Extraction** finds numeric occurrences in a generated claim, with exact
  character spans of the original text and a stable ``value_index``. It never
  edits the text and does not judge meaning.
- **A numeric match is not correctness.** A number that also occurs in a
  passage proves nothing. A value is compared only against a *verified
  reference fact* that a named person explicitly associated with that
  occurrence (:class:`FactAssociation`). Without one the result is
  ``unverified``, never ``correct`` or ``incorrect``.
- **Attribution** (company, segment, period, basis) is checked only where the
  claim states it in terms the frozen scope analyzer recognizes. Silence or
  ambiguity is ``unverified`` or ``ambiguous``; a document's reporting period is
  never assumed for a value; fiscal and calendar periods are never equated;
  GAAP/non-GAAP keywords are a signal, never verification.
- **Calculations** are checked only for an explicitly declared formula over
  verified facts. Mathematical validity is reported separately from factual
  correctness; correct arithmetic on wrong operands never passes.
- All arithmetic uses :class:`~decimal.Decimal`. There is no global tolerance:
  a coarser claim may match only by rounding the verified value to the claim's
  own stated precision (half-up), with at least two significant digits.

The guardrail numeric trace (policy ``3f202ebc205a``) is not reused: it is part
of a frozen policy and has no spans, signs, percentage points or ratios. The
frozen scope analyzer and vocabulary are reused unchanged for period, company
and segment recognition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Final

from founderlens.evaluation.judgments import (
    ClaimValueUnit,
    EvidenceReference,
    Judgment,
    NumericAspect,
    Provenance,
    ReferenceBasis,
    Role,
    Source,
    rubric_id,
)
from founderlens.guardrails.scope import CompanyScope, CorpusVocabulary, analyze_question

EXTRACTOR_VERSION: Final = 1
CHECKER_VERSION: Final = 2  # v2: a reference basis may be 'unverified'
ROUNDING_MODE: Final = ROUND_HALF_UP
MINUS_SIGNS: Final = ("-", "\u2212")  # hyphen-minus, minus sign
DASHES: Final = "-\u2013"  # hyphen-minus, en dash (ranges)
MIN_SIGNIFICANT_DIGITS_FOR_ROUNDING: Final = 2


class FinancialCheckError(ValueError):
    """Malformed input to the checker (never quotes claim text)."""


# --------------------------------------------------------------------------- #
# Units, scales and occurrence kinds
# --------------------------------------------------------------------------- #


class Unit(StrEnum):
    MONEY = "money"
    PERCENT = "percent"
    PERCENTAGE_POINTS = "percentage_points"
    RATIO = "ratio"  # "2.5x", "3 times"
    UNITLESS = "unitless"


class OccurrenceKind(StrEnum):
    FINANCIAL_QUANTITY = "financial_quantity"
    UNITLESS_NUMBER = "unitless_number"  # a number with no unit, currency or scale
    PERIOD_REFERENCE = "period_reference"  # fiscal/calendar year, quarter, period-end date
    YEAR = "year"  # a bare four-digit year
    PAGE_REFERENCE = "page_reference"
    AMBIGUOUS = "ambiguous"


class Reason(StrEnum):
    UNIT_ATTACHED = "unit_attached"  # "26.3billion": parsed, flagged
    RANGE_EXPRESSION = "range_expression"
    DOLLAR_SIGN_UNQUALIFIED = "dollar_sign_unqualified"
    MISSING_UNIT = "missing_unit"
    MISSING_CURRENCY = "missing_currency"
    CURRENCY_MISMATCH = "currency_mismatch_requires_conversion"
    UNIT_KIND_MISMATCH = "unit_kind_mismatch"
    INSUFFICIENT_PRECISION = "insufficient_precision"
    CLAIM_MORE_PRECISE = "claim_more_precise_than_reference"
    SIGN_NOT_EXPLICIT = "sign_not_explicit"
    PARENTHESES_SIGN_AMBIGUOUS = "parenthesized_sign_ambiguous"  # accounting negative?
    NO_VERIFIED_ASSOCIATION = "no_verified_association"
    ASSOCIATION_MISMATCH = "association_does_not_match_occurrence"
    METRIC_MISMATCH = "association_metric_differs_from_fact"
    NOT_A_FINANCIAL_QUANTITY = "not_a_financial_quantity"
    COMPANY_NOT_STATED = "company_not_stated"
    COMPANY_MISMATCH = "company_mismatch"
    MULTIPLE_COMPANIES = "multiple_companies"
    SEGMENT_VS_CONSOLIDATED = "segment_vs_consolidated"
    SEGMENT_NOT_STATED = "segment_not_stated"
    PERIOD_NOT_STATED = "period_not_stated"
    PERIOD_MISMATCH = "period_mismatch"
    PERIOD_BASIS_DIFFERS = "fiscal_calendar_basis_differs"
    MULTIPLE_PERIODS = "multiple_periods"
    BASIS_SIGNAL_ONLY = "basis_keyword_signal_only"
    BASIS_SIGNAL_CONFLICT = "basis_keyword_conflicts_with_fact"
    BASIS_NOT_STATED = "basis_not_stated"
    BASIS_NOT_VERIFIED = "reference_basis_not_verified"
    MISSING_OPERAND = "missing_verified_operand"
    INCOMPATIBLE_OPERANDS = "incompatible_operands"
    DIVISION_BY_ZERO = "division_by_zero"
    INCORRECT_OPERANDS = "incorrect_operands"
    ARITHMETIC_ERROR = "arithmetic_error"
    UNDECLARED_FORMULA = "formula_not_declared"


class Outcome(StrEnum):
    EXACT_MATCH = "exact_match"
    ROUNDING_MATCH = "rounding_match"
    MISMATCH = "mismatch"
    AMBIGUOUS = "ambiguous"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"


PASSING: Final = frozenset({Outcome.EXACT_MATCH, Outcome.ROUNDING_MATCH, Outcome.NOT_APPLICABLE})

SCALES: Final[dict[str, tuple[str, Decimal]]] = {
    # spelled-out or abbreviated scale word -> (canonical scale, multiplier)
    "thousand": ("thousand", Decimal(10) ** 3),
    "million": ("million", Decimal(10) ** 6),
    "mn": ("million", Decimal(10) ** 6),
    "mm": ("million", Decimal(10) ** 6),
    "billion": ("billion", Decimal(10) ** 9),
    "bn": ("billion", Decimal(10) ** 9),
    "trillion": ("trillion", Decimal(10) ** 12),
    "tn": ("trillion", Decimal(10) ** 12),
}
LETTER_SCALES: Final[dict[str, tuple[str, Decimal]]] = {
    # single letters count only when attached to the number ("$35.6B")
    "k": ("thousand", Decimal(10) ** 3),
    "m": ("million", Decimal(10) ** 6),
    "b": ("billion", Decimal(10) ** 9),
    "t": ("trillion", Decimal(10) ** 12),
}
CURRENCY_MARKERS: Final[dict[str, str | None]] = {
    # explicit marker -> ISO currency; "$" alone is a dollar of unstated country
    "us$": "USD",
    "usd": "USD",
    "$": None,
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<open>\()?"
    r"(?P<sign>[-\u2212+])?"
    r"(?P<cur>US\$|USD\s?|\$|€|£|EUR\s?|GBP\s?)?"
    r"(?P<sign2>[-\u2212])?"
    r"(?P<num>" + _NUM + r")"
    r"(?:"
    r"(?P<sp_pp>\s?(?:percentage\s+points?|pp|ppts?)\b)"
    r"|(?P<sp_bps>\s?(?:basis\s+points?|bps)\b)"
    r"|(?P<sp_pct>\s?(?:%|percent\b|per\s+cent\b))"
    r"|(?P<sp_ratio>(?:x\b|\s?times\b))"
    r"|(?P<sp_scale>\s?(?:thousand|million|billion|trillion|bn|mn|mm|tn)\b)"
    r"|(?P<letter>[KkMmBbTt])(?![A-Za-z0-9])"
    r")?"
    r"(?P<close>\))?"
    r"(?P<post_scale>\s?(?:thousand|million|billion|trillion)\b)?"
    r"(?P<curpost>\s?(?:USD|EUR|GBP)\b)?",
    re.IGNORECASE,
)
_PAGE_BEFORE = re.compile(r"(?:\bpage|\bp\.|\bpp\.)\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Quantity:
    """A stated number in canonical form. ``value`` is in base units (dollars, %, pp, x)."""

    magnitude: Decimal  # as written, unsigned, unscaled ("26.3")
    sign: int  # -1, 1
    explicit_sign: bool  # minus sign or accounting parentheses
    unit: Unit
    scale: str | None  # thousand / million / billion / trillion
    currency_marker: str | None  # "$", "US$", "€", ... as normalized lower-case key
    currency: str | None  # ISO code when explicit; None for "$" alone or none
    decimals: int  # decimal places as written
    sign_ambiguous: bool = False  # parenthesized: prose aside or accounting negative

    @property
    def multiplier(self) -> Decimal:
        if self.scale is None:
            return Decimal(1)
        return next(m for name, m in SCALES.values() if name == self.scale)

    @property
    def value(self) -> Decimal:
        base = self.magnitude * self.multiplier
        if self.unit is Unit.PERCENTAGE_POINTS and self.scale == "bps":  # pragma: no cover
            return base
        return base * self.sign

    @property
    def quantum(self) -> Decimal:
        """Stated precision in base units (e.g. 0.1 billion = 10^8)."""
        return Decimal(1).scaleb(-self.decimals) * self.multiplier

    @property
    def significant_digits(self) -> int:
        digits = self.magnitude.normalize().as_tuple().digits
        stated = len(str(self.magnitude).replace(".", "").lstrip("0")) or 1
        return max(stated, len(digits))


@dataclass(frozen=True, slots=True)
class NumericOccurrence:
    """One numeric token of a claim. Spans index the original claim text (code points)."""

    value_index: int  # 1-based, by start offset, over every numeric token of the claim
    start: int
    end: int
    kind: OccurrenceKind
    quantity: Quantity | None
    reasons: tuple[Reason, ...] = ()

    def public(self) -> dict[str, Any]:
        """Span, kind and codes; no text and no value."""
        return {
            "value_index": self.value_index,
            "start": self.start,
            "end": self.end,
            "kind": self.kind.value,
            "unit": None if self.quantity is None else self.quantity.unit.value,
            "reasons": [r.value for r in self.reasons],
        }


def _number(raw: str) -> tuple[Decimal, int]:
    cleaned = raw.replace(",", "")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:  # pragma: no cover - the pattern only admits digits
        raise FinancialCheckError("unparseable number") from None
    decimals = len(cleaned.split(".", 1)[1]) if "." in cleaned else 0
    return value, decimals


def extract_occurrences(text: str, vocabulary: CorpusVocabulary) -> tuple[NumericOccurrence, ...]:
    """Every numeric token of ``text``, classified, in source order.

    ``value_index`` convention (extractor v1): every maximal numeric token that is
    not embedded in an alphanumeric identifier (``FY2025``, ``Q4``, ``E01``,
    ``H100``) gets an index, 1-based by start offset, including non-financial
    tokens (years, periods, pages), so indexes do not depend on how a token is
    classified. Repeated values get distinct indexes by position.
    """
    if not isinstance(text, str):
        raise FinancialCheckError("claim text must be a string")
    scope = analyze_question(text, vocabulary)
    periods = [(p.start, p.end) for p in scope.periods]
    periods += [(u.start, u.end) for u in scope.unresolved]
    found: list[NumericOccurrence] = []
    for match in _TOKEN.finditer(text):
        start, end = match.start(), match.end()
        # trim trailing whitespace and an unmatched opening parenthesis from the span
        while end > start and text[end - 1].isspace():
            end -= 1
        if match.group("open") and not match.group("close"):
            start += 1
        if match.group("close") and not match.group("open") and not match.group("post_scale"):
            end = match.start("close")  # a closing bracket of surrounding text
            while end > start and text[end - 1].isspace():
                end -= 1
        found.append(_classify(text, match, start, end, periods, len(found) + 1))
    return tuple(found)


def _classify(
    text: str,
    match: re.Match[str],
    start: int,
    end: int,
    periods: Sequence[tuple[int, int]],
    index: int,
) -> NumericOccurrence:
    g = match.groupdict()
    num_start = match.start("num")
    magnitude, decimals = _number(g["num"])
    reasons: list[Reason] = []

    def occurrence(kind: OccurrenceKind, quantity: Quantity | None = None) -> NumericOccurrence:
        return NumericOccurrence(index, start, end, kind, quantity, tuple(reasons))

    if any(s < match.end("num") and num_start < e for s, e in periods):
        return occurrence(OccurrenceKind.PERIOD_REFERENCE)
    if _PAGE_BEFORE.search(text[:start]):
        return occurrence(OccurrenceKind.PAGE_REFERENCE)
    before = text[max(0, num_start - 1) : num_start]
    after = text[match.end("num") : match.end("num") + 2]
    if (before and before in DASHES and num_start >= 2 and text[num_start - 2].isdigit()) or (
        len(after) == 2 and after[0] in DASHES and after[1].isdigit()
    ):
        reasons.append(Reason.RANGE_EXPRESSION)
        return occurrence(OccurrenceKind.AMBIGUOUS)
    unit = Unit.UNITLESS
    scale: str | None = None
    if g["sp_pp"]:
        unit = Unit.PERCENTAGE_POINTS
    elif g["sp_bps"]:
        unit, magnitude, decimals = Unit.PERCENTAGE_POINTS, magnitude / 100, decimals + 2
    elif g["sp_pct"]:
        unit = Unit.PERCENT
    elif g["sp_ratio"]:
        unit = Unit.RATIO
    elif g["sp_scale"]:
        word = g["sp_scale"].strip().casefold()
        scale = SCALES[word][0]
        if not g["sp_scale"][:1].isspace():
            reasons.append(Reason.UNIT_ATTACHED)
    elif g["letter"]:
        scale = LETTER_SCALES[g["letter"].casefold()][0]
    elif g["post_scale"] and g["open"] and g["close"]:  # "($1.2) million"
        scale = SCALES[g["post_scale"].strip().casefold()][0]
    marker_raw = (g["cur"] or g["curpost"] or "").strip().casefold()
    marker = marker_raw or None
    currency = None if marker is None else CURRENCY_MARKERS.get(marker)
    if marker is not None and unit is Unit.UNITLESS:
        unit = Unit.MONEY
    if marker == "$":
        reasons.append(Reason.DOLLAR_SIGN_UNQUALIFIED)
    negative = bool(g["sign"] in MINUS_SIGNS or g["sign2"])
    parenthesized = bool(g["open"] and g["close"]) and not negative
    if parenthesized:
        reasons.append(Reason.PARENTHESES_SIGN_AMBIGUOUS)
    quantity = Quantity(
        magnitude=magnitude,
        sign=-1 if negative else 1,
        explicit_sign=negative or g["sign"] == "+",
        unit=unit,
        scale=scale,
        currency_marker=marker,
        currency=currency,
        decimals=decimals,
        sign_ambiguous=parenthesized,
    )
    if unit is Unit.UNITLESS and scale is None:
        plain = g["num"]
        if re.fullmatch(r"(19|20)\d{2}", plain):
            return occurrence(OccurrenceKind.YEAR)
        reasons.append(Reason.MISSING_UNIT)
        return occurrence(OccurrenceKind.UNITLESS_NUMBER, quantity)
    if unit is Unit.UNITLESS:  # a scale without currency ("26.3 billion")
        reasons.append(Reason.MISSING_CURRENCY)
    return occurrence(OccurrenceKind.FINANCIAL_QUANTITY, quantity)


# --------------------------------------------------------------------------- #
# Verified reference facts and explicit associations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ReportingPeriod:
    label: str  # canonical scope-analyzer label, e.g. "FY2025 Q4", "CY2024 Q3"
    basis: str  # fiscal | calendar
    company: str | None = None  # fiscal labels are company-relative

    def __post_init__(self) -> None:
        if self.basis not in ("fiscal", "calendar"):
            raise FinancialCheckError("period basis must be fiscal or calendar")


@dataclass(frozen=True, slots=True)
class VerifiedReferenceFact:
    """A financial fact verified by a named person against trusted source evidence.

    Construction requires verification provenance and at least one evidence
    reference; nothing here verifies facts automatically.
    """

    fact_id: str
    metric: str
    company: str
    segment: str | None  # None: the consolidated company
    period: ReportingPeriod
    value: Quantity
    basis: str  # GAAP | non-GAAP | not_applicable | unverified (source does not establish it)
    evidence: tuple[EvidenceReference, ...]
    verified_by: str  # pseudonymous annotator ID
    verification_role: Role

    def __post_init__(self) -> None:
        if not self.evidence:
            raise FinancialCheckError("a verified fact needs trusted source evidence")
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", self.verified_by):
            raise FinancialCheckError("verification provenance must name an annotator")
        if self.basis not in ("GAAP", "non-GAAP", "not_applicable", "unverified"):
            raise FinancialCheckError("unknown accounting basis")
        if self.value.unit is Unit.UNITLESS:
            raise FinancialCheckError("a verified fact states its unit")
        if self.value.unit is Unit.MONEY and self.value.currency is None:
            raise FinancialCheckError("a verified monetary fact states its currency")


@dataclass(frozen=True, slots=True)
class FactAssociation:
    """An explicit, attributable link from one claim occurrence to one verified fact."""

    question_id: str
    generation_record_sha256: str
    claim_position: int
    value_index: int
    start: int
    end: int
    fact_id: str
    metric: str  # the metric the associator says the claim value expresses
    associated_by: str  # pseudonymous annotator ID; never automatic

    def __post_init__(self) -> None:
        problems = []
        if not re.fullmatch(r"dev-\d{3}", self.question_id):
            problems.append("question ID")
        if not re.fullmatch(r"[0-9a-f]{64}", self.generation_record_sha256):
            problems.append("generation record digest")
        if self.claim_position < 1 or self.value_index < 1:
            problems.append("claim position or value index")
        if not 0 <= self.start < self.end:
            problems.append("span")
        for name in ("fact_id", "metric", "associated_by"):
            if not re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", getattr(self, name)):
                problems.append(name)
        if problems:
            raise FinancialCheckError("malformed association: " + ", ".join(problems))


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One deterministic check of one aspect of one claim value."""

    question_id: str
    claim_position: int
    value_index: int
    aspect: str  # a NumericAspect value, or company / segment / metric
    outcome: Outcome
    reasons: tuple[Reason, ...]
    fact_id: str | None
    checker_id: str

    def public(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "claim_position": self.claim_position,
            "value_index": self.value_index,
            "aspect": self.aspect,
            "outcome": self.outcome.value,
            "reasons": [r.value for r in self.reasons],
            "fact_id": self.fact_id,
            "checker_id": self.checker_id,
        }


# --------------------------------------------------------------------------- #
# Value comparison (Decimal, stated-precision rounding)
# --------------------------------------------------------------------------- #


def _round_to(value: Decimal, quantum: Decimal) -> Decimal:
    return (value / quantum).quantize(Decimal(1), rounding=ROUNDING_MODE) * quantum


def compare_values(claim: Quantity, fact: Quantity) -> tuple[Outcome, tuple[Reason, ...]]:
    """Compare a claimed value with a verified value; never a universal tolerance."""
    if claim.unit is Unit.UNITLESS:
        return Outcome.UNVERIFIED, (Reason.MISSING_UNIT,)
    if claim.unit is not fact.unit:
        return Outcome.MISMATCH, (Reason.UNIT_KIND_MISMATCH,)
    if fact.unit is Unit.MONEY and claim.scale is None and fact.scale is not None:
        return Outcome.UNVERIFIED, (Reason.MISSING_UNIT,)
    reasons: list[Reason] = []
    claimed = claim.value
    verified = fact.value
    if claim.sign_ambiguous and abs(claimed) == abs(verified):
        return Outcome.AMBIGUOUS, (Reason.PARENTHESES_SIGN_AMBIGUOUS,)
    if not claim.explicit_sign and abs(claimed) == abs(verified) and claimed != verified:
        return Outcome.AMBIGUOUS, (Reason.SIGN_NOT_EXPLICIT,)
    if claimed == verified:
        return Outcome.EXACT_MATCH, ()
    if claim.quantum > fact.quantum:  # the claim is coarser: round the verified value
        if _round_to(verified, claim.quantum) != claimed:
            return Outcome.MISMATCH, ()
        if claim.significant_digits < MIN_SIGNIFICANT_DIGITS_FOR_ROUNDING:
            return Outcome.AMBIGUOUS, (Reason.INSUFFICIENT_PRECISION,)
        return Outcome.ROUNDING_MATCH, ()
    # A finer claim has digits the reference cannot confirm.
    if claim.quantum < fact.quantum and _round_to(claimed, fact.quantum) == verified:
        reasons.append(Reason.CLAIM_MORE_PRECISE)
        return Outcome.AMBIGUOUS, tuple(reasons)
    return Outcome.MISMATCH, ()


# --------------------------------------------------------------------------- #
# Claim-level checks
# --------------------------------------------------------------------------- #


_NON_GAAP = re.compile(r"\bnon[-\s]?gaap\b|\badjusted\b", re.IGNORECASE)
_GAAP = re.compile(r"(?<![-\w])gaap\b", re.IGNORECASE)


def basis_signal(text: str) -> str:
    """A keyword signal only: ``non-GAAP`` / ``GAAP`` / ``both`` / ``none``."""
    non = bool(_NON_GAAP.search(text))
    plain = bool(_GAAP.search(_NON_GAAP.sub(" ", text)))
    return "both" if non and plain else "non-GAAP" if non else "GAAP" if plain else "none"


def _period_basis(kind: str) -> str | None:
    if kind.startswith("fiscal"):
        return "fiscal"
    if kind.startswith("calendar"):
        return "calendar"
    return None  # basis-unspecified or a date: never equated with either


def check_value(
    *,
    question_id: str,
    generation_record_sha256: str,
    claim_position: int,
    claim_text: str,
    occurrences: Sequence[NumericOccurrence],
    value_index: int,
    association: FactAssociation | None,
    facts: Mapping[str, VerifiedReferenceFact],
    vocabulary: CorpusVocabulary,
) -> tuple[CheckResult, ...]:
    """Every applicable aspect of one claim value against its associated fact."""
    cid = checker_id()

    def result(
        aspect: str, outcome: Outcome, reasons: Sequence[Reason], fact: str | None
    ) -> CheckResult:
        return CheckResult(
            question_id, claim_position, value_index, aspect, outcome, tuple(reasons), fact, cid
        )

    occurrence = next((o for o in occurrences if o.value_index == value_index), None)
    if occurrence is None:
        raise FinancialCheckError("value_index does not name an occurrence of this claim")
    checkable = occurrence.kind is OccurrenceKind.FINANCIAL_QUANTITY or (
        # an explicitly associated unitless number is checked (and stays unverified)
        occurrence.kind is OccurrenceKind.UNITLESS_NUMBER and association is not None
    )
    if not checkable or occurrence.quantity is None:
        return (
            result(
                NumericAspect.VALUE.value,
                Outcome.NOT_APPLICABLE,
                (Reason.NOT_A_FINANCIAL_QUANTITY,),
                None,
            ),
        )
    if association is None:
        return (
            result(
                NumericAspect.VALUE.value,
                Outcome.UNVERIFIED,
                (Reason.NO_VERIFIED_ASSOCIATION,),
                None,
            ),
        )
    fact = facts.get(association.fact_id)
    binding = (
        association.question_id == question_id
        and association.generation_record_sha256 == generation_record_sha256
        and association.claim_position == claim_position
        and association.value_index == value_index
        and (association.start, association.end) == (occurrence.start, occurrence.end)
    )
    if fact is None or not binding:
        return (
            result(
                NumericAspect.VALUE.value,
                Outcome.UNVERIFIED,
                (Reason.ASSOCIATION_MISMATCH,),
                association.fact_id,
            ),
        )
    claim = occurrence.quantity
    out: list[CheckResult] = []
    value_outcome, value_reasons = compare_values(claim, fact.value)
    out.append(result(NumericAspect.VALUE.value, value_outcome, value_reasons, fact.fact_id))
    # unit and scale
    if claim.unit is Unit.UNITLESS:
        unit_outcome, unit_reasons = Outcome.UNVERIFIED, [Reason.MISSING_UNIT]
    elif claim.unit is not fact.value.unit:
        unit_outcome, unit_reasons = Outcome.MISMATCH, [Reason.UNIT_KIND_MISMATCH]
    else:
        unit_outcome, unit_reasons = Outcome.EXACT_MATCH, []
    out.append(result(NumericAspect.UNIT_SCALE.value, unit_outcome, unit_reasons, fact.fact_id))
    # currency
    if fact.value.unit is not Unit.MONEY:
        out.append(result(NumericAspect.CURRENCY.value, Outcome.NOT_APPLICABLE, (), fact.fact_id))
    elif claim.currency_marker is None:
        out.append(
            result(
                NumericAspect.CURRENCY.value,
                Outcome.UNVERIFIED,
                (Reason.MISSING_CURRENCY,),
                fact.fact_id,
            )
        )
    elif claim.currency is None:
        out.append(
            result(
                NumericAspect.CURRENCY.value,
                Outcome.UNVERIFIED,
                (Reason.DOLLAR_SIGN_UNQUALIFIED,),
                fact.fact_id,
            )
        )
    elif claim.currency != fact.value.currency:
        out.append(
            result(
                NumericAspect.CURRENCY.value,
                Outcome.MISMATCH,
                (Reason.CURRENCY_MISMATCH,),
                fact.fact_id,
            )
        )
    else:
        out.append(result(NumericAspect.CURRENCY.value, Outcome.EXACT_MATCH, (), fact.fact_id))
    scope = analyze_question(claim_text, vocabulary)
    # metric (asserted by the associator; a correct number for the wrong metric fails)
    metric_ok = association.metric == fact.metric
    out.append(
        result(
            "metric",
            Outcome.EXACT_MATCH if metric_ok else Outcome.MISMATCH,
            () if metric_ok else (Reason.METRIC_MISMATCH,),
            fact.fact_id,
        )
    )
    # company and segment
    companies = {c.company for c in scope.companies}
    segments = {c.segment for c in scope.companies if c.scope is CompanyScope.SEGMENT}
    if not companies:
        out.append(
            result("company", Outcome.UNVERIFIED, (Reason.COMPANY_NOT_STATED,), fact.fact_id)
        )
    elif len(companies) > 1:
        out.append(result("company", Outcome.AMBIGUOUS, (Reason.MULTIPLE_COMPANIES,), fact.fact_id))
    elif fact.company in companies:
        out.append(result("company", Outcome.EXACT_MATCH, (), fact.fact_id))
    else:
        out.append(result("company", Outcome.MISMATCH, (Reason.COMPANY_MISMATCH,), fact.fact_id))
    if fact.segment is None and segments:
        out.append(
            result("segment", Outcome.MISMATCH, (Reason.SEGMENT_VS_CONSOLIDATED,), fact.fact_id)
        )
    elif fact.segment is not None and fact.segment not in segments:
        out.append(
            result(
                "segment",
                Outcome.UNVERIFIED if not segments else Outcome.MISMATCH,
                (Reason.SEGMENT_NOT_STATED if not segments else Reason.SEGMENT_VS_CONSOLIDATED,),
                fact.fact_id,
            )
        )
    else:
        out.append(
            result(
                "segment",
                Outcome.EXACT_MATCH if fact.segment else Outcome.NOT_APPLICABLE,
                (),
                fact.fact_id,
            )
        )
    # reporting period: only what the claim states; never the document's period
    stated = list(scope.periods)
    if not stated:
        period_outcome, period_reasons = Outcome.UNVERIFIED, [Reason.PERIOD_NOT_STATED]
    elif len({p.canonical for p in stated}) > 1:
        period_outcome, period_reasons = Outcome.AMBIGUOUS, [Reason.MULTIPLE_PERIODS]
    else:
        period = stated[0]
        basis = _period_basis(period.kind)
        if basis != fact.period.basis:
            period_outcome, period_reasons = Outcome.AMBIGUOUS, [Reason.PERIOD_BASIS_DIFFERS]
        elif period.canonical == fact.period.label:
            period_outcome, period_reasons = Outcome.EXACT_MATCH, []
        else:
            period_outcome, period_reasons = Outcome.MISMATCH, [Reason.PERIOD_MISMATCH]
    out.append(
        result(NumericAspect.REPORTING_PERIOD.value, period_outcome, period_reasons, fact.fact_id)
    )
    # accounting basis: keywords are a signal only, never verification
    signal = basis_signal(claim_text)
    if fact.basis == "not_applicable":
        out.append(
            result(NumericAspect.FINANCIAL_BASIS.value, Outcome.NOT_APPLICABLE, (), fact.fact_id)
        )
    elif fact.basis == "unverified":
        out.append(
            result(
                NumericAspect.FINANCIAL_BASIS.value,
                Outcome.UNVERIFIED,
                (Reason.BASIS_NOT_VERIFIED,),
                fact.fact_id,
            )
        )
    elif signal == "none":
        out.append(
            result(
                NumericAspect.FINANCIAL_BASIS.value,
                Outcome.UNVERIFIED,
                (Reason.BASIS_NOT_STATED,),
                fact.fact_id,
            )
        )
    elif signal == fact.basis:
        out.append(
            result(
                NumericAspect.FINANCIAL_BASIS.value,
                Outcome.UNVERIFIED,
                (Reason.BASIS_SIGNAL_ONLY,),
                fact.fact_id,
            )
        )
    else:
        out.append(
            result(
                NumericAspect.FINANCIAL_BASIS.value,
                Outcome.AMBIGUOUS,
                (Reason.BASIS_SIGNAL_CONFLICT,),
                fact.fact_id,
            )
        )
    return tuple(out)


def overall(results: Sequence[CheckResult]) -> Outcome:
    """A value passes only if every applicable aspect passes; any mismatch fails it."""
    outcomes = {r.outcome for r in results}
    if Outcome.MISMATCH in outcomes:
        return Outcome.MISMATCH
    if outcomes <= PASSING:
        value = next((r.outcome for r in results if r.aspect == NumericAspect.VALUE.value), None)
        return (
            value
            if value in (Outcome.EXACT_MATCH, Outcome.ROUNDING_MATCH)
            else Outcome.NOT_APPLICABLE
        )
    if Outcome.AMBIGUOUS in outcomes:
        return Outcome.AMBIGUOUS
    return Outcome.UNVERIFIED


# --------------------------------------------------------------------------- #
# Declared calculations
# --------------------------------------------------------------------------- #


class Formula(StrEnum):
    ABSOLUTE_DIFFERENCE = "absolute_difference"  # comparison - base
    PERCENT_CHANGE = "percent_change"  # (comparison - base) / |base| * 100, in percent
    RELATIVE_CHANGE = "relative_change"  # comparison / base, as a ratio
    PERCENTAGE_POINT_DIFFERENCE = "percentage_point_difference"  # comparison% - base%


@dataclass(frozen=True, slots=True)
class CalculationSpec:
    """An explicitly declared formula; never inferred from wording."""

    formula: Formula
    base_fact_id: str
    comparison_fact_id: str
    result_value_index: int
    stated_base_index: int | None = None  # claim occurrences restating the operands
    stated_comparison_index: int | None = None
    declared_by: str = field(default="")

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", self.declared_by):
            raise FinancialCheckError("a calculation is declared by a named annotator")


@dataclass(frozen=True, slots=True)
class CalculationResult:
    mathematical: Outcome  # stated operands -> stated result, independent of facts
    factual: Outcome  # verified facts -> stated result
    overall: Outcome
    reasons: tuple[Reason, ...]


def _apply(
    formula: Formula, base: Quantity, comparison: Quantity
) -> tuple[Quantity | None, Reason | None]:
    def make(value: Decimal, unit: Unit, currency: str | None) -> Quantity:
        sign = -1 if value < 0 else 1
        exponent = value.as_tuple().exponent
        decimals = -exponent if isinstance(exponent, int) and exponent < 0 else 0
        return Quantity(abs(value), sign, True, unit, None, None, currency, decimals)

    if formula is Formula.PERCENTAGE_POINT_DIFFERENCE:
        if base.unit is not Unit.PERCENT or comparison.unit is not Unit.PERCENT:
            return None, Reason.INCOMPATIBLE_OPERANDS
        return make(comparison.value - base.value, Unit.PERCENTAGE_POINTS, None), None
    if (
        base.unit is not comparison.unit
        or base.unit in (Unit.UNITLESS,)
        or (base.unit is Unit.MONEY and base.currency != comparison.currency)
    ):
        return None, Reason.INCOMPATIBLE_OPERANDS
    if formula is Formula.ABSOLUTE_DIFFERENCE:
        return make(comparison.value - base.value, base.unit, base.currency), None
    if base.value == 0:
        return None, Reason.DIVISION_BY_ZERO
    if formula is Formula.PERCENT_CHANGE:
        return make(
            (comparison.value - base.value) / abs(base.value) * 100, Unit.PERCENT, None
        ), None
    return make(comparison.value / base.value, Unit.RATIO, None), None


def _compare_result(stated: Quantity, expected: Quantity) -> Outcome:
    """A computed value has no stated precision: match at the claim's own precision."""
    if stated.unit is not expected.unit:
        return Outcome.MISMATCH
    if stated.value == expected.value:
        return Outcome.EXACT_MATCH
    if _round_to(expected.value, stated.quantum) != stated.value:
        return Outcome.MISMATCH
    if stated.significant_digits < MIN_SIGNIFICANT_DIGITS_FOR_ROUNDING:
        return Outcome.AMBIGUOUS
    return Outcome.ROUNDING_MATCH


def check_calculation(
    spec: CalculationSpec,
    occurrences: Sequence[NumericOccurrence],
    facts: Mapping[str, VerifiedReferenceFact],
) -> CalculationResult:
    by_index = {o.value_index: o for o in occurrences}

    def quantity(index: int | None) -> Quantity | None:
        occ = None if index is None else by_index.get(index)
        return None if occ is None else occ.quantity

    result = quantity(spec.result_value_index)
    if result is None:
        raise FinancialCheckError("the declared result is not a numeric occurrence")
    base_fact, comp_fact = facts.get(spec.base_fact_id), facts.get(spec.comparison_fact_id)
    reasons: list[Reason] = []
    # mathematical validity from the operands the claim itself states
    stated_base, stated_comp = (
        quantity(spec.stated_base_index),
        quantity(spec.stated_comparison_index),
    )
    if stated_base is None or stated_comp is None:
        mathematical = Outcome.NOT_APPLICABLE
    else:
        expected_math, problem = _apply(spec.formula, stated_base, stated_comp)
        if expected_math is None:
            mathematical, reasons = Outcome.AMBIGUOUS, [*reasons, problem] if problem else reasons
        else:
            mathematical = _compare_result(result, expected_math)
            if mathematical is Outcome.MISMATCH:
                reasons.append(Reason.ARITHMETIC_ERROR)
    # factual correctness from verified facts only
    if base_fact is None or comp_fact is None:
        factual = Outcome.UNVERIFIED
        reasons.append(Reason.MISSING_OPERAND)
    else:
        expected, problem = _apply(spec.formula, base_fact.value, comp_fact.value)
        if expected is None:
            factual = Outcome.AMBIGUOUS
            reasons.append(problem or Reason.INCOMPATIBLE_OPERANDS)
        else:
            factual = _compare_result(result, expected)
        for stated, fact in ((stated_base, base_fact), (stated_comp, comp_fact)):
            if stated is not None and compare_values(stated, fact.value)[0] not in (
                Outcome.EXACT_MATCH,
                Outcome.ROUNDING_MATCH,
            ):
                reasons.append(Reason.INCORRECT_OPERANDS)
    if Reason.INCORRECT_OPERANDS in reasons or Outcome.MISMATCH in (mathematical, factual):
        verdict = Outcome.MISMATCH
    elif factual in (Outcome.EXACT_MATCH, Outcome.ROUNDING_MATCH) and mathematical in PASSING:
        verdict = factual
    elif Outcome.AMBIGUOUS in (mathematical, factual):
        verdict = Outcome.AMBIGUOUS
    else:
        verdict = Outcome.UNVERIFIED
    return CalculationResult(mathematical, factual, verdict, tuple(dict.fromkeys(reasons)))


# --------------------------------------------------------------------------- #
# Judgment mapping
# --------------------------------------------------------------------------- #

LABELS: Final = {
    Outcome.EXACT_MATCH: "correct",
    Outcome.ROUNDING_MATCH: "correct",
    Outcome.MISMATCH: "incorrect",
    Outcome.AMBIGUOUS: "unverified",
    Outcome.UNVERIFIED: "unverified",
    Outcome.NOT_APPLICABLE: "not_applicable",
}
NUMERIC_ASPECTS: Final = frozenset(a.value for a in NumericAspect)


IDENTITY_ASPECTS: Final = ("metric", "company", "segment")


def to_judgment(
    result: CheckResult,
    *,
    context: Sequence[CheckResult],
    association: FactAssociation | None,
    fact: VerifiedReferenceFact | None,
    generation_run_id: str,
    role: Role,
) -> Judgment | None:
    """A deterministic numerical-correctness judgment, or ``None`` when one is not warranted.

    Only results with an explicit association to a verified fact that carries
    trusted evidence become judgments, each for exactly its own aspect; company,
    segment and metric results stay diagnostics. ``context`` is every result of the
    same claim value: when the claim's stated metric, company or segment contradicts
    the associated fact, that fact does not describe the claim, so no aspect is
    judged correct or incorrect against it (the label becomes ``unverified``).
    """
    if association is None or fact is None or not fact.evidence:
        return None
    if result.aspect not in NUMERIC_ASPECTS or result.fact_id != fact.fact_id:
        return None
    key = (result.question_id, result.claim_position, result.value_index, result.fact_id)
    siblings = {
        r.aspect: r
        for r in context
        if (r.question_id, r.claim_position, r.value_index, r.fact_id) == key
    }
    if any(aspect not in siblings for aspect in IDENTITY_ASPECTS):
        raise FinancialCheckError("judgment context must hold the value's identity checks")
    label = LABELS[result.outcome]
    if label in ("correct", "incorrect") and any(
        siblings[a].outcome is Outcome.MISMATCH for a in IDENTITY_ASPECTS
    ):
        label = "unverified"  # the associated fact is not about what the claim states
    return Judgment.model_validate(
        {
            "question_id": result.question_id,
            "generation_run_id": generation_run_id,
            "generation_record_sha256": association.generation_record_sha256,
            "dimension": "numerical_correctness",
            "unit": ClaimValueUnit(
                claim_position=result.claim_position,
                value_index=result.value_index,
                aspect=NumericAspect(result.aspect),
                fact_id=fact.fact_id,
            ).model_dump(),
            "label": label,
            "evidence": [e.model_dump() for e in fact.evidence],
            "reference_basis": ReferenceBasis.REFERENCE_FACT.value
            if label in ("correct", "incorrect")
            else None,
            "provenance": Provenance(
                source=Source.DETERMINISTIC,
                role=role,
                check_id=f"fin.{result.checker_id}.{result.aspect}",
                rubric_id=rubric_id(),
            ).model_dump(),
        }
    )


# --------------------------------------------------------------------------- #
# Identities
# --------------------------------------------------------------------------- #


def extractor_definition() -> dict[str, Any]:
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "token_pattern": _TOKEN.pattern,
        "page_pattern": _PAGE_BEFORE.pattern,
        "scales": {k: [name, str(m)] for k, (name, m) in SCALES.items()},
        "letter_scales_attached_only": {
            k: [name, str(m)] for k, (name, m) in LETTER_SCALES.items()
        },
        "currency_markers": dict(CURRENCY_MARKERS),
        "kinds": [k.value for k in OccurrenceKind],
        "units": [u.value for u in Unit],
        "value_index": "1-based by start offset over every numeric token not embedded in an "
        "alphanumeric identifier, including non-financial tokens",
        "classification": "scope-analyzer period and unresolved spans -> period_reference; "
        "page markers -> page_reference; digit-hyphen-digit -> ambiguous range; bare "
        "four-digit 19xx/20xx without unit -> year; no unit, currency or scale -> "
        "unitless_number; scale without currency -> money-less quantity flagged "
        "missing_currency; basis points -> percentage points / 100",
        "signs": (
            "a leading minus is an explicit negative; parentheses are sign-ambiguous "
            "(prose aside or accounting negative) and never match on sign"
        ),
    }


def checker_definition() -> dict[str, Any]:
    return {
        "checker_version": CHECKER_VERSION,
        "extractor_id": extractor_id(),
        "rubric_id": rubric_id(),
        "arithmetic": "decimal.Decimal, no binary floating point",
        "rounding": {
            "mode": "ROUND_HALF_UP",
            "rule": "a claim coarser than the verified value matches only if the verified "
            "value rounded to the claim's stated precision equals it; a non-matching value is "
            "a mismatch at any precision",
            "min_significant_digits": MIN_SIGNIFICANT_DIGITS_FOR_ROUNDING,
            "finer_claim": "ambiguous (claim_more_precise_than_reference)",
            "universal_tolerance": None,
        },
        "outcomes": [o.value for o in Outcome],
        "reasons": [r.value for r in Reason],
        "formulas": [f.value for f in Formula],
        "labels": {o.value: label for o, label in LABELS.items()},
        "association": "explicit, attributable, bound to question, record digest, claim "
        "position, value index and span; never inferred from equal numbers",
        "periods": "only periods the claim states; fiscal and calendar never equated",
        "basis": "GAAP/non-GAAP keywords are a signal, never verification; a reference fact "
        "whose basis its verifier could not establish is 'unverified', never guessed, and "
        "every basis check against it is unverified",
        "currency": "'$' alone is unqualified; no currency or unit is inferred",
        "overall": "a value passes only if every applicable aspect passes",
        "judgment_scope": "one judgment per numeric aspect, for exactly that aspect; a "
        "metric, company or segment mismatch makes every aspect judged against that fact "
        "unverified; company, segment and metric results are diagnostics only",
        "association_validation": "dev question ID, 64-hex record digest, positive claim "
        "position and value index, 0 <= start < end, safe-token fact ID, metric and "
        "associator",
    }


def _hash(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def extractor_id() -> str:
    return _hash(extractor_definition())


def checker_id() -> str:
    return _hash(checker_definition())


# --------------------------------------------------------------------------- #
# Read-only baseline smoke
# --------------------------------------------------------------------------- #


def baseline_summary(run_id: str, project_root: str | None = None) -> dict[str, Any]:
    """Extract and check every eligible claim of a verified run. Counts and codes only."""
    from founderlens.config import load_settings
    from founderlens.corpus.manifest import load_manifest
    from founderlens.evaluation.judging_input import load_verified_run

    settings = load_settings(project_root)
    run, _ = load_verified_run(settings, run_id)
    vocabulary = CorpusVocabulary.from_manifest(load_manifest(settings.corpus_manifest_path))
    kinds: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    reasons: dict[str, int] = {}
    claims_with_numbers = 0
    stable = True
    for question in run.questions:
        if not question.eligible:
            continue
        for claim in question.claims:
            occurrences = extract_occurrences(claim.text, vocabulary)
            stable &= occurrences == extract_occurrences(claim.text, vocabulary)
            stable &= [o.value_index for o in occurrences] == list(range(1, len(occurrences) + 1))
            claims_with_numbers += bool(occurrences)
            for occ in occurrences:
                kinds[occ.kind.value] = kinds.get(occ.kind.value, 0) + 1
                for check in check_value(
                    question_id=question.question_id,
                    generation_record_sha256=question.record_sha256,
                    claim_position=claim.position,
                    claim_text=claim.text,
                    occurrences=occurrences,
                    value_index=occ.value_index,
                    association=None,  # no verified reference facts exist for real data yet
                    facts={},
                    vocabulary=vocabulary,
                ):
                    outcomes[check.outcome.value] = outcomes.get(check.outcome.value, 0) + 1
                    for r in check.reasons:
                        reasons[r.value] = reasons.get(r.value, 0) + 1
                for r in occ.reasons:
                    key = f"occurrence:{r.value}"
                    reasons[key] = reasons.get(key, 0) + 1
    return {
        "run_id": run.run_id,
        "extractor_id": extractor_id(),
        "checker_id": checker_id(),
        "rubric_id": rubric_id(),
        "scheduled": len(run.questions),
        "eligible": sum(q.eligible for q in run.questions),
        "ineligible": sorted(q.question_id for q in run.questions if not q.eligible),
        "claims": sum(len(q.claims) for q in run.questions),
        "claims_with_numeric_occurrences": claims_with_numbers,
        "occurrence_kinds": dict(sorted(kinds.items())),
        "check_outcomes": dict(sorted(outcomes.items())),
        "reason_codes": dict(sorted(reasons.items())),
        "deterministic_and_indexed": stable,
        "judgments_created": 0,
        "verified_reference_facts": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.financial_checks",
        description="Read-only deterministic financial-check smoke over a verified run.",
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    from founderlens.config import ConfigError
    from founderlens.evaluation.judging_input import JudgingInputError

    try:
        summary = baseline_summary(args.run, args.project_root)
    except (ConfigError, JudgingInputError, FinancialCheckError) as exc:
        print(f"Financial check smoke FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
