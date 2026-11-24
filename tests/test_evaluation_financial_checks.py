"""Deterministic financial checks (Commit 7.2), offline and synthetic.

Uses the tracked corpus manifest for the frozen vocabulary (no PDFs), synthetic
claim strings and synthetic verified reference facts. No model, key, network,
real record or held-out file is used, and no label is created for real data.
"""

from __future__ import annotations

import json
import socket
from decimal import Decimal
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.corpus.manifest import load_manifest
from founderlens.evaluation import financial_checks as fc
from founderlens.evaluation.financial_checks import (
    CalculationSpec,
    FactAssociation,
    FinancialCheckError,
    Formula,
    OccurrenceKind,
    Outcome,
    Quantity,
    Reason,
    ReportingPeriod,
    Unit,
    VerifiedReferenceFact,
)
from founderlens.evaluation.judgments import (
    EvidenceKind,
    EvidenceReference,
    ReferenceBasis,
    Role,
    rubric_id,
)
from founderlens.guardrails.scope import CorpusVocabulary

VOCAB = CorpusVocabulary.from_manifest(load_manifest(load_settings().corpus_manifest_path))
NVIDIA, ALPHABET = "NVIDIA Corporation", "Alphabet Inc."
DIGEST = "d" * 64
EVIDENCE = (
    EvidenceReference(
        kind=EvidenceKind.REFERENCE_SOURCE,
        document_id="nvda-fy2025-q4",
        page_number=1,
        chunk_id="nvda-fy2025-q4:p0001:c0000:0c2f87411627",
        collection_artifact_sha256="f" * 64,
        start=0,
        end=10,
        span_sha256="a" * 64,
    ),
)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def money(value: str, scale: str | None = "billion", currency: str | None = "USD") -> Quantity:
    magnitude = Decimal(value)
    decimals = -magnitude.as_tuple().exponent if "." in value else 0  # type: ignore[operator]  # int exponent
    return Quantity(
        abs(magnitude),
        -1 if magnitude < 0 else 1,
        magnitude < 0,
        Unit.MONEY,
        scale,
        "usd" if currency else None,
        currency,
        int(decimals),
    )


def percent(value: str, unit: Unit = Unit.PERCENT) -> Quantity:
    magnitude = Decimal(value)
    decimals = len(value.split(".")[1]) if "." in value else 0
    return Quantity(
        abs(magnitude), -1 if magnitude < 0 else 1, magnitude < 0, unit, None, None, None, decimals
    )


def fact(
    fact_id: str = "syn-f1", value: Quantity | None = None, **over: Any
) -> VerifiedReferenceFact:
    base: dict[str, Any] = {
        "fact_id": fact_id,
        "metric": "revenue",
        "company": NVIDIA,
        "segment": None,
        "period": ReportingPeriod("FY2025 Q4", "fiscal", NVIDIA),
        "value": value or money("39.3"),
        "basis": "GAAP",
        "evidence": EVIDENCE,
        "verified_by": "synthetic-annotator",
        "verification_role": Role.CALIBRATION,
    }
    base.update(over)
    return VerifiedReferenceFact(**base)


def extract(text: str) -> tuple[fc.NumericOccurrence, ...]:
    return fc.extract_occurrences(text, VOCAB)


def check(
    text: str,
    index: int = 1,
    facts: dict[str, VerifiedReferenceFact] | None = None,
    associate: bool = True,
    metric: str = "revenue",
    fact_id: str = "syn-f1",
    **assoc: Any,
) -> dict[str, fc.CheckResult]:
    occurrences = extract(text)
    occ = next(o for o in occurrences if o.value_index == index)
    association = None
    if associate:
        fields: dict[str, Any] = {
            "question_id": "dev-001",
            "generation_record_sha256": DIGEST,
            "claim_position": 1,
            "value_index": index,
            "start": occ.start,
            "end": occ.end,
            "fact_id": fact_id,
            "metric": metric,
            "associated_by": "synthetic-annotator",
        }
        fields.update(assoc)
        association = FactAssociation(**fields)
    results = fc.check_value(
        question_id="dev-001",
        generation_record_sha256=DIGEST,
        claim_position=1,
        claim_text=text,
        occurrences=occurrences,
        value_index=index,
        association=association,
        facts=facts if facts is not None else {"syn-f1": fact()},
        vocabulary=VOCAB,
    )
    return {r.aspect: r for r in results}


# --- 1-3, 8: extraction, indexing and classification ----------------------------------------


def test_value_indexes_follow_source_order_with_exact_spans() -> None:
    text = "NVIDIA revenue was $39.3 billion in FY2025 Q4, up 78% from $22.1 billion."
    occurrences = extract(text)
    assert [o.value_index for o in occurrences] == list(range(1, len(occurrences) + 1))
    assert [text[o.start : o.end] for o in occurrences] == ["$39.3 billion", "78%", "$22.1 billion"]
    assert all(o.start < o.end for o in occurrences)
    assert [o.start for o in occurrences] == sorted(o.start for o in occurrences)
    assert extract(text) == occurrences  # deterministic


def test_repeated_values_get_distinct_positions() -> None:
    text = "Revenue was $5 billion and cost was $5 billion."
    first, second = extract(text)
    assert (first.value_index, second.value_index) == (1, 2)
    assert first.start != second.start
    assert first.quantity == second.quantity


def test_thousands_separators_and_decimal_precision() -> None:
    (occ,) = extract("Revenue was $39,331.25 million.")
    assert occ.quantity is not None
    assert occ.quantity.magnitude == Decimal("39331.25")
    assert occ.quantity.decimals == 2
    assert occ.quantity.value == Decimal("39331250000.00")


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Revenue in fiscal 2025 grew.", OccurrenceKind.PERIOD_REFERENCE),
        ("In FY2025 Q4 revenue grew.", None),  # embedded in identifiers: not a token
        ("For the quarter ended January 26, 2025, revenue grew.", OccurrenceKind.PERIOD_REFERENCE),
        ("See page 12 of the release.", OccurrenceKind.PAGE_REFERENCE),
        ("Evidence E01 supports this.", None),
        ("The model shipped 2024 units.", OccurrenceKind.PERIOD_REFERENCE),  # bare year
        ("It employs 36,000 people.", OccurrenceKind.UNITLESS_NUMBER),
        ("Margins of 10-12% were guided.", OccurrenceKind.AMBIGUOUS),
    ],
)
def test_non_financial_numbers_are_never_amounts(text: str, kind: OccurrenceKind | None) -> None:
    occurrences = extract(text)
    if kind is None:
        assert occurrences == ()
    else:
        assert occurrences[0].kind is kind
        assert occurrences[0].kind is not OccurrenceKind.FINANCIAL_QUANTITY


def test_attached_units_scales_and_ratios() -> None:
    attached, pp, bps, ratio, letter = extract(
        "It added 26.3billion, 2.5 percentage points, 150 bps, a 3.2x multiple and $35.6B."
    )
    assert attached.quantity is not None
    assert (attached.quantity.scale, attached.reasons) == (
        "billion",
        (Reason.UNIT_ATTACHED, Reason.MISSING_CURRENCY),
    )
    assert pp.quantity is not None
    assert pp.quantity.unit is Unit.PERCENTAGE_POINTS
    assert bps.quantity is not None
    assert bps.quantity.value == Decimal("1.50")
    assert ratio.quantity is not None
    assert ratio.quantity.unit is Unit.RATIO
    assert letter.quantity is not None
    assert letter.quantity.scale == "billion"


def test_detached_single_letters_are_not_scales() -> None:
    (occ,) = extract("Item 5 b of the filing.")
    assert occ.kind is OccurrenceKind.UNITLESS_NUMBER


# --- 4-7, 9, 18: values, units, currency, signs and rounding ---------------------------------


def test_billion_to_million_normalization_is_exact() -> None:
    assert (
        check("NVIDIA revenue was $39,300 million in FY2025 Q4.")["value"].outcome
        is Outcome.EXACT_MATCH
    )


def test_dollar_sign_is_not_assumed_to_be_usd() -> None:
    results = check("NVIDIA revenue was $39.3 billion in FY2025 Q4.")
    assert results["value"].outcome is Outcome.EXACT_MATCH
    assert results["currency"].outcome is Outcome.UNVERIFIED
    assert results["currency"].reasons == (Reason.DOLLAR_SIGN_UNQUALIFIED,)
    assert check("NVIDIA revenue was US$39.3 billion in FY2025 Q4.")["currency"].outcome is (
        Outcome.EXACT_MATCH
    )
    euro = check("NVIDIA revenue was €39.3 billion in FY2025 Q4.")["currency"]
    assert (euro.outcome, euro.reasons) == (Outcome.MISMATCH, (Reason.CURRENCY_MISMATCH,))


def test_percent_and_percentage_points_are_never_equated() -> None:
    facts = {"syn-f1": fact(value=percent("2.5"))}
    results = check("NVIDIA margin rose 2.5 percentage points in FY2025 Q4.", facts=facts)
    assert results["value"].outcome is Outcome.MISMATCH
    assert results["value"].reasons == (Reason.UNIT_KIND_MISMATCH,)


def test_signs_and_accounting_parentheses() -> None:
    loss = {"syn-f1": fact(value=money("-1.2"))}
    assert check("NVIDIA lost -$1.2 billion in FY2025 Q4.", facts=loss)["value"].outcome is (
        Outcome.EXACT_MATCH
    )
    paren = check("NVIDIA reported ($1.2) billion in FY2025 Q4.", facts=loss)["value"]
    assert (paren.outcome, paren.reasons) == (
        Outcome.AMBIGUOUS,
        (Reason.PARENTHESES_SIGN_AMBIGUOUS,),
    )
    unsigned = check("NVIDIA reported $1.2 billion in FY2025 Q4.", facts=loss)["value"]
    assert (unsigned.outcome, unsigned.reasons) == (Outcome.AMBIGUOUS, (Reason.SIGN_NOT_EXPLICIT,))
    assert check("NVIDIA lost -$1.3 billion in FY2025 Q4.", facts=loss)["value"].outcome is (
        Outcome.MISMATCH
    )


@pytest.mark.parametrize(
    ("claim", "outcome", "reasons"),
    [
        ("$39.3 billion", Outcome.EXACT_MATCH, ()),
        ("$39 billion", Outcome.ROUNDING_MATCH, ()),  # coarser claim rounds the verified value
        ("$40 billion", Outcome.MISMATCH, ()),  # 39.3 rounds to 39, not 40
        ("$39.4 billion", Outcome.MISMATCH, ()),
        ("$39.31 billion", Outcome.AMBIGUOUS, (Reason.CLAIM_MORE_PRECISE,)),  # finer
        ("$39.36 billion", Outcome.MISMATCH, ()),  # finer, and does not round back
        ("$39.29 billion", Outcome.AMBIGUOUS, (Reason.CLAIM_MORE_PRECISE,)),
        ("$4 billion", Outcome.MISMATCH, ()),
    ],
)
def test_rounding_respects_stated_precision(
    claim: str, outcome: Outcome, reasons: tuple[Reason, ...]
) -> None:
    result = check(f"NVIDIA revenue was {claim} in FY2025 Q4.")["value"]
    assert (result.outcome, result.reasons) == (outcome, reasons)


def test_one_significant_digit_never_rounds_to_a_match() -> None:
    facts = {"syn-f1": fact(value=money("4.2"))}
    result = check("NVIDIA revenue was $4 billion in FY2025 Q4.", facts=facts)["value"]
    assert (result.outcome, result.reasons) == (Outcome.AMBIGUOUS, (Reason.INSUFFICIENT_PRECISION,))


def test_missing_unit_or_currency_stays_unverified() -> None:
    unitless = check("NVIDIA revenue was 39,300 in FY2025 Q4.")["value"]
    assert (unitless.outcome, unitless.reasons) == (Outcome.UNVERIFIED, (Reason.MISSING_UNIT,))
    scaled = check("NVIDIA revenue was 39.3 billion in FY2025 Q4.")
    assert scaled["currency"].outcome is Outcome.UNVERIFIED
    assert scaled["currency"].reasons == (Reason.MISSING_CURRENCY,)
    assert scaled["value"].outcome is Outcome.UNVERIFIED  # no currency: not a money match


# --- 12-17, 19-20: attribution, association and conservative outcomes ------------------------


def test_company_mismatch_fails_the_value() -> None:
    results = check("Alphabet revenue was $39.3 billion in FY2025 Q4.")
    assert results["value"].outcome is Outcome.EXACT_MATCH  # the number alone matches
    assert results["company"].outcome is Outcome.MISMATCH
    assert fc.overall(tuple(results.values())) is Outcome.MISMATCH
    assert check("Revenue was $39.3 billion.")["company"].reasons == (Reason.COMPANY_NOT_STATED,)


def test_reporting_periods_are_checked_only_as_stated() -> None:
    assert check("NVIDIA revenue was $39.3 billion in FY2025 Q4.")["reporting_period"].outcome is (
        Outcome.EXACT_MATCH
    )
    other = check("NVIDIA revenue was $39.3 billion in FY2025 Q3.")["reporting_period"]
    assert (other.outcome, other.reasons) == (Outcome.MISMATCH, (Reason.PERIOD_MISMATCH,))
    calendar = check("NVIDIA revenue was $39.3 billion in CY2024 Q4.")["reporting_period"]
    assert (calendar.outcome, calendar.reasons) == (
        Outcome.AMBIGUOUS,
        (Reason.PERIOD_BASIS_DIFFERS,),
    )
    silent = check("NVIDIA revenue was $39.3 billion.")["reporting_period"]
    assert (silent.outcome, silent.reasons) == (Outcome.UNVERIFIED, (Reason.PERIOD_NOT_STATED,))
    both = check("NVIDIA revenue was $39.3 billion in FY2025 Q4 vs FY2024 Q4.")["reporting_period"]
    assert both.outcome is Outcome.AMBIGUOUS


def test_gaap_keywords_are_signals_never_verification() -> None:
    agree = check("NVIDIA GAAP revenue was $39.3 billion in FY2025 Q4.")["financial_basis"]
    assert (agree.outcome, agree.reasons) == (Outcome.UNVERIFIED, (Reason.BASIS_SIGNAL_ONLY,))
    conflict = check("NVIDIA non-GAAP revenue was $39.3 billion in FY2025 Q4.")["financial_basis"]
    assert (conflict.outcome, conflict.reasons) == (
        Outcome.AMBIGUOUS,
        (Reason.BASIS_SIGNAL_CONFLICT,),
    )
    assert fc.basis_signal("non-GAAP and GAAP measures") == "both"


def test_segment_is_never_the_consolidated_company() -> None:
    facts = {
        "syn-f1": fact(
            company=ALPHABET,
            value=money("96.5"),
            period=ReportingPeriod("FY2024 Q4", "fiscal", ALPHABET),
        )
    }
    results = check("Google Cloud revenue was $96.5 billion in FY2024 Q4.", facts=facts)
    assert results["segment"].outcome is Outcome.MISMATCH
    assert results["segment"].reasons == (Reason.SEGMENT_VS_CONSOLIDATED,)


def test_a_correct_number_for_the_wrong_metric_fails() -> None:
    results = check("NVIDIA revenue was $39.3 billion in FY2025 Q4.", metric="operating_income")
    assert results["value"].outcome is Outcome.EXACT_MATCH
    assert results["metric"].reasons == (Reason.METRIC_MISMATCH,)
    assert fc.overall(tuple(results.values())) is Outcome.MISMATCH


def test_no_or_mismatched_association_is_unverified_never_scored() -> None:
    none = check("NVIDIA revenue was $39.3 billion in FY2025 Q4.", associate=False)
    assert list(none) == ["value"]
    assert (none["value"].outcome, none["value"].reasons) == (
        Outcome.UNVERIFIED,
        (Reason.NO_VERIFIED_ASSOCIATION,),
    )
    for wrong in (
        {"start": 0},
        {"generation_record_sha256": "0" * 64},
        {"claim_position": 2},
        {"fact_id": "unknown"},
    ):
        result = check("NVIDIA revenue was $39.3 billion in FY2025 Q4.", **wrong)["value"]
        assert result.reasons == (Reason.ASSOCIATION_MISMATCH,)
        assert result.outcome is Outcome.UNVERIFIED


def test_equal_numbers_are_never_linked_automatically() -> None:
    text = "NVIDIA revenue was $39.3 billion in FY2025 Q4; guidance was $39.3 billion."
    associated = check(text, index=1)
    unassociated = check(text, index=2, associate=False)
    assert associated["value"].outcome is Outcome.EXACT_MATCH
    assert unassociated["value"].outcome is Outcome.UNVERIFIED  # same number, no link


def test_overall_is_conservative() -> None:
    results = check("NVIDIA revenue was $39.3 billion in FY2025 Q4.")
    assert fc.overall(tuple(results.values())) is Outcome.UNVERIFIED  # "$" and basis unverified
    clean = check(
        "NVIDIA revenue was US$39.3 billion in FY2025 Q4.",
        facts={"syn-f1": fact(basis="not_applicable")},
    )
    assert fc.overall(tuple(clean.values())) is Outcome.EXACT_MATCH
    assert check("Revenue fell in fiscal 2025.", associate=False)  # no financial value at index


def test_non_financial_occurrences_are_not_applicable() -> None:
    result = fc.check_value(
        question_id="dev-001",
        generation_record_sha256=DIGEST,
        claim_position=1,
        claim_text="Revenue in fiscal 2025 grew.",
        occurrences=extract("Revenue in fiscal 2025 grew."),
        value_index=1,
        association=None,
        facts={},
        vocabulary=VOCAB,
    )
    assert result[0].outcome is Outcome.NOT_APPLICABLE
    with pytest.raises(FinancialCheckError, match="value_index"):
        fc.check_value(
            question_id="dev-001",
            generation_record_sha256=DIGEST,
            claim_position=1,
            claim_text="Revenue was $5 billion.",
            occurrences=extract("Revenue was $5 billion."),
            value_index=9,
            association=None,
            facts={},
            vocabulary=VOCAB,
        )


# --- 10-11: declared calculations -----------------------------------------------------------


def _calc(
    text: str,
    formula: Formula,
    result: int,
    base: int | None = None,
    comparison: int | None = None,
    facts: dict[str, VerifiedReferenceFact] | None = None,
) -> fc.CalculationResult:
    spec = CalculationSpec(formula, "base", "comp", result, base, comparison, "synthetic-annotator")
    default = {
        "base": fact("base", money("22.1"), period=ReportingPeriod("FY2024 Q4", "fiscal", NVIDIA)),
        "comp": fact("comp", money("39.3")),
    }
    return fc.check_calculation(spec, extract(text), facts if facts is not None else default)


def test_valid_growth_and_difference_calculations() -> None:
    growth = _calc(
        "Revenue rose from $22.1 billion to $39.3 billion, up 78%.", Formula.PERCENT_CHANGE, 3, 1, 2
    )
    assert (growth.mathematical, growth.factual, growth.overall) == (
        Outcome.ROUNDING_MATCH,
        Outcome.ROUNDING_MATCH,
        Outcome.ROUNDING_MATCH,
    )
    diff = _calc("Revenue rose by $17.2 billion.", Formula.ABSOLUTE_DIFFERENCE, 1)
    assert diff.factual is Outcome.EXACT_MATCH
    assert diff.mathematical is Outcome.NOT_APPLICABLE
    pp_facts = {"base": fact("base", percent("73.0")), "comp": fact("comp", percent("75.5"))}
    pp = _calc(
        "Margin rose 2.5 percentage points.", Formula.PERCENTAGE_POINT_DIFFERENCE, 1, facts=pp_facts
    )
    assert pp.factual is Outcome.EXACT_MATCH
    ratio = _calc("Revenue was 1.8x the prior year.", Formula.RELATIVE_CHANGE, 1)
    assert ratio.factual is Outcome.ROUNDING_MATCH


def test_invalid_arithmetic_and_wrong_operands_never_pass() -> None:
    wrong_math = _calc(
        "Revenue rose from $22.1 billion to $39.3 billion, up 90%.", Formula.PERCENT_CHANGE, 3, 1, 2
    )
    assert wrong_math.overall is Outcome.MISMATCH
    assert Reason.ARITHMETIC_ERROR in wrong_math.reasons
    # internally consistent arithmetic on operands the verified facts contradict
    wrong_operands = _calc(
        "Revenue rose from $20.0 billion to $35.6 billion, up 78%.", Formula.PERCENT_CHANGE, 3, 1, 2
    )
    assert wrong_operands.mathematical is Outcome.EXACT_MATCH  # 20.0 -> 35.6 is +78%
    assert Reason.INCORRECT_OPERANDS in wrong_operands.reasons
    assert wrong_operands.overall is Outcome.MISMATCH


def test_division_by_zero_and_unsupported_calculations() -> None:
    zero = {"base": fact("base", money("0")), "comp": fact("comp", money("39.3"))}
    result = _calc("Revenue grew 78%.", Formula.PERCENT_CHANGE, 1, facts=zero)
    assert (result.factual, result.reasons) == (Outcome.AMBIGUOUS, (Reason.DIVISION_BY_ZERO,))
    unsupported = _calc(
        "Margin rose 2.5 percentage points.", Formula.PERCENTAGE_POINT_DIFFERENCE, 1
    )
    assert Reason.INCOMPATIBLE_OPERANDS in unsupported.reasons
    missing = _calc("Revenue grew 78%.", Formula.PERCENT_CHANGE, 1, facts={})
    assert (missing.overall, missing.reasons) == (Outcome.UNVERIFIED, (Reason.MISSING_OPERAND,))
    with pytest.raises(FinancialCheckError, match="named annotator"):
        CalculationSpec(Formula.PERCENT_CHANGE, "a", "b", 1)


# --- 21-24: identities, judgments, malformed inputs and privacy ------------------------------


def test_check_identities_are_stable_and_normative(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor, checker = fc.extractor_id(), fc.checker_id()
    assert (fc.extractor_id(), fc.checker_id()) == (extractor, checker)
    assert rubric_id() == "e8b840e4f4af"  # rubric v1 is unchanged by Commit 7.2
    monkeypatch.setattr(fc, "SCALES", {**fc.SCALES, "lakh": ("lakh", Decimal(10) ** 5)})
    assert fc.extractor_id() != extractor
    assert fc.checker_id() != checker  # the checker identity includes the extractor's
    monkeypatch.undo()
    monkeypatch.setattr(fc, "MIN_SIGNIFICANT_DIGITS_FOR_ROUNDING", 3)
    assert fc.checker_id() != checker


def test_judgments_only_from_explicit_verified_associations() -> None:
    dataset_fact = fact("dev-001-f1")  # judgment units use dataset-style fact IDs
    results = check(
        "NVIDIA revenue was US$39.3 billion in FY2025 Q4.",
        facts={"dev-001-f1": dataset_fact},
        fact_id="dev-001-f1",
    )
    text = "NVIDIA revenue was US$39.3 billion in FY2025 Q4."
    occ = extract(text)[0]
    association = FactAssociation(
        "dev-001", DIGEST, 1, 1, occ.start, occ.end, "dev-001-f1", "revenue", "synthetic-annotator"
    )
    judgment = fc.to_judgment(
        results["value"],
        context=tuple(results.values()),
        association=association,
        fact=dataset_fact,
        generation_run_id="live-synthetic",
        role=Role.CALIBRATION,
    )
    assert judgment is not None
    assert (judgment.label, judgment.reference_basis) == (
        "correct",
        ReferenceBasis.REFERENCE_FACT,
    )
    assert judgment.provenance.source.value == "deterministic"
    assert judgment.provenance.check_id == f"fin.{fc.checker_id()}.value"
    assert judgment.unit.model_dump()["fact_id"] == "dev-001-f1"
    unverified = fc.to_judgment(
        results["financial_basis"],
        context=tuple(results.values()),
        association=association,
        fact=dataset_fact,
        generation_run_id="live-synthetic",
        role=Role.CALIBRATION,
    )
    assert unverified is not None
    assert unverified.label == "unverified"
    for blocked in (
        fc.to_judgment(
            results["value"],
            context=tuple(results.values()),
            association=None,
            fact=dataset_fact,
            generation_run_id="live-synthetic",
            role=Role.EVALUATION,
        ),
        fc.to_judgment(
            results["company"],
            context=tuple(results.values()),
            association=association,
            fact=dataset_fact,
            generation_run_id="live-synthetic",
            role=Role.EVALUATION,
        ),
    ):
        assert blocked is None


def test_verified_facts_require_trusted_evidence_and_provenance() -> None:
    with pytest.raises(FinancialCheckError, match="trusted source evidence"):
        fact(evidence=())
    with pytest.raises(FinancialCheckError, match="annotator"):
        fact(verified_by="")
    with pytest.raises(FinancialCheckError, match="currency"):
        fact(value=money("39.3", currency=None))
    with pytest.raises(FinancialCheckError, match="basis"):
        fact(basis="IFRS")
    with pytest.raises(FinancialCheckError, match="fiscal or calendar"):
        ReportingPeriod("Q4 2024", "unspecified")
    with pytest.raises(Exception):  # noqa: B017, PT011 - pydantic rejects the foreign reference
        EvidenceReference(
            kind=EvidenceKind.REFERENCE_SOURCE,
            document_id="x",
            page_number=1,
            chunk_id="c",
            collection_artifact_sha256="nothex",
            start=0,
            end=1,
            span_sha256="0" * 64,
        )


def test_public_diagnostics_hold_no_text_or_values() -> None:
    text = "NVIDIA revenue was $39.3 billion in FY2025 Q4."
    payload = json.dumps([o.public() for o in extract(text)])
    payload += json.dumps([r.public() for r in check(text).values()])
    for leak in ("NVIDIA revenue", "39.3", "39300000000"):
        assert leak not in payload
    assert "dollar_sign_unqualified" in payload


# --- final verification: judgment scope, association validation, boundaries ----------------


def _judge(text: str, aspect: str = "value", **kw: Any) -> Any:
    dataset_fact = kw.pop("fact_obj", fact("dev-001-f1"))
    metric = kw.pop("metric", "revenue")
    results = check(text, facts={"dev-001-f1": dataset_fact}, fact_id="dev-001-f1", metric=metric)
    occ = extract(text)[0]
    association = FactAssociation(
        "dev-001", DIGEST, 1, 1, occ.start, occ.end, "dev-001-f1", metric, "synthetic-annotator"
    )
    return fc.to_judgment(
        results[aspect],
        context=tuple(results.values()),
        association=association,
        fact=dataset_fact,
        generation_run_id="live-synthetic",
        role=Role.EVALUATION,
    )


@pytest.mark.parametrize(
    ("text", "metric"),
    [
        ("NVIDIA revenue was US$39.3 billion in FY2025 Q4.", "operating_income"),  # wrong metric
        ("Alphabet revenue was US$39.3 billion in FY2025 Q4.", "revenue"),  # wrong company
        ("Google Cloud revenue was US$39.3 billion in FY2025 Q4.", "revenue"),  # segment
    ],
)
def test_no_aspect_is_certified_against_a_fact_the_claim_contradicts(
    text: str, metric: str
) -> None:
    for aspect in ("value", "unit_scale", "currency"):
        judgment = _judge(text, aspect, metric=metric)
        assert judgment is not None
        assert judgment.label == "unverified", aspect
        assert judgment.reference_basis is None


def test_each_judgment_covers_exactly_its_own_aspect() -> None:
    text = "NVIDIA revenue was US$39.3 billion in FY2025 Q3."  # right number, wrong period
    value = _judge(text, "value")
    period = _judge(text, "reporting_period")
    basis = _judge(text, "financial_basis")
    assert (value.label, period.label, basis.label) == ("correct", "incorrect", "unverified")
    assert value.unit.model_dump()["aspect"] == "value"
    assert period.unit.model_dump()["aspect"] == "reporting_period"
    assert value.evidence == fact("dev-001-f1").evidence
    assert value.reference_basis is not None
    assert value.reference_basis.value == "reference_fact"


def test_judgments_need_the_value_identity_context() -> None:
    text = "NVIDIA revenue was US$39.3 billion in FY2025 Q4."
    results = check(text, facts={"dev-001-f1": fact("dev-001-f1")}, fact_id="dev-001-f1")
    occ = extract(text)[0]
    association = FactAssociation(
        "dev-001", DIGEST, 1, 1, occ.start, occ.end, "dev-001-f1", "revenue", "synthetic-annotator"
    )
    with pytest.raises(FinancialCheckError, match="identity checks"):
        fc.to_judgment(
            results["value"],
            context=(results["value"],),
            association=association,
            fact=fact("dev-001-f1"),
            generation_run_id="live-synthetic",
            role=Role.EVALUATION,
        )


def test_overall_never_succeeds_vacuously_or_hides_mismatches() -> None:
    assert fc.overall(()) is Outcome.NOT_APPLICABLE
    ambiguous_and_mismatch = check("Alphabet revenue was $39.3 billion in CY2024 Q4.")
    assert ambiguous_and_mismatch["reporting_period"].outcome is Outcome.AMBIGUOUS
    assert fc.overall(tuple(ambiguous_and_mismatch.values())) is Outcome.MISMATCH


@pytest.mark.parametrize(
    "fields",
    [
        {"question_id": "test-001"},
        {"generation_record_sha256": "stale"},
        {"claim_position": 0},
        {"value_index": 0},
        {"start": 9, "end": 3},
        {"fact_id": ""},
        {"metric": "revenue; drop"},
        {"associated_by": ""},
    ],
)
def test_malformed_associations_are_rejected(fields: dict[str, Any]) -> None:
    base: dict[str, Any] = {
        "question_id": "dev-001",
        "generation_record_sha256": DIGEST,
        "claim_position": 1,
        "value_index": 1,
        "start": 0,
        "end": 5,
        "fact_id": "dev-001-f1",
        "metric": "revenue",
        "associated_by": "synthetic-annotator",
    }
    with pytest.raises(FinancialCheckError, match="malformed association"):
        FactAssociation(**{**base, **fields})


@pytest.mark.parametrize(
    ("claim", "fact_value", "outcome"),
    [
        ("150 bps", percent("1.50", Unit.PERCENTAGE_POINTS), Outcome.EXACT_MATCH),
        ("1.5 percentage points", percent("1.5", Unit.PERCENTAGE_POINTS), Outcome.EXACT_MATCH),
        ("1.5%", percent("1.5", Unit.PERCENTAGE_POINTS), Outcome.MISMATCH),
        ("US$0 billion", money("0"), Outcome.EXACT_MATCH),
        ("US$0.04 trillion", money("39.3"), Outcome.AMBIGUOUS),  # rounds, one significant digit
        ("US$0.05 trillion", money("39.3"), Outcome.MISMATCH),
        ("US$35,600 million", money("35.6"), Outcome.EXACT_MATCH),
    ],
)
def test_numeric_boundaries(claim: str, fact_value: Quantity, outcome: Outcome) -> None:
    facts = {"syn-f1": fact(value=fact_value)}
    assert (
        check(f"NVIDIA margin was {claim} in FY2025 Q4.", facts=facts)["value"].outcome is outcome
    )


def test_repeated_values_keep_offsets_and_indexes_under_checks() -> None:
    text = "NVIDIA revenue was $39.3 billion in FY2025 Q4 and $39.3 billion again."
    first, second = extract(text)
    assert text[first.start : first.end] == text[second.start : second.end]
    assert (first.value_index, second.value_index) == (1, 2)
    assert check(text, index=2)["value"].outcome is Outcome.EXACT_MATCH  # bound to index 2 only
    assert check(text, index=2, start=first.start)["value"].reasons == (
        Reason.ASSOCIATION_MISMATCH,
    )
