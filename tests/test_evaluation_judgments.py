"""Rubric v1 and typed judgment contracts (Commit 7.1), offline and synthetic.

No labels are created here beyond synthetic fixtures that exercise the schema.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from founderlens.evaluation import judgments as jd
from founderlens.evaluation.judgments import (
    AnswerUnit,
    ClaimAspectUnit,
    ClaimCitationUnit,
    ClaimDecomposition,
    ClaimUnit,
    ClaimValueUnit,
    Dimension,
    Judgment,
    RequiredFactUnit,
    SubclaimRef,
    SubclaimUnit,
)

RUN = "live-20990101T000000000000Z-00000000"
DIGEST = "a" * 64
REF: dict[str, Any] = {
    "kind": "selected_context",
    "evidence_id": "E01",
    "document_id": "nvda-fy2025-q4",
    "page_number": 1,
    "chunk_id": "nvda-fy2025-q4:p0001:c0000:0c2f87411627",
    "collection_artifact_sha256": "f" * 64,
    "start": 0,
    "end": 10,
    "span_sha256": "b" * 64,
}
HUMAN = {
    "source": "human",
    "role": "calibration",
    "annotator_id": "owner",
    "rubric_id": jd.rubric_id(),
}
UNITS: dict[Dimension, dict[str, Any]] = {
    Dimension.CLAIM_FAITHFULNESS: {"kind": "claim", "claim_position": 1},
    Dimension.CITATION_SUPPORT: {
        "kind": "claim_citation",
        "claim_position": 1,
        "citation_number": 1,
        "evidence_id": "E01",
    },
    Dimension.FACTUAL_CORRECTNESS: {"kind": "required_fact", "fact_id": "dev-001-f1"},
    Dimension.FINANCIAL_ATTRIBUTION: {
        "kind": "claim_aspect",
        "claim_position": 1,
        "aspect": "reporting_period",
    },
    Dimension.NUMERICAL_CORRECTNESS: {
        "kind": "claim_value",
        "claim_position": 1,
        "value_index": 1,
        "aspect": "unit_scale",
    },
    Dimension.ANSWER_COMPLETENESS: {"kind": "answer"},
    Dimension.ABSTENTION_APPROPRIATENESS: {"kind": "answer"},
    Dimension.ANSWER_RELEVANCE: {"kind": "answer"},
}


def _payload(dimension: Dimension, label: str, **over: Any) -> dict[str, Any]:
    spec = jd.RUBRIC[dimension]
    payload: dict[str, Any] = {
        "question_id": "dev-001",
        "generation_run_id": RUN,
        "generation_record_sha256": DIGEST,
        "dimension": dimension.value,
        "unit": UNITS[dimension],
        "label": label,
        "provenance": HUMAN,
    }
    if label == jd.NOT_ASSESSABLE:
        payload["not_assessable_reason"] = "evidence_unreadable"
    if label in spec.get("evidence_required_for", []):
        kind = "cited" if dimension is Dimension.CITATION_SUPPORT else "selected_context"
        payload["evidence"] = [{**REF, "kind": kind}]
    if label in spec.get("reference_basis_required_for", []):
        payload["reference_basis"] = "reference_fact"
    if dimension is Dimension.ABSTENTION_APPROPRIATENESS and label != jd.NOT_ASSESSABLE:
        payload["evidence_sufficiency"] = "retrieved_sufficient"
    payload.update(over)
    return payload


def _judgment(dimension: Dimension, label: str, **over: Any) -> Judgment:
    return Judgment.model_validate(_payload(dimension, label, **over))


ALL = [(d, label) for d in Dimension for label in jd.labels_for(d)]


# --- rubric identity ------------------------------------------------------------------------


def test_rubric_identity_is_stable_and_content_addressed(monkeypatch: pytest.MonkeyPatch) -> None:
    first = jd.rubric_id()
    assert first == jd.rubric_id()
    assert len(first) == 12
    definition = jd.rubric_definition()
    assert set(definition["dimensions"]) == {d.value for d in Dimension}
    assert definition["rubric_version"] == 1
    changed = {
        **jd.RUBRIC,
        Dimension.ANSWER_RELEVANCE: {**jd.RUBRIC[Dimension.ANSWER_RELEVANCE], "question": "edited"},
    }
    monkeypatch.setattr(jd, "RUBRIC", changed)
    assert jd.rubric_id() != first


def test_rubric_holds_the_required_dimensions_and_labels() -> None:
    assert jd.labels_for(Dimension.CLAIM_FAITHFULNESS) == (
        "supported",
        "unsupported",
        "contradicted",
        "not_assessable",
    )
    assert jd.labels_for(Dimension.CITATION_SUPPORT) == (
        "supports",
        "partially_supports",
        "does_not_support",
        "not_assessable",
    )
    assert jd.labels_for(Dimension.FACTUAL_CORRECTNESS) == (
        "correct",
        "incorrect",
        "missing",
        "not_assessable",
    )
    assert jd.labels_for(Dimension.ANSWER_COMPLETENESS) == (
        "complete",
        "partial",
        "none",
        "not_assessable",
    )
    assert jd.labels_for(Dimension.ABSTENTION_APPROPRIATENESS) == (
        "correct_abstention",
        "correct_attempt",
        "false_abstention",
        "missed_abstention",
        "not_assessable",
    )
    assert jd.RUBRIC[Dimension.ANSWER_RELEVANCE]["ordinal"] == {
        "directly_answers": 2,
        "partially_relevant": 1,
        "off_topic": 0,
    }
    assert "incorrect" in jd.labels_for(Dimension.FINANCIAL_ATTRIBUTION)
    assert "unverified" in jd.labels_for(Dimension.NUMERICAL_CORRECTNESS)


# --- dimension / label contract -------------------------------------------------------------


@pytest.mark.parametrize(("dimension", "label"), ALL)
def test_every_permitted_dimension_label_combination(dimension: Dimension, label: str) -> None:
    judgment = _judgment(dimension, label)
    assert judgment.label == label
    assert judgment.unit_id.startswith("dev-001/")


@pytest.mark.parametrize(
    ("dimension", "label"),
    [
        (Dimension.CLAIM_FAITHFULNESS, "correct"),  # another dimension's label
        (Dimension.CITATION_SUPPORT, "supported"),
        (Dimension.FACTUAL_CORRECTNESS, "partially_correct"),
        (Dimension.ANSWER_RELEVANCE, "2"),
        (Dimension.ABSTENTION_APPROPRIATENESS, "insufficient_evidence"),
    ],
)
def test_unknown_or_foreign_labels_are_rejected(dimension: Dimension, label: str) -> None:
    with pytest.raises(ValidationError, match="unknown label"):
        _judgment(dimension, label)


def test_unit_kinds_must_fit_the_dimension() -> None:
    with pytest.raises(ValidationError, match="does not take"):
        _judgment(
            Dimension.CITATION_SUPPORT, "does_not_support", unit=UNITS[Dimension.CLAIM_FAITHFULNESS]
        )
    with pytest.raises(ValidationError, match="does not take"):
        _judgment(Dimension.ANSWER_COMPLETENESS, "none", unit=UNITS[Dimension.FACTUAL_CORRECTNESS])


def test_not_assessable_needs_exactly_one_reason() -> None:
    with pytest.raises(ValidationError, match="requires a reason"):
        _judgment(Dimension.CLAIM_FAITHFULNESS, "not_assessable", not_assessable_reason=None)
    with pytest.raises(ValidationError, match="requires a reason"):
        _judgment(
            Dimension.CLAIM_FAITHFULNESS, "unsupported", not_assessable_reason="claim_not_atomic"
        )
    with pytest.raises(ValidationError):
        _judgment(Dimension.CLAIM_FAITHFULNESS, "not_assessable", not_assessable_reason="vibes")
    atomic = _judgment(
        Dimension.CLAIM_FAITHFULNESS, "not_assessable", not_assessable_reason="claim_not_atomic"
    )
    assert atomic.not_assessable_reason is jd.NotAssessableReason.CLAIM_NOT_ATOMIC


def test_evidence_and_reference_basis_requirements() -> None:
    with pytest.raises(ValidationError, match="requires evidence"):
        _judgment(Dimension.CLAIM_FAITHFULNESS, "supported", evidence=[])
    unsupported = _judgment(Dimension.CLAIM_FAITHFULNESS, "unsupported")
    assert unsupported.evidence == ()
    with pytest.raises(ValidationError, match="requires a reference basis"):
        _judgment(Dimension.FACTUAL_CORRECTNESS, "correct", reference_basis=None)
    with pytest.raises(ValidationError, match="source_evidence basis requires evidence"):
        _judgment(Dimension.FACTUAL_CORRECTNESS, "incorrect", reference_basis="source_evidence")
    sourced = _judgment(
        Dimension.FACTUAL_CORRECTNESS,
        "correct",
        reference_basis="source_evidence",
        evidence=[{**REF, "kind": "reference_source", "evidence_id": None}],
    )
    assert sourced.evidence[0].evidence_id is None  # source evidence outside the context
    with pytest.raises(ValidationError, match="takes no reference basis"):
        _judgment(Dimension.CLAIM_FAITHFULNESS, "unsupported", reference_basis="reference_fact")
    with pytest.raises(ValidationError, match="does not accept that evidence kind"):
        _judgment(
            Dimension.CLAIM_FAITHFULNESS,
            "supported",
            evidence=[{**REF, "kind": "reference_source", "evidence_id": None}],
        )


def test_abstention_records_evidence_sufficiency_separately() -> None:
    judged = _judgment(
        Dimension.ABSTENTION_APPROPRIATENESS,
        "false_abstention",
        evidence_sufficiency="retrieved_insufficient",
    )
    assert judged.evidence_sufficiency is jd.EvidenceSufficiency.INSUFFICIENT
    with pytest.raises(ValidationError, match="evidence_sufficiency"):
        _judgment(
            Dimension.ABSTENTION_APPROPRIATENESS, "correct_abstention", evidence_sufficiency=None
        )
    with pytest.raises(ValidationError, match="evidence_sufficiency"):
        _judgment(Dimension.ANSWER_RELEVANCE, "off_topic", evidence_sufficiency="unknown")


def test_citation_evidence_must_come_from_the_cited_record() -> None:
    with pytest.raises(ValidationError, match="cited record only"):
        _judgment(
            Dimension.CITATION_SUPPORT,
            "supports",
            evidence=[{**REF, "kind": "cited", "evidence_id": "E02"}],
        )


# --- provenance -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provenance",
    [
        {"source": "human", "role": "evaluation", "annotator_id": "a1"},
        {"source": "judge", "role": "calibration", "judge_config_id": "0123456789ab"},
        {"source": "deterministic", "role": "evaluation", "check_id": "numeric.exact.v1"},
    ],
)
def test_each_source_names_exactly_its_identity(provenance: dict[str, Any]) -> None:
    full = {**provenance, "rubric_id": jd.rubric_id()}
    judgment = _judgment(Dimension.ANSWER_RELEVANCE, "directly_answers", provenance=full)
    assert judgment.provenance.source.value == provenance["source"]


@pytest.mark.parametrize(
    "provenance",
    [
        {"source": "human", "role": "evaluation"},
        {
            "source": "human",
            "role": "evaluation",
            "annotator_id": "a1",
            "judge_config_id": "0123456789ab",
        },
        {"source": "judge", "role": "evaluation", "annotator_id": "a1"},
        {"source": "deterministic", "role": "evaluation"},
        {"source": "llm", "role": "evaluation", "check_id": "x"},
        {"source": "human", "role": "tuning", "annotator_id": "a1"},
    ],
)
def test_inconsistent_provenance_is_rejected(provenance: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _judgment(
            Dimension.ANSWER_RELEVANCE,
            "off_topic",
            provenance={**provenance, "rubric_id": jd.rubric_id()},
        )


def test_provenance_must_name_this_rubric() -> None:
    with pytest.raises(ValidationError, match="different rubric"):
        _judgment(
            Dimension.ANSWER_RELEVANCE, "off_topic", provenance={**HUMAN, "rubric_id": "0" * 12}
        )


# --- units and identifiers ------------------------------------------------------------------


def test_unit_identifiers_are_stable() -> None:
    sub = SubclaimRef(decomposition_id="0123456789ab", index=2)
    assert ClaimUnit(claim_position=3).unit_id("dev-002") == "dev-002/claim:3"
    assert (
        ClaimCitationUnit(claim_position=1, citation_number=2, evidence_id="E04").unit_id("dev-002")
        == "dev-002/claim:1/citation:2:E04"
    )
    assert RequiredFactUnit(fact_id="dev-002-f2").unit_id("dev-002") == "dev-002/fact:dev-002-f2"
    assert ClaimAspectUnit(claim_position=1, aspect=jd.AttributionAspect.COMPANY).unit_id(
        "dev-002"
    ) == ("dev-002/claim:1/aspect:company")
    assert (
        ClaimValueUnit(claim_position=1, value_index=2, aspect=jd.NumericAspect.CURRENCY).unit_id(
            "dev-002"
        )
        == "dev-002/claim:1/value:2/aspect:currency"
    )
    assert SubclaimUnit(claim_position=1, subclaim=sub).unit_id("dev-002") == (
        "dev-002/claim:1/sub:0123456789ab:2"
    )
    assert AnswerUnit().unit_id("dev-002") == "dev-002/answer"


def test_malformed_units_and_references_are_rejected() -> None:
    for bad_unit in (
        {"kind": "claim", "claim_position": 0},
        {"kind": "claim_citation", "claim_position": 1, "citation_number": 1, "evidence_id": "E1"},
        {"kind": "required_fact", "fact_id": "test-001-f1"},
        {"kind": "sentence", "index": 1},
    ):
        with pytest.raises(ValidationError):
            _judgment(Dimension.CLAIM_FAITHFULNESS, "unsupported", unit=bad_unit)
    for bad_ref in (
        {**REF, "end": 0},
        {**REF, "start": 5, "end": 5},
        {**REF, "span_sha256": "xyz"},
        {**REF, "page_number": 0},
        {**REF, "text": "copied passage"},  # extra fields (such as copied text) are forbidden
    ):
        with pytest.raises(ValidationError):
            _judgment(Dimension.CLAIM_FAITHFULNESS, "supported", evidence=[bad_ref])


def test_held_out_question_ids_are_rejected_by_construction() -> None:
    with pytest.raises(ValidationError):
        _judgment(Dimension.ANSWER_RELEVANCE, "off_topic", question_id="test-001")


# --- compound claims ------------------------------------------------------------------------


def _decomposition(text: str, **over: Any) -> ClaimDecomposition:
    payload: dict[str, Any] = {
        "question_id": "dev-001",
        "generation_record_sha256": DIGEST,
        "claim_position": 1,
        "claim_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "method": "human",
        "subclaims": [{"index": 1, "start": 0, "end": 9}, {"index": 2, "start": 10, "end": 21}],
    }
    payload.update(over)
    return ClaimDecomposition.model_validate(payload)


def test_decompositions_are_spans_of_the_original_claim() -> None:
    text = "Revenue A and revenue B."
    decomposition = _decomposition(text)
    decomposition.check_against(text)
    assert decomposition.decomposition_id == _decomposition(text).decomposition_id
    with pytest.raises(ValueError, match="original claim text"):
        decomposition.check_against(text + " ")
    with pytest.raises(ValueError, match="outside the claim text"):
        _decomposition(
            "short",
            subclaims=[{"index": 1, "start": 0, "end": 3}, {"index": 2, "start": 3, "end": 40}],
        ).check_against("short")
    with pytest.raises(ValidationError, match="names its rule"):
        _decomposition(text, method="deterministic_rule")
    with pytest.raises(ValidationError):
        _decomposition(text, method="llm")
    with pytest.raises(ValidationError):
        _decomposition(text, subclaims=[{"index": 1, "start": 0, "end": 9}])  # at least two
    with pytest.raises(ValidationError, match=r"1\.\.n"):
        _decomposition(
            text,
            subclaims=[{"index": 2, "start": 0, "end": 9}, {"index": 1, "start": 10, "end": 21}],
        )


def test_subclaim_judgments_keep_parent_provenance() -> None:
    sub = {"decomposition_id": "0123456789ab", "index": 1}
    judged = _judgment(
        Dimension.CLAIM_FAITHFULNESS,
        "supported",
        unit={"kind": "subclaim", "claim_position": 2, "subclaim": sub},
    )
    assert isinstance(judged.unit, SubclaimUnit)
    assert judged.unit.claim_position == 2
    with pytest.raises(ValidationError, match="aspect is required"):
        _judgment(
            Dimension.FINANCIAL_ATTRIBUTION,
            "correct",
            unit={"kind": "subclaim", "claim_position": 2, "subclaim": sub},
        )
    attribution = _judgment(
        Dimension.FINANCIAL_ATTRIBUTION,
        "unverified",
        unit={"kind": "subclaim", "claim_position": 2, "subclaim": sub, "aspect": "company"},
    )
    assert attribution.unit_id.endswith("/aspect:company")


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (("supported", "supported"), "supported"),
        (("supported", "unsupported"), "unsupported"),  # never silently fully supported
        (("supported", "contradicted", "unsupported"), "contradicted"),
        (("supported", "not_assessable"), "not_assessable"),
    ],
)
def test_parent_faithfulness_is_derived_conservatively(
    labels: tuple[str, ...], expected: str
) -> None:
    spans = [{"index": i, "start": 2 * i, "end": 2 * i + 1} for i in range(1, len(labels) + 1)]
    decomposition = _decomposition("x" * 20, subclaims=spans)
    assert jd.derive_faithfulness(decomposition, dict(enumerate(labels, start=1))) == expected


def test_missing_subclaim_labels_never_yield_supported() -> None:
    decomposition = _decomposition("Revenue A and revenue B.")
    assert jd.derive_faithfulness(decomposition, {1: "supported"}) == "not_assessable"
    assert jd.derive_faithfulness(decomposition, {}) == "not_assessable"
    with pytest.raises(ValueError, match="outside the decomposition"):
        jd.derive_faithfulness(decomposition, {1: "supported", 2: "supported", 3: "supported"})
    with pytest.raises(ValueError, match="faithfulness labels"):
        jd.derive_faithfulness(decomposition, {1: "correct", 2: "supported"})


# --- privacy and immutability ---------------------------------------------------------------


def test_public_view_holds_no_text() -> None:
    judgment = _judgment(Dimension.CLAIM_FAITHFULNESS, "supported")
    view = jd.public_view(judgment)
    assert set(view) == {
        "question_id",
        "unit_id",
        "dimension",
        "label",
        "not_assessable_reason",
        "source",
        "role",
        "evidence_count",
        "rubric_id",
    }
    assert "span" not in json.dumps(view)
    with pytest.raises(ValidationError):
        judgment.label = "unsupported"  # frozen model


# --- final verification: rubric completeness, evidence kinds, exact spans ------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.setattr(
            jd,
            "EVIDENCE_KINDS",
            {**jd.EVIDENCE_KINDS, Dimension.CITATION_SUPPORT: tuple(jd.EvidenceKind)},
        ),
        lambda m: m.setattr(jd, "_QUESTION_ID", r"^(dev|test)-\d{3}$"),
    ],
)
def test_normative_definitions_cannot_change_silently(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    before = jd.rubric_id()
    mutate(monkeypatch)
    assert jd.rubric_id() != before


def test_rubric_definition_names_every_vocabulary() -> None:
    definition = jd.rubric_definition()
    assert definition["attribution_aspects"] == [a.value for a in jd.AttributionAspect]
    assert definition["numeric_aspects"] == [a.value for a in jd.NumericAspect]
    assert definition["evidence_sufficiency"] == [e.value for e in jd.EvidenceSufficiency]
    assert definition["reference_bases"] == [b.value for b in jd.ReferenceBasis]
    assert definition["unit_kinds"] == [u.value for u in jd.UnitKind]
    assert set(definition["evidence_kinds"]) == {k.value for k in jd.EvidenceKind}
    assert definition["evidence_kinds_by_dimension"]["citation_support"] == ["cited"]


def test_evidence_kinds_are_explicit_and_cannot_masquerade() -> None:
    with pytest.raises(ValidationError, match="name an evidence ID"):
        _judgment(
            Dimension.FACTUAL_CORRECTNESS,
            "correct",
            reference_basis="source_evidence",
            evidence=[{**REF, "kind": "reference_source"}],
        )  # keeps an evidence ID
    with pytest.raises(ValidationError, match="name an evidence ID"):
        _judgment(
            Dimension.CLAIM_FAITHFULNESS, "supported", evidence=[{**REF, "evidence_id": None}]
        )
    with pytest.raises(ValidationError, match="does not accept that evidence kind"):
        _judgment(Dimension.CITATION_SUPPORT, "supports", evidence=[REF])  # selected, not cited
    with pytest.raises(ValidationError):
        _judgment(
            Dimension.CLAIM_FAITHFULNESS, "supported", evidence=[{**REF, "kind": "retrieved"}]
        )


def test_decomposition_spans_reproduce_exact_substrings() -> None:
    text = "Revenue A and revenue B."
    decomposition = _decomposition(text)
    assert decomposition.subclaim_texts(text) == (text[0:9], text[10:21])
    accented = "Cafe" + chr(0x301) + " sales rose and costs fell"
    split = _decomposition(
        accented,
        subclaims=[{"index": 1, "start": 0, "end": 4}, {"index": 2, "start": 5, "end": 10}],
    )
    with pytest.raises(ValueError, match="combining character"):
        split.check_against(accented)


def test_validation_errors_never_echo_private_input() -> None:
    secret = "SENTINEL-PRIVATE-PASSAGE-TEXT"
    with pytest.raises(ValidationError) as info:
        _judgment(Dimension.CLAIM_FAITHFULNESS, "supported", evidence=[{**REF, "text": secret}])
    assert secret not in str(info.value)
    with pytest.raises(ValidationError) as label_error:
        _judgment(Dimension.CLAIM_FAITHFULNESS, secret)
    assert secret not in str(label_error.value)
