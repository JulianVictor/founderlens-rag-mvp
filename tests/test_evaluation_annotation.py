"""Human annotation and calibration workflow (Commit 7.3), offline.

A synthetic live run is produced by the real harness with a mocked provider over
the synthetic NVIDIA/Alphabet collection, verified through the judging-input
gate, and annotated with mocked human input. No model, key, network, real
corpus or held-out file is used, and nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import copy
import json
import pathlib
import socket
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from founderlens.evaluation import annotation as an
from founderlens.evaluation import calibration as cal
from founderlens.evaluation import financial_checks as fc
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation import partition as pt
from founderlens.evaluation.judgments import EvidenceKind, Role, rubric_id
from tests.test_evaluation_generation_eval import SPLIT
from tests.test_evaluation_judging_input import CLEAN, _run, _verify
from tests.test_guardrails_response_checks import COLLECTION, VOCAB

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
NVIDIA = "NVIDIA Corporation"
Q3 = "nvda-fy2025-q3:p0001:c0000:0c2f87411627"
Q4 = "nvda-fy2025-q4:p0001:c0000:0c2f87411627"


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(ge, "_git_state", lambda root=None: dict(CLEAN))


@pytest.fixture
def inputs(tmp_path: pathlib.Path) -> dict[str, Any]:
    return _run(tmp_path / "run")


@pytest.fixture
def ctx(inputs: dict[str, Any]) -> an.AnnotationContext:
    return an.make_context(_verify(inputs), COLLECTION, SPLIT, inputs["records"], VOCAB)


@pytest.fixture
def paths(tmp_path: pathlib.Path, ctx: an.AnnotationContext) -> an.AnnotationPaths:
    return an.AnnotationPaths(tmp_path / "annotation" / ctx.run.run_id)


# The synthetic run's allocation: dev-003 (temporal, answered) is calibration; the other
# eligible answers (dev-002, dev-004, dev-005, dev-006) are evaluation; dev-001 failed.
SYNTHETIC_POLICY = {
    **pt.POLICY_V1,
    "seed": "founderlens-synthetic-test",
    "quotas": {"temporal_multi_evidence/answered": 1},
}


def manifest(ctx: an.AnnotationContext) -> pt.PartitionManifest:
    return pt.verify_manifest(ctx, pt.build_manifest(ctx, SYNTHETIC_POLICY))


# --- helpers ----------------------------------------------------------------------------------


def sheet(
    ctx: an.AnnotationContext,
    qid: str = "dev-003",
    annotator: str = "ann-a1",
    role: Role = Role.CALIBRATION,
) -> dict[str, Any]:
    return an.worksheet_template(an.build_packet(ctx, qid), annotator, role)


def find(ws: dict[str, Any], dimension: str, **unit: Any) -> dict[str, Any]:
    return next(
        e
        for e in ws["entries"]
        if e.get("dimension") == dimension and unit.items() <= e["unit"].items()
    )


def span(chunk_id: str, needle: str, kind: str, evidence_id: str | None = None) -> dict[str, Any]:
    text = next(c.text for c in COLLECTION.chunks if c.chunk_id == chunk_id)
    start = text.index(needle)
    return {
        "kind": kind,
        "evidence_id": evidence_id,
        "chunk_id": chunk_id,
        "start": start,
        "end": start + len(needle),
    }


def plan(ctx: an.AnnotationContext, ledger: an.Ledger, ws: dict[str, Any]) -> an.ImportResult:
    return an.plan_import(ctx, ledger, ws, worksheet_sha256=an.digest(ws), now=NOW)


def commit(ctx: an.AnnotationContext, ledger: an.Ledger, ws: dict[str, Any]) -> an.Ledger:
    return an.Ledger(ledger.entries + plan(ctx, ledger, ws).accepted)


def supported(ws: dict[str, Any], position: int = 1, evidence: str = "E02") -> dict[str, Any]:
    entry = find(ws, "claim_faithfulness", claim_position=position)
    needle, chunk = ("$35.1 billion", Q3) if position == 1 else ("$39.3 billion", Q4)
    entry.update(
        review_action="confirmed",
        label="supported",
        evidence=[span(chunk, needle, "selected_context", evidence)],
    )
    return entry


def fact_entry(**over: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "entry_type": "reference_fact",
        "review_action": "confirmed",
        "fact_id": "dev-003-f1",
        "metric": "revenue",
        "company": NVIDIA,
        "segment": None,
        "period": {"label": "FY2025 Q3", "basis": "fiscal", "company": NVIDIA},
        "value": {"magnitude": "35.1", "unit": "money", "scale": "billion", "currency": "USD"},
        "basis": "GAAP",
        "evidence": [span(Q3, "$35.1 billion", "reference_source")],
    }
    entry.update(over)
    return entry


def with_fact(ws: dict[str, Any], **over: Any) -> dict[str, Any]:
    ws["entries"] = [e for e in ws["entries"] if e["entry_type"] != "reference_fact"]
    ws["entries"].append(fact_entry(**over))
    return ws


def association(ctx: an.AnnotationContext, **over: Any) -> dict[str, Any]:
    claim = next(c for c in ctx.question("dev-003").claims if c.position == 1)
    occ = fc.extract_occurrences(claim.text, VOCAB)[0]
    entry: dict[str, Any] = {
        "entry_type": "fact_association",
        "review_action": "confirmed",
        "claim_position": 1,
        "value_index": occ.value_index,
        "start": occ.start,
        "end": occ.end,
        "fact_id": "dev-003-f1",
        "metric": "revenue",
    }
    entry.update(over)
    return entry


def only(ws: dict[str, Any], *entries: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(ws)
    out["entries"] = list(entries)
    return out


def refused(ctx: an.AnnotationContext, ledger: an.Ledger, ws: dict[str, Any], match: str) -> str:
    with pytest.raises(an.AnnotationError, match=match) as info:
        plan(ctx, ledger, ws)
    return str(info.value)


def private_texts(ctx: an.AnnotationContext) -> list[str]:
    texts = [c.text for q in ctx.run.questions for c in q.claims]
    texts += [c.text for c in COLLECTION.chunks]
    return [*texts, "SENTINEL-QUESTION", "SENTINEL-LIMITATION"]


# --- packets ----------------------------------------------------------------------------------


def test_packets_are_deterministic_and_bound_to_the_generation_record(
    ctx: an.AnnotationContext,
) -> None:
    packet = an.build_packet(ctx, "dev-003")
    assert packet == an.build_packet(ctx, "dev-003")
    body = {k: v for k, v in packet.items() if k != "packet_sha256"}
    assert packet["packet_sha256"] == an.digest(body)
    q = ctx.question("dev-003")
    assert packet["generation_record_sha256"] == q.record_sha256
    assert packet["run_id"] == ctx.run.run_id
    assert (packet["rubric_id"], packet["extractor_id"], packet["checker_id"]) == (
        rubric_id(),
        fc.extractor_id(),
        fc.checker_id(),
    )
    assert packet["question"]["category"] == "temporal_multi_evidence"
    assert packet["question"]["answerable"] is True
    assert packet["response"]["status"] == "answered"
    assert [c["position"] for c in packet["response"]["claims"]] == [1, 2]
    claim = packet["response"]["claims"][0]
    assert claim["text"] == q.claims[0].text
    value = claim["numeric_values"][0]
    assert value["span_text"] == claim["text"][value["start"] : value["end"]]
    assert {c["outcome"] for c in value["deterministic_checks"]} <= {"unverified", "ambiguous"}
    assert an.run_counts(ctx) == {
        "scheduled_outcomes": 6,
        "eligible_answers": 5,
        "answers_with_claims": 3,
        "answers_without_claims": 2,
        "claims": 4,
        "claim_evidence_associations": 4,
        "distinct_citations": 4,
        "financial_quantities": 4,
        "ineligible": ["dev-001"],
    }


def test_packets_separate_selected_cited_and_reference_evidence(
    ctx: an.AnnotationContext,
) -> None:
    evidence = an.build_packet(ctx, "dev-003")["evidence"]
    selected = evidence["selected_context"]
    assert [e["evidence_id"] for e in selected] == ["E01", "E02", "E03", "E04", "E05"]
    assert [e["evidence_id"] for e in selected if e["cited_by_answer"]] == ["E02", "E03"]
    cited = evidence["cited"]
    assert [(c["citation_number"], c["evidence_id"], c["cited_by_claims"]) for c in cited] == [
        (1, "E02", [1]),
        (2, "E03", [2]),
    ]
    assert cited[0]["chunk_id"] == Q3
    assert "35.1" in cited[0]["text"]
    reference = {r["fact_id"]: r for r in evidence["reference_source"]}
    assert set(reference) == {"dev-003-f1", "dev-003-f2"}
    location = reference["dev-003-f1"]["trusted_locations"][0]
    assert (location["document_id"], location["page_number"]) == ("nvda-fy2025-q3", 1)
    assert [c["chunk_id"] for c in location["chunks"]] == [Q3]
    assert location["chunks"][0]["in_selected_context"] is True
    # an unanswerable question has no reference source and states its expected reason
    assert an.build_packet(ctx, "dev-004")["evidence"]["reference_source"] == []


def test_templates_list_every_unit_without_any_label(ctx: an.AnnotationContext) -> None:
    ws = sheet(ctx)
    assert all(e["review_action"] == "pending" for e in ws["entries"])
    assert all(e.get("label") is None for e in ws["entries"])
    kinds: dict[str, int] = {}
    for e in ws["entries"]:
        key = e.get("dimension", e["entry_type"])
        kinds[key] = kinds.get(key, 0) + 1
    assert kinds == {
        "claim_faithfulness": 2,
        "citation_support": 2,  # one per claim-to-evidence association
        "financial_attribution": 6,
        "numerical_correctness": 2,
        "factual_correctness": 2,
        "reference_fact": 2,
        "answer_completeness": 1,
        "abstention_appropriateness": 1,
        "answer_relevance": 1,
    }
    assert not [e for e in ws["entries"] if e["entry_type"] == "fact_association"]
    # an insufficient_evidence answer is not pre-marked as a correct abstention
    abstention = find(sheet(ctx, "dev-006"), "abstention_appropriateness")
    assert (abstention["review_action"], abstention["label"]) == ("pending", None)
    an.Worksheet.model_validate(ws)


# --- human labels -----------------------------------------------------------------------------


def test_partial_progress_round_trips_and_reimports_idempotently(
    ctx: an.AnnotationContext, paths: an.AnnotationPaths
) -> None:
    ws = sheet(ctx)
    supported(ws)
    path = paths.worksheets / "dev-003.ann-a1.json"
    an.write_json(path, ws)
    dry = an.import_worksheet(ctx, paths, path, manifest=manifest(ctx), dry_run=True, now=NOW)
    assert len(dry.accepted) == 1
    assert not paths.ledger.exists()
    first = an.import_worksheet(ctx, paths, path, manifest=manifest(ctx), now=NOW)
    assert (len(first.accepted), first.pending) == (1, len(ws["entries"]) - 1)
    again = an.import_worksheet(ctx, paths, path, manifest=manifest(ctx), now=NOW)
    assert (len(again.accepted), again.duplicates) == (0, 1)  # no duplicate judgment
    supported(ws, 2, "E03")
    an.write_json(path, ws)
    later = an.import_worksheet(ctx, paths, path, manifest=manifest(ctx), now=NOW)
    assert (len(later.accepted), later.duplicates) == (1, 1)
    ledger = an.load_ledger(paths.ledger)
    assert len(ledger.entries) == 2
    assert {j.unit_id for j in ledger.judgments()} == {"dev-003/claim:1", "dev-003/claim:2"}


def test_labels_need_an_explicit_action_and_the_rubric_contract(
    ctx: an.AnnotationContext,
) -> None:
    ledger = an.Ledger()
    entry = find(sheet(ctx), "claim_faithfulness", claim_position=1)
    ws = only(sheet(ctx), {**entry, "review_action": "confirmed"})
    refused(ctx, ledger, ws, "malformed")  # confirmed without a label
    ws = only(sheet(ctx), {**entry, "review_action": "confirmed", "label": "supported"})
    refused(ctx, ledger, ws, "requires evidence")
    ws = only(sheet(ctx), {**entry, "review_action": "confirmed", "label": "not_assessable"})
    refused(ctx, ledger, ws, "requires a reason")
    ws = only(sheet(ctx), {**entry, "review_action": "confirmed", "label": "correct"})
    refused(ctx, ledger, ws, "unknown label")
    na = {
        **entry,
        "review_action": "confirmed",
        "label": "not_assessable",
        "not_assessable_reason": "claim_not_atomic",
    }
    assert len(plan(ctx, ledger, only(sheet(ctx), na)).accepted) == 1
    citation = find(sheet(ctx), "citation_support", claim_position=1)
    wrong_chunk = {
        **citation,
        "review_action": "confirmed",
        "label": "supports",
        "evidence": [span(Q4, "$39.3 billion", "cited", "E02")],
    }
    refused(ctx, ledger, only(sheet(ctx), wrong_chunk), "does not name a chunk")
    value = find(sheet(ctx), "numerical_correctness", claim_position=1)
    bad_index = {
        **value,
        "review_action": "confirmed",
        "label": "unverified",
        "unit": {**value["unit"], "value_index": 7},
    }
    refused(ctx, ledger, only(sheet(ctx), bad_index), "value index outside")


def test_provenance_binds_annotator_record_and_identities(ctx: an.AnnotationContext) -> None:
    ws = sheet(ctx, annotator="ann-b2", role=Role.EVALUATION)
    entry = supported(ws)
    entry["rationale"] = "matches the selected passage"
    (accepted,) = plan(ctx, an.Ledger(), only(ws, entry)).accepted
    q = ctx.question("dev-003")
    assert accepted.annotator_id == "ann-b2"
    assert accepted.role is Role.EVALUATION
    assert (accepted.run_id, accepted.question_id) == (ctx.run.run_id, "dev-003")
    assert accepted.generation_record_sha256 == q.record_sha256
    assert accepted.packet_sha256 == an.build_packet(ctx, "dev-003")["packet_sha256"]
    assert (accepted.rubric_id, accepted.extractor_id, accepted.checker_id) == (
        rubric_id(),
        fc.extractor_id(),
        fc.checker_id(),
    )
    assert accepted.unit_key == "judgment:dev-003/claim:1#claim_faithfulness"
    assert (accepted.revision, accepted.supersedes) == (1, None)
    assert accepted.payload["provenance"] == {
        "source": "human",
        "role": "evaluation",
        "annotator_id": "ann-b2",
        "judge_config_id": None,
        "check_id": None,
        "rubric_id": rubric_id(),
    }
    assert accepted.payload["evidence"][0]["kind"] == "selected_context"


@pytest.mark.parametrize("annotator", ["jane.doe@example.com", "Jane", "ann-", "annotator1"])
def test_annotators_must_be_pseudonymous(ctx: an.AnnotationContext, annotator: str) -> None:
    with pytest.raises(an.AnnotationError, match="pseudonym"):
        sheet(ctx, annotator=annotator)
    ws = sheet(ctx)
    ws["annotator_id"] = annotator
    refused(ctx, an.Ledger(), ws, "annotator_id")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("generation_record_sha256", "c" * 64, "stale or foreign record"),
        ("packet_sha256", "c" * 64, "stale packet"),
        ("collection_artifact_sha256", "c" * 64, "source artifact"),
        ("run_id", "live-other-run", "binding failed: run"),
        ("rubric_id", "0" * 12, "rubric"),
        ("question_id", "test-001", "question_id"),
        ("question_id", "dev-099", "outside the verified run"),
        ("question_id", "dev-001", "ineligible"),
    ],
)
def test_stale_foreign_or_ineligible_bindings_are_refused(
    ctx: an.AnnotationContext, field: str, value: str, match: str
) -> None:
    ws = sheet(ctx)
    supported(ws)
    ws[field] = value
    refused(ctx, an.Ledger(), ws, match)


def test_duplicates_conflicts_revisions_and_withdrawals(ctx: an.AnnotationContext) -> None:
    ws = sheet(ctx)
    entry = supported(ws)
    refused(ctx, an.Ledger(), only(ws, entry, entry), "duplicate entry")
    ledger = commit(ctx, an.Ledger(), only(ws, entry))
    (first,) = ledger.entries
    changed = {
        **entry,
        "label": "unsupported",
        "evidence": [],
        "error_type": "unsupported_by_evidence",
    }
    message = refused(ctx, ledger, only(ws, changed), "conflicts with current entry")
    assert first.entry_id in message
    stale = {**changed, "supersedes": "f" * 24, "revision_reason": "correction"}
    refused(ctx, ledger, only(ws, stale), "conflicts with current entry")
    revised = {**changed, "supersedes": first.entry_id, "revision_reason": "correction"}
    ledger = commit(ctx, ledger, only(ws, revised))
    assert [e.revision for e in ledger.entries] == [1, 2]  # history kept
    assert [j.label for j in ledger.judgments()] == ["unsupported"]
    assert plan(ctx, ledger, only(ws, revised)).duplicates == 1
    # a second annotator disagrees: both kept, the disagreement is reported
    other = sheet(ctx, annotator="ann-c3")
    ledger = commit(ctx, ledger, only(other, supported(other)))
    assert ledger.disagreements() == ["judgment:dev-003/claim:1#claim_faithfulness"]
    head = ledger.heads()[("ann-a1", "judgment:dev-003/claim:1#claim_faithfulness")]
    withdraw = {
        **entry,
        "review_action": "withdrawn",
        "supersedes": head.entry_id,
        "revision_reason": "entry_error",
    }
    ledger = commit(ctx, ledger, only(ws, withdraw))
    assert [j.provenance.annotator_id for j in ledger.judgments()] == ["ann-c3"]
    assert len(ledger.entries) == 4
    nothing = {**find(sheet(ctx), "answer_relevance"), "review_action": "withdrawn"}
    refused(ctx, ledger, only(ws, nothing), "nothing to withdraw")


def test_ledger_digests_catch_accidental_edits(
    ctx: an.AnnotationContext, paths: an.AnnotationPaths
) -> None:
    ws = sheet(ctx)
    supported(ws)
    path = paths.worksheets / "dev-003.ann-a1.json"
    an.write_json(path, ws)
    an.import_worksheet(ctx, paths, path, manifest=manifest(ctx), now=NOW)
    assert len(an.load_ledger(paths.ledger).entries) == 1
    text = paths.ledger.read_text(encoding="utf-8")
    paths.ledger.write_text(text.replace('"supported"', '"unsupported"'), encoding="utf-8")
    with pytest.raises(an.AnnotationError, match="digest does not match"):
        an.load_ledger(paths.ledger)
    paths.ledger.write_text(text + text, encoding="utf-8")
    with pytest.raises(an.AnnotationError, match="duplicated entry"):
        an.load_ledger(paths.ledger)


# --- reference facts and associations ----------------------------------------------------------


def test_reference_facts_need_explicit_human_verification(ctx: an.AnnotationContext) -> None:
    ws = sheet(ctx)
    pending = [e for e in ws["entries"] if e["entry_type"] == "reference_fact"]
    assert pending
    assert all(e["review_action"] == "pending" for e in pending)
    ledger = commit(ctx, an.Ledger(), ws)
    assert ledger.verified_facts() == {}  # parsed or pending fields verify nothing
    ledger = commit(ctx, an.Ledger(), with_fact(sheet(ctx)))
    fact = ledger.verified_facts()["dev-003-f1"]
    assert (fact.verified_by, fact.verification_role) == ("ann-a1", Role.CALIBRATION)
    assert (fact.metric, fact.company, fact.period.label, fact.basis) == (
        "revenue",
        NVIDIA,
        "FY2025 Q3",
        "GAAP",
    )
    assert (
        fact.value.value
        == fc.Quantity(Decimal("35.1"), 1, False, fc.Unit.MONEY, "billion", "usd", "USD", 1).value
    )
    assert fact.evidence[0].kind is EvidenceKind.REFERENCE_SOURCE
    assert ledger.associations() == ()  # never created automatically


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"evidence": [span(Q4, "$39.3 billion", "reference_source")]}, "trusted reference page"),
        ({"evidence": [span(Q3, "$35.1 billion", "cited", "E02")]}, "reference_source evidence"),
        ({"fact_id": "dev-002-f1"}, "not a required fact"),
        ({"company": "Acme Corp"}, "not a corpus company"),
        ({"segment": "Google Cloud"}, "approved segment"),
        ({"value": {"magnitude": "35.1", "unit": "money", "scale": "billion"}}, "currency"),
        ({"metric": None}, "malformed"),
        ({"period": {"label": "FY2025 Q3", "basis": "calendar"}}, "malformed"),
    ],
)
def test_wrong_or_incomplete_reference_facts_are_refused(
    ctx: an.AnnotationContext, over: dict[str, Any], match: str
) -> None:
    refused(ctx, an.Ledger(), with_fact(sheet(ctx), **over), match)


def test_unresolved_or_disputed_facts_are_not_verified(ctx: an.AnnotationContext) -> None:
    unresolved = {
        "entry_type": "reference_fact",
        "review_action": "unresolved",
        "fact_id": "dev-003-f2",
        "unresolved_reason": "reference_ambiguous",
    }
    ledger = commit(ctx, an.Ledger(), only(sheet(ctx), unresolved))
    assert (ledger.verified_facts(), ledger.unresolved_facts()) == ({}, ["dev-003-f2"])
    ledger = commit(ctx, an.Ledger(), with_fact(sheet(ctx)))
    other = with_fact(
        sheet(ctx, annotator="ann-c3"),
        value={"magnitude": "35.0", "unit": "money", "scale": "billion", "currency": "USD"},
    )
    ledger = commit(ctx, ledger, other)
    assert ledger.verified_facts() == {}  # annotators disagree: not verified
    assert ledger.disagreements() == ["fact:dev-003-f1"]


def test_fact_associations_are_explicit_and_validated(ctx: an.AnnotationContext) -> None:
    ledger = commit(ctx, an.Ledger(), with_fact(sheet(ctx)))
    ws = sheet(ctx)
    assert ledger.associations() == ()
    ledger = commit(ctx, ledger, only(ws, association(ctx)))
    (link,) = ledger.associations()
    q = ctx.question("dev-003")
    assert (link.question_id, link.generation_record_sha256) == ("dev-003", q.record_sha256)
    assert (link.fact_id, link.metric, link.associated_by) == ("dev-003-f1", "revenue", "ann-a1")
    claim = q.claims[0]
    occurrences = fc.extract_occurrences(claim.text, VOCAB)
    results = fc.check_value(
        question_id="dev-003",
        generation_record_sha256=q.record_sha256,
        claim_position=1,
        claim_text=claim.text,
        occurrences=occurrences,
        value_index=link.value_index,
        association=link,
        facts=ledger.verified_facts(),
        vocabulary=VOCAB,
    )
    assert {r.aspect: r.outcome for r in results}["value"] is fc.Outcome.EXACT_MATCH


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"start": 0}, "span does not match"),
        ({"value_index": 9}, "value index outside"),
        ({"claim_position": 5}, "claim position does not exist"),
        ({"fact_id": "dev-003-f2"}, "unknown, unresolved"),
        ({"fact_id": "dev-002-f1"}, "another question"),
        ({"metric": "net_income"}, "metric differs"),
    ],
)
def test_wrong_fact_associations_are_refused(
    ctx: an.AnnotationContext, over: dict[str, Any], match: str
) -> None:
    ledger = commit(ctx, an.Ledger(), with_fact(sheet(ctx)))
    refused(ctx, ledger, only(sheet(ctx), association(ctx, **over)), match)


def test_associations_need_a_verified_fact_even_when_numbers_are_equal(
    ctx: an.AnnotationContext,
) -> None:
    # the claim states the same number as the source, but nothing is verified yet
    refused(ctx, an.Ledger(), only(sheet(ctx), association(ctx)), "unknown, unresolved")


# --- decompositions and calibration ------------------------------------------------------------


def _compound_ledger(ctx: an.AnnotationContext) -> an.Ledger:
    ws = sheet(ctx)
    decomposition = {
        "entry_type": "decomposition",
        "review_action": "confirmed",
        "claim_position": 1,
        "subclaims": [[0, 6], [7, 20]],
    }
    ledger = commit(ctx, an.Ledger(), only(ws, decomposition))
    (decomposition_id,) = ledger.decompositions()
    subs = []
    for index, label in ((1, "supported"), (2, "unsupported")):
        subs.append(
            {
                "entry_type": "judgment",
                "review_action": "confirmed",
                "dimension": "claim_faithfulness",
                "unit": {
                    "kind": "subclaim",
                    "claim_position": 1,
                    "subclaim": {"decomposition_id": decomposition_id, "index": index},
                },
                "label": label,
                "evidence": [span(Q3, "NVIDIA", "selected_context", "E02")]
                if label == "supported"
                else [],
            }
        )
    return commit(ctx, ledger, only(ws, *subs))


def test_decompositions_bind_to_the_claim_text(ctx: an.AnnotationContext) -> None:
    bad = {
        "entry_type": "decomposition",
        "review_action": "confirmed",
        "claim_position": 1,
        "subclaims": [[0, 6], [7, 90]],
    }
    refused(ctx, an.Ledger(), only(sheet(ctx), bad), "do not fit")
    ledger = _compound_ledger(ctx)
    candidates, _ = cal.natural_candidates(ctx, ledger)
    compound = [x for x in candidates if x.category is cal.Category.COMPOUND_MIXED_SUPPORT]
    assert [(x.question_id, x.claim_position, x.label) for x in compound] == [
        ("dev-003", 1, "unsupported")
    ]


def _labelled_ledger(ctx: an.AnnotationContext) -> an.Ledger:
    ws = sheet(ctx)
    citation = find(ws, "citation_support", claim_position=1)
    citation.update(
        review_action="confirmed",
        label="supports",
        evidence=[span(Q3, "$35.1 billion", "cited", "E02")],
    )
    attempt = find(ws, "abstention_appropriateness")
    attempt.update(
        review_action="confirmed",
        label="correct_attempt",
        evidence_sufficiency="retrieved_sufficient",
    )
    ledger = commit(
        ctx, an.Ledger(), only(ws, supported(ws), supported(ws, 2, "E03"), citation, attempt)
    )
    false_abstention = find(sheet(ctx, "dev-006"), "abstention_appropriateness")
    false_abstention.update(
        review_action="confirmed",
        label="false_abstention",
        evidence_sufficiency="retrieved_sufficient",
    )
    ledger = commit(ctx, ledger, only(sheet(ctx, "dev-006"), false_abstention))
    missed = find(sheet(ctx, "dev-004"), "abstention_appropriateness")
    missed.update(
        review_action="confirmed",
        label="missed_abstention",
        evidence_sufficiency="retrieved_insufficient",
    )
    wrong_metric = find(sheet(ctx, "dev-002"), "claim_faithfulness", claim_position=1)
    wrong_metric.update(review_action="confirmed", label="unsupported", error_type="wrong_metric")
    ledger = commit(ctx, ledger, only(sheet(ctx, "dev-004"), missed))
    return commit(ctx, ledger, only(sheet(ctx, "dev-002"), wrong_metric))


def test_natural_examples_are_categorized_and_selection_is_deterministic(
    ctx: an.AnnotationContext,
) -> None:
    ledger = _labelled_ledger(ctx)
    candidates, excluded = cal.natural_candidates(ctx, ledger)
    assert {x.category.value for x in candidates} == {
        "supported_claim",
        "supported_citation",
        "correct_attempt",
        "false_abstention",
        "missed_abstention",
        "wrong_metric",
    }
    assert set(excluded.values()) == {0}
    assert {x.agreement for x in candidates} == {"single_reviewed"}  # never "agreement"
    selected = cal.select(candidates)
    assert selected == cal.select(list(reversed(candidates)))
    assert {x.question_category for x in selected} >= {"unanswerable", "temporal_multi_evidence"}
    first = cal.build_calibration_set(ctx, ledger)
    assert first == cal.build_calibration_set(ctx, ledger)
    assert first["selection_policy_id"] == cal.selection_policy_id()
    assert first["disclosure"]["eligible_answers"] == 5
    assert "Small development sample" in first["disclosure"]["note"]


def test_selection_caps_each_category_and_mixes_strata() -> None:
    def example(i: int, category: cal.Category, qcat: str) -> cal.NaturalExample:
        return cal.NaturalExample(
            f"{i:016x}", f"dev-{i:03d}", qcat, "answered", "reported", 1, f"u{i}",
            "claim_faithfulness", "supported", category, "single_annotator", (),
        )  # fmt: skip

    easy = [example(i, cal.Category.SUPPORTED_CLAIM, "single_doc_factual") for i in range(30)]
    hard = [example(100 + i, cal.Category.WRONG_SCALE, "cross_doc_comparison") for i in range(3)]
    mixed = [example(200 + i, cal.Category.SUPPORTED_CLAIM, "unanswerable") for i in range(2)]
    selected = cal.select(easy + hard + mixed)
    cap = cal.SELECTION_POLICY["max_per_category"]
    supported_ = [x for x in selected if x.category is cal.Category.SUPPORTED_CLAIM]
    assert len(supported_) == cap
    assert sum(x.question_category == "unanswerable" for x in supported_) == 2  # strata mixed
    assert sum(x.category is cal.Category.WRONG_SCALE for x in selected) == 3


def test_disagreements_not_assessable_and_evaluation_labels_are_excluded(
    ctx: an.AnnotationContext,
) -> None:
    ws = sheet(ctx)
    ledger = commit(ctx, an.Ledger(), only(ws, supported(ws)))
    other = sheet(ctx, annotator="ann-c3")
    entry = find(other, "claim_faithfulness", claim_position=1)
    entry.update(review_action="confirmed", label="unsupported")
    relevance = find(other, "answer_relevance")
    relevance.update(
        review_action="confirmed",
        label="not_assessable",
        not_assessable_reason="annotator_uncertain",
    )
    ledger = commit(ctx, ledger, only(other, entry, relevance))
    evaluation = sheet(ctx, "dev-002", role=Role.EVALUATION)
    labelled = find(evaluation, "claim_faithfulness", claim_position=1)
    labelled.update(
        review_action="confirmed",
        label="supported",
        evidence=[span(Q4, "$39.3 billion", "selected_context", "E03")],
    )
    ledger = commit(ctx, ledger, only(evaluation, labelled))
    candidates, excluded = cal.natural_candidates(ctx, ledger)
    assert candidates == []
    assert excluded == {
        "disputed": 1,
        "withdrawn": 0,
        "not_assessable": 1,
        "unverified_or_not_applicable": 0,
        "evaluation_partition": 1,
    }
    listed = cal.build_calibration_set(ctx, ledger)["excluded_units"]
    assert sorted(x["reason"] for x in listed) == [
        "disputed",
        "evaluation_partition",
        "not_assessable",
    ]  # difficult and non-assessable units stay visible to diagnostics
    assert all(set(x) == {"unit_key", "question_id", "reason", "review_state"} for x in listed)


def test_synthetic_variants_keep_provenance_and_never_touch_the_source(
    ctx: an.AnnotationContext, inputs: dict[str, Any]
) -> None:
    before = json.dumps(inputs["records"], sort_keys=True)
    ledger = _labelled_ledger(ctx)
    variants = cal.synthetic_variants(ctx, ledger)
    assert variants == cal.synthetic_variants(ctx, ledger)
    rules = {v.rule_id for v in variants}
    assert {"scale_shift_v1", "company_swap_v1", "citation_swap_v1", "abstention_flip_v1"} <= rules
    candidates = {x.example_id: x for x in cal.natural_candidates(ctx, ledger)[0]}
    for v in variants:
        record = v.record(private=True)
        assert record["synthetic"] is True
        assert record["human_review"] == "pending"
        assert record["transformation_id"] == cal.transformation_id()
        assert record["expected_label_source"].startswith("transformation_rule")
        assert all(s in candidates for s in v.source_example_ids)
    scale = next(v for v in variants if v.rule_id == "scale_shift_v1")
    source = ctx.question(scale.question_id).claims[(scale.claim_position or 1) - 1]
    assert scale.variant_text is not None
    assert "million" in scale.variant_text
    assert "billion" in source.text
    assert scale.source_text_sha256 == source.text_sha256
    company = next(v for v in variants if v.rule_id == "company_swap_v1")
    assert company.variant_text is not None
    assert company.variant_text.startswith("Alphabet Inc.")
    swap = next(v for v in variants if v.rule_id == "citation_swap_v1")
    assert swap.variant_evidence_ids == ("E01",)
    assert swap.variant_text is None
    assert json.dumps(inputs["records"], sort_keys=True) == before
    assert len(ledger.judgments()) == 7  # variants create no human judgment


def test_synthetic_expected_labels_need_human_review_and_stay_separate(
    ctx: an.AnnotationContext,
) -> None:
    ledger = _labelled_ledger(ctx)
    scale = next(v for v in cal.synthetic_variants(ctx, ledger) if v.rule_id == "scale_shift_v1")
    review = {
        "entry_type": "synthetic_review",
        "review_action": "confirmed",
        "variant_id": scale.variant_id,
        "decision": "accept_expected_label",
    }
    unknown = {**review, "variant_id": "0" * 16}
    refused(ctx, ledger, only(sheet(ctx, scale.question_id), unknown), "variant does not exist")
    ledger = commit(ctx, ledger, only(sheet(ctx, scale.question_id), review))
    calibration = cal.build_calibration_set(ctx, ledger)
    reviewed = {v["variant_id"]: v["human_review"] for v in calibration["synthetic"]}
    assert reviewed[scale.variant_id] == "accepted"
    assert all(x["synthetic"] is False for x in calibration["natural"])
    assert all(v["synthetic"] is True for v in calibration["synthetic"])
    natural_ids = {x["example_id"] for x in calibration["natural"]}
    assert not natural_ids & set(reviewed)
    summary = cal.public_summary(calibration)
    assert set(summary["synthetic_review"]) <= {"accepted", "pending"}
    assert sum(summary["natural_by_category"].values()) == len(calibration["natural"])
    assert len(ledger.judgments()) == 7


# --- privacy, failures and held-out protection --------------------------------------------------


def test_public_outputs_hold_no_private_text(
    ctx: an.AnnotationContext, paths: an.AnnotationPaths
) -> None:
    summary = an.write_packets(ctx, paths)
    ws = sheet(ctx)
    entry = supported(ws)
    entry["rationale"] = "SENTINEL-RATIONALE"
    ledger = commit(ctx, an.Ledger(), only(ws, entry))
    outputs = [
        summary,
        an.progress(ctx, ledger),
        plan(ctx, an.Ledger(), only(ws, entry)).public(),
        an.history(ledger, "dev-003"),
        cal.public_summary(cal.build_calibration_set(ctx, _labelled_ledger(ctx))),
    ]
    blob = json.dumps(outputs)
    for text in [*private_texts(ctx), "SENTINEL-RATIONALE"]:
        assert text not in blob
    with pytest.raises(an.AnnotationError, match="private field"):
        an.ensure_public({"nested": [{"text": "x"}]})
    with pytest.raises(an.AnnotationError, match="free text"):
        an.ensure_public({"label": "x" * 200})
    packet_files = sorted(p.name for p in paths.packets.iterdir())
    assert packet_files == ["dev-002.json", "dev-003.json", "dev-004.json", "dev-005.json",
                            "dev-006.json", "index.json"]  # fmt: skip
    index = json.loads((paths.packets / "index.json").read_text(encoding="utf-8"))
    assert all(t not in json.dumps(index) for t in private_texts(ctx))


def test_errors_never_echo_private_input(ctx: an.AnnotationContext) -> None:
    ws = sheet(ctx)
    entry = find(ws, "claim_faithfulness", claim_position=1)
    entry.update(
        review_action="confirmed",
        label="SENTINEL-LABEL",
        rationale="SENTINEL-RATIONALE",
        evidence=[{"kind": "selected_context", "chunk_id": "SENTINEL-CHUNK", "start": 0, "end": 3}],
    )
    message = refused(ctx, an.Ledger(), only(ws, entry), "entry 1")
    malformed = only(ws, {**entry, "unit": "SENTINEL-UNIT", "label": 3})
    message += refused(ctx, an.Ledger(), malformed, "malformed")
    for sentinel in ("SENTINEL-LABEL", "SENTINEL-RATIONALE", "SENTINEL-CHUNK", "SENTINEL-UNIT"):
        assert sentinel not in message


def test_model_output_failures_get_no_packet_units_or_labels(
    ctx: an.AnnotationContext, paths: an.AnnotationPaths
) -> None:
    with pytest.raises(an.AnnotationError, match="ineligible"):
        an.build_packet(ctx, "dev-001")
    an.write_packets(ctx, paths)
    assert not (paths.packets / "dev-001.json").exists()
    ws = sheet(ctx)
    ws.update(question_id="dev-001", generation_record_sha256=ctx.question("dev-001").record_sha256)
    abstention = find(ws, "abstention_appropriateness")
    abstention.update(
        review_action="confirmed", label="correct_abstention", evidence_sufficiency="unknown"
    )
    refused(ctx, an.Ledger(), only(ws, abstention), "ineligible")
    progress = an.progress(ctx, an.Ledger())
    assert progress["ineligible"] == ["dev-001"]
    failed = ctx.question("dev-001")
    assert (failed.claims, failed.execution_outcome) == ((), "model_output_failure")


def test_held_out_split_is_never_opened_and_nothing_is_labelled_automatically(
    ctx: an.AnnotationContext, paths: an.AnnotationPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    for name in ("read_bytes", "read_text", "open"):
        original = getattr(pathlib.Path, name)

        def spy(self: pathlib.Path, *args: Any, _original: Any = original, **kw: Any) -> Any:
            opened.append(self.name)
            if self.name == "questions.test.jsonl":
                raise AssertionError("held-out file opened")
            return _original(self, *args, **kw)

        monkeypatch.setattr(pathlib.Path, name, spy)
    summary = an.write_packets(ctx, paths)
    an.create_worksheet(ctx, paths, "dev-003", "ann-a1", Role.CALIBRATION, manifest=manifest(ctx))
    with pytest.raises(an.AnnotationError, match="already exists"):
        an.create_worksheet(
            ctx, paths, "dev-003", "ann-a1", Role.CALIBRATION, manifest=manifest(ctx)
        )
    result = an.import_worksheet(
        ctx, paths, paths.worksheets / "dev-003.ann-a1.json", manifest=manifest(ctx), now=NOW
    )
    assert (result.accepted, summary["human_judgments_created"]) == ((), 0)
    assert summary["verified_reference_facts_created"] == 0
    assert not paths.ledger.exists()
    calibration = cal.build_calibration_set(ctx, an.load_ledger(paths.ledger))
    assert (calibration["natural"], calibration["synthetic"]) == ([], [])
    assert "questions.test.jsonl" not in opened
    assert an.main(["status", "--run", "test-run"]) == 1  # not a live development run


def test_existing_packets_are_never_silently_overwritten(
    ctx: an.AnnotationContext, paths: an.AnnotationPaths
) -> None:
    an.write_packets(ctx, paths)
    assert an.write_packets(ctx, paths)["packets"] == 5  # identical inputs: unchanged
    target = paths.packets / "dev-003.json"
    packet = json.loads(target.read_text(encoding="utf-8"))
    packet["response"]["status"] = "partially_answered"
    an.write_json(target, packet)
    with pytest.raises(an.AnnotationError, match="inputs changed"):
        an.write_packets(ctx, paths)


def test_context_refuses_records_that_differ_from_the_verified_run(
    inputs: dict[str, Any],
) -> None:
    run = _verify(inputs)
    records = copy.deepcopy(inputs["records"])
    records[2]["status"] = "insufficient_evidence"  # an edit breaks the seal
    with pytest.raises(an.AnnotationError, match="not sealed"):
        an.make_context(run, COLLECTION, SPLIT, records, VOCAB)
    with pytest.raises(an.AnnotationError, match="records do not match"):
        an.make_context(run, COLLECTION, SPLIT, inputs["records"][:-1], VOCAB)


# --- final verification: review states, partitions, coverage, integrity --------------------------


def _judge(ws: dict[str, Any], dimension: str, label: str, **unit: Any) -> dict[str, Any]:
    entry = copy.deepcopy(find(ws, dimension, **unit))
    entry.update(review_action="confirmed", label=label)
    return entry


CLAIM1 = "judgment:dev-003/claim:1#claim_faithfulness"
CLAIM2 = "judgment:dev-003/claim:2#claim_faithfulness"


def _states(ledger: an.Ledger) -> dict[str, an.ReviewState]:
    return {u: s for u, (s, _) in ledger.resolution("judgment").items()}


def test_review_states_never_report_one_reviewer_as_agreement(ctx: an.AnnotationContext) -> None:
    unlabeled = an.review_state_counts(ctx, an.Ledger())
    assert unlabeled["unlabeled"] == len(an.template_judgment_units(ctx))
    assert sum(unlabeled.values()) == unlabeled["unlabeled"]
    a1, c3, d4 = (sheet(ctx, annotator=a) for a in ("ann-a1", "ann-c3", "ann-d4"))
    ledger = commit(ctx, an.Ledger(), only(a1, supported(a1), supported(a1, 2, "E03")))
    assert _states(ledger)[CLAIM1] is an.ReviewState.SINGLE_REVIEWED
    assert ledger.independently_reviewed_units() == []  # no reviewer pair exists
    unsupported = _judge(c3, "claim_faithfulness", "unsupported", claim_position=2)
    ledger = commit(ctx, ledger, only(c3, supported(c3), unsupported))
    assert _states(ledger)[CLAIM1] is an.ReviewState.INDEPENDENTLY_AGREED
    assert _states(ledger)[CLAIM2] is an.ReviewState.DISPUTED
    assert ledger.independently_reviewed_units() == [CLAIM1, CLAIM2]
    own = _judge(a1, "claim_faithfulness", "unsupported", claim_position=2)
    own.update(
        adjudication=True,
        supersedes=ledger.heads()[("ann-a1", CLAIM2)].entry_id,
        revision_reason="correction",
    )
    refused(ctx, ledger, only(a1, own), "must not be one of the unit's independent reviewers")
    refused(ctx, ledger, only(d4, {**supported(d4), "adjudication": True}), "needs a dispute")
    adjudicated = _judge(d4, "claim_faithfulness", "unsupported", claim_position=2)
    adjudicated["adjudication"] = True
    ledger = commit(ctx, ledger, only(d4, adjudicated))
    assert _states(ledger)[CLAIM2] is an.ReviewState.ADJUDICATED
    assert ledger.disagreements() == []
    candidates = {x.unit_key: x for x in cal.natural_candidates(ctx, ledger)[0]}
    assert candidates[CLAIM1].agreement == "independently_agreed"
    assert (candidates[CLAIM2].agreement, candidates[CLAIM2].label) == (
        "adjudicated",
        "unsupported",
    )
    relevance = _judge(a1, "answer_relevance", "directly_answers")
    ledger = commit(ctx, ledger, only(a1, relevance))
    head = ledger.heads()[("ann-a1", "judgment:dev-003/answer#answer_relevance")]
    withdraw = {
        **relevance,
        "review_action": "withdrawn",
        "supersedes": head.entry_id,
        "revision_reason": "entry_error",
    }
    ledger = commit(ctx, ledger, only(a1, withdraw))
    counts = an.review_state_counts(ctx, ledger)
    assert counts["withdrawn"] == 1
    assert (counts["single_reviewed"], counts["independently_agreed"]) == (0, 1)
    assert (counts["adjudicated"], counts["disputed"]) == (1, 0)
    assert ledger.superseded() == 1


def test_no_agreement_statistic_is_computed_without_pairs(ctx: an.AnnotationContext) -> None:
    ws = sheet(ctx)
    ledger = commit(ctx, an.Ledger(), only(ws, supported(ws)))
    calibration = cal.build_calibration_set(ctx, ledger)
    disclosure = calibration["disclosure"]
    assert disclosure["independently_reviewed_units"] == 0
    assert disclosure["selected_by_review_state"] == {"single_reviewed": 1}
    assert "two independent reviews" in disclosure["agreement_statistics"]
    assert "kappa" not in json.dumps(calibration).lower()
    assert an.progress(ctx, ledger)["agreement_statistics"] == an.AGREEMENT_NOTE


def test_calibration_and_evaluation_never_share_a_question(ctx: an.AnnotationContext) -> None:
    calibration = sheet(ctx)
    ledger = commit(ctx, an.Ledger(), only(calibration, supported(calibration)))
    evaluation = sheet(ctx, annotator="ann-c3", role=Role.EVALUATION)
    other_claim = supported(evaluation, 2, "E03")
    refused(ctx, ledger, only(evaluation, other_claim), "calibration partition")
    head = ledger.heads()[("ann-a1", CLAIM1)]
    withdraw = {
        **supported(sheet(ctx)),
        "review_action": "withdrawn",
        "supersedes": head.entry_id,
        "revision_reason": "entry_error",
    }
    ledger = commit(ctx, ledger, only(calibration, withdraw))
    refused(ctx, ledger, only(evaluation, other_claim), "calibration partition")  # history
    assert ledger.partitions() == {"dev-003": Role.CALIBRATION}
    ledger = _labelled_ledger(ctx)
    variants = cal.synthetic_variants(ctx, ledger)
    for variant in variants:
        record = variant.record(private=False)
        assert record["partition"] == "calibration"
        assert record["source_question_id"] == variant.question_id
        assert ledger.partitions()[variant.question_id] is Role.CALIBRATION
    review = {
        "entry_type": "synthetic_review",
        "review_action": "confirmed",
        "variant_id": variants[0].variant_id,
        "decision": "accept_expected_label",
    }
    qid = variants[0].question_id
    refused(ctx, ledger, only(sheet(ctx, qid, role=Role.EVALUATION), review), "partition")


def test_every_claim_citation_association_is_its_own_bound_unit(
    ctx: an.AnnotationContext,
) -> None:
    units = [u for u in an.template_judgment_units(ctx) if u.endswith("#citation_support")]
    assert len(units) == len(set(units)) == an.run_counts(ctx)["claim_evidence_associations"]
    assert an.progress(ctx, an.Ledger())["pending_citation_support_units"] == len(units)
    ws = sheet(ctx)
    entry = _judge(ws, "citation_support", "does_not_support", claim_position=1)
    wrong_claim = {**entry, "unit": {**entry["unit"], "claim_position": 2}}
    refused(ctx, an.Ledger(), only(ws, wrong_claim), "citation the claim does not carry")
    wrong_evidence = {**entry, "unit": {**entry["unit"], "evidence_id": "E03"}}
    refused(ctx, an.Ledger(), only(ws, wrong_evidence), "differs from the cited record")
    ledger = commit(ctx, an.Ledger(), only(ws, entry))
    assert ledger.entries[0].unit_key == (
        "judgment:dev-003/claim:1/citation:1:E02#citation_support"
    )


def test_inapplicable_or_unknown_fact_attributes_are_never_invented(
    ctx: an.AnnotationContext,
) -> None:
    percent = {"magnitude": "78", "unit": "percent", "currency": "USD"}
    refused(ctx, an.Ledger(), with_fact(sheet(ctx), value=percent), "monetary values only")
    scaled = {"magnitude": "78", "unit": "percent", "scale": "billion"}
    refused(ctx, an.Ledger(), with_fact(sheet(ctx), value=scaled), "monetary values only")
    ledger = commit(ctx, an.Ledger(), with_fact(sheet(ctx), basis="unverified"))
    assert ledger.verified_facts()["dev-003-f1"].basis == "unverified"
    ledger = commit(ctx, ledger, only(sheet(ctx), association(ctx)))
    q = ctx.question("dev-003")
    claim = q.claims[0]
    results = fc.check_value(
        question_id="dev-003",
        generation_record_sha256=q.record_sha256,
        claim_position=1,
        claim_text=claim.text,
        occurrences=fc.extract_occurrences(claim.text, VOCAB),
        value_index=1,
        association=ledger.associations()[0],
        facts=ledger.verified_facts(),
        vocabulary=VOCAB,
    )
    basis = next(r for r in results if r.aspect == "financial_basis")
    assert basis.outcome is fc.Outcome.UNVERIFIED
    assert basis.reasons == (fc.Reason.BASIS_NOT_VERIFIED,)


def test_disputed_facts_need_adjudication_before_use(ctx: an.AnnotationContext) -> None:
    ledger = commit(ctx, an.Ledger(), with_fact(sheet(ctx)))
    other = {"magnitude": "35.0", "unit": "money", "scale": "billion", "currency": "USD"}
    ledger = commit(ctx, ledger, with_fact(sheet(ctx, annotator="ann-c3"), value=other))
    assert ledger.verified_facts() == {}
    refused(ctx, ledger, only(sheet(ctx), association(ctx)), "unknown, unresolved")
    adjudicator = with_fact(sheet(ctx, annotator="ann-d4"), adjudication=True)
    ledger = commit(ctx, ledger, adjudicator)
    assert ledger.verified_facts()["dev-003-f1"].verified_by == "ann-d4"


def test_failed_or_interrupted_imports_leave_the_ledger_unchanged(
    ctx: an.AnnotationContext, paths: an.AnnotationPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = sheet(ctx)
    path = paths.worksheets / "dev-003.ann-a1.json"
    an.write_json(path, only(ws, supported(ws)))
    an.import_worksheet(ctx, paths, path, manifest=manifest(ctx), now=NOW)
    before = paths.ledger.read_bytes()
    bad = {**_judge(ws, "answer_relevance", "directly_answers"), "label": "wrong"}
    an.write_json(path, only(ws, supported(ws, 2, "E03"), bad))  # one valid, one invalid
    with pytest.raises(an.AnnotationError, match="entry 2"):
        an.import_worksheet(ctx, paths, path, manifest=manifest(ctx), now=NOW)
    assert paths.ledger.read_bytes() == before  # no partial write
    an.write_json(path, only(ws, supported(ws, 2, "E03")))

    def interrupted(self: pathlib.Path, target: Any) -> Any:
        raise OSError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "replace", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            an.import_worksheet(ctx, paths, path, manifest=manifest(ctx), now=NOW)
    assert paths.ledger.read_bytes() == before
    assert not paths.ledger.with_name("ledger.jsonl.tmp").exists()
    original = an.plan_import

    def concurrent(*args: Any, **kwargs: Any) -> an.ImportResult:
        result = original(*args, **kwargs)
        paths.ledger.write_bytes(before + b"\n")
        return result

    monkeypatch.setattr(an, "plan_import", concurrent)
    with pytest.raises(an.AnnotationError, match="changed during import"):
        an.import_worksheet(ctx, paths, path, manifest=manifest(ctx), now=NOW)


def test_worksheets_outside_the_private_directory_are_refused(
    ctx: an.AnnotationContext, paths: an.AnnotationPaths, tmp_path: pathlib.Path
) -> None:
    outside = tmp_path / "elsewhere.json"
    an.write_json(outside, sheet(ctx))
    with pytest.raises(an.AnnotationError, match="private annotation directory"):
        an.import_worksheet(ctx, paths, outside, manifest=manifest(ctx), now=NOW)


def test_reprs_hide_private_text(ctx: an.AnnotationContext) -> None:
    ws = sheet(ctx)
    entry = supported(ws)
    entry["rationale"] = "SENTINEL-RATIONALE"
    ledger = commit(ctx, an.Ledger(), only(ws, entry))
    parsed = an.Worksheet.model_validate(only(ws, entry))
    blob = repr(ctx) + repr(ledger) + repr(parsed)
    for text in [*private_texts(ctx), "SENTINEL-RATIONALE"]:
        assert text not in blob


def test_all_eight_transformations_are_versioned_lineaged_and_reviewable(
    ctx: an.AnnotationContext, inputs: dict[str, Any]
) -> None:
    assert set(cal.TRANSFORMATIONS) == {
        "scale_shift_v1", "company_swap_v1", "period_shift_v1", "metric_swap_v1",
        "percent_shift_v1", "citation_swap_v1", "abstention_flip_v1", "compound_join_v1",
    }  # fmt: skip
    # text rules on fixed strings (the synthetic claims state no quarter, metric or percent)
    assert cal._period_shift("FY2025 Q4 revenue") == "FY2025 Q1 revenue"
    assert cal._metric_swap("Revenue rose") == "Net income rose"
    text = "Margin rose 1.5% in FY2025 Q4."
    shifted = cal._percent_shift(text, fc.extract_occurrences(text, VOCAB))
    assert shifted == "Margin rose 1.6% in FY2025 Q4."
    before = json.dumps(inputs["records"], sort_keys=True)
    ledger = _labelled_ledger(ctx)
    a1 = sheet(ctx)
    head = ledger.heads()[("ann-a1", CLAIM2)]
    unsupported = _judge(a1, "claim_faithfulness", "unsupported", claim_position=2)
    unsupported.update(supersedes=head.entry_id, revision_reason="correction")
    ledger = commit(ctx, ledger, only(a1, unsupported))
    variants = cal.synthetic_variants(ctx, ledger)
    assert variants == cal.synthetic_variants(ctx, ledger)  # stable identities
    assert {
        "scale_shift_v1", "company_swap_v1", "citation_swap_v1", "abstention_flip_v1",
        "compound_join_v1",
    } <= {v.rule_id for v in variants}  # fmt: skip
    joined = next(v for v in variants if v.rule_id == "compound_join_v1")
    assert len(joined.source_example_ids) == 2
    natural_before = cal.natural_candidates(ctx, ledger)
    reject = {
        "entry_type": "synthetic_review",
        "review_action": "confirmed",
        "variant_id": joined.variant_id,
        "decision": "reject_expected_label",
    }
    ledger = commit(ctx, ledger, only(sheet(ctx), reject))
    reviewed = {v.variant_id: v.review for v in cal.synthetic_variants(ctx, ledger)}
    assert reviewed[joined.variant_id] == "rejected"
    assert set(reviewed.values()) <= {"pending", "rejected"}  # none auto-accepted
    assert cal.natural_candidates(ctx, ledger) == natural_before  # no contamination
    assert json.dumps(inputs["records"], sort_keys=True) == before
