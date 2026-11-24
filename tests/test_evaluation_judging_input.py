"""Verified judging input and judgment binding (Commit 7.1), offline.

A synthetic live run is produced by the real harness with a mocked provider
(no model, key or network) over the synthetic NVIDIA/Alphabet collection, then
verified without any further retrieval or provider call.
"""

from __future__ import annotations

import json
import pathlib
import socket
from types import SimpleNamespace
from typing import Any

import pytest

from founderlens.evaluation import completion as cp
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation import judging_input as ji
from founderlens.evaluation.benchmark import DevSplit
from founderlens.evaluation.judgments import (
    ClaimDecomposition,
    EvidenceKind,
    Judgment,
    rubric_id,
)
from founderlens.retrieval.policy import PolicyRetriever
from tests.test_evaluation_failure_diagnostics import INVALID, SECRET, ContractProvider
from tests.test_evaluation_generation_eval import IDS, PIPELINE, SPLIT, ScriptProvider
from tests.test_guardrails_response_checks import COLLECTION

CLEAN = {"commit": "c0ffee", "working_tree_clean": True}


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    # The report records Git state itself; pin it to the state given to the harness.
    monkeypatch.setattr(ge, "_git_state", lambda root=None: dict(CLEAN))


def _run(
    tmp_path: pathlib.Path, provider: ScriptProvider | None = None, mode: ge.Mode = "live"
) -> dict[str, Any]:
    outputs = ge.execute(
        SPLIT,
        PIPELINE,
        tmp_path,
        mode=mode,
        provider_factory=lambda: provider or ContractProvider(INVALID["unknown_id"][0]),
        max_requests=24,
        expected_identities=IDS,
        git=CLEAN,
    )
    entries, _ = cp.read_run_log(tmp_path / ge.REPORT_SUBDIR / cp.RUN_LOG_FILENAME)
    return {
        "run_id": outputs.report["run_id"],
        "records": ge.load_records(outputs.records_path),
        "report": json.loads(outputs.json_path.read_text(encoding="utf-8")),
        "run_log": entries,
    }


def _verify(inputs: dict[str, Any], **over: Any) -> ji.VerifiedRun:
    args = {**inputs, "split": SPLIT, "collection": COLLECTION, "expected_identities": IDS, **over}
    return ji.verify_generation_run(**args)


@pytest.fixture
def run_inputs(tmp_path: pathlib.Path) -> dict[str, Any]:
    return _run(tmp_path)


# --- accepted input -------------------------------------------------------------------------


def test_execution_complete_run_verifies_with_failures_ineligible(
    run_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_retrieval(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("verification must not retrieve")

    monkeypatch.setattr(PolicyRetriever, "search", no_retrieval)
    run = _verify(run_inputs)
    assert len(run.questions) == 6
    assert run.counts() == {"local_abstention": 1, "model_output_failure": 1, "valid_answer": 4}
    failed = run.question("dev-001")
    assert failed is not None
    assert (failed.eligible, failed.ineligibility_reason) == (False, "model_output_failure")
    assert failed.failure_codes == ("unknown_evidence_id",)
    assert (failed.claims, failed.citations) == ((), ())  # nothing invented
    assert failed.execution_outcome != "local_abstention"  # never turned into an abstention
    local = run.question("dev-005")
    assert local is not None
    assert (local.eligible, local.execution_outcome) == (True, "local_abstention")
    valid = run.question("dev-003")
    assert valid is not None
    assert valid.eligible
    assert [c.position for c in valid.claims] == [1, 2]
    assert valid.selected_evidence
    assert run.generation_protocol == "1.3"  # the input keeps its own protocol
    assert run.completion_state == "execution_complete"


def test_public_summary_holds_no_private_content(run_inputs: dict[str, Any]) -> None:
    run = _verify(run_inputs)
    summary = json.dumps(run.public_summary())
    for record in run_inputs["records"]:
        for claim in (record.get("answer") or {}).get("claims", []):
            assert claim["text"] not in summary
    for leak in ("SENTINEL-QUESTION", SECRET, "SENTINEL-LIMITATION"):
        assert leak not in summary
    assert run.public_summary()["rubric_id"] == rubric_id()
    assert "text=" not in repr(run.questions[2].claims)  # private text excluded from repr


# --- refusals -------------------------------------------------------------------------------


def _resealed(records: list[dict[str, Any]], qid: str, **changes: Any) -> list[dict[str, Any]]:
    return [ge.seal({**r, **changes}) if r["question_id"] == qid else r for r in records]


def test_tampered_or_incompatible_records_are_refused(run_inputs: dict[str, Any]) -> None:
    records = run_inputs["records"]
    cases = {
        "record digest": [{**records[0], "status": "answered"}, *records[1:]],
        "frozen identities": _resealed(
            records, "dev-002", identities={**IDS, "prompt_id": "0" * 12}
        ),
        "run id": _resealed(records, "dev-002", run_id="live-other"),
        "record schema 3": _resealed(records, "dev-002", record_schema_version=2),
        "generation protocol 1.3": _resealed(records, "dev-002", protocol_version="1.2"),
        "live origin": _resealed(records, "dev-002", synthetic=True),
        "every scheduled question": records[:-1],
    }
    for expected, tampered in cases.items():
        with pytest.raises(ji.JudgingInputError, match=expected):
            _verify(run_inputs, records=tampered)


def test_broken_answer_and_citation_associations_are_refused(run_inputs: dict[str, Any]) -> None:
    records = run_inputs["records"]
    valid = next(r for r in records if r["question_id"] == "dev-003")
    answer = json.loads(json.dumps(valid["answer"]))
    answer["claims"][0]["evidence_ids"] = ["E09"]
    with pytest.raises(ji.JudgingInputError, match="response invariants"):
        _verify(run_inputs, records=_resealed(records, "dev-003", answer=answer))
    citations = [{**c, "page_number": 9} for c in valid["citations"]]
    with pytest.raises(ji.JudgingInputError, match="does not match selected evidence"):
        _verify(run_inputs, records=_resealed(records, "dev-003", citations=citations))
    evidence = [{**e, "chunk_id": "missing"} for e in valid["selected_evidence"]]
    with pytest.raises(ji.JudgingInputError, match="not in the verified collection"):
        _verify(run_inputs, records=_resealed(records, "dev-003", selected_evidence=evidence))
    failed = next(r for r in records if r["question_id"] == "dev-001")
    with pytest.raises(ji.JudgingInputError, match="failed record carries an answer"):
        _verify(run_inputs, records=_resealed(records, "dev-001", answer=valid["answer"]))
    assert failed["answer"] is None


def test_wrong_corpus_report_or_log_is_refused(
    run_inputs: dict[str, Any], tmp_path: pathlib.Path
) -> None:
    other = SimpleNamespace(chunks=COLLECTION.chunks, artifact_sha256="0" * 64)
    with pytest.raises(ji.JudgingInputError, match="frozen V2 artifact"):
        _verify(run_inputs, collection=other)
    report = run_inputs["report"]
    with pytest.raises(ji.JudgingInputError, match="dataset hash"):
        _verify(run_inputs, split=DevSplit(SPLIT.questions, "x", "c" * 64, SPLIT.split_sha256))
    with pytest.raises(ji.JudgingInputError, match="reportable"):
        _verify(
            run_inputs,
            report={**report, "reportability": {"reportable_as_hosted_model_diagnostic": False}},
        )
    with pytest.raises(ji.JudgingInputError, match="clean tree"):
        _verify(
            run_inputs,
            report={
                **report,
                "operational": {"git": {"commit": "c0ffee", "working_tree_clean": False}},
            },
        )
    with pytest.raises(ji.JudgingInputError, match="preregistered and one finished"):
        _verify(run_inputs, run_log=[])
    with pytest.raises(ji.JudgingInputError, match="frozen identities"):
        _verify(run_inputs, expected_identities={**IDS, "model": "other"})


def test_synthetic_or_incomplete_runs_are_refused(tmp_path: pathlib.Path) -> None:
    dry = _run(tmp_path / "dry", mode="dry_run")
    dry["run_log"] = [
        {"event": "preregistered", "run_id": dry["run_id"], "git": CLEAN},
        {"event": "finished", "run_id": dry["run_id"]},
    ]
    with pytest.raises(ji.JudgingInputError, match="live origin"):
        _verify(dry)
    from founderlens.generation import errors as gerr

    fail = {SPLIT.questions[1].question: gerr.ProviderTimeoutError("synthetic")}
    infra = _run(tmp_path / "infra", ScriptProvider(fail=fail))
    with pytest.raises(ji.JudgingInputError, match="accepted completion state"):
        _verify(infra)


# --- evidence references and judgment binding -----------------------------------------------


def _evidence_ref(run: ji.VerifiedRun, qid: str, n: int = 0, start: int = 0, end: int = 7) -> Any:
    question = run.question(qid)
    assert question is not None
    ev = question.selected_evidence[n]
    return ji.make_reference(
        COLLECTION, ev.chunk_id, start, end, EvidenceKind.SELECTED_CONTEXT, ev.evidence_id
    )


def _faithfulness(run: ji.VerifiedRun, qid: str, refs: list[Any], **over: Any) -> Judgment:
    question = run.question(qid)
    assert question is not None
    payload: dict[str, Any] = {
        "question_id": qid,
        "generation_run_id": run.run_id,
        "generation_record_sha256": question.record_sha256,
        "dimension": "claim_faithfulness",
        "unit": {"kind": "claim", "claim_position": 1},
        "label": "supported",
        "evidence": [r.model_dump() for r in refs],
        "provenance": {
            "source": "human",
            "role": "calibration",
            "annotator_id": "synthetic",
            "rubric_id": rubric_id(),
        },
    }
    payload.update(over)
    return Judgment.model_validate(payload)


def test_a_well_formed_judgment_binds_to_the_verified_run(run_inputs: dict[str, Any]) -> None:
    run = _verify(run_inputs)
    judgment = _faithfulness(run, "dev-003", [_evidence_ref(run, "dev-003")])
    assert ji.check_judgment(judgment, run, COLLECTION) == []


def test_judgments_on_failures_or_unknown_units_are_refused(run_inputs: dict[str, Any]) -> None:
    run = _verify(run_inputs)
    ref = _evidence_ref(run, "dev-003")
    failed_q = run.question("dev-001")
    assert failed_q is not None
    on_failure = _faithfulness(
        run,
        "dev-003",
        [ref],
        question_id="dev-001",
        generation_record_sha256=failed_q.record_sha256,
    )
    assert any("ineligible" in p for p in ji.check_judgment(on_failure, run, COLLECTION))
    wrong_digest = _faithfulness(run, "dev-003", [ref], generation_record_sha256="0" * 64)
    assert "judgment names a different generation record" in ji.check_judgment(
        wrong_digest, run, COLLECTION
    )
    no_claim = _faithfulness(run, "dev-003", [ref], unit={"kind": "claim", "claim_position": 9})
    assert "judgment names a claim that does not exist" in ji.check_judgment(
        no_claim, run, COLLECTION
    )
    question = run.question("dev-003")
    assert question is not None
    citation = question.citations[0]
    wrong_citation = _faithfulness(
        run,
        "dev-003",
        [],
        dimension="citation_support",
        label="does_not_support",
        unit={
            "kind": "claim_citation",
            "claim_position": 1,
            "citation_number": 9,
            "evidence_id": citation.evidence_id,
        },
    )
    assert any("does not carry" in p for p in ji.check_judgment(wrong_citation, run, COLLECTION))
    foreign_fact = _faithfulness(
        run,
        "dev-003",
        [],
        dimension="factual_correctness",
        label="missing",
        unit={"kind": "required_fact", "fact_id": "dev-002-f1"},
    )
    assert any("another question" in p for p in ji.check_judgment(foreign_fact, run, COLLECTION))
    local_claim = _faithfulness(run, "dev-005", [], label="unsupported")
    problems = ji.check_judgment(local_claim, run, COLLECTION)
    assert any("answer-level dimensions only" in p for p in problems)


def test_citation_support_must_cite_the_cited_chunk(run_inputs: dict[str, Any]) -> None:
    run = _verify(run_inputs)
    question = run.question("dev-003")
    assert question is not None
    citation = question.citations[0]
    claim = next(c for c in question.claims if citation.number in c.citation_numbers)
    other = next(e for e in question.selected_evidence if e.chunk_id != citation.chunk_id)
    good = ji.make_reference(
        COLLECTION, citation.chunk_id, 0, 5, EvidenceKind.CITED, citation.evidence_id
    )
    unit = {
        "kind": "claim_citation",
        "claim_position": claim.position,
        "citation_number": citation.number,
        "evidence_id": citation.evidence_id,
    }
    ok = _faithfulness(
        run, "dev-003", [good], dimension="citation_support", label="supports", unit=unit
    )
    assert ji.check_judgment(ok, run, COLLECTION) == []
    wrong = ji.make_reference(
        COLLECTION, other.chunk_id, 0, 5, EvidenceKind.CITED, citation.evidence_id
    )
    bad = _faithfulness(
        run, "dev-003", [wrong], dimension="citation_support", label="supports", unit=unit
    )
    problems = ji.check_judgment(bad, run, COLLECTION)
    assert "cited reference does not name a chunk this answer cites" in problems
    assert "citation_support evidence must be spans of the cited chunk" in problems


def test_reference_offsets_hashes_and_artifacts_are_checked(run_inputs: dict[str, Any]) -> None:
    run = _verify(run_inputs)
    question = run.question("dev-003")
    assert question is not None
    ref = _evidence_ref(run, "dev-003")
    chunk = next(c for c in COLLECTION.chunks if c.chunk_id == ref.chunk_id)
    cases = {
        "reference offsets outside the chunk text": ref.model_copy(
            update={"end": len(chunk.text) + 1}
        ),
        "reference span hash does not match the chunk text": ref.model_copy(update={"end": 8}),
        "reference names a different source artifact": ref.model_copy(
            update={"collection_artifact_sha256": "0" * 64}
        ),
        "reference chunk not in the verified collection": ref.model_copy(update={"chunk_id": "x"}),
        "reference document or page differs from its chunk": ref.model_copy(
            update={"page_number": 99}
        ),
    }
    for expected, bad in cases.items():
        assert expected in ji.reference_problems(bad, question, COLLECTION), expected


def test_offsets_are_code_points_on_base_characters() -> None:
    text = "Cafe" + chr(0x301) + " revenue rose"  # "e" + combining acute accent
    chunk = SimpleNamespace(chunk_id="c1", document_id="d", page_number=1, text=text)
    collection = SimpleNamespace(chunks=[chunk], artifact_sha256="f" * 64)
    question = SimpleNamespace(selected_evidence=())
    source = EvidenceKind.REFERENCE_SOURCE
    whole = ji.make_reference(collection, "c1", 0, 5, source)  # type: ignore[arg-type]  # fake
    assert whole.span_sha256 == ji.span_sha256(text[0:5])  # code points, no re-normalization
    assert ji.reference_problems(whole, question, collection) == []  # type: ignore[arg-type]  # fakes
    split = ji.make_reference(collection, "c1", 0, 4, source)  # type: ignore[arg-type]  # fake
    problems = ji.reference_problems(split, question, collection)  # type: ignore[arg-type]  # fakes
    assert "reference offsets split a combining character sequence" in problems


def test_no_provider_or_generation_call_happens_during_verification(
    run_inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no generation, provider or retrieval call is allowed")

    monkeypatch.setattr(ge.Pipeline, "service", forbidden)
    monkeypatch.setattr(ge.BudgetedProvider, "generate_structured", forbidden)
    monkeypatch.setattr(PolicyRetriever, "search", forbidden)
    run = _verify(run_inputs)
    judgment = _faithfulness(run, "dev-003", [_evidence_ref(run, "dev-003")])
    assert ji.check_judgment(judgment, run, COLLECTION) == []


# --- final verification: count reconciliation, evidence kinds, subclaim binding -------------


def test_claim_citation_counts_reconcile_through_shared_references(
    run_inputs: dict[str, Any],
) -> None:
    run = _verify(run_inputs)
    for question in run.questions:
        if not question.eligible:
            continue
        associations = [e for c in question.claims for e in c.evidence_ids]
        assert all(c.evidence_ids for c in question.claims)  # every claim cites evidence
        assert {c.evidence_id for c in question.citations} == set(associations)
        assert len(question.citations) == len(set(associations))  # one per distinct record
        selected = {e.evidence_id for e in question.selected_evidence}
        assert set(associations) <= selected


def test_evidence_kinds_bind_to_context_citations_or_reference_sources(
    run_inputs: dict[str, Any],
) -> None:
    run = _verify(run_inputs)
    question = run.question("dev-003")
    assert question is not None
    cited_chunks = {c.chunk_id for c in question.citations}
    uncited = next(e for e in question.selected_evidence if e.chunk_id not in cited_chunks)
    as_cited = ji.make_reference(
        COLLECTION, uncited.chunk_id, 0, 5, EvidenceKind.CITED, uncited.evidence_id
    )
    assert "cited reference does not name a chunk this answer cites" in ji.reference_problems(
        as_cited, question, COLLECTION
    )
    as_context = ji.make_reference(
        COLLECTION, uncited.chunk_id, 0, 5, EvidenceKind.SELECTED_CONTEXT, uncited.evidence_id
    )
    assert ji.reference_problems(as_context, question, COLLECTION) == []
    # Any frozen chunk may serve as human-verified reference text; it is labelled as such.
    source = ji.make_reference(
        COLLECTION, COLLECTION.chunks[0].chunk_id, 0, 5, EvidenceKind.REFERENCE_SOURCE
    )
    assert ji.reference_problems(source, question, COLLECTION) == []
    assert source.evidence_id is None  # a reference span never claims retrieval or citation
    factual = _faithfulness(
        run,
        "dev-003",
        [source],
        dimension="factual_correctness",
        label="correct",
        reference_basis="source_evidence",
        unit={"kind": "required_fact", "fact_id": "dev-003-f1"},
    )
    assert ji.check_judgment(factual, run, COLLECTION) == []
    with pytest.raises(Exception, match="evidence kind"):
        _faithfulness(run, "dev-003", [source])  # faithfulness cannot use a reference span


def test_subclaim_judgments_bind_only_through_their_decomposition(
    run_inputs: dict[str, Any],
) -> None:
    run = _verify(run_inputs)
    question = run.question("dev-003")
    assert question is not None
    claim = question.claims[0]
    cut = max(1, len(claim.text) // 2)
    while cut < len(claim.text) and claim.text[cut] != " ":
        cut += 1
    decomposition = ClaimDecomposition.model_validate(
        {
            "question_id": "dev-003",
            "generation_record_sha256": question.record_sha256,
            "claim_position": claim.position,
            "claim_text_sha256": claim.text_sha256,
            "method": "human",
            "subclaims": [
                {"index": 1, "start": 0, "end": cut},
                {"index": 2, "start": cut, "end": len(claim.text)},
            ],
        }
    )
    unit = {
        "kind": "subclaim",
        "claim_position": claim.position,
        "subclaim": {"decomposition_id": decomposition.decomposition_id, "index": 2},
    }
    judged = _faithfulness(run, "dev-003", [], label="unsupported", unit=unit)
    registry = {decomposition.decomposition_id: decomposition}
    assert ji.check_judgment(judged, run, COLLECTION, registry) == []
    assert "subclaim judgment needs its registered decomposition" in ji.check_judgment(
        judged, run, COLLECTION
    )
    tampered = decomposition.model_copy(update={"claim_text_sha256": "0" * 64})
    assert tampered.decomposition_id != decomposition.decomposition_id  # content-addressed
    assert "subclaim judgment needs its registered decomposition" in ji.check_judgment(
        judged, run, COLLECTION, {decomposition.decomposition_id: tampered}
    )
    wrong_unit = {**unit, "subclaim": {"decomposition_id": tampered.decomposition_id, "index": 2}}
    wrong = _faithfulness(run, "dev-003", [], label="unsupported", unit=wrong_unit)
    problems = ji.check_judgment(wrong, run, COLLECTION, {tampered.decomposition_id: tampered})
    assert "decomposition does not pin the original claim text" in problems
    beyond = {**unit, "subclaim": {"decomposition_id": decomposition.decomposition_id, "index": 3}}
    out_of_range = _faithfulness(run, "dev-003", [], label="unsupported", unit=beyond)
    assert "subclaim index outside the decomposition" in ji.check_judgment(
        out_of_range, run, COLLECTION, registry
    )
