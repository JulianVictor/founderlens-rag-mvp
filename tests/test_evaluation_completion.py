"""Protocol v1.3 completion states, failure taxonomy, denominators and run log, offline.

Synthetic questions, the synthetic NVIDIA/Alphabet pipeline and mocked providers
only. No model, key, network or real report is used; files go under ``tmp_path``.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import replace
from typing import Any

import pytest

from founderlens.evaluation import completion as cp
from founderlens.evaluation import generation_eval as ge
from founderlens.generation import errors as gerr
from founderlens.generation.schema import TokenUsage
from founderlens.guardrails.response_checks import GuardrailError
from founderlens.retrieval.policy import PolicyRetriever
from tests.test_evaluation_failure_diagnostics import INVALID, SECRET, ContractProvider
from tests.test_evaluation_generation_eval import IDS, PIPELINE, QUESTIONS, SPLIT, ScriptProvider

FIRST = QUESTIONS[0].question  # dev-001, answerable
CLEAN = {"commit": "c0ffee", "working_tree_clean": True}


def _execute(
    tmp_path: pathlib.Path,
    provider: ScriptProvider,
    *,
    mode: ge.Mode = "live",
    max_requests: int = 24,
    pipeline: ge.Pipeline = PIPELINE,
    replay: list[dict[str, Any]] | None = None,
) -> ge.RunOutputs:
    return ge.execute(
        SPLIT,
        pipeline,
        tmp_path,
        mode=mode,
        provider_factory=lambda: provider,
        max_requests=max_requests,
        expected_identities=IDS,
        replay_records=replay,
        git=CLEAN,
    )


def _records(outputs: ge.RunOutputs) -> dict[str, dict[str, Any]]:
    return {r["question_id"]: r for r in ge.load_records(outputs.records_path)}


def _state(outputs: ge.RunOutputs) -> tuple[str | None, bool]:
    report = outputs.report
    return report["completion_state"], report["reportability"][
        "reportable_as_hosted_model_diagnostic"
    ]


def _contract_failure() -> ContractProvider:
    return ContractProvider(INVALID["unknown_id"][0])


# --- taxonomy -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "category", "expected"),
    [
        ("answer_validation", "response_contract", cp.FailureClass.MODEL_OUTPUT),
        ("generation", "refusal", cp.FailureClass.MODEL_OUTPUT),
        ("generation", "incomplete", cp.FailureClass.MODEL_OUTPUT),
        ("generation", "structured_output_invalid", cp.FailureClass.MODEL_OUTPUT),
        ("generation", "timeout", cp.FailureClass.INFRASTRUCTURE),
        ("generation", "rate_limited", cp.FailureClass.INFRASTRUCTURE),
        ("generation", "authentication", cp.FailureClass.INFRASTRUCTURE),
        ("generation", "configuration", cp.FailureClass.INFRASTRUCTURE),
        # the same category at another stage is an application inconsistency
        ("citation_resolution", "response_contract", cp.FailureClass.INTEGRITY),
        ("citation_resolution", "citation_resolution", cp.FailureClass.INTEGRITY),
        ("generation", "response_contract", cp.FailureClass.INTEGRITY),
        ("answer_validation", "refusal", cp.FailureClass.INTEGRITY),
        ("replay_verification", "replay_mismatch", cp.FailureClass.INTEGRITY),
        ("generation", "unexpected_response", cp.FailureClass.INTEGRITY),
        ("generation", "invalid_question", cp.FailureClass.INTEGRITY),
        ("generation", "something_new", cp.FailureClass.INTEGRITY),
        (None, None, cp.FailureClass.INTEGRITY),
    ],
)
def test_failures_are_classified_by_stage_and_category(
    stage: str | None, category: str | None, expected: cp.FailureClass
) -> None:
    assert cp.classify_failure(stage, category) is expected


# --- completion states ----------------------------------------------------------------------


def test_fully_valid_run_is_answer_complete(tmp_path: pathlib.Path) -> None:
    outputs = _execute(tmp_path, ScriptProvider())
    assert _state(outputs) == ("answer_complete", True)
    assert outputs.report["protocol_version"] == "1.3"
    assert outputs.report["run_complete"] is True
    records = _records(outputs)
    assert records["dev-005"]["execution_outcome"] == "local_abstention"
    assert records["dev-005"]["provider_attempted"] is False  # no provider call
    assert {r["record_schema_version"] for r in records.values()} == {3}


@pytest.mark.parametrize(
    "error",
    [
        gerr.ModelRefusalError("synthetic", refusal="SENTINEL-REFUSAL"),
        gerr.StructuredOutputError("synthetic"),
        gerr.IncompleteGenerationError("synthetic", reason="max_output_tokens"),
    ],
)
def test_model_output_failures_keep_the_run_execution_complete(
    tmp_path: pathlib.Path, error: gerr.GenerationError
) -> None:
    outputs = _execute(tmp_path, ScriptProvider(fail={FIRST: error}))
    assert _state(outputs) == ("execution_complete", True)
    record = _records(outputs)["dev-001"]
    assert record["execution_outcome"] == "model_output_failure"
    assert record["failure"]["classification"] == "model_output"
    assert outputs.report["run_complete"] is False  # not every answer is valid
    assert "SENTINEL-REFUSAL" not in outputs.records_path.read_text(encoding="utf-8")


def test_response_contract_failure_is_execution_complete(tmp_path: pathlib.Path) -> None:
    outputs = _execute(tmp_path, _contract_failure())
    assert _state(outputs) == ("execution_complete", True)
    assert "model-output failures counted" in outputs.report["reportability"]["reason"]
    summary = outputs.markdown_path.read_text(encoding="utf-8")
    assert "Completion state: `execution_complete`" in summary
    assert "INCOMPLETE RUN" not in summary
    assert "model_output_failure 1" in summary
    reliability = outputs.report["overall"]["reliability"]
    assert reliability["failure_counts"] == {"model_output_failure": {"response_contract": 1}}


def test_internal_citation_errors_are_integrity_failures(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        def resolve(self, run: Any) -> Any:
            from founderlens.generation.answer import ResponseContractError

            raise ResponseContractError(["synthetic internal revalidation"])  # same category

    pipeline = replace(PIPELINE, resolver=Broken())  # type: ignore[arg-type]  # fake resolver
    outputs = _execute(tmp_path, ScriptProvider(), pipeline=pipeline)
    record = _records(outputs)["dev-001"]
    assert (record["error_category"], record["failure"]["stage"]) == (
        "response_contract",
        "citation_resolution",
    )
    assert record["execution_outcome"] == "integrity_failure"
    assert _state(outputs) == ("incomplete", False)


def test_guardrail_failures_make_the_run_incomplete(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise GuardrailError("synthetic")

    monkeypatch.setattr(ge, "check_response", broken)
    outputs = _execute(tmp_path, ScriptProvider())
    assert _state(outputs) == ("incomplete", False)
    assert any("guardrail" in p for p in outputs.report["completion_problems"])


@pytest.mark.parametrize(
    "error",
    [
        gerr.ProviderTimeoutError("synthetic"),
        gerr.RateLimitError("synthetic"),
        gerr.UnexpectedResponseError("synthetic"),
    ],
)
def test_infrastructure_and_unknown_failures_make_the_run_incomplete(
    tmp_path: pathlib.Path, error: gerr.GenerationError
) -> None:
    outputs = _execute(tmp_path, ScriptProvider(fail={FIRST: error}))
    assert _state(outputs) == ("incomplete", False)
    assert outputs.report["run_complete"] is False


def test_interrupted_and_budget_exhausted_runs_are_incomplete(tmp_path: pathlib.Path) -> None:
    def interrupt(call: int) -> None:
        if call == 2:
            raise KeyboardInterrupt

    interrupted = _execute(tmp_path / "a", ScriptProvider(on_call=interrupt))
    assert _state(interrupted) == ("incomplete", False)
    budget = _execute(tmp_path / "b", ScriptProvider(), max_requests=2)
    assert _state(budget) == ("incomplete", False)
    assert (
        budget.report["overall"]["reliability"]["unconditional_outcome_distribution"]["not_run"]
        == 3
    )


# --- denominators ---------------------------------------------------------------------------


def test_conditional_and_unconditional_denominators(tmp_path: pathlib.Path) -> None:
    overall = _execute(tmp_path, _contract_failure()).report["overall"]
    # dev-001 (answerable) is a model-output failure; dev-005 is a local abstention.
    attempt = overall["answer_attempt_rate"]
    assert (attempt["numerator"], attempt["denominator"]) == (2, 3)
    assert attempt["excluded_scheduled"] == {
        "scheduled": 4,
        "with_valid_outcome": 3,
        "excluded_by_outcome": {"model_output_failure": 1},
    }
    rel = overall["reliability"]
    assert rel["unconditional_outcome_distribution"] == {
        "valid_answer": 4,
        "local_abstention": 1,
        "model_output_failure": 1,
        "infrastructure_failure": 0,
        "integrity_failure": 0,
        "not_run": 0,
    }

    def nd(name: str) -> tuple[int, int]:
        return rel[name]["numerator"], rel[name]["denominator"]

    assert nd("contract_valid_outcome_rate") == (5, 6)
    assert nd("model_output_failure_rate") == (1, 5)  # the local abstention made no request
    assert nd("valid_answer_attempt_rate_unconditional") == (2, 4)
    assert rel["valid_answer_attempt_rate_unconditional"]["model_output_failures"] == 1
    assert nd("abstention_on_answerable_rate_unconditional") == (1, 4)
    assert nd("abstention_on_unanswerable_rate_unconditional") == (1, 2)
    assert nd("answered_on_unanswerable_rate_unconditional") == (0, 2)
    assert nd("partially_answered_on_unanswerable_rate_unconditional") == (1, 2)
    unanswerable = overall["abstention_on_unanswerable_rate"]["excluded_scheduled"]
    assert unanswerable["excluded_by_outcome"] == {}
    assert overall["status_distribution"]["excluded_scheduled"]["excluded_by_outcome"] == {
        "model_output_failure": 1
    }


def test_zero_denominators_are_not_applicable() -> None:
    empty = cp.reliability_metrics([])
    assert empty["contract_valid_outcome_rate"]["rate"] == "not_applicable"
    assert empty["model_output_failure_rate"]["denominator"] == 0


def test_failed_attempts_without_usage_stay_unknown(tmp_path: pathlib.Path) -> None:
    provider = ContractProvider(INVALID["unknown_id"][0], usage=TokenUsage())
    usage = _execute(tmp_path, provider).report["overall"]["provider_usage"]
    assert (usage["attempted_requests"], usage["requests_with_usage"]) == (5, 4)
    assert usage["usage_complete"] is False


# --- retrieval provenance -------------------------------------------------------------------


def test_failed_questions_keep_the_sent_evidence_without_retrieving_again(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    searches: list[str] = []
    original = PolicyRetriever.search

    def counted(self: PolicyRetriever, query: str) -> Any:
        searches.append(query)
        return original(self, query)

    monkeypatch.setattr(PolicyRetriever, "search", counted)
    seen: list[list[str]] = []

    def builder(ids: list[str]) -> dict[str, Any]:
        seen.append(ids)
        return INVALID["unknown_id"][0](ids)

    outputs = _execute(tmp_path, ContractProvider(builder))
    assert len(searches) == len(QUESTIONS)  # exactly one retrieval per question
    record = _records(outputs)["dev-001"]
    assert record["selected_evidence_source"] == "model_facing_request"
    assert [e["evidence_id"] for e in record["selected_evidence"]] == seen[0]
    assert set(record["selected_evidence"][0]) == {"evidence_id", "document_id", "page_number"}
    assert len(record["request"]["input_sha256"]) == 64
    assert record["answer"] is None
    assert record["citations"] == []
    recall = outputs.report["overall"]["selected_evidence_page_recall"]
    assert recall["excluded_scheduled"]["with_recorded_selected_evidence"] == 4
    assert recall["denominator"] == 4  # the failed question's retrieval is still observed
    attempted = outputs.report["overall"]["cited_evidence_page_recall"]
    assert attempted["denominator"] == 2  # answer-dependent metrics stay conditional


def test_missing_retrieval_observations_are_excluded_not_reconstructed(
    tmp_path: pathlib.Path,
) -> None:
    fail = {FIRST: gerr.ProviderTimeoutError("synthetic")}
    outputs = _execute(tmp_path, ScriptProvider(fail=fail))
    record = _records(outputs)["dev-001"]
    assert record["selected_evidence_source"] == "model_facing_request"  # the request was sent
    budget = _execute(tmp_path / "b", ScriptProvider(), max_requests=1)
    unrun = _records(budget)["dev-002"]
    assert (unrun["selected_evidence"], unrun["selected_evidence_source"]) == ([], None)
    excluded = budget.report["overall"]["selected_evidence_page_recall"]["excluded_scheduled"]
    assert excluded["excluded_by_outcome"] == {"not_run": 3}


# --- run log --------------------------------------------------------------------------------


def _log(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / ge.REPORT_SUBDIR / cp.RUN_LOG_FILENAME


def test_live_runs_are_preregistered_before_any_provider_exists(tmp_path: pathlib.Path) -> None:
    provider = ScriptProvider()

    def factory() -> Any:
        entries, _ = cp.read_run_log(_log(tmp_path))
        assert [e["event"] for e in entries] == ["preregistered"]
        assert entries[0]["selection_rule"] == cp.SELECTION_RULE
        assert entries[0]["protocol_version"] == "1.3"
        assert entries[0]["git"] == CLEAN
        assert entries[0]["identities"] == IDS
        assert entries[0]["max_requests"] == 24
        return provider

    outputs = ge.execute(
        SPLIT,
        PIPELINE,
        tmp_path,
        mode="live",
        provider_factory=factory,
        max_requests=24,
        expected_identities=IDS,
        git=CLEAN,
    )
    entries, skipped = cp.read_run_log(_log(tmp_path))
    assert skipped == 0
    assert [e["event"] for e in entries] == ["preregistered", "finished"]
    assert entries[1]["completion_state"] == "answer_complete"
    assert entries[1]["run_id"] == outputs.report["run_id"]
    assert entries[1]["outcomes"]["valid_answer"] == 5


def test_run_log_is_append_only_and_tolerates_interruptions(tmp_path: pathlib.Path) -> None:
    _execute(tmp_path, ScriptProvider())
    first = _log(tmp_path).read_bytes()
    _execute(tmp_path, ScriptProvider(fail={FIRST: gerr.ProviderTimeoutError("synthetic")}))
    assert _log(tmp_path).read_bytes().startswith(first)  # earlier entries are untouched

    class Crash(ScriptProvider):
        def generate_structured(self, request: Any, schema: Any) -> Any:
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError):
        _execute(tmp_path, Crash())
    with _log(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write('{"event": "finished", "run_')  # a torn final line
    entries, skipped = cp.read_run_log(_log(tmp_path))
    assert skipped == 1
    status = cp.run_log_status(entries)
    assert sorted(v["finished"] for v in status.values()) == [
        "answer_complete",
        "incomplete",
        "unfinished",
    ]
    assert all(v["preregistered"] for v in status.values())


def test_dry_runs_and_replays_do_not_write_the_run_log(tmp_path: pathlib.Path) -> None:
    dry = _execute(tmp_path, ScriptProvider(), mode="dry_run")
    ge.execute(
        SPLIT,
        PIPELINE,
        tmp_path,
        mode="replay",
        provider_factory=ScriptProvider,
        expected_identities=IDS,
        replay_records=ge.load_records(dry.records_path),
    )
    assert not _log(tmp_path).exists()


def test_run_log_holds_no_question_answer_or_secret(tmp_path: pathlib.Path) -> None:
    _execute(tmp_path, _contract_failure())
    text = _log(tmp_path).read_text(encoding="utf-8")
    for leak in ("SENTINEL-QUESTION", SECRET, "SENTINEL-LIMITATION", "test-key", "sk-"):
        assert leak not in text


# --- replay semantics -----------------------------------------------------------------------


def test_live_origin_v3_replay_keeps_state_and_reportability(tmp_path: pathlib.Path) -> None:
    live = _execute(tmp_path, _contract_failure())
    replayed = _execute(
        tmp_path, ScriptProvider(), mode="replay", replay=ge.load_records(live.records_path)
    )
    assert _state(replayed) == ("execution_complete", True)
    assert replayed.report["origin_run_id"] == live.report["run_id"]
    record = _records(replayed)["dev-001"]
    assert (record["outcome"], record["execution_outcome"]) == (
        "not_replayable",
        "model_output_failure",
    )
    assert record["protocol_version"] == "1.3"


def test_synthetic_origin_replay_is_never_reportable(tmp_path: pathlib.Path) -> None:
    dry = _execute(tmp_path, _contract_failure(), mode="dry_run")
    assert _state(dry) == ("execution_complete", False)
    replayed = _execute(
        tmp_path, ScriptProvider(), mode="replay", replay=ge.load_records(dry.records_path)
    )
    assert _state(replayed) == ("execution_complete", False)
    assert replayed.report["synthetic"] is True


def _as_schema(records: list[dict[str, Any]], version: int) -> list[dict[str, Any]]:
    """Faithful pre-v1.3 records: no protocol 1.3 fields (and no failure block for v1)."""
    drop = {"record_sha256", "protocol_version", "execution_outcome", "selected_evidence_source"}
    out = []
    for record in records:
        body = {k: v for k, v in record.items() if k not in drop}
        if body.get("failure"):
            body["failure"] = {k: v for k, v in body["failure"].items() if k != "classification"}
        if record["outcome"] == "failed":
            body.update(selected_evidence=[], request=None)
            if version == 1:
                body.update(failure=None, usage=None)
        if version == 1:
            body.pop("failure", None)
        out.append(ge.seal({**body, "record_schema_version": version}))
    return out


@pytest.mark.parametrize("version", [1, 2])
def test_historical_v12_runs_are_never_upgraded(tmp_path: pathlib.Path, version: int) -> None:
    live = _execute(tmp_path, _contract_failure())
    legacy = _as_schema(ge.load_records(live.records_path), version)
    replayed = _execute(tmp_path / str(version), ScriptProvider(), mode="replay", replay=legacy)
    report = replayed.report
    assert report["protocol_version"] == "1.2"
    assert report["completion_state"] is None
    assert report["completion_semantics"].startswith("protocol 1.2")
    assert report["run_complete"] is False
    assert report["reportability"] == {
        "reportable_as_hosted_model_diagnostic": False,
        "reason": "incomplete run: not a completed 24-question diagnostic result",
    }
    assert report["selection_rule"] is None
    assert {r["protocol_version"] for r in _records(replayed).values()} == {"1.2"}


def test_mixed_protocol_records_are_refused_and_never_reportable(tmp_path: pathlib.Path) -> None:
    live = _execute(tmp_path, ScriptProvider())
    records = ge.load_records(live.records_path)
    mixed = [ge.seal({**records[0], "protocol_version": "1.2"}), *records[1:]]
    report = ge.aggregate(
        mixed,
        SPLIT,
        run_id="x",
        mode="live",
        state=ge.RunState(),
        max_requests=24,
        attempted_requests=5,
    )
    assert report["completion_state"] is None
    assert report["reportability"]["reportable_as_hosted_model_diagnostic"] is False
    assert any("protocol version" in p for p in report["completion_problems"])


def test_stored_outcomes_match_their_derivation(tmp_path: pathlib.Path) -> None:
    outputs = _execute(tmp_path, _contract_failure())
    for record in ge.load_records(outputs.records_path):
        assert record["execution_outcome"] == cp.execution_outcome(record).value
    assert json.loads(outputs.json_path.read_text("utf-8"))["failure_classification"] == (
        cp.classification_table()
    )


# --- final verification: failed-record replay and artifact contents -------------------------


def test_replayed_model_output_failure_stays_a_failure(tmp_path: pathlib.Path) -> None:
    live = _execute(tmp_path, _contract_failure())
    source = _records(live)["dev-001"]

    def factory() -> Any:
        raise AssertionError("replay must not build a provider")

    replayed = ge.execute(
        SPLIT,
        PIPELINE,
        tmp_path,
        mode="replay",
        provider_factory=factory,
        expected_identities=IDS,
        git=CLEAN,
        replay_records=ge.load_records(live.records_path),
    )
    record = _records(replayed)["dev-001"]
    assert record["outcome"] == "not_replayable"  # never re-inferred, never a valid answer
    assert record["execution_outcome"] == "model_output_failure"
    assert (record["status"], record["answer"], record["claims"]) == (None, None, [])
    assert record["error_category"] == "response_contract"
    assert record["failure"] == source["failure"]
    assert record["failure"]["stage"] == "answer_validation"
    assert record["failure"]["classification"] == "model_output"
    assert record["selected_evidence"] == source["selected_evidence"]
    assert record["selected_evidence_source"] == "model_facing_request"
    assert (record["provider_attempted"], record["usage"]) == (False, None)
    assert record["replay_of"]["usage"] == source["usage"]
    assert replayed.report["request_budget"]["attempted_requests"] == 0
    diagnostics = replayed.report["failure_diagnostics"]
    assert diagnostics["failed"] == 1
    assert diagnostics["questions"][0]["carried_from_source_run"] is True
    assert diagnostics["questions"][0]["usage_recorded"] is True
    for key in (
        "reliability",
        "selected_evidence_page_recall",
        "status_distribution",
        "answer_attempt_rate",
    ):
        assert replayed.report["overall"][key] == live.report["overall"][key], key
    assert _state(replayed) == _state(live) == ("execution_complete", True)
    assert ge.REPLAY_NOTE in replayed.report["provenance_notes"]


def test_artifact_contents_match_the_privacy_contract(tmp_path: pathlib.Path) -> None:
    live = _execute(tmp_path, _contract_failure())
    records = _records(live)
    valid = records["dev-002"]
    claim_texts = [c["text"] for c in valid["answer"]["claims"]]
    assert claim_texts  # successful records keep the validated answer, for replay
    assert records["dev-001"]["answer"] is None  # the rejected answer is never stored
    detailed = live.records_path.read_text(encoding="utf-8")
    summaries = "".join(
        p.read_text(encoding="utf-8") for p in (live.json_path, live.markdown_path, _log(tmp_path))
    )
    for text in claim_texts:
        assert text in detailed
        assert text not in summaries
    for leak in ("SENTINEL-QUESTION", SECRET, "test-key"):
        assert leak not in detailed
        assert leak not in summaries
    note = live.report["overall"]["reliability"]["contract_valid_outcome_rate"]["note"]
    assert "not factual correctness" in note


def test_a_later_run_never_hides_an_earlier_incomplete_one(tmp_path: pathlib.Path) -> None:
    first = _execute(tmp_path, ScriptProvider(fail={FIRST: gerr.ProviderTimeoutError("t")}))
    second = _execute(tmp_path, ScriptProvider())
    status = cp.run_log_status(cp.read_run_log(_log(tmp_path))[0])
    assert status[first.report["run_id"]]["finished"] == "incomplete"
    assert status[second.report["run_id"]]["finished"] == "answer_complete"
    assert list(status) == [first.report["run_id"], second.report["run_id"]]  # log order kept
