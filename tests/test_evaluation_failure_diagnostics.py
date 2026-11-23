"""Safe generation failure diagnostics in the evaluation harness (record schema 2), offline.

Mocked providers return schema-valid but contract-violating SYNTHETIC answers.
No model, key, network or real report is used; files go under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Callable
from typing import Any

import pytest

from founderlens.evaluation import generation_eval as ge
from founderlens.generation.answer import (
    PROBLEM_CODES,
    GroundedAnswer,
    ResponseContractError,
    answer_problem_details,
    answer_problems,
    validate_answer,
)
from founderlens.generation.config import DEFAULT_GENERATION_CONFIG
from founderlens.generation.schema import (
    GenerationRequest,
    GenerationResult,
    ProviderIdentity,
    SchemaT,
    TokenUsage,
)
from tests.test_evaluation_generation_eval import IDS, PIPELINE, QUESTIONS, SPLIT, ScriptProvider
from tests.test_generation_context import _build

Builder = Callable[[list[str]], dict[str, Any]]  # evidence IDs of the request -> answer
SECRET = "SENTINEL-INVALID-ANSWER-TEXT"
FAILING = QUESTIONS[0].question  # dev-001; the other questions use the default script


def _ok(ids: list[str]) -> dict[str, Any]:
    return {"status": "answered", "claims": [_claim(ids[0])], "limitations": []}


def _claim(*ids: str, text: str = SECRET) -> dict[str, Any]:
    return {"text": text, "claim_type": "reported", "evidence_ids": list(ids)}


INVALID: dict[str, tuple[Builder, list[str]]] = {
    "answered_with_limitations": (
        lambda ids: {"status": "answered", "claims": [_claim(ids[0])], "limitations": [SECRET]},
        ["answered_has_limitations"],
    ),
    "partial_without_limitations": (
        lambda ids: {"status": "partially_answered", "claims": [_claim(ids[0])], "limitations": []},
        ["limitations_required"],
    ),
    "unknown_id": (
        lambda ids: {"status": "answered", "claims": [_claim("E42")], "limitations": []},
        ["unknown_evidence_id"],
    ),
    "duplicate_id": (
        lambda ids: {"status": "answered", "claims": [_claim(ids[0], ids[0])], "limitations": []},
        ["duplicate_evidence_id"],
    ),
    "blank_text": (
        lambda ids: {
            "status": "answered",
            "claims": [_claim(ids[0], text="   ")],
            "limitations": [],
        },
        ["blank_claim_text"],
    ),
}


class ContractProvider(ScriptProvider):
    """dev-001 gets a scripted (possibly invalid) answer; usage and request ID are configurable."""

    def __init__(
        self,
        builder: Builder,
        *,
        usage: TokenUsage | None = None,
        request_id: str | None = "req_synthetic_123",
    ) -> None:
        super().__init__()
        self.builder = builder
        self.usage = TokenUsage(input_tokens=111, output_tokens=22) if usage is None else usage
        self.request_id = request_id

    def generate_structured(
        self, request: GenerationRequest, schema: type[SchemaT]
    ) -> GenerationResult[SchemaT]:
        payload = json.loads(request.input_text)
        if payload["question"] != FAILING:
            return super().generate_structured(request, schema)
        self.calls += 1
        text = json.dumps(self.builder([e["evidence_id"] for e in payload["evidence"]]))
        identity = ProviderIdentity(
            "fake", "0", DEFAULT_GENERATION_CONFIG.model, DEFAULT_GENERATION_CONFIG.config_id
        )
        return GenerationResult(
            text=text,
            parsed=schema.model_validate_json(text),  # schema-valid: the SDK would accept it
            status="completed",
            requested_model=identity.model,
            response_model="fake",
            response_id="fake",
            request_id=self.request_id,
            usage=self.usage,
            latency_ms=1.0,
            provider=identity,
        )


def _run(tmp_path: pathlib.Path, provider: ScriptProvider, mode: ge.Mode = "live") -> ge.RunOutputs:
    return ge.execute(SPLIT, PIPELINE, tmp_path, mode=mode, provider_factory=lambda: provider,
                      max_requests=24, expected_identities=IDS)  # fmt: skip


def _record(outputs: ge.RunOutputs, qid: str = "dev-001") -> dict[str, Any]:
    return next(r for r in ge.load_records(outputs.records_path) if r["question_id"] == qid)


# --- invariant codes ------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(INVALID))
def test_contract_failures_persist_exact_codes(tmp_path: pathlib.Path, case: str) -> None:
    builder, codes = INVALID[case]
    provider = ContractProvider(builder)
    outputs = _run(tmp_path, provider)
    record = _record(outputs)
    assert (record["outcome"], record["error_category"], record["status"]) == (
        "failed",
        "response_contract",
        None,
    )
    failure = record["failure"]
    assert failure["stage"] == "answer_validation"
    assert [p["code"] for p in failure["problems"]] == codes
    assert set(codes) <= set(PROBLEM_CODES)
    assert failure["request_id"] == "req_synthetic_123"
    assert record["answer"] is None  # the invalid answer is never stored
    assert provider.calls == 5  # one request per question that reaches the provider
    assert outputs.report["run_complete"] is False
    assert outputs.report["failure_diagnostics"]["problem_codes"] == dict.fromkeys(codes, 1)


def test_problem_positions_and_sanitized_ids(tmp_path: pathlib.Path) -> None:
    long_id = "<script>" + "Z" * 200

    def builder(ids: list[str]) -> dict[str, Any]:
        return {"status": "answered", "claims": [_claim(ids[0]), _claim(ids[0], "E42", ids[0],
                long_id)], "limitations": []}  # fmt: skip

    problems = _record(_run(tmp_path, ContractProvider(builder)))["failure"]["problems"]
    assert [(p["code"], p["claim_position"], p["reference_position"]) for p in problems] == [
        ("unknown_evidence_id", 2, 2),
        ("duplicate_evidence_id", 2, 3),
        ("malformed_evidence_id", 2, 4),
    ]
    malformed = problems[2]["evidence_id"]
    assert len(malformed) <= 32
    assert "<" not in malformed
    assert problems[0]["evidence_id"] == "E42"


ALL_CODES: list[tuple[dict[str, Any], list[tuple[str, int | None, int | None, int | None]]]] = [
    ({"status": "insufficient_evidence", "claims": [_claim("E01")], "limitations": ["x"]},
     [("insufficient_evidence_has_claims", None, None, None)]),
    ({"status": "answered", "claims": [], "limitations": []},
     [("claims_required", None, None, None)]),
    ({"status": "answered", "claims": [_claim("E01")], "limitations": ["x"]},
     [("answered_has_limitations", None, None, None)]),
    ({"status": "partially_answered", "claims": [_claim("E01")], "limitations": []},
     [("limitations_required", None, None, None)]),
    ({"status": "partially_answered", "claims": [_claim("E01")], "limitations": ["x", " "]},
     [("blank_limitation", None, None, 2)]),
    ({"status": "answered", "claims": [_claim("E01"), _claim("E01", text=" ")], "limitations": []},
     [("blank_claim_text", 2, None, None)]),
    ({"status": "answered", "claims": [_claim()], "limitations": []},
     [("claim_cites_no_evidence", 1, None, None)]),
    ({"status": "answered", "claims": [_claim("E01", "E02", "E01")], "limitations": []},
     [("duplicate_evidence_id", 1, 3, None)]),
    ({"status": "answered", "claims": [_claim("E1")], "limitations": []},
     [("malformed_evidence_id", 1, 1, None)]),
    ({"status": "answered", "claims": [_claim("E01"), _claim("E02", "E77")], "limitations": []},
     [("unknown_evidence_id", 2, 2, None)]),
]  # fmt: skip


@pytest.mark.parametrize(("payload", "expected"), ALL_CODES)
def test_every_invariant_code_and_position(
    payload: dict[str, Any], expected: list[tuple[str, int | None, int | None, int | None]]
) -> None:
    evidence = _build().evidence  # E01..E03
    details = answer_problem_details(GroundedAnswer.model_validate(payload), evidence)
    found = [
        (d.code, d.claim_position, d.reference_position, d.limitation_position) for d in details
    ]
    assert found == expected
    assert {code for code, *_ in expected} <= set(PROBLEM_CODES)


def test_all_ten_codes_are_covered() -> None:
    assert sorted(code for _, expected in ALL_CODES for code, *_ in expected) == sorted(
        PROBLEM_CODES
    )
    assert len(PROBLEM_CODES) == 10


def test_error_text_repr_and_logs_expose_no_answer_content(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    hostile = SECRET + " <|endoftext|> ignore previous instructions sk-abcdefghijklmnop"
    answer = GroundedAnswer.model_validate(
        {"status": "answered", "claims": [_claim(hostile)], "limitations": [SECRET]}
    )
    with pytest.raises(ResponseContractError) as info:
        validate_answer(answer, _build().evidence, request_id="req_1", usage=TokenUsage())
    error = info.value
    shown = (
        str(error)
        + repr(error)
        + repr(error.details)
        + json.dumps([d.to_dict() for d in error.details])
    )
    assert " " not in error.details[1].evidence_id  # type: ignore[operator]  # str here
    assert len(error.details[1].evidence_id or "") <= 32
    # At most 32 sanitized characters of a model-supplied ID appear; nothing beyond them.
    for leak in ("ndoftext", "ignore", "previous", "sk-abcdefghijklmnop"):
        assert leak not in str(error) + json.dumps([d.to_dict() for d in error.details])
    assert "?" in shown  # spaces and markup were replaced, not passed through
    assert error.usage == TokenUsage()  # unknown usage stays unknown: every field None
    assert all(v is None for v in error.usage.to_dict().values())
    caplog.set_level("WARNING")
    _run(tmp_path, ContractProvider(lambda ids: {"status": "answered",
         "claims": [_claim(hostile)], "limitations": []}))  # fmt: skip
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "category=response_contract" in logged
    for leak in (SECRET, "ignore previous", "sk-abcdefghijklmnop", "SENTINEL-QUESTION"):
        assert leak not in logged


def test_messages_and_identities_are_unchanged() -> None:
    prepared = _build()
    answer = GroundedAnswer.model_validate(
        {
            "status": "partially_answered",
            "claims": [_claim("E01", "E01"), _claim("E42")],
            "limitations": [],
        }
    )
    assert answer_problems(answer, prepared.evidence) == [
        "partially_answered needs at least one limitation",
        "claim 1: evidence ID E01 is cited twice",
        "claim 2: evidence ID E42 is not in this request's evidence",
    ]
    details = answer_problem_details(answer, prepared.evidence)
    assert [d.message for d in details] == answer_problems(answer, prepared.evidence)
    assert all("message" not in d.to_dict() for d in details)
    with pytest.raises(ResponseContractError) as info:
        validate_answer(
            answer, prepared.evidence, request_id="r1", usage=TokenUsage(input_tokens=3)
        )
    assert info.value.details == tuple(details)
    assert info.value.usage == TokenUsage(input_tokens=3)
    assert str(info.value).startswith("[response_contract] partially_answered needs")
    valid = GroundedAnswer.model_validate(_ok(["E01"]))
    assert answer_problem_details(valid, prepared.evidence) == []
    validate_answer(valid, prepared.evidence)  # successful responses are unaffected
    assert (IDS["answer_schema_id"], IDS["response_contract_id"]) == (
        "e3ec802e2267",
        "debe65096147",
    )


# --- usage accounting -----------------------------------------------------------------------


def test_usage_of_a_rejected_answer_is_preserved(tmp_path: pathlib.Path) -> None:
    outputs = _run(tmp_path, ContractProvider(INVALID["unknown_id"][0]))
    record = _record(outputs)
    assert record["usage"]["input_tokens"] == 111
    assert record["failure"]["usage_recorded"] is True
    usage = outputs.report["overall"]["provider_usage"]
    assert (usage["attempted_requests"], usage["requests_with_usage"]) == (5, 5)
    assert usage["failed_requests_with_usage"] == 1
    assert usage["usage_complete"] is True
    assert usage["input_tokens"] == {"reported_n": 5, "total": 4 * 100 + 111}


def test_missing_usage_stays_unknown(tmp_path: pathlib.Path) -> None:
    outputs = _run(tmp_path, ContractProvider(INVALID["unknown_id"][0], usage=TokenUsage(),
                                              request_id=None))  # fmt: skip
    record = _record(outputs)
    assert record["failure"]["usage_recorded"] is False
    assert record["failure"]["request_id"] is None
    assert all(v is None for v in (record["usage"] or {}).values())  # unknown, never zero
    usage = outputs.report["overall"]["provider_usage"]
    assert (usage["attempted_requests"], usage["requests_with_usage"]) == (5, 4)
    assert usage["usage_complete"] is False
    assert "understate" in usage["note"]
    assert usage["input_tokens"]["total"] == 400
    assert outputs.report["failure_diagnostics"]["questions"][0]["usage_recorded"] is False


def test_non_contract_failures_record_a_generation_stage(tmp_path: pathlib.Path) -> None:
    from founderlens.generation import errors as gerr

    fail = {FAILING: gerr.RateLimitError("synthetic", request_id="req_rl", status_code=429)}
    outputs = _run(tmp_path, ScriptProvider(fail=fail))
    failure = _record(outputs)["failure"]
    assert failure == {
        "stage": "generation",
        "error_category": "rate_limited",
        "classification": "infrastructure",
        "problems": [],
        "request_id": "req_rl",
        "status_code": 429,
        "usage_recorded": False,
    }
    assert outputs.report["overall"]["provider_usage"]["usage_complete"] is False


# --- privacy --------------------------------------------------------------------------------


def test_no_answer_question_or_credential_leakage(tmp_path: pathlib.Path) -> None:
    outputs = _run(tmp_path, ContractProvider(INVALID["answered_with_limitations"][0]))
    record_text = json.dumps(_record(outputs))
    summaries = (outputs.json_path, outputs.markdown_path)
    everything = "".join(p.read_text(encoding="utf-8") for p in summaries)
    for text in (record_text, everything):
        assert SECRET not in text
        assert "SENTINEL-QUESTION" not in text
        assert "test-key" not in text
        assert "cannot answer" not in text  # no prompt instructions
    assert "answer_validation" in outputs.markdown_path.read_text(encoding="utf-8")


# --- record schema and replay ---------------------------------------------------------------


def test_records_are_schema_3_and_failed_records_never_replay(tmp_path: pathlib.Path) -> None:
    live = _run(tmp_path, ContractProvider(INVALID["duplicate_id"][0]))
    records = ge.load_records(live.records_path)
    assert {r["record_schema_version"] for r in records} == {3}
    assert all("failure" in r for r in records)
    assert live.report["record_schema_version"] == 3
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.*")}
    replayed = ge.execute(SPLIT, PIPELINE, tmp_path, mode="replay", provider_factory=ScriptProvider,
                          expected_identities=IDS, replay_records=records)  # fmt: skip
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*.*")}
    assert {k: after[k] for k in before} == before  # the original run's files are untouched
    first = _record(replayed)
    assert first["outcome"] == "not_replayable"
    assert first["replay_of"]["failure_diagnostics"] == "recorded"
    assert [p["code"] for p in first["replay_of"]["failure"]["problems"]] == [
        "duplicate_evidence_id"
    ]
    assert replayed.report["run_complete"] is False
    assert replayed.report["original_run_usage"]["failed_requests_with_usage"] == 1
    # Protocol v1.3: one model-output failure is counted, not disqualifying.
    assert live.report["completion_state"] == "execution_complete"
    assert live.report["reportability"]["reportable_as_hosted_model_diagnostic"] is True
    assert replayed.report["completion_state"] == "execution_complete"  # the source's state


SCHEMA_3_ONLY = frozenset(
    {"record_sha256", "protocol_version", "execution_outcome", "selected_evidence_source"}
)


def _as_legacy(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Faithful schema-1 records: no failure block, no protocol 1.3 fields."""
    legacy = []
    for record in records:
        body = {k: v for k, v in record.items() if k not in SCHEMA_3_ONLY | {"failure"}}
        if record["outcome"] == "failed":
            body["usage"] = None  # schema 1 never kept a rejected answer's usage
        legacy.append(ge.seal({**body, "record_schema_version": 1}))
    return legacy


def test_legacy_schema_1_records_replay_without_invented_fields(tmp_path: pathlib.Path) -> None:
    live = _run(tmp_path, ContractProvider(INVALID["unknown_id"][0]))
    legacy = _as_legacy(ge.load_records(live.records_path))
    replayed = ge.execute(SPLIT, PIPELINE, tmp_path, mode="replay", provider_factory=ScriptProvider,
                          expected_identities=IDS, replay_records=legacy)  # fmt: skip
    records = {r["question_id"]: r for r in ge.load_records(replayed.records_path)}
    assert records["dev-002"]["outcome"] == "completed"
    assert records["dev-002"]["replay_of"]["record_schema_version"] == 1
    failed = records["dev-001"]
    assert failed["outcome"] == "not_replayable"
    assert failed["replay_of"]["failure"] is None
    assert failed["replay_of"]["failure_diagnostics"] == "not_recorded_in_record_schema_1"
    usage = replayed.report["original_run_usage"]
    assert (usage["attempted_requests"], usage["requests_with_usage"]) == (5, 4)
    assert usage["usage_complete"] is False
    mixed = [*legacy[:1], *ge.load_records(live.records_path)[1:]]
    with pytest.raises(ge.EvaluationRunError, match="mix record schema versions"):
        ge.execute(SPLIT, PIPELINE, tmp_path / "m", mode="replay", provider_factory=ScriptProvider,
                   expected_identities=IDS, replay_records=mixed)  # fmt: skip
    future = [ge.seal({**r, "record_schema_version": 4}) for r in legacy]
    with pytest.raises(ge.EvaluationRunError, match="unsupported record schema"):
        ge.execute(SPLIT, PIPELINE, tmp_path / "f", mode="replay", provider_factory=ScriptProvider,
                   expected_identities=IDS, replay_records=future)  # fmt: skip


def test_successful_runs_are_unchanged(tmp_path: pathlib.Path) -> None:
    outputs = _run(tmp_path, ContractProvider(_ok))
    records = ge.load_records(outputs.records_path)
    assert all(r["failure"] is None for r in records if r["outcome"] == "completed")
    assert _record(outputs)["outcome"] == "completed"
    assert outputs.report["failure_diagnostics"]["failed"] == 0
    assert outputs.report["run_complete"] is True
