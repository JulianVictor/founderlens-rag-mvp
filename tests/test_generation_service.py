"""Structured grounded generation (Commit 5.3), offline.

The real policy retriever, context builder, answer schema and invariants run
over the synthetic V2 collection from the context tests. The provider is
either a small fake implementing ``LLMProvider`` or the real ``OpenAIProvider``
over ``httpx2.MockTransport``. No key, network, tokenizer asset or paid call.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from founderlens.generation import errors as ge
from founderlens.generation import prompts
from founderlens.generation.answer import (
    AnswerStatus,
    ClaimType,
    GroundedAnswer,
    ResponseContractError,
    answer_problems,
    answer_schema_definition,
    answer_schema_id,
    answer_text_format,
    schema_token_report,
    validate_answer,
)
from founderlens.generation.config import DEFAULT_GENERATION_CONFIG, DEFAULT_MODEL, GenerationConfig
from founderlens.generation.context import (
    ContextCompatibilityError,
    ContextConfig,
    ContextStatus,
    EvidenceRegistry,
    InvalidQuestionError,
)
from founderlens.generation.openai_provider import OpenAIProvider, strict_output_format
from founderlens.generation.prompts import GROUNDED_INSTRUCTIONS, serialize_input
from founderlens.generation.schema import (
    GenerationRequest,
    GenerationResult,
    ProviderIdentity,
    SchemaT,
    TokenUsage,
)
from founderlens.generation.service import (
    LOCAL_LIMITATIONS,
    AnswerOrigin,
    GroundedGenerationService,
    LocalAbstentionReason,
)
from founderlens.retrieval.policy import SELECTED_POLICY
from tests.test_generation_context import CHARS, COLLECTION, RETRIEVER, _build, _candidates
from tests.test_generation_openai import FAKE_KEY, Transport, _body, _message, _text

QUESTION = "revenue growth SENTINEL-QUESTION"
# A budget wide enough for the one-token-per-character test counter, including the schema.
SERVICE_CONFIG = ContextConfig(
    tokenizer="test-chars-v1",
    total_token_target=200_000,
    input_token_limit=190_000,
    framing_allowance_tokens=8_976,
)
NO_RETRIES = GenerationConfig(max_retries=0)  # same config_id: retries are operational


def _answer(
    status: str = "answered",
    claims: list[tuple[str, str, list[str]]] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    rows = [("Revenue grew SENTINEL-CLAIM.", "reported", ["E01"])] if claims is None else claims
    return {
        "status": status,
        "claims": [{"text": t, "claim_type": k, "evidence_ids": ids} for t, k, ids in rows],
        "limitations": [] if limitations is None else limitations,
    }


class FakeProvider:
    """Returns a fixed JSON answer through the real schema, recording each call."""

    def __init__(self, answer: dict[str, Any], config_id: str | None = None) -> None:
        self.payload = json.dumps(answer)
        self.calls: list[tuple[GenerationRequest, type[BaseModel]]] = []
        self._identity = ProviderIdentity(
            "fake", "0", DEFAULT_MODEL, config_id or DEFAULT_GENERATION_CONFIG.config_id
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def generate_text(self, request: GenerationRequest) -> GenerationResult[None]:
        raise AssertionError("the service must use structured generation")

    def generate_structured(
        self, request: GenerationRequest, schema: type[SchemaT]
    ) -> GenerationResult[SchemaT]:
        self.calls.append((request, schema))
        return GenerationResult(
            text=self.payload,
            parsed=schema.model_validate_json(self.payload),
            status="completed",
            requested_model=DEFAULT_MODEL,
            response_model=DEFAULT_MODEL,
            response_id="resp_fake_1",
            request_id="req_fake_1",
            usage=TokenUsage(900, 0, 60, 0, 960),
            latency_ms=1.0,
            provider=self._identity,
        )

    def close(self) -> None:
        return None


def _service(provider: Any, **kwargs: Any) -> GroundedGenerationService:
    return GroundedGenerationService(
        RETRIEVER, CHARS, provider, context_config=kwargs.pop("config", SERVICE_CONFIG), **kwargs
    )


def _openai(*replies: Any) -> tuple[OpenAIProvider, Transport]:
    transport = Transport(*replies)
    client = httpx2.Client(transport=httpx2.MockTransport(transport))
    return OpenAIProvider(NO_RETRIES, api_key=FAKE_KEY, http_client=client), transport


def _ok(answer: dict[str, Any]) -> Any:
    return Transport.ok(_body([_message(_text(json.dumps(answer)))]))


REGISTRY = _build(candidates=_candidates(0, 1)).evidence  # E01, E02


# --- schema -----------------------------------------------------------------------------------


def test_schema_is_strict_and_contains_only_answer_content() -> None:
    text_format = answer_text_format()
    assert text_format == dict(strict_output_format(GroundedAnswer))
    schema = text_format["schema"]
    assert text_format["strict"] is True
    assert list(schema["properties"]) == ["status", "claims", "limitations"]
    assert schema["required"] == ["status", "claims", "limitations"]
    claim = schema["$defs"]["Claim"]
    assert list(claim["properties"]) == ["text", "claim_type", "evidence_ids"]
    assert schema["additionalProperties"] is False
    assert claim["additionalProperties"] is False
    assert schema["$defs"]["AnswerStatus"]["enum"] == [s.value for s in AnswerStatus]
    assert schema["$defs"]["ClaimType"]["enum"] == [t.value for t in ClaimType]
    dumped = json.dumps(schema).lower()
    for trusted in ("document_id", "page", "url", "offset", "score", "sha", "hash", "config_id"):
        assert f'"{trusted}' not in dumped  # no model-generated provenance fields
    for name in ("claims", "limitations"):
        assert schema["properties"][name]["description"]
    assert "not cited" in schema["properties"]["limitations"]["description"]
    assert "insufficient_evidence" in schema["$defs"]["AnswerStatus"]["description"]
    assert "calculated" in schema["$defs"]["ClaimType"]["description"]


def test_schema_identity_is_stable_and_separate() -> None:
    assert answer_schema_id() == "e3ec802e2267"
    definition = answer_schema_definition()
    assert (definition["answer_schema_version"], definition["response_invariant_version"]) == (1, 1)
    assert definition["duplicate_reference_policy"] == "reject"
    others = {
        prompts.prompt_id(),
        SERVICE_CONFIG.config_id,
        SELECTED_POLICY.policy_id,
        DEFAULT_GENERATION_CONFIG.config_id,
    }
    assert answer_schema_id() not in others
    assert prompts.prompt_id() == "2a45c8422c20"  # the committed prompt is unchanged


def test_answer_model_rejects_extra_and_missing_fields_and_is_frozen() -> None:
    good = _answer()
    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate_json(json.dumps({**good, "answer": "free text"}))
    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate_json(json.dumps({"status": "answered", "claims": []}))
    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate_json(json.dumps({**good, "status": "maybe"}))
    claim = dict(good["claims"][0], document_id="d")
    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate_json(json.dumps({**good, "claims": [claim]}))
    answer = GroundedAnswer.model_validate_json(json.dumps(good))
    with pytest.raises(ValidationError):
        answer.status = AnswerStatus.INSUFFICIENT_EVIDENCE  # frozen model: raises at runtime


class RefWithDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: AnswerStatus = Field(description="a sibling of $ref")


def test_ref_with_sibling_keys_is_rejected_before_any_request() -> None:
    with pytest.raises(ge.GenerationConfigError, match="must not have sibling keys"):
        strict_output_format(RefWithDescription)


def test_schema_token_report_counts_the_sent_parameter() -> None:
    report = schema_token_report(CHARS, 1024)
    text_format = answer_text_format()
    compact = json.dumps({"format": text_format}, separators=(",", ":"), ensure_ascii=False)
    assert report.text_parameter_tokens == len(compact)
    assert report.schema_tokens == len(
        json.dumps(text_format["schema"], separators=(",", ":"), ensure_ascii=False)
    )
    assert report.remaining_allowance_tokens == 1024 - len(compact)


# --- response invariants ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        _answer(),
        _answer("answered", [("a", "calculated", ["E01", "E02"]), ("b", "synthesis", ["E02"])]),
        _answer("partially_answered", limitations=["The second period is not covered."]),
        _answer("insufficient_evidence", claims=[], limitations=["No evidence covers it."]),
    ],
)
def test_valid_answers_pass(answer: dict[str, Any]) -> None:
    parsed = GroundedAnswer.model_validate_json(json.dumps(answer))
    assert answer_problems(parsed, REGISTRY) == []
    assert parsed.abstained is (parsed.status is AnswerStatus.INSUFFICIENT_EVIDENCE)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (_answer("answered", claims=[]), "answered needs at least one claim"),
        (_answer("answered", limitations=["x"]), "answered must have no limitations"),
        (_answer("partially_answered"), "partially_answered needs at least one limitation"),
        (_answer("partially_answered", claims=[], limitations=["x"]), "needs at least one claim"),
        (_answer("insufficient_evidence", limitations=["x"]), "must have no claims"),
        (_answer("insufficient_evidence", claims=[]), "needs at least one limitation"),
        (_answer("partially_answered", limitations=["  "]), "limitation 1 is blank"),
        (_answer(claims=[("   ", "reported", ["E01"])]), "claim 1: text is blank"),
        (_answer(claims=[("a", "reported", [])]), "claim 1: cites no evidence"),
        (_answer(claims=[("a", "reported", ["E01", "E01"])]), "E01 is cited twice"),
        (_answer(claims=[("a", "reported", ["E1"])]), "E1 is not an evidence ID"),
        (_answer(claims=[("a", "reported", ["e01"])]), "e01 is not an evidence ID"),
        (_answer(claims=[("a", "reported", ["E001"])]), "E001 is not an evidence ID"),
        (_answer(claims=[("a", "reported", ["S01"])]), "S01 is not an evidence ID"),
        (_answer(claims=[("a", "reported", ["E03"])]), "E03 is not in this request's evidence"),
    ],
)
def test_invariant_violations_are_reported(answer: dict[str, Any], message: str) -> None:
    parsed = GroundedAnswer.model_validate_json(json.dumps(answer))
    with pytest.raises(ResponseContractError, match=message) as caught:
        validate_answer(parsed, REGISTRY, request_id="req_1")
    assert caught.value.request_id == "req_1"
    assert "SENTINEL-CLAIM" not in str(caught.value)


def test_excluded_or_absent_evidence_ids_are_not_citable() -> None:
    from tests.test_generation_context import _config, _limit_for

    limit = _limit_for("What was revenue?", [COLLECTION.chunks[0]])
    prepared = _build("What was revenue?", _candidates(0, 3), config=_config(limit))
    assert prepared.evidence.ids == ("E01",)  # rank 2 was excluded and has no ID
    parsed = GroundedAnswer.model_validate_json(
        json.dumps(_answer(claims=[("a", "reported", ["E02"])]))
    )
    assert any("E02 is not in" in p for p in answer_problems(parsed, prepared.evidence))
    assert any("E02 is not in" in p for p in answer_problems(parsed, EvidenceRegistry()))


# --- orchestration ----------------------------------------------------------------------------


def test_selected_evidence_makes_one_structured_call_on_the_prepared_request() -> None:
    provider = FakeProvider(_answer("answered", [("Revenue grew.", "reported", ["E01", "E03"])]))
    result = _service(provider).answer(QUESTION)
    assert len(provider.calls) == 1
    request, schema = provider.calls[0]
    assert schema is GroundedAnswer
    assert request is result.prepared.request
    assert result.prepared.status is ContextStatus.EVIDENCE_SELECTED
    assert len(result.prepared.evidence) == 10
    assert (result.origin, result.local_abstention_reason) == (AnswerOrigin.MODEL, None)
    assert result.provider_called
    assert result.answer.claims[0].evidence_ids == ("E01", "E03")
    assert result.provider_result is not None
    assert result.provider_result.usage.total_tokens == 960
    ids = result.identities()
    assert ids["answer_schema_id"] == answer_schema_id()
    assert ids["prompt_id"] == "2a45c8422c20"
    assert ids["context_config_id"] == SERVICE_CONFIG.config_id
    assert ids["generation_config_id"] == "cbc4e415c848"
    assert ids["retrieval"]["policy_id"] == "f20913f1c1ec"
    assert result.total_ms >= result.context_ms >= 0


def test_real_provider_sends_the_prepared_input_and_answer_schema() -> None:
    answer = _answer("partially_answered", [("Revenue grew.", "reported", ["E02"])], ["No Q4."])
    provider, transport = _openai(_ok(answer))
    result = _service(provider, generation_config=NO_RETRIES).answer(QUESTION)
    assert len(transport.requests) == 1
    body = transport.body()
    assert body["instructions"] == GROUNDED_INSTRUCTIONS
    assert body["input"] == result.prepared.request.input_text
    assert body["text"] == {"format": answer_text_format()}
    assert body["store"] is False
    assert (body["model"], body["max_output_tokens"]) == (DEFAULT_MODEL, 1024)
    assert (body["reasoning"], body["stream"]) == ({"effort": "none"}, False)
    assert set(body) == {
        "model",
        "instructions",
        "input",
        "max_output_tokens",
        "reasoning",
        "store",
        "stream",
        "text",
    }  # no tools, background, conversation or sampling parameters
    assert result.answer.status is AnswerStatus.PARTIALLY_ANSWERED
    assert result.provider_result is not None
    assert result.provider_result.response_id == "resp_synthetic_1"
    assert result.provider_result.request_id == "req_synthetic_1"
    runtime = result.metadata()["runtime"]["provider"]
    assert runtime["requested_model"] == runtime["response_model"] == DEFAULT_MODEL
    provider.close()


def test_empty_retrieval_abstains_locally_without_calling_the_provider() -> None:
    provider = FakeProvider(_answer())
    result = _service(provider).answer("zyzzyva quokka")
    assert provider.calls == []
    assert result.prepared.status is ContextStatus.EMPTY_RETRIEVAL
    assert (result.origin, result.local_abstention_reason) == (
        AnswerOrigin.LOCAL,
        LocalAbstentionReason.EMPTY_RETRIEVAL,
    )
    assert result.answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.answer.claims == ()
    assert result.answer.limitations == (LOCAL_LIMITATIONS[LocalAbstentionReason.EMPTY_RETRIEVAL],)
    assert "does not show" in result.answer.limitations[0]
    assert result.abstained
    assert not result.provider_called
    assert result.provider_ms is None
    metadata = result.metadata()
    assert metadata["runtime"]["provider"] is None  # no fabricated usage or IDs
    assert metadata["generation"]["local_abstention_reason"] == "empty_retrieval"


def test_no_evidence_fits_abstains_locally_with_its_own_reason() -> None:
    base = CHARS.count(GROUNDED_INSTRUCTIONS) + CHARS.count(serialize_input(QUESTION, []))
    config = ContextConfig(
        tokenizer="test-chars-v1",
        total_token_target=base + 8_976 + 1024,
        input_token_limit=base,
        framing_allowance_tokens=8_976,
    )
    provider = FakeProvider(_answer())
    result = _service(provider, config=config).answer(QUESTION)
    assert provider.calls == []
    assert result.prepared.status is ContextStatus.NO_EVIDENCE_FITS
    assert result.local_abstention_reason is LocalAbstentionReason.NO_EVIDENCE_FITS
    assert result.answer.limitations == (LOCAL_LIMITATIONS[LocalAbstentionReason.NO_EVIDENCE_FITS],)
    assert result.prepared.candidate_count == 10


def test_invalid_question_is_an_error_not_an_abstention() -> None:
    provider = FakeProvider(_answer())
    with pytest.raises(InvalidQuestionError):
        _service(provider).answer("?!")
    assert provider.calls == []


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (Transport.error(401, "invalid_api_key"), ge.AuthenticationError),
        (Transport.error(403, "unsupported_country"), ge.PermissionDeniedError),
        (Transport.error(429, "rate_limit_exceeded"), ge.RateLimitError),
        (Transport.error(429, "insufficient_quota"), ge.QuotaExceededError),
        (httpx2.ReadTimeout("synthetic"), ge.ProviderTimeoutError),
        (httpx2.ConnectError("synthetic"), ge.ProviderConnectionError),
        (
            Transport.ok(_body([_message({"type": "refusal", "refusal": "no"})])),
            ge.ModelRefusalError,
        ),
        (
            Transport.ok(
                _body(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
            ),
            ge.IncompleteGenerationError,
        ),
        (Transport.ok(_body(model="gpt-5.4-mini")), ge.UnexpectedResponseError),
        (Transport.ok(_body([_message(_text('{"status":"answered"}'))])), ge.StructuredOutputError),
    ],
)
def test_provider_errors_propagate_without_retry_or_abstention(
    reply: Any, error: type[ge.GenerationError]
) -> None:
    provider, transport = _openai(reply)
    with pytest.raises(error):
        _service(provider, generation_config=NO_RETRIES).answer(QUESTION)
    assert len(transport.requests) == 1  # one call; no orchestration retry or repair
    provider.close()


def test_contract_violation_fails_explicitly_after_one_call() -> None:
    answer = _answer(claims=[("Revenue grew SENTINEL-CLAIM.", "reported", ["E11"])])
    provider, transport = _openai(_ok(answer))
    with pytest.raises(ResponseContractError, match="E11 is not in") as caught:
        _service(provider, generation_config=NO_RETRIES).answer(QUESTION)
    assert len(transport.requests) == 1
    assert caught.value.request_id == "req_synthetic_1"
    assert "SENTINEL-CLAIM" not in str(caught.value)
    provider.close()


def test_incompatible_provider_or_schema_allowance_is_rejected_at_construction() -> None:
    with pytest.raises(ContextCompatibilityError, match="provider configuration"):
        _service(FakeProvider(_answer(), config_id="000000000000"))
    narrow = ContextConfig(
        tokenizer="test-chars-v1",
        total_token_target=200_000,
        input_token_limit=190_000,
        framing_allowance_tokens=1024,
    )
    with pytest.raises(ContextCompatibilityError, match="framing allowance"):
        _service(FakeProvider(_answer()), config=narrow)  # one token per character is too many


def test_result_metadata_and_logs_hold_no_question_evidence_or_answer_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = FakeProvider(_answer())
    with caplog.at_level(logging.DEBUG):
        result = _service(provider).answer(QUESTION)
    dumped = json.dumps(result.metadata())
    for secret in ("SENTINEL-QUESTION", "SENTINEL-CLAIM", result.prepared.request.input_text):
        assert secret not in dumped
        assert secret not in caplog.text
    assert all(r.text not in dumped for r in result.prepared.evidence)
    assert result.answer_schema_id in caplog.text
    generation = result.metadata()["generation"]
    assert generation["cited_evidence_ids"] == ["E01"]
    assert generation["claim_types"] == ["reported"]
    with pytest.raises(AttributeError):
        result.answer = result.answer  # type: ignore[misc]  # frozen result


# --- final 5.3 verification: immutability, request scoping, privacy ---------------------------


def test_validated_answer_and_result_cannot_be_mutated() -> None:
    answer = _answer(
        "partially_answered",
        [("Revenue grew.", "reported", ["E01"]), ("Margin rose.", "synthesis", ["E02"])],
        ["The guidance period is not covered."],
    )
    result = _service(FakeProvider(answer)).answer(QUESTION)
    validated = result.answer
    claim = validated.claims[0]
    with pytest.raises(AttributeError):
        validated.claims.append(claim)  # type: ignore[attr-defined]  # tuples have no append
    with pytest.raises(AttributeError):
        claim.evidence_ids.append("E99")  # type: ignore[attr-defined]  # tuples have no append
    with pytest.raises(AttributeError):
        validated.limitations.append("x")  # type: ignore[attr-defined]  # tuples have no append
    with pytest.raises(ValidationError):
        claim.evidence_ids = ("E99",)  # frozen model: raises at runtime
    with pytest.raises(ValidationError):
        validated.limitations = ()  # frozen model: raises at runtime
    with pytest.raises(AttributeError):
        result.prepared = result.prepared  # type: ignore[misc]  # frozen result
    copy = validated.model_copy(update={"status": AnswerStatus.ANSWERED})
    assert copy is not validated
    assert result.answer.status is AnswerStatus.PARTIALLY_ANSWERED  # the result is unchanged
    assert result.answer.claims[0].evidence_ids == ("E01",)


def test_evidence_ids_resolve_only_against_their_own_run() -> None:
    provider = FakeProvider(_answer("answered", [("Stated.", "reported", ["E01"])]))
    service = _service(provider)
    first = service.answer("cloud revenue search")
    second = service.answer("cash flow capital expenditures")
    e01_first = first.prepared.evidence.get("E01")
    e01_second = second.prepared.evidence.get("E01")
    assert e01_first is not None
    assert e01_second is not None
    assert e01_first.chunk_id != e01_second.chunk_id  # same label, different chunks
    assert provider.calls[0][0] is first.prepared.request
    assert provider.calls[1][0] is second.prepared.request
    narrow = service.answer("advertising")
    assert narrow.prepared.evidence.ids == ("E01",)
    later = FakeProvider(_answer("answered", [("Stated.", "reported", ["E02"])]))
    assert _service(later).answer("cloud revenue search").answer.claims[0].evidence_ids == ("E02",)
    with pytest.raises(ResponseContractError, match="E02 is not in this request's evidence"):
        _service(later).answer("advertising")  # valid elsewhere, absent here


def test_failure_and_success_paths_keep_text_out_of_logs_and_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from tests.test_generation_openai import INPUT as PROVIDER_SENTINEL

    answer = _answer(
        "partially_answered",
        [("Revenue SENTINEL-CLAIM.", "reported", ["E01"])],
        ["SENTINEL-LIMITATION not covered."],
    )
    with caplog.at_level(logging.DEBUG):
        result = _service(FakeProvider(answer)).answer(QUESTION)
        provider, _ = _openai(Transport.error(401, "invalid_api_key"))
        with pytest.raises(ge.AuthenticationError) as caught:
            _service(provider, generation_config=NO_RETRIES).answer(QUESTION)
        provider.close()
    dumped = json.dumps(result.metadata())
    texts = (
        "SENTINEL-QUESTION",
        "SENTINEL-CLAIM",
        "SENTINEL-LIMITATION",
        "SECRET-PROVIDER-MESSAGE",
        PROVIDER_SENTINEL,
        FAKE_KEY,
    )
    for text in texts:
        assert text not in dumped
        assert text not in caplog.text
        assert text not in str(caught.value)
    assert "category=authentication" in caplog.text
    assert result.prepared.identities()["generation_config_id"] == (
        FakeProvider(answer).identity.config_id
    )


@pytest.mark.parametrize(
    "payload",
    [
        {**_answer(), "claims": [{"text": "a", "claim_type": "estimate", "evidence_ids": ["E01"]}]},
        {**_answer(), "answer": "free-form text outside the schema"},
        {
            **_answer(),
            "claims": [
                {"text": "a", "claim_type": "reported", "evidence_ids": ["E01"], "page_number": 3}
            ],
        },
    ],
    ids=["invalid-claim-type", "extra-top-level-field", "model-generated-provenance"],
)
def test_schema_violations_from_the_model_are_structured_output_errors(
    payload: dict[str, Any],
) -> None:
    provider, transport = _openai(_ok(payload))
    with pytest.raises(ge.StructuredOutputError):
        _service(provider, generation_config=NO_RETRIES).answer(QUESTION)
    assert len(transport.requests) == 1
    provider.close()


def test_live_smoke_harness_caps_paid_requests_offline() -> None:
    from tests.test_generation_service_live import MAX_PAID_REQUESTS, CountingProvider, _observe

    answer = _answer("answered", [("Stated.", "reported", ["E01"])])
    inner, transport = _openai(_ok(answer))
    counting = CountingProvider(inner)
    service = _service(counting, generation_config=NO_RETRIES)
    first = _observe(service, "evidence_present", QUESTION)
    empty = _observe(service, "empty_retrieval", "zyzzyva quokka")
    _observe(service, "evidence_absent", QUESTION)
    assert (first["outcome"], first["parsed"], first["cited_evidence_ids"]) == (
        "completed",
        True,
        ["E01"],
    )
    assert (empty["origin"], empty["usage"]) == ("local", None)
    with pytest.raises(AssertionError, match="more than 2 paid requests"):
        service.answer(QUESTION)
    assert counting.calls == MAX_PAID_REQUESTS == len(transport.requests) == 2
    bad = _openai(_ok(_answer(claims=[("x", "reported", ["E42"])])))[0]
    semantic = _observe(
        _service(CountingProvider(bad), generation_config=NO_RETRIES), "s", QUESTION
    )
    assert (semantic["outcome"], semantic["category"]) == ("semantic_failure", "response_contract")
    counting.close()
    bad.close()
