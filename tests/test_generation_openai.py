"""OpenAI provider (Commit 5.1), fully offline.

The real SDK runs against ``httpx2.MockTransport``: request serialization,
SDK retries, status-error mapping and response parsing are all exercised;
only the network is replaced. No API key, no network, no paid call.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import openai
import pytest
from pydantic import BaseModel, ConfigDict

from founderlens.generation import errors as ge
from founderlens.generation.config import (
    DEFAULT_GENERATION_CONFIG,
    DEFAULT_MODEL,
    GenerationConfig,
)
from founderlens.generation.openai_provider import OpenAIProvider, strict_output_format
from founderlens.generation.schema import GenerationRequest, TokenUsage

FAKE_KEY = "sk-test-fake-0000000000000000"  # synthetic placeholder, never a real key
INSTRUCTIONS = "SYNTH-INSTRUCTIONS: answer briefly."
INPUT = "SYNTH-INPUT: name a colour and a number."
REQUEST = GenerationRequest(INSTRUCTIONS, INPUT)
FAST = {"retry-after-ms": "1"}  # keeps SDK retry backoff out of the test runtime


class SyntheticFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    colour: str
    count: int


class Inner(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str


class Outer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[Inner]
    note: str | None


# --- synthetic SDK traffic --------------------------------------------------------------------


def _message(*parts: dict[str, Any], phase: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": list(parts),
    }
    if phase is not None:
        item["phase"] = phase
    return item


def _text(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": []}


USAGE = {
    "input_tokens": 120,
    "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 0},
    "output_tokens": 30,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 150,
}


def _body(output: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "resp_synthetic_1",
        "object": "response",
        "created_at": 1_800_000_000,
        "status": "completed",
        "model": DEFAULT_MODEL,
        "output": [_message(_text("blue 3"))] if output is None else output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": USAGE,
    }
    body.update(overrides)
    return body


Reply = httpx2.Response | Exception


class Transport:
    """Records every HTTP request and replays queued replies (the last one repeats)."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies) or [self.ok()]
        self.requests: list[httpx2.Request] = []

    @staticmethod
    def ok(body: dict[str, Any] | None = None, request_id: str = "req_synthetic_1") -> Reply:
        return httpx2.Response(
            200, json=_body() if body is None else body, headers={"x-request-id": request_id}
        )

    @staticmethod
    def error(status: int, code: str | None = None, **extra: Any) -> Reply:
        error = {"message": f"SECRET-PROVIDER-MESSAGE {INPUT} {FAKE_KEY}", "type": "x", **extra}
        if code is not None:
            error["code"] = code
        return httpx2.Response(
            status, json={"error": error}, headers={"x-request-id": "req_err_1", **FAST}
        )

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        reply = self.replies[min(len(self.requests), len(self.replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def body(self, index: int = -1) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[index].content)
        return data


@pytest.fixture
def make_provider() -> Iterator[Callable[..., tuple[OpenAIProvider, Transport]]]:
    providers: list[OpenAIProvider] = []

    def factory(
        *replies: Reply, config: GenerationConfig = DEFAULT_GENERATION_CONFIG
    ) -> tuple[OpenAIProvider, Transport]:
        transport = Transport(*replies)
        client = httpx2.Client(transport=httpx2.MockTransport(transport))
        provider = OpenAIProvider(config, api_key=FAKE_KEY, http_client=client)
        providers.append(provider)
        return provider, transport

    yield factory
    for provider in providers:
        provider.close()


Factory = Callable[..., tuple[OpenAIProvider, Transport]]


# --- configuration ----------------------------------------------------------------------------


def test_default_configuration_and_identity() -> None:
    c = DEFAULT_GENERATION_CONFIG
    assert (c.model, c.reasoning_effort, c.max_output_tokens) == (DEFAULT_MODEL, "none", 1024)
    assert (c.timeout_seconds, c.max_retries) == (30.0, 1)
    assert c.model == "gpt-5.4-mini-2026-03-17"
    assert c.semantic_dict()["store"] is False
    assert c.semantic_dict()["tools"] == []
    assert c.config_id == GenerationConfig().config_id == "cbc4e415c848"
    # Operational settings are reported but are not part of the identity.
    assert GenerationConfig(timeout_seconds=5, max_retries=0).config_id == c.config_id
    assert c.operational_dict() == {"timeout_seconds_per_attempt": 30.0, "sdk_max_retries": 1}
    assert GenerationConfig(max_output_tokens=512).config_id != c.config_id
    assert GenerationConfig(reasoning_effort="low").config_id != c.config_id
    other = GenerationConfig(model="gpt-5.4-nano-2026-03-17", reasoning_effort="minimal")
    assert other.config_id != c.config_id  # another model is allowed deliberately


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"max_output_tokens": 15}, "max_output_tokens"),
        ({"max_output_tokens": 128_001}, "max_output_tokens"),
        ({"max_output_tokens": True}, "max_output_tokens"),
        ({"max_output_tokens": "1024"}, "max_output_tokens"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": -1.0}, "timeout_seconds"),
        ({"timeout_seconds": float("nan")}, "timeout_seconds"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
        ({"timeout_seconds": 601}, "timeout_seconds"),
        ({"timeout_seconds": True}, "timeout_seconds"),
        ({"max_retries": -1}, "max_retries"),
        ({"max_retries": 4}, "max_retries"),
        ({"max_retries": 1.0}, "max_retries"),
        ({"reasoning_effort": "max"}, "reasoning_effort"),
        ({"reasoning_effort": "minimal"}, "supports reasoning_effort"),
        ({"model": ""}, "model"),
        ({"model": "gpt 5"}, "model"),
        ({"model": None}, "model"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ge.GenerationConfigError, match=message):
        GenerationConfig(**kwargs)


def test_credentials_are_required_only_for_the_live_provider() -> None:
    with pytest.raises(ge.GenerationConfigError, match="OPENAI_API_KEY is not set"):
        OpenAIProvider(environ={})
    with pytest.raises(ge.GenerationConfigError, match="OPENAI_API_KEY"):
        OpenAIProvider(environ={"OPENAI_API_KEY": "   "})
    GenerationConfig()  # configuration and requests need no key
    GenerationRequest("i", "x")


def test_api_key_never_appears_in_identity_repr_or_errors(make_provider: Factory) -> None:
    provider, _ = make_provider(Transport.error(401, "invalid_api_key"))
    surfaces = [
        repr(provider),
        json.dumps(provider.config.semantic_dict()),
        json.dumps(provider.identity.to_dict()),
        repr(provider.config),
        repr(REQUEST),
    ]
    with pytest.raises(ge.AuthenticationError) as caught:
        provider.generate_text(REQUEST)
    surfaces.append(str(caught.value))
    assert not any(FAKE_KEY in s for s in surfaces)
    assert not any(INPUT in s or INSTRUCTIONS in s for s in surfaces)
    assert not hasattr(provider, "_api_key")
    assert "SECRET-PROVIDER-MESSAGE" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True  # raw SDK error not chained


# --- request construction ---------------------------------------------------------------------


def test_text_request_is_stateless_and_minimal(make_provider: Factory) -> None:
    provider, transport = make_provider()
    provider.generate_text(REQUEST)
    request = transport.requests[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/responses")
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert transport.body() == {
        "model": "gpt-5.4-mini-2026-03-17",
        "instructions": INSTRUCTIONS,
        "input": INPUT,
        "max_output_tokens": 1024,
        "reasoning": {"effort": "none"},
        "store": False,
        "stream": False,
    }  # no tools, background, previous_response_id, conversation or sampling parameters


def test_structured_request_uses_native_strict_json_schema(make_provider: Factory) -> None:
    provider, transport = make_provider(
        Transport.ok(_body([_message(_text('{"colour":"blue","count":3}'))]))
    )
    provider.generate_structured(REQUEST, SyntheticFact)
    text = transport.body()["text"]
    assert text == {
        "format": {
            "type": "json_schema",
            "name": "SyntheticFact",
            "schema": SyntheticFact.model_json_schema(),
            "strict": True,
        }
    }
    assert text["format"]["schema"]["additionalProperties"] is False
    assert text["format"]["schema"]["required"] == ["colour", "count"]


def test_configured_model_and_budget_are_sent(make_provider: Factory) -> None:
    config = GenerationConfig(model="gpt-5.4-nano-2026-03-17", max_output_tokens=64)
    reply = Transport.ok(_body(model="gpt-5.4-nano-2026-03-17"))
    provider, transport = make_provider(reply, config=config)
    result = provider.generate_text(REQUEST)
    assert transport.body()["model"] == "gpt-5.4-nano-2026-03-17"
    assert transport.body()["max_output_tokens"] == 64
    assert result.requested_model == "gpt-5.4-nano-2026-03-17"


class WithDefault(BaseModel):
    model_config = ConfigDict(extra="forbid")
    colour: str = "blue"


class Permissive(BaseModel):
    colour: str


class NestedPermissive(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inner: Permissive


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (WithDefault, "every property must be required"),
        (Permissive, "additionalProperties must be false"),
        (NestedPermissive, "additionalProperties must be false"),
        (dict, "pydantic BaseModel subclass"),
    ],
)
def test_non_strict_schemas_are_rejected_before_any_request(
    make_provider: Factory, schema: Any, message: str
) -> None:
    provider, transport = make_provider()
    with pytest.raises(ge.GenerationConfigError, match=message):
        provider.generate_structured(REQUEST, schema)
    assert transport.requests == []


def test_nested_strict_schema_is_accepted() -> None:
    fmt = strict_output_format(Outer)
    assert fmt["strict"] is True
    defs = fmt["schema"]["$defs"]
    assert isinstance(defs, dict)
    assert "Inner" in defs


# --- successful responses ---------------------------------------------------------------------


def test_text_result_is_normalized(make_provider: Factory) -> None:
    provider, _ = make_provider()
    result = provider.generate_text(REQUEST)
    assert (result.text, result.parsed, result.status) == ("blue 3", None, "completed")
    assert (result.requested_model, result.response_model) == (DEFAULT_MODEL, DEFAULT_MODEL)
    assert (result.response_id, result.request_id) == ("resp_synthetic_1", "req_synthetic_1")
    assert result.usage == TokenUsage(120, 100, 30, 0, 150)
    assert result.latency_ms >= 0
    assert result.provider.to_dict() == {
        "provider": "openai",
        "sdk_version": openai.__version__,
        "model": DEFAULT_MODEL,
        "config_id": DEFAULT_GENERATION_CONFIG.config_id,
    }
    assert openai.__version__ == "3.55.0"
    metadata = json.dumps(result.metadata())
    assert "blue 3" not in metadata
    assert '"total_tokens": 150' in metadata  # provider total; subsets not added again


def test_structured_result_is_parsed_and_typed(make_provider: Factory) -> None:
    provider, _ = make_provider(
        Transport.ok(_body([_message(_text('{"colour":"blue","count":3}'))]))
    )
    result = provider.generate_structured(REQUEST, SyntheticFact)
    assert result.parsed == SyntheticFact(colour="blue", count=3)
    assert isinstance(result.parsed, SyntheticFact)


def test_reasoning_items_and_commentary_are_not_the_answer(make_provider: Factory) -> None:
    output = [
        {"id": "rs_1", "type": "reasoning", "summary": []},
        _message(_text("thinking aloud"), phase="commentary"),
        _message(_text("final"), phase="final_answer"),
    ]
    provider, _ = make_provider(Transport.ok(_body(output)))
    assert provider.generate_text(REQUEST).text == "final"


@pytest.mark.parametrize("returned", ["gpt-5.4-mini", "gpt-5.4-mini-2026-09-01", ""])
def test_pinned_snapshot_mismatch_is_rejected_not_switched(
    make_provider: Factory, caplog: pytest.LogCaptureFixture, returned: str
) -> None:
    provider, transport = make_provider(Transport.ok(_body(model=returned)))
    with (
        caplog.at_level(logging.WARNING, logger="founderlens.generation"),
        pytest.raises(ge.UnexpectedResponseError, match="does not match the pinned snapshot"),
    ):
        provider.generate_text(REQUEST)
    assert len(transport.requests) == 1  # never retried, never another model
    assert transport.body()["model"] == DEFAULT_MODEL
    assert "category=unexpected_response" in caplog.text


def test_alias_records_the_resolved_model(make_provider: Factory) -> None:
    config = GenerationConfig(model="gpt-5.4-mini")
    provider, transport = make_provider(Transport.ok(_body(model=DEFAULT_MODEL)), config=config)
    result = provider.generate_text(REQUEST)
    assert transport.body()["model"] == "gpt-5.4-mini"
    assert (result.requested_model, result.response_model) == ("gpt-5.4-mini", DEFAULT_MODEL)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, TokenUsage()),
        (
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            TokenUsage(10, None, 5, None, 15),
        ),
    ],
)
def test_missing_usage_stays_unknown(
    make_provider: Factory, usage: dict[str, int] | None, expected: TokenUsage
) -> None:
    provider, _ = make_provider(Transport.ok(_body(usage=usage)))
    result = provider.generate_text(REQUEST)
    assert result.usage == expected
    assert result.usage.reported is (usage is not None)


# --- failures ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "error", "attempts"),
    [
        (Transport.error(401, "invalid_api_key"), ge.AuthenticationError, 1),
        (Transport.error(403, "unsupported_country"), ge.PermissionDeniedError, 1),
        (
            Transport.error(400, "invalid_value", param="reasoning.effort"),
            ge.InvalidRequestError,
            1,
        ),
        (Transport.error(404, "model_not_found"), ge.ModelUnavailableError, 1),
        (Transport.error(400, "model_not_found"), ge.ModelUnavailableError, 1),
        (Transport.error(429, "rate_limit_exceeded"), ge.RateLimitError, 2),
        (Transport.error(429, "insufficient_quota"), ge.QuotaExceededError, 2),
        (Transport.error(500), ge.ProviderServerError, 2),
        (Transport.error(503), ge.ProviderServerError, 2),
    ],
)
def test_status_errors_map_to_categories_with_sdk_retries_only(
    make_provider: Factory, reply: Reply, error: type[ge.GenerationError], attempts: int
) -> None:
    provider, transport = make_provider(reply)
    with pytest.raises(error) as caught:
        provider.generate_text(REQUEST)
    assert len(transport.requests) == attempts  # 1 + max_retries for retryable statuses only
    assert caught.value.request_id == "req_err_1"
    assert "SECRET-PROVIDER-MESSAGE" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_invalid_request_reports_the_parameter(make_provider: Factory) -> None:
    provider, _ = make_provider(Transport.error(400, "invalid_value", param="reasoning.effort"))
    with pytest.raises(ge.InvalidRequestError, match=r"param reasoning\.effort.*status 400"):
        provider.generate_text(REQUEST)


def test_retryable_error_then_success(make_provider: Factory) -> None:
    provider, transport = make_provider(Transport.error(503), Transport.ok())
    assert provider.generate_text(REQUEST).text == "blue 3"
    assert len(transport.requests) == 2


def test_no_retries_when_disabled(make_provider: Factory) -> None:
    provider, transport = make_provider(
        Transport.error(500), config=GenerationConfig(max_retries=0)
    )
    with pytest.raises(ge.ProviderServerError):
        provider.generate_text(REQUEST)
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    ("exception", "error"),
    [
        (httpx2.ReadTimeout("synthetic timeout"), ge.ProviderTimeoutError),
        (httpx2.ConnectError("synthetic connection failure"), ge.ProviderConnectionError),
    ],
)
def test_timeouts_and_connection_failures(
    make_provider: Factory, exception: Exception, error: type[ge.GenerationError]
) -> None:
    provider, transport = make_provider(exception)
    with pytest.raises(error):
        provider.generate_text(REQUEST)
    assert len(transport.requests) == 2  # the original attempt plus one SDK retry


def test_refusal_is_its_own_outcome(make_provider: Factory) -> None:
    refusal = {"type": "refusal", "refusal": "SYNTH-REFUSAL-TEXT"}
    provider, _ = make_provider(Transport.ok(_body([_message(refusal)])))
    with pytest.raises(ge.ModelRefusalError) as caught:
        provider.generate_structured(REQUEST, SyntheticFact)
    assert caught.value.refusal == "SYNTH-REFUSAL-TEXT"
    assert "SYNTH-REFUSAL-TEXT" not in str(caught.value)
    assert caught.value.request_id == "req_synthetic_1"


@pytest.mark.parametrize(
    ("reason", "exhausted"), [("max_output_tokens", True), ("content_filter", False)]
)
def test_incomplete_generation_is_detected_before_parsing(
    make_provider: Factory, reason: str, exhausted: bool
) -> None:
    truncated = _body(
        [_message(_text('{"colour":"bl'))],
        status="incomplete",
        incomplete_details={"reason": reason},
    )
    provider, _ = make_provider(Transport.ok(truncated))
    with pytest.raises(ge.IncompleteGenerationError) as caught:
        provider.generate_structured(REQUEST, SyntheticFact)  # not a schema error
    assert caught.value.reason == reason
    assert caught.value.output_tokens_exhausted is exhausted


@pytest.mark.parametrize(
    "text", ['{"colour": 1, "count": "x"}', '{"colour":"blue"}', "{not json", "blue 3"]
)
def test_invalid_structured_output_fails_without_fallback(
    make_provider: Factory, text: str
) -> None:
    provider, _ = make_provider(Transport.ok(_body([_message(_text(text))])))
    with pytest.raises(ge.StructuredOutputError, match="SyntheticFact") as caught:
        provider.generate_structured(REQUEST, SyntheticFact)
    assert text not in str(caught.value)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (_body([]), "no output text"),
        (_body([_message(_text("  "))]), "no output text"),
        (
            _body([{"id": "ws_1", "type": "web_search_call", "status": "completed"}]),
            "web_search_call",
        ),
        (_body(status="in_progress"), "unexpected response status"),
    ],
)
def test_unexpected_responses_are_errors(
    make_provider: Factory, body: dict[str, Any], message: str
) -> None:
    provider, _ = make_provider(Transport.ok(body))
    with pytest.raises(ge.UnexpectedResponseError, match=message):
        provider.generate_text(REQUEST)


def test_failed_response_is_a_provider_error(make_provider: Factory) -> None:
    body = _body([], status="failed", error={"code": "server_error", "message": "x"})
    provider, _ = make_provider(Transport.ok(body))
    with pytest.raises(ge.ProviderServerError, match="code server_error"):
        provider.generate_text(REQUEST)


# --- lifecycle, logging and imports -----------------------------------------------------------


def test_one_client_is_reused_and_closed(
    make_provider: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[int] = []
    original = openai.OpenAI

    class Counting(original):  # type: ignore[misc, valid-type]  # subclassing the SDK client to count constructions
        def __init__(self, **kwargs: Any) -> None:
            built.append(1)
            super().__init__(**kwargs)

    monkeypatch.setattr(openai, "OpenAI", Counting)
    transport = Transport()
    client = httpx2.Client(transport=httpx2.MockTransport(transport))
    with OpenAIProvider(api_key=FAKE_KEY, http_client=client) as provider:
        for _ in range(3):
            provider.generate_text(REQUEST)
    assert built == [1]
    assert len(transport.requests) == 3
    assert provider.closed
    assert client.is_closed
    provider.close()  # idempotent
    with pytest.raises(ge.GenerationConfigError, match="closed"):
        provider.generate_text(REQUEST)


def test_logs_contain_metadata_but_no_content(
    make_provider: Factory, caplog: pytest.LogCaptureFixture
) -> None:
    provider, _ = make_provider(Transport.ok(), Transport.error(401, "invalid_api_key"))
    with caplog.at_level(logging.DEBUG):
        provider.generate_text(REQUEST)
        with pytest.raises(ge.AuthenticationError):
            provider.generate_text(REQUEST)
    ours = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("founderlens"))
    assert DEFAULT_GENERATION_CONFIG.config_id in ours
    assert "total_tokens=150" in ours
    assert "category=authentication" in ours
    everything = caplog.text
    for secret in (INSTRUCTIONS, INPUT, "blue 3", FAKE_KEY, "SECRET-PROVIDER-MESSAGE"):
        assert secret not in everything


def test_imports_need_no_key_and_retrieval_never_loads_the_provider() -> None:
    script = textwrap.dedent(
        """
        import socket, sys
        def blocked(*args, **kwargs):
            raise AssertionError("network access attempted")
        socket.socket.connect = blocked
        socket.create_connection = blocked
        import founderlens.__main__
        import founderlens.retrieval.policy
        assert "openai" not in sys.modules, "retrieval imported the OpenAI SDK"
        assert not any(m.startswith("founderlens.generation") for m in sys.modules)
        import founderlens.generation.openai_provider as op
        from founderlens.generation.config import GenerationConfig
        GenerationConfig()
        try:
            op.OpenAIProvider()
        except op.ge.GenerationConfigError as exc:
            assert "OPENAI_API_KEY" in str(exc)
        else:
            raise AssertionError("provider built without a key")
        print("ok")
        """
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


# --- focused security and structure checks (final 5.1 verification) ---------------------------

INJECTION = "x\nINJECTED log line 'quoted' " + FAKE_KEY


def test_safe_token_neutralizes_untrusted_identifiers() -> None:
    assert ge.safe_token("req_abc-1.2:3/4") == "req_abc-1.2:3/4"
    assert ge.safe_token("a\nb c'd") == "a?b?c?d"
    assert ge.safe_token(None) is None
    assert ge.safe_token("code " + FAKE_KEY) == "code?sk-REDACTED"
    assert ge.safe_token("sk-proj-abcdefgh12345678") == "sk-REDACTED"
    assert len(ge.safe_token("x" * 500) or "") == ge.MAX_TOKEN_LENGTH


def test_untrusted_provider_fields_cannot_inject_into_errors_or_logs(
    make_provider: Factory, caplog: pytest.LogCaptureFixture
) -> None:
    fields = {"message": INJECTION, "type": INJECTION, "code": INJECTION, "param": INJECTION}
    reply = httpx2.Response(
        400, json={"error": fields}, headers={"x-request-id": "req with spaces 'and quotes'"}
    )
    provider, _ = make_provider(reply)
    with caplog.at_level(logging.DEBUG), pytest.raises(ge.InvalidRequestError) as caught:
        provider.generate_text(REQUEST)
    message = str(caught.value)
    assert "\n" not in message
    assert "INJECTED log line" not in message
    assert FAKE_KEY not in message
    assert caught.value.request_id == "req?with?spaces??and?quotes?"
    ours = [r.getMessage() for r in caplog.records if r.name.startswith("founderlens")]
    assert ours
    assert all("\n" not in line and "INJECTED log line" not in line for line in ours)
    assert FAKE_KEY not in caplog.text


def test_untrusted_response_identifiers_are_sanitized_on_success(
    make_provider: Factory, caplog: pytest.LogCaptureFixture
) -> None:
    provider, _ = make_provider(Transport.ok(_body(id="resp_1\nINJECTED")))
    with caplog.at_level(logging.INFO, logger="founderlens.generation"):
        result = provider.generate_text(REQUEST)
    assert result.response_id == "resp_1?INJECTED"
    assert all("\n" not in r.getMessage() for r in caplog.records)


def test_refusal_hidden_in_commentary_is_still_a_refusal(make_provider: Factory) -> None:
    output = [
        _message({"type": "refusal", "refusal": "SYNTH-REFUSAL"}, phase="commentary"),
        _message(_text('{"colour":"blue","count":3}'), phase="final_answer"),
    ]
    provider, _ = make_provider(Transport.ok(_body(output)))
    with pytest.raises(ge.ModelRefusalError):
        provider.generate_structured(REQUEST, SyntheticFact)


UNKNOWN_PART = {"type": "audio_blob", "data": "x"}


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            [_message(UNKNOWN_PART, phase="commentary"), _message(_text("ok"))],
            "unexpected message content type audio_blob",
        ),
        (
            [{**_message(_text("partial")), "status": "incomplete"}],
            "output message status incomplete",
        ),
        ([{"id": "rs_1", "type": "reasoning", "summary": []}], "no output text"),
    ],
)
def test_skipped_items_cannot_hide_failures(
    make_provider: Factory, output: list[dict[str, Any]], message: str
) -> None:
    provider, _ = make_provider(Transport.ok(_body(output)))
    with pytest.raises(ge.UnexpectedResponseError, match=message):
        provider.generate_text(REQUEST)


def test_nested_schema_with_definitions_round_trips(make_provider: Factory) -> None:
    payload = '{"items":[{"label":"a"},{"label":"b"}],"note":null}'
    provider, transport = make_provider(Transport.ok(_body([_message(_text(payload))])))
    result = provider.generate_structured(REQUEST, Outer)
    assert result.parsed == Outer(items=[Inner(label="a"), Inner(label="b")], note=None)
    schema = transport.body()["text"]["format"]["schema"]
    assert schema["properties"]["items"]["items"] == {"$ref": "#/$defs/Inner"}
    assert schema["$defs"]["Inner"]["additionalProperties"] is False
    bad = '{"items":[{"label":"a","extra":1}],"note":null}'
    provider, _ = make_provider(Transport.ok(_body([_message(_text(bad))])))
    with pytest.raises(ge.StructuredOutputError) as caught:
        provider.generate_structured(REQUEST, Outer)
    assert bad not in str(caught.value)
    assert "extra" not in str(caught.value)


class OpenMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    counts: dict[str, int]


class InnerWithDefault(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = "x"


class OuterWithNestedDefault(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inner: InnerWithDefault


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (OpenMapping, r"\$\.properties\.counts: additionalProperties must be false"),
        (OuterWithNestedDefault, r"\$\.\$defs\.InnerWithDefault: every property must be required"),
    ],
)
def test_open_ended_and_nested_default_schemas_are_rejected(schema: Any, message: str) -> None:
    with pytest.raises(ge.GenerationConfigError, match=message):
        strict_output_format(schema)


def test_invalid_output_is_not_logged(
    make_provider: Factory, caplog: pytest.LogCaptureFixture
) -> None:
    raw = '{"colour": "SYNTH-INVALID-OUTPUT", "count": "not-a-number"}'
    provider, _ = make_provider(Transport.ok(_body([_message(_text(raw))])))
    with caplog.at_level(logging.DEBUG), pytest.raises(ge.StructuredOutputError) as caught:
        provider.generate_structured(REQUEST, SyntheticFact)
    assert "SYNTH-INVALID-OUTPUT" not in caplog.text
    assert "SYNTH-INVALID-OUTPUT" not in str(caught.value)
    assert caught.value.__cause__ is None
