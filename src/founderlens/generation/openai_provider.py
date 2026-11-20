"""OpenAI Responses API adapter (Commit 5.1).

One synchronous ``openai.OpenAI`` client per provider instance, created
explicitly (never at import), reused for every request and closed with
:meth:`OpenAIProvider.close` or a ``with`` block. Each request is stateless
and non-streaming: ``store=False``, no ``previous_response_id``, no
conversation, no background mode, no tools, and no temperature, top-p or seed.
The configured model is always sent; there is no model fallback.

**Structured Outputs.** :meth:`OpenAIProvider.generate_structured` sends the
API's native strict ``json_schema`` text format built from a Pydantic model.
It uses ``responses.create`` rather than the SDK's ``responses.parse`` helper
because the helper validates the output text before the response status can be
inspected. With that helper, a truncated (``incomplete``) response would surface
as a schema error. Here the status, refusals and output items are checked
first. Only a completed, non-refused output is validated with
``schema.model_validate_json``. There is no fallback to unconstrained JSON and
no repair of malformed output. Schemas must be strict-mode compatible
(``extra="forbid"``, every field required, no defaults); others are rejected
before any request.

**Timeouts and retries.** The SDK applies ``timeout_seconds`` per attempt and
retries up to ``max_retries`` times (default 1). It retries connection errors,
timeouts and HTTP 408, 409, 429 and 5xx with exponential backoff, and never
retries other 4xx such as 400, 401, 403 or 404. There is no application retry
loop. A per-attempt timeout is not a total deadline: two attempts plus
backoff can exceed it. A timed-out or retried request may still have been
processed and billed. A 429 for exhausted quota (``insufficient_quota``) is
also retried once by the SDK, unless the server says not to. End-to-end
deadlines and public rate limits belong to the serving layer.

**Model identity.** The configured model is always the requested model. For a
dated snapshot the response must report exactly that snapshot; any other
returned model raises ``UnexpectedResponseError`` and the output is discarded.
For an alias the resolved model is recorded as reported.

**Logging.** Only metadata is logged (provider, model, configuration ID,
status, latency, token usage, response and request IDs, error category).
Provider-controlled identifiers pass through ``errors.safe_token``, so they
cannot add log lines. Prompts, inputs, generated text, refusal text and raw
invalid output are never logged or put in an error. The API key is never kept
on this object, logged or put in an error. The SDK client does hold it in
memory to authenticate requests. Separate production safeguards are still
needed:
- the SDK's own debug logging (``OPENAI_LOG``) and HTTP-library debug logs
  can record request details;
- error trackers that capture traceback locals can see request objects and
  the constructor's ``api_key`` argument.

``store=False``
stops OpenAI keeping the response as application state. It does not by itself
remove OpenAI's abuse-monitoring retention (up to 30 days unless the account
has approved zero data retention).
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import re
import time
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self, TypeVar

import httpx2
import openai
from openai.types.responses import (
    Response,
    ResponseFormatTextJSONSchemaConfigParam,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextConfigParam,
)
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, ValidationError

from founderlens.generation import errors as ge
from founderlens.generation.config import DEFAULT_GENERATION_CONFIG, PROVIDER, GenerationConfig
from founderlens.generation.schema import (
    GenerationRequest,
    GenerationResult,
    ProviderIdentity,
    SchemaT,
    TokenUsage,
)

API_KEY_ENV_VAR = "OPENAI_API_KEY"
SDK_DISTRIBUTION = "openai"
logger = logging.getLogger(__name__)
_ParsedT = TypeVar("_ParsedT")

# JSON Schema keywords Structured Outputs does not support in strict mode, plus
# "default" (a field with a default is optional, and strict mode needs every field).
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {"allOf", "not", "if", "then", "else", "dependentRequired", "dependentSchemas", "default"}
)
_SCHEMA_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
# A dated snapshot such as gpt-5.4-mini-2026-03-17 (as opposed to an alias such as gpt-5.4-mini).
_SNAPSHOT = re.compile(r".+-\d{4}-\d{2}-\d{2}")


def sdk_version() -> str:
    return importlib.metadata.version(SDK_DISTRIBUTION)


# --------------------------------------------------------------------------- #
# Structured output format
# --------------------------------------------------------------------------- #


def _schema_problems(node: object, path: str, problems: list[str]) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _schema_problems(item, f"{path}[{index}]", problems)
        return
    if not isinstance(node, dict):
        return
    for key in sorted(_UNSUPPORTED_SCHEMA_KEYWORDS & set(node)):
        problems.append(f"{path}: keyword {key!r} is not supported in strict mode")
    if "$ref" in node and len(node) > 1:
        # The SDK's own strict helper unrolls these because the API rejects them.
        problems.append(
            f"{path}: $ref must not have sibling keys (move descriptions to the target)"
        )
    properties = node.get("properties")
    if node.get("type") == "object" or isinstance(properties, dict):
        names = sorted(properties) if isinstance(properties, dict) else []
        if node.get("additionalProperties") is not False:
            problems.append(f"{path}: additionalProperties must be false (extra='forbid')")
        if sorted(node.get("required", [])) != names:
            problems.append(f"{path}: every property must be required (no defaults)")
    for key, value in node.items():
        if key in {"properties", "$defs"} and isinstance(value, dict):
            for name, sub in value.items():
                _schema_problems(sub, f"{path}.{key}.{name}", problems)
        elif isinstance(value, dict | list):
            _schema_problems(value, f"{path}.{key}", problems)


def strict_output_format(schema: type[BaseModel]) -> ResponseFormatTextJSONSchemaConfigParam:
    """The native ``json_schema`` text format for ``schema`` in strict mode.

    Raises :class:`GenerationConfigError` for a schema strict mode cannot enforce
    exactly, instead of silently transforming it.
    """
    candidate: object = schema  # runtime check: callers may pass anything
    if not (isinstance(candidate, type) and issubclass(candidate, BaseModel)):
        raise ge.GenerationConfigError("schema must be a pydantic BaseModel subclass")
    name = schema.__name__
    json_schema = schema.model_json_schema()
    problems: list[str] = []
    if not _SCHEMA_NAME.fullmatch(name):
        problems.append(f"schema name {name!r} must match [A-Za-z0-9_-]{{1,64}}")
    if json_schema.get("type") != "object":
        problems.append("the root of the schema must be an object")
    _schema_problems(json_schema, "$", problems)
    if problems:
        raise ge.GenerationConfigError(
            f"schema {name} is not strict-mode compatible: " + "; ".join(problems)
        )
    return {"type": "json_schema", "name": name, "schema": json_schema, "strict": True}


# --------------------------------------------------------------------------- #
# Error and response normalization
# --------------------------------------------------------------------------- #


def map_provider_error(exc: openai.OpenAIError) -> ge.GenerationError:
    """A sanitized :class:`GenerationError` for an SDK exception (no message, no headers)."""
    if isinstance(exc, openai.APITimeoutError):
        return ge.ProviderTimeoutError("the request timed out on every attempt")
    if isinstance(exc, openai.APIConnectionError):
        return ge.ProviderConnectionError("could not connect to the provider")
    if isinstance(exc, openai.APIResponseValidationError):
        return ge.UnexpectedResponseError(
            "the provider response did not match the SDK schema",
            status_code=exc.status_code,
            request_id=exc.response.headers.get("x-request-id"),
        )
    if not isinstance(exc, openai.APIStatusError):
        return ge.UnexpectedResponseError(f"provider client error ({type(exc).__name__})")
    code = exc.code or exc.type
    meta: dict[str, Any] = {
        "request_id": exc.request_id,
        "status_code": exc.status_code,
        "provider_code": code,
    }
    if isinstance(exc, openai.AuthenticationError):
        return ge.AuthenticationError(f"authentication failed; check {API_KEY_ENV_VAR}", **meta)
    if isinstance(exc, openai.PermissionDeniedError):
        return ge.PermissionDeniedError("the API key may not use this resource or model", **meta)
    if isinstance(exc, openai.NotFoundError) or exc.code == "model_not_found":
        return ge.ModelUnavailableError(
            "the configured model is not available to this account; no other model is tried",
            **meta,
        )
    if isinstance(exc, openai.RateLimitError):
        if exc.code == "insufficient_quota":
            return ge.QuotaExceededError("quota or billing limit exhausted", **meta)
        return ge.RateLimitError("rate limited by the provider", **meta)
    if isinstance(exc, openai.BadRequestError | openai.UnprocessableEntityError):
        param = f" (param {ge.safe_token(exc.param)})" if exc.param else ""
        return ge.InvalidRequestError(f"the provider rejected the request{param}", **meta)
    if exc.status_code >= 500:
        return ge.ProviderServerError("the provider returned a server error", **meta)
    return ge.UnexpectedResponseError("unexpected provider status", **meta)


def normalize_usage(response: Response) -> TokenUsage:
    """Provider-reported usage; fields the provider omitted stay ``None``."""
    usage = response.usage
    if usage is None:
        return TokenUsage()

    def count(obj: object, name: str) -> int | None:
        value = getattr(obj, name, None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return TokenUsage(
        input_tokens=count(usage, "input_tokens"),
        cached_input_tokens=count(getattr(usage, "input_tokens_details", None), "cached_tokens"),
        output_tokens=count(usage, "output_tokens"),
        reasoning_tokens=count(getattr(usage, "output_tokens_details", None), "reasoning_tokens"),
        total_tokens=count(usage, "total_tokens"),
    )


def check_model_identity(requested: str, returned: object, request_id: str | None) -> str:
    """The returned model name, or an error if it breaks the pinned-snapshot contract.

    When the configured model is a dated snapshot, the response must report exactly
    that snapshot; anything else raises :class:`UnexpectedResponseError` and the
    output is not used. For an alias the returned (resolved) model is recorded as
    reported. The provider never requests another model.
    """
    name = ge.safe_token(returned) or ""
    if _SNAPSHOT.fullmatch(requested) and name != requested:
        raise ge.UnexpectedResponseError(
            f"returned model {name or '(none)'} does not match the pinned snapshot {requested}",
            request_id=request_id,
        )
    return name


def output_text(response: Response, request_id: str | None) -> str:
    """The final output text of a completed, non-refused response, or a typed error."""
    status: str | None = response.status
    if status == "incomplete":
        details = response.incomplete_details
        reason = details.reason if details is not None else None
        raise ge.IncompleteGenerationError(
            "the response is incomplete"
            + (" (output-token budget exhausted)" if reason == "max_output_tokens" else ""),
            reason=reason,
            request_id=request_id,
        )
    if status == "failed":
        code = response.error.code if response.error is not None else None
        raise ge.ProviderServerError(
            "the provider marked the response as failed", provider_code=code, request_id=request_id
        )
    if status != "completed":
        raise ge.UnexpectedResponseError(
            f"unexpected response status {ge.safe_token(status)}", request_id=request_id
        )
    texts: list[str] = []
    refusals: list[str] = []
    items: list[object] = list(response.output)  # widened: the SDK may construct unknown types
    for item in items:
        # The SDK builds unknown discriminator values as a known class, so the
        # ``type`` field is checked as well as the class.
        item_type: object = getattr(item, "type", None)
        if isinstance(item, ResponseReasoningItem) and item_type == "reasoning":
            continue
        if not (isinstance(item, ResponseOutputMessage) and item_type == "message"):
            kind = ge.safe_token(item_type if item_type is not None else type(item).__name__)
            raise ge.UnexpectedResponseError(
                f"unexpected output item type {kind}", request_id=request_id
            )
        item_status: str | None = item.status
        if item_status != "completed":
            raise ge.UnexpectedResponseError(
                f"output message status {ge.safe_token(item_status)} in a completed response",
                request_id=request_id,
            )
        # Every message is checked for refusals and unknown content, including
        # commentary, so a skipped item cannot hide a refusal or a failure. Only
        # final-answer (or unphased) text is the answer.
        is_answer = item.phase in (None, "final_answer")
        parts: list[object] = list(item.content)
        for part in parts:
            part_type: object = getattr(part, "type", None)
            if isinstance(part, ResponseOutputRefusal) and part_type == "refusal":
                refusals.append(part.refusal)
            elif isinstance(part, ResponseOutputText) and part_type == "output_text":
                if is_answer:
                    texts.append(part.text)
            else:
                kind = ge.safe_token(part_type if part_type is not None else type(part).__name__)
                raise ge.UnexpectedResponseError(
                    f"unexpected message content type {kind}", request_id=request_id
                )
    if refusals:
        raise ge.ModelRefusalError(
            "the model refused the request", refusal="\n".join(refusals), request_id=request_id
        )
    text = "".join(texts)
    if not text.strip():
        raise ge.UnexpectedResponseError(
            "the completed response contains no output text", request_id=request_id
        )
    return text


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class OpenAIProvider:
    """The live OpenAI provider. Needs ``OPENAI_API_KEY`` (or ``api_key``) when constructed."""

    def __init__(
        self,
        config: GenerationConfig = DEFAULT_GENERATION_CONFIG,
        *,
        api_key: str | None = None,
        environ: Mapping[str, str] | None = None,
        http_client: httpx2.Client | None = None,
    ) -> None:
        if not isinstance(config, GenerationConfig):
            raise ge.GenerationConfigError("config must be a GenerationConfig")
        env = os.environ if environ is None else environ
        key = api_key if api_key is not None else env.get(API_KEY_ENV_VAR)
        if key is None or not key.strip():
            raise ge.GenerationConfigError(
                f"{API_KEY_ENV_VAR} is not set; it is needed only for live generation "
                "(see .env.example)"
            )
        self._config = config
        self._identity = ProviderIdentity(PROVIDER, sdk_version(), config.model, config.config_id)
        # The key goes to the SDK client only; this object never keeps it.
        self._client = openai.OpenAI(
            api_key=key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            http_client=http_client,
        )
        self._closed = False

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self._config.model!r}, config_id={self._config.config_id})"

    @property
    def config(self) -> GenerationConfig:
        return self._config

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release the HTTP connection pool. Safe to call twice."""
        if not self._closed:
            self._client.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def generate_text(self, request: GenerationRequest) -> GenerationResult[None]:
        """Plain text generation."""
        response, request_id, latency_ms = self._create(request, None)
        text = self._checked_text(response, request_id, latency_ms)
        return self._result(response, request_id, latency_ms, text, None)

    def generate_structured(
        self, request: GenerationRequest, schema: type[SchemaT]
    ) -> GenerationResult[SchemaT]:
        """Schema-constrained generation, validated against ``schema``."""
        output_format = strict_output_format(schema)
        response, request_id, latency_ms = self._create(request, output_format)
        text = self._checked_text(response, request_id, latency_ms)
        try:
            parsed = schema.model_validate_json(text)
        except ValidationError as exc:
            error = ge.StructuredOutputError(
                f"output does not match schema {schema.__name__} "
                f"({exc.error_count()} validation error(s))",
                request_id=request_id,
            )
            self._log_failure(error, latency_ms)
            raise error from None
        return self._result(response, request_id, latency_ms, text, parsed)

    # ----------------------------------------------------------------------- #

    def _create(
        self,
        request: GenerationRequest,
        output_format: ResponseFormatTextJSONSchemaConfigParam | None,
    ) -> tuple[Response, str | None, float]:
        if self._closed:
            raise ge.GenerationConfigError("the provider is closed")
        if not isinstance(request, GenerationRequest):
            raise ge.GenerationConfigError("request must be a GenerationRequest")
        config = self._config
        reasoning: Reasoning = {"effort": config.reasoning_effort}
        text: ResponseTextConfigParam | openai.Omit = (
            {"format": output_format} if output_format is not None else openai.omit
        )
        started = time.perf_counter()
        try:
            raw = self._client.responses.with_raw_response.create(
                model=config.model,
                instructions=request.instructions,
                input=request.input_text,
                max_output_tokens=config.max_output_tokens,
                reasoning=reasoning,
                store=False,
                stream=False,
                text=text,
            )
            response = raw.parse()
        except openai.OpenAIError as exc:
            error = map_provider_error(exc)
            self._log_failure(error, (time.perf_counter() - started) * 1000)
            raise error from None
        return response, ge.safe_token(raw.request_id), (time.perf_counter() - started) * 1000

    def _checked_text(self, response: Response, request_id: str | None, latency_ms: float) -> str:
        try:
            check_model_identity(self._config.model, response.model, request_id)
            return output_text(response, request_id)
        except ge.GenerationError as error:
            self._log_failure(error, latency_ms, response_id=response.id)
            raise

    def _result(
        self,
        response: Response,
        request_id: str | None,
        latency_ms: float,
        text: str,
        parsed: _ParsedT,
    ) -> GenerationResult[_ParsedT]:
        result = GenerationResult(
            text=text,
            parsed=parsed,
            status="completed",
            requested_model=self._config.model,
            response_model=check_model_identity(self._config.model, response.model, request_id),
            response_id=ge.safe_token(response.id) or "",
            request_id=request_id,
            usage=normalize_usage(response),
            latency_ms=round(latency_ms, 3),
            provider=self._identity,
        )
        usage = result.usage
        logger.info(
            "generation completed provider=%s model=%s config_id=%s response_model=%s "
            "latency_ms=%.1f input_tokens=%s cached_input_tokens=%s output_tokens=%s "
            "reasoning_tokens=%s total_tokens=%s response_id=%s request_id=%s",
            PROVIDER,
            self._config.model,
            self._config.config_id,
            result.response_model,
            latency_ms,
            usage.input_tokens,
            usage.cached_input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            usage.total_tokens,
            result.response_id,
            request_id,
        )
        return result

    def _log_failure(
        self, error: ge.GenerationError, latency_ms: float, *, response_id: str | None = None
    ) -> None:
        logger.warning(
            "generation failed provider=%s model=%s config_id=%s category=%s status_code=%s "
            "provider_code=%s latency_ms=%.1f response_id=%s request_id=%s",
            PROVIDER,
            self._config.model,
            self._config.config_id,
            error.category,
            error.status_code,
            error.provider_code,
            latency_ms,
            ge.safe_token(response_id),
            error.request_id,
        )
