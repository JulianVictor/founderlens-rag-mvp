"""Sanitized generation errors (Commit 5.1).

Every provider failure becomes one :class:`GenerationError` subclass with a
stable ``category``. Messages carry only sanitized metadata: the category,
HTTP status, the provider's error ``code`` / ``type`` / ``param`` and the
request ID. They never contain API keys, request headers, prompts, document
text, generated text or the provider's raw exception, which is not chained
(``raise ... from None``), so tracebacks do not print it either.

Provider refusals, incomplete generations and invalid structured output are
model outcomes, not transport failures, and are kept as separate categories.
None of them means "insufficient financial evidence": answerability and
abstention belong to the later grounded generation service.
"""

from __future__ import annotations

import re
from typing import ClassVar

MAX_TOKEN_LENGTH = 128
_UNSAFE_TOKEN_CHARACTERS = re.compile(r"[^A-Za-z0-9._:/-]")
# OpenAI-style secret keys (sk-..., sk-proj-...), in case a provider field echoes one.
_SECRET_KEY = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


def safe_token(value: object) -> str | None:
    """A provider-controlled identifier made safe for messages and logs.

    Request IDs, error codes, parameter names, model names, response IDs and
    item types come from the provider and are untrusted. Every character outside
    ``[A-Za-z0-9._:/-]`` (including newlines, spaces and quotes) becomes ``?``,
    anything shaped like an OpenAI secret key becomes ``sk-REDACTED``, and the
    result is cut to 128 characters. It therefore cannot add log lines, and
    cannot echo a key that a provider field reflects back.
    """
    if value is None:
        return None
    text = _SECRET_KEY.sub("sk-REDACTED", str(value))
    return _UNSAFE_TOKEN_CHARACTERS.sub("?", text)[:MAX_TOKEN_LENGTH]


class GenerationError(RuntimeError):
    """Base class. ``str(error)`` is safe to show and log."""

    category: ClassVar[str] = "generation_error"

    def __init__(
        self,
        detail: str,
        *,
        request_id: str | None = None,
        status_code: int | None = None,
        provider_code: str | None = None,
    ) -> None:
        # ``detail`` is written by this package; provider values in it go through
        # safe_token at the call site. Provider-supplied fields are sanitized here.
        self.detail = detail
        self.request_id = safe_token(request_id)
        self.status_code = status_code if isinstance(status_code, int) else None
        self.provider_code = safe_token(provider_code)
        request_id, provider_code = self.request_id, self.provider_code
        parts = [f"[{self.category}] {detail}"]
        if status_code is not None:
            parts.append(f"status {status_code}")
        if provider_code is not None:
            parts.append(f"code {provider_code}")
        if request_id is not None:
            parts.append(f"request_id {request_id}")
        super().__init__("; ".join(parts))


class GenerationConfigError(GenerationError, ValueError):
    """Invalid generation configuration or missing credentials (never sent to the provider)."""

    category = "configuration"


class AuthenticationError(GenerationError):
    category = "authentication"


class PermissionDeniedError(GenerationError):
    category = "permission_denied"


class InvalidRequestError(GenerationError):
    """The provider rejected the request (for example an unsupported parameter or schema)."""

    category = "invalid_request"


class ModelUnavailableError(GenerationError):
    """The configured model does not exist or is not available to this account."""

    category = "model_unavailable"


class RateLimitError(GenerationError):
    category = "rate_limited"


class QuotaExceededError(GenerationError):
    """Quota or billing exhausted (``insufficient_quota``); waiting will not help."""

    category = "quota_exceeded"


class ProviderTimeoutError(GenerationError):
    """Every attempt timed out. The provider may still have done (and billed) the work."""

    category = "timeout"


class ProviderConnectionError(GenerationError):
    category = "connection"


class ProviderServerError(GenerationError):
    """5xx responses, or a response the provider marked ``failed``."""

    category = "provider_server_error"


class ModelRefusalError(GenerationError):
    """The model returned a refusal. ``refusal`` holds its text for the caller; never logged."""

    category = "refusal"

    def __init__(self, detail: str, *, refusal: str, request_id: str | None = None) -> None:
        super().__init__(detail, request_id=request_id)
        self.refusal = refusal


class IncompleteGenerationError(GenerationError):
    """``status == "incomplete"``; ``reason`` is the provider's reason, if any."""

    category = "incomplete"

    def __init__(self, detail: str, *, reason: str | None, request_id: str | None = None) -> None:
        super().__init__(detail, request_id=request_id, provider_code=reason)
        self.reason = safe_token(reason)

    @property
    def output_tokens_exhausted(self) -> bool:
        return self.reason == "max_output_tokens"


class StructuredOutputError(GenerationError):
    """Completed output does not validate against the requested schema."""

    category = "structured_output_invalid"


class UnexpectedResponseError(GenerationError):
    """Empty output, unknown output item types, unexpected status, an unparseable response,
    or a returned model that differs from a pinned snapshot."""

    category = "unexpected_response"
