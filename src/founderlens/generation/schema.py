"""Provider-neutral generation types and the provider protocol (Commit 5.1).

These types know nothing about retrieval: a request is trusted application
instructions plus input text, and a result is the normalized provider
outcome. Valid structured output only means the output matches the schema;
it says nothing about factual correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from founderlens.generation.errors import GenerationConfigError

SchemaT = TypeVar("SchemaT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """``instructions`` are trusted application instructions; ``input_text`` is the user input.

    ``repr`` shows lengths only, so a request can be logged without its content.
    """

    instructions: str = field(repr=False)
    input_text: str = field(repr=False)

    def __post_init__(self) -> None:
        instructions: object = self.instructions
        input_text: object = self.input_text
        if not isinstance(instructions, str) or not instructions.strip():
            raise GenerationConfigError("instructions must be a non-empty string")
        if not isinstance(input_text, str) or not input_text.strip():
            raise GenerationConfigError("input_text must be a non-empty string")

    def __repr__(self) -> str:
        return (
            f"GenerationRequest(instructions=<{len(self.instructions)} chars>, "
            f"input_text=<{len(self.input_text)} chars>)"
        )


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-reported usage. ``None`` means not reported, never an estimate.

    ``cached_input_tokens`` is a subset of ``input_tokens`` and ``reasoning_tokens``
    a subset of ``output_tokens``; ``total_tokens`` is the provider's own total
    (input + output). Subsets are never added again.
    """

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def reported(self) -> bool:
        return self.total_tokens is not None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Who generated: provider, SDK version, configured model and configuration ID."""

    provider: str
    sdk_version: str
    model: str
    config_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "sdk_version": self.sdk_version,
            "model": self.model,
            "config_id": self.config_id,
        }


@dataclass(frozen=True, slots=True)
class GenerationResult[ParsedT]:
    """A completed, non-refused generation. ``parsed`` is ``None`` for text requests."""

    text: str = field(repr=False)
    parsed: ParsedT = field(repr=False)
    status: str
    requested_model: str
    response_model: str
    response_id: str
    request_id: str | None
    usage: TokenUsage
    latency_ms: float
    provider: ProviderIdentity

    def metadata(self) -> dict[str, Any]:
        """Everything except generated content; safe to log or store."""
        return {
            **self.provider.to_dict(),
            "status": self.status,
            "requested_model": self.requested_model,
            "response_model": self.response_model,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "usage": self.usage.to_dict(),
            "latency_ms": self.latency_ms,
        }


class LLMProvider(Protocol):
    """The contract later generation and evaluation code depends on."""

    @property
    def identity(self) -> ProviderIdentity: ...

    def generate_text(self, request: GenerationRequest) -> GenerationResult[None]: ...

    def generate_structured(
        self, request: GenerationRequest, schema: type[SchemaT]
    ) -> GenerationResult[SchemaT]: ...

    def close(self) -> None: ...
