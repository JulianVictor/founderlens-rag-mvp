"""Versioned generation configuration (Commit 5.1).

The **semantic** part decides what is requested from the model and is hashed
into :attr:`GenerationConfig.config_id`: provider, model snapshot, reasoning
effort, output-token budget and the fixed request policy (no storage, no
streaming, no background mode, no tools, no persistent conversation, no
sampling parameters, no model fallback). The **operational** part (per-attempt
timeout and SDK retries) changes reliability and latency, not the request, so
it is reported separately and is not part of the identity.

Neither part contains credentials, timestamps, request IDs or paths.

Defaults follow the OpenAI model page for ``gpt-5.4-mini`` (snapshot
``gpt-5.4-mini-2026-03-17``; ``reasoning.effort`` none, low, medium, high or
xhigh; at most 128,000 output tokens). Temperature, top-p and seed are never
sent. Another model can be configured deliberately; whether it accepts the
chosen effort is then enforced by the API (a rejected request raises
:class:`~founderlens.generation.errors.InvalidRequestError`), never by
switching models.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, TypeGuard

from founderlens.generation.errors import GenerationConfigError

CONFIG_SCHEMA_VERSION = 1
PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh")
# Documented for the default snapshot ("minimal" is not listed for gpt-5.4-mini).
DEFAULT_MODEL_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")
MIN_OUTPUT_TOKENS = 16  # Responses API minimum for max_output_tokens
MAX_OUTPUT_TOKENS = 128_000  # gpt-5.4-mini maximum output tokens
MAX_TIMEOUT_SECONDS = 600.0
MAX_RETRIES_LIMIT = 3
_MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Immutable generation settings. Construct with keywords; validated on creation."""

    model: str = DEFAULT_MODEL
    reasoning_effort: ReasoningEffort = "none"
    max_output_tokens: int = 1024
    timeout_seconds: float = 30.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        problems: list[str] = []
        model: object = self.model
        effort: object = self.reasoning_effort
        tokens: object = self.max_output_tokens
        timeout: object = self.timeout_seconds
        retries: object = self.max_retries
        if not isinstance(model, str) or not _MODEL_PATTERN.fullmatch(model):
            problems.append(f"model must be an explicit model identifier, got {model!r}")
        if effort not in REASONING_EFFORTS:
            problems.append(f"reasoning_effort must be one of {REASONING_EFFORTS}, got {effort!r}")
        elif model == DEFAULT_MODEL and effort not in DEFAULT_MODEL_EFFORTS:
            problems.append(f"{DEFAULT_MODEL} supports reasoning_effort {DEFAULT_MODEL_EFFORTS}")
        if not (_is_int(tokens) and MIN_OUTPUT_TOKENS <= tokens <= MAX_OUTPUT_TOKENS):
            problems.append(
                f"max_output_tokens must be an integer in [{MIN_OUTPUT_TOKENS}, "
                f"{MAX_OUTPUT_TOKENS}], got {tokens!r}"
            )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or not 0 < timeout <= MAX_TIMEOUT_SECONDS
        ):
            problems.append(
                f"timeout_seconds must be a finite number in (0, {MAX_TIMEOUT_SECONDS}], "
                f"got {timeout!r}"
            )
        if not (_is_int(retries) and 0 <= retries <= MAX_RETRIES_LIMIT):
            problems.append(
                f"max_retries must be an integer in [0, {MAX_RETRIES_LIMIT}], got {retries!r}"
            )
        if problems:
            raise GenerationConfigError("; ".join(problems))

    def semantic_dict(self) -> dict[str, Any]:
        """What is requested from the provider. Hashed into :attr:`config_id`."""
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "provider": PROVIDER,
            "api": "responses",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "stream": False,
            "background": False,
            "tools": [],
            "previous_response_id": None,
            "sampling_parameters": None,
            "model_fallback": None,
        }

    def operational_dict(self) -> dict[str, Any]:
        """Reliability settings; reported, not part of the identity."""
        return {
            "timeout_seconds_per_attempt": float(self.timeout_seconds),
            "sdk_max_retries": self.max_retries,
        }

    @property
    def config_id(self) -> str:
        """First 12 hex characters of the SHA-256 of the canonical semantic settings."""
        canonical = json.dumps(self.semantic_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


DEFAULT_GENERATION_CONFIG = GenerationConfig()
