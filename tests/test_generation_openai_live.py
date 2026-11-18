"""Opt-in live smoke test of the OpenAI provider (Commit 5.1). Makes PAID API calls.

Skipped unless ``FOUNDERLENS_RUN_LIVE_LLM_TESTS=1``; an API key alone never
enables it. Exactly two tiny requests (64 output tokens each, no SDK retry)
with synthetic content written for this test: no PDFs, chunks, retrieval
artifacts or evaluation questions. Run manually::

    $env:FOUNDERLENS_RUN_LIVE_LLM_TESTS = "1"   # and OPENAI_API_KEY set
    python -m pytest tests/test_generation_openai_live.py -q

Hosted model output is not bitwise reproducible, so only shape, status,
identities and reported usage are checked.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, ConfigDict

from founderlens.generation.config import DEFAULT_MODEL, GenerationConfig
from founderlens.generation.openai_provider import OpenAIProvider
from founderlens.generation.schema import GenerationRequest

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_RUN_LIVE_LLM_TESTS") != "1",
    reason="set FOUNDERLENS_RUN_LIVE_LLM_TESTS=1 (and OPENAI_API_KEY) to make paid live calls",
)

# max_retries=0: exactly two HTTP requests in total, never a billed SDK retry.
CONFIG = GenerationConfig(max_output_tokens=64, max_retries=0)


class ColourCount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    colour: str
    count: int


def test_live_text_and_structured_generation() -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        pytest.skip("OPENAI_API_KEY is not set")
    with OpenAIProvider(CONFIG) as provider:
        text = provider.generate_text(
            GenerationRequest("Reply with one lowercase word.", "Say the word: ready")
        )
        structured = provider.generate_structured(
            GenerationRequest(
                "Extract the colour and the number from the sentence.",
                "The synthetic box holds 3 blue marbles.",
            ),
            ColourCount,
        )
    for result in (text, structured):
        assert result.status == "completed"
        assert result.requested_model == DEFAULT_MODEL
        assert result.response_model.startswith("gpt-5.4-mini")
        assert result.response_id
        assert result.usage.reported
        assert result.usage.output_tokens is not None
        assert result.usage.output_tokens <= CONFIG.max_output_tokens
    assert text.text.strip()
    assert isinstance(structured.parsed, ColourCount)
