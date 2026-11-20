"""Opt-in LIVE grounded-generation smoke test (Commit 5.3). Makes PAID API calls.

Skipped unless ``FOUNDERLENS_RUN_LIVE_RAG_TESTS=1``. Neither an API key nor the
provider's own ``FOUNDERLENS_RUN_LIVE_LLM_TESTS`` flag enables it. It also
needs ``OPENAI_API_KEY``, the prepared tokenizer asset and the local V2
collection. Running it sends the question, the grounded instructions and up to
ten public-source evidence chunks per request to OpenAI (``store=False``).

Scenarios, with questions written for this smoke test without consulting the
evaluation files:

1. a financial question whose evidence is in the corpus (one paid request);
2. a question that retrieves candidates but asks for something absent from
   them (one paid request);
3. an empty retrieval, which must make no request.

There are at most two paid requests, with SDK retries disabled
(``max_retries=0``; the semantic configuration and ``config_id`` are
unchanged). The test checks live API and schema compatibility. It does not
require a particular answer or abstention. Refusals, incomplete outputs and
response-contract violations are printed as semantic observations, not
failures. Transport, authentication, schema-rejection and other provider
errors fail the test. Observations (statuses, cited IDs, usage, latency) are
printed without prompts, evidence text, answer text or keys. A successful call
does not show the answer is correct. Run manually::

    $env:FOUNDERLENS_RUN_LIVE_RAG_TESTS = "1"   # OPENAI_API_KEY must be set
    python -m pytest tests/test_generation_service_live.py -q
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.generation import errors as ge
from founderlens.generation.answer import ResponseContractError
from founderlens.generation.config import GenerationConfig
from founderlens.generation.openai_provider import OpenAIProvider
from founderlens.generation.schema import (
    GenerationRequest,
    GenerationResult,
    ProviderIdentity,
    SchemaT,
)
from founderlens.generation.service import AnswerOrigin, GroundedGenerationService
from founderlens.generation.token_budget import load_generation_tokenizer
from founderlens.retrieval.policy import load_policy_retriever

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_RUN_LIVE_RAG_TESTS") != "1",
    reason="set FOUNDERLENS_RUN_LIVE_RAG_TESTS=1 (and OPENAI_API_KEY) to make paid live calls",
)

MAX_PAID_REQUESTS = 2
# Retries disabled for the smoke only: operational, so config_id stays cbc4e415c848.
LIVE_CONFIG = GenerationConfig(max_retries=0)
SCENARIOS: tuple[tuple[str, str], ...] = (
    (
        "evidence_present",
        "What revenue did NVIDIA report for its Data Center business in the fourth quarter "
        "of fiscal 2025?",
    ),
    ("evidence_absent", "What was NVIDIA's revenue from customers in Antarctica in fiscal 2025?"),
    ("empty_retrieval", "zyzzyva quokka"),
)
SEMANTIC_OUTCOMES = (ge.ModelRefusalError, ge.IncompleteGenerationError, ResponseContractError)


class CountingProvider:
    """Wraps the live provider and refuses a third paid request before sending it."""

    def __init__(self, inner: OpenAIProvider) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def identity(self) -> ProviderIdentity:
        return self._inner.identity

    def generate_text(self, request: GenerationRequest) -> GenerationResult[None]:
        raise AssertionError("the smoke test uses structured generation only")

    def generate_structured(
        self, request: GenerationRequest, schema: type[SchemaT]
    ) -> GenerationResult[SchemaT]:
        if self.calls >= MAX_PAID_REQUESTS:
            raise AssertionError(f"more than {MAX_PAID_REQUESTS} paid requests attempted")
        self.calls += 1
        return self._inner.generate_structured(request, schema)

    def close(self) -> None:
        self._inner.close()


def _observe(service: GroundedGenerationService, scenario: str, question: str) -> dict[str, Any]:
    try:
        result = service.answer(question)
    except SEMANTIC_OUTCOMES as error:  # model outcomes, recorded rather than failed
        return {
            "scenario": scenario,
            "outcome": "semantic_failure",
            "category": error.category,
            "request_id": error.request_id,
            "parsed": isinstance(error, ResponseContractError),
        }
    metadata = result.metadata()
    provider = metadata["runtime"]["provider"]
    return {
        "scenario": scenario,
        "outcome": "completed",
        "parsed": result.provider_called,
        "origin": result.origin.value,
        "preparation_status": metadata["generation"]["preparation_status"],
        "selected_evidence": metadata["context"]["selected_count"],
        "answer_status": metadata["generation"]["answer_status"],
        "claim_types": metadata["generation"]["claim_types"],
        "cited_evidence_ids": metadata["generation"]["cited_evidence_ids"],
        "limitation_count": metadata["generation"]["limitation_count"],
        "response_model": None if provider is None else provider["response_model"],
        "usage": None if provider is None else provider["usage"],
        "provider_latency_ms": None if provider is None else provider["latency_ms"],
    }


def test_live_grounded_generation_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        pytest.skip("OPENAI_API_KEY is not set")
    settings = load_settings()
    counter = load_generation_tokenizer(settings)
    retriever = load_policy_retriever(settings)
    with OpenAIProvider(LIVE_CONFIG) as live:
        provider = CountingProvider(live)
        service = GroundedGenerationService(
            retriever, counter, provider, generation_config=LIVE_CONFIG
        )
        observations = [_observe(service, scenario, q) for scenario, q in SCENARIOS]
    with capsys.disabled():
        print("\nLive grounded-generation smoke (retries disabled; no prompts or answer text):")
        for observation in observations:
            print(json.dumps(observation, sort_keys=True))
        print(f"paid requests attempted: {provider.calls} of at most {MAX_PAID_REQUESTS}")
    assert provider.calls <= MAX_PAID_REQUESTS
    empty = observations[2]
    assert empty["outcome"] == "completed"
    assert empty["origin"] == AnswerOrigin.LOCAL.value  # no request for empty retrieval
    assert empty["usage"] is None
    # API and schema compatibility: at least one live response was parsed as GroundedAnswer.
    assert any(o["parsed"] for o in observations[:2]), observations
