"""Opt-in checks of the generation service with the real tokenizer and V2 corpus (Commit 5.3).

Skipped unless ``FOUNDERLENS_REAL_TOKENIZER=1`` (prepared asset) and
``FOUNDERLENS_REAL_CORPUS=1``. The provider is the offline fake from the unit
tests: nothing is sent to OpenAI, no key is needed and the network is blocked.
Run locally::

    $env:FOUNDERLENS_REAL_TOKENIZER = "1"; $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_generation_service_real.py -q
"""

from __future__ import annotations

import os
import socket
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.generation.answer import AnswerStatus, schema_token_report
from founderlens.generation.context import DEFAULT_CONTEXT_CONFIG, ContextStatus
from founderlens.generation.service import AnswerOrigin, GroundedGenerationService
from founderlens.generation.token_budget import load_generation_tokenizer
from founderlens.retrieval.policy import load_policy_retriever
from tests.test_generation_service import FakeProvider, _answer

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_TOKENIZER") != "1"
    or os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_TOKENIZER=1 and FOUNDERLENS_REAL_CORPUS=1",
)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.mark.usefixtures("no_network")
def test_real_schema_fits_the_framing_allowance() -> None:
    counter = load_generation_tokenizer(load_settings())
    report = schema_token_report(counter, DEFAULT_CONTEXT_CONFIG.framing_allowance_tokens)
    assert (report.schema_tokens, report.wrapper_tokens, report.text_parameter_tokens) == (
        461,
        22,
        481,
    )
    assert report.remaining_allowance_tokens == 543


@pytest.mark.usefixtures("no_network")
def test_real_service_composes_retrieval_context_and_provider_offline() -> None:
    settings = load_settings()
    counter = load_generation_tokenizer(settings)
    retriever = load_policy_retriever(settings)
    provider = FakeProvider(_answer("answered", [("Blackwell is cited.", "reported", ["E01"])]))
    service = GroundedGenerationService(retriever, counter, provider)
    answered = service.answer("NVIDIA Blackwell")
    assert answered.origin is AnswerOrigin.MODEL
    assert answered.prepared.status is ContextStatus.EVIDENCE_SELECTED
    assert len(answered.prepared.evidence) == 10
    assert provider.calls[0][0] is answered.prepared.request
    local = service.answer("zyzzyva quokka")
    assert local.origin is AnswerOrigin.LOCAL
    assert local.answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert len(provider.calls) == 1  # the empty retrieval made no provider call
