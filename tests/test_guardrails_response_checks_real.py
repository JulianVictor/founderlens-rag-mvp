"""Opt-in guardrail smoke on the real corpus with SYNTHETIC answers (Commit 6.2).

Skipped unless ``FOUNDERLENS_REAL_TOKENIZER=1`` (prepared asset) and
``FOUNDERLENS_REAL_CORPUS=1``. The frozen V2 BM25 policy retrieves real
evidence; a scripted provider writes the answers. Nothing is sent to OpenAI,
no key is needed and the network is blocked. Run locally::

    $env:FOUNDERLENS_REAL_TOKENIZER = "1"; $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_guardrails_response_checks_real.py -q
"""

from __future__ import annotations

import os
import socket
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.guardrails.response_checks import finding_label, run_guardrail_smoke

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_TOKENIZER") != "1"
    or os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_TOKENIZER=1 and FOUNDERLENS_REAL_CORPUS=1",
)

EXPECTED = {
    "single_company_factual": [],
    "cross_company_one_cited": [
        "warning:named_company_not_cited[Alphabet Inc.]",
        "warning:selected_but_not_cited[Alphabet Inc.]",
    ],
    "cross_company_both_cited": [],
    "temporal_comparison": [],
    "unsupported_period": [
        "notice:selected_but_not_cited[NVIDIA Corporation]",
        "notice:unassessable_references",
    ],
    "ambiguous_period": ["notice:basis_unspecified_periods"],
    "mismatched_number": ["notice:value_not_found_verbatim[99.9billion]"],
}


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.mark.usefixtures("no_network")
def test_real_corpus_smoke_findings_are_expected_and_deterministic() -> None:
    settings = load_settings()
    first = run_guardrail_smoke(settings)
    second = run_guardrail_smoke(settings)
    found = {name: [finding_label(f) for f in g.report.findings] for name, _, g in first}
    assert found == EXPECTED
    assert [g.report for _, _, g in first] == [g.report for _, _, g in second]
    assert [g.response.render() for _, _, g in first] == [g.response.render() for _, _, g in second]
