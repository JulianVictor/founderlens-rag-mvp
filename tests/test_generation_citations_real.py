"""Opt-in end-to-end check of the offline demo on the real corpus (Commit 5.4).

Skipped unless ``FOUNDERLENS_REAL_TOKENIZER=1`` (prepared asset) and
``FOUNDERLENS_REAL_CORPUS=1``. The answers come from the SYNTHETIC scripted
provider: nothing is sent to OpenAI, no key is needed and the network is
blocked. Run locally::

    $env:FOUNDERLENS_REAL_TOKENIZER = "1"; $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_generation_citations_real.py -q
"""

from __future__ import annotations

import os
import socket
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.corpus.manifest import load_manifest
from founderlens.generation.demo import run_offline_demo

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
def test_offline_demo_resolves_real_citations_deterministically() -> None:
    settings = load_settings()
    manifest = load_manifest(settings.corpus_manifest_path)
    first = run_offline_demo(settings)
    second = run_offline_demo(settings)
    labels = [label for label, _ in first]
    assert labels == [
        "answered",
        "partially_answered",
        "insufficient_evidence",
        "local_empty_retrieval",
    ]
    assert [r.render() for _, r in first] == [r.render() for _, r in second]
    for _, response in first:
        assert response.identities()["corpus"]["corpus_version"] == manifest.corpus_version
        for citation in response.citations:
            entry = manifest.get(citation.document_id)
            assert entry is not None
            assert (citation.document_title, citation.source_url) == (entry.title, entry.source_url)
            record = response.run.prepared.evidence.get(citation.evidence_id)
            assert record is not None
            assert citation.text == record.chunk.text
    answered, partial, insufficient, local = (r for _, r in first)
    assert [c.citation_numbers for c in answered.claims] == [(1, 2), (1,)]
    assert len(partial.citations) == 2
    assert insufficient.citations == ()
    assert local.usage() is None
    assert local.run.provider_result is None
