"""Opt-in checks with the real o200k_base tokenizer and the real V2 collection (Commit 5.2).

Skipped unless ``FOUNDERLENS_REAL_TOKENIZER=1``; it needs the prepared asset
(``python -m founderlens.generation.token_budget prepare``). The corpus check
also needs ``FOUNDERLENS_REAL_CORPUS=1`` and the local V2 artifact. Both only
read existing files: nothing is written, nothing is downloaded and no LLM is
called. Run locally::

    $env:FOUNDERLENS_REAL_TOKENIZER = "1"; $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_generation_context_real.py -q
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest

from founderlens.config import Settings, load_settings
from founderlens.generation import token_budget as tb
from founderlens.generation.context import (
    CONTEXT_SMOKE_QUERIES,
    DEFAULT_CONTEXT_CONFIG,
    ContextStatus,
    InvalidQuestionError,
    build_generation_input,
    build_smoke_report,
    prepare_generation_input,
)
from founderlens.generation.token_budget import load_generation_tokenizer, tokenizer_identity
from founderlens.retrieval.bm25 import BM25Config, BM25Retriever
from founderlens.retrieval.policy import PolicyRetriever, load_policy_retriever
from founderlens.retrieval.schema import RetrievalResult

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_TOKENIZER") != "1",
    reason="set FOUNDERLENS_REAL_TOKENIZER=1 (after token_budget prepare) for real-tokenizer tests",
)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.mark.usefixtures("no_network")
def test_real_tokenizer_loads_offline_and_treats_special_tokens_as_text() -> None:
    counter = load_generation_tokenizer(load_settings())
    assert counter.identity == tokenizer_identity() == DEFAULT_CONTEXT_CONFIG.tokenizer
    assert counter.count("hello world") == 2
    assert counter.count("<|endoftext|>") > 1  # ordinary text, not the control token
    assert counter.count("€ 39,331 million — 收入") > 0
    assert counter.count("") == 0


@pytest.mark.usefixtures("no_network")
@pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 to use the local V2 collection",
)
def test_real_corpus_context_smoke_is_reproducible() -> None:
    report = build_smoke_report(load_settings())
    assert report["reproducible"] is True
    assert report["retrieval_policy_id"] == "f20913f1c1ec"
    assert report["context_config_id"] == DEFAULT_CONTEXT_CONFIG.config_id
    entries = {e["category"] + "|" + e.get("query", ""): e for e in report["queries"]}
    assert len(report["queries"]) == len(CONTEXT_SMOKE_QUERIES)
    for entry in report["queries"]:
        if "error" in entry:
            assert entry["error"] == "invalid_question"
            continue
        budget = entry["budget"]
        assert budget["local_total_tokens"] <= budget["input_token_limit"]
        assert entry["selected_count"] + entry["excluded_count"] == entry["candidate_count"]
        assert [e["evidence_id"] for e in entry["evidence"]] == [
            f"E{i:02d}" for i in range(1, entry["selected_count"] + 1)
        ]
    assert entries["company and product terminology|NVIDIA Blackwell"]["candidate_count"] == 10
    assert entries["out of vocabulary|zyzzyva quokka"]["status"] == "empty_retrieval"


REAL_CORPUS = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 to use the local V2 collection",
)


@pytest.mark.usefixtures("no_network")
def test_real_tokenizer_matches_tiktoken_and_from_file_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tiktoken

    source = load_settings()
    ours = load_generation_tokenizer(source)
    # tiktoken's own construction from the same verified file (cached, so no download).
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tb.asset_cache_dir(source)))
    reference = tiktoken.get_encoding("o200k_base")
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR")
    samples = ["hello world", "<|endoftext|>", "Revenue was $35.1 billion, up 94%.", "x" * 3000]
    for text in samples:
        assert ours.count(text) == len(
            reference.encode(text, allowed_special=set(), disallowed_special=())
        )
    fresh = Settings(tmp_path)
    copied = tb.prepare_asset(fresh, from_file=tb.asset_path(source))
    assert copied.read_bytes() == tb.asset_path(source).read_bytes()
    assert tb.main(["--project-root", str(tmp_path), "check"]) == 0
    assert load_generation_tokenizer(fresh).count("hello world") == 2


def _question_with_tokens(counter: Any, tokens: int) -> str:
    for words in range(tokens - 20, tokens + 20):
        question = " ".join(["revenue"] * words)
        if counter.count(question) == tokens:
            return question
    raise AssertionError(f"no {tokens}-token question found")


@REAL_CORPUS
@pytest.mark.usefixtures("no_network")
def test_real_question_boundary_and_synthetic_stress_case() -> None:
    settings = load_settings()
    counter = load_generation_tokenizer(settings)
    retriever = load_policy_retriever(settings)
    q512 = _question_with_tokens(counter, 512)
    accepted = prepare_generation_input(q512, retriever, counter)
    assert accepted.budget.question_tokens == 512
    assert accepted.status is ContextStatus.EVIDENCE_SELECTED
    assert accepted.budget.local_total_tokens <= DEFAULT_CONTEXT_CONFIG.input_token_limit
    assert accepted.question == q512  # never truncated

    class Spy(PolicyRetriever):
        searched = 0

        def search(self, query: str) -> tuple[RetrievalResult, ...]:
            Spy.searched += 1
            return super().search(query)

    spy = Spy(
        retriever.policy,
        retriever.collection,
        BM25Retriever(retriever.collection.chunks, BM25Config(top_k=10)),
    )
    with pytest.raises(InvalidQuestionError, match="513 tokens"):
        prepare_generation_input(_question_with_tokens(counter, 513), spy, counter)
    assert Spy.searched == 0  # rejected before retrieval

    # Synthetic stress case: the ten largest V2 chunks as if retrieved, with a valid
    # 512-token question. Not a real retrieval result and not a proven upper bound.
    largest = sorted(retriever.collection.chunks, key=lambda c: -counter.count(c.text))[:10]
    candidates = [
        RetrievalResult("bm25", rank, float(100 - rank), chunk)
        for rank, chunk in enumerate(largest, start=1)
    ]
    stress = build_generation_input(
        q512,
        candidates,
        collection=retriever.collection,
        retrieval_identities=retriever.identities(),
        counter=counter,
    )
    assert len(stress.evidence) == 10
    assert stress.budget.local_total_tokens <= DEFAULT_CONTEXT_CONFIG.input_token_limit
    assert str(settings.project_root) not in str(stress.metadata())
