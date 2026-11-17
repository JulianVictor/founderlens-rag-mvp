"""Reciprocal Rank Fusion: math, validation, ranking, lifecycle, results and CLI.

Uses the real RRF implementation, a real BM25 index and a real dense retriever
over synthetic V2-identified chunks with a deterministic fake query encoder.
No model is loaded and nothing touches the network.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from founderlens.ingestion.chunk import ChunkingConfig, ChunkRecord, chunk_page
from founderlens.ingestion.normalize import NormalizedPageRecord, page_warnings
from founderlens.ingestion.schema import make_page_id, status_for_text
from founderlens.retrieval import embeddings as em
from founderlens.retrieval import hybrid as hy
from founderlens.retrieval.bm25 import BM25Retriever
from founderlens.retrieval.collections import CHUNKING_V2_CONFIG
from founderlens.retrieval.dense import DenseRetriever, QueryTooLongError
from founderlens.retrieval.embeddings import EmbeddingArtifact, EmbeddingConfig
from founderlens.retrieval.hybrid import (
    HybridConfig,
    HybridRetriever,
    ranking_problems,
    rrf_fuse,
)
from founderlens.retrieval.model import MODEL_ID, MODEL_REVISION
from founderlens.retrieval.schema import ChunkCollection, RetrievalError, RetrievalResult

A, B = "aaa-2024-q1", "bbb-2024-q1"
DIM = 4
EMBEDDING = EmbeddingConfig(dimension=DIM, max_seq_length=256)


class FakeCounter:
    """Whitespace words plus two special tokens, 256-token limit (for V2 chunking)."""

    max_seq_length = 256

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) + 2 for t in texts]


class FakeEncoder(FakeCounter):
    """Deterministic query vectors; counts calls. Pinned model identity."""

    def __init__(self, limit: int = 256) -> None:
        self.max_seq_length = limit
        self.calls: dict[str, int] = {"describe": 0, "encode": 0}
        self.encoded: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return DIM

    def encode(self, texts: Sequence[str]) -> Any:
        self.calls["encode"] += 1
        self.encoded.append(list(texts))
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append([digest[i] - 127.5 for i in range(DIM)])
        return np.asarray(rows, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        self.calls["describe"] += 1
        return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}


PAGES = {
    (A, 1): "revenue growth data center Blackwell demand",
    (A, 2): "operating income margin expenses research",
    (B, 1): "cloud revenue search advertising growth",
    (B, 2): "cash flow capital expenditures dividends buyback",
}


def _collection(pages: dict[tuple[str, int], str] = PAGES) -> ChunkCollection:
    chunks: list[ChunkRecord] = []
    for (document_id, number), text in pages.items():
        status = status_for_text(text)
        page = NormalizedPageRecord(
            document_id=document_id,
            document_sha256="a" * 64,
            company="Example",
            fiscal_period="Q1 2024",
            source_url="https://example.com/x.pdf",
            page_number=number,
            page_id=make_page_id(document_id, number),
            extraction_status=status,
            raw_character_count=len(text),
            normalized_text=text,
            normalized_character_count=len(text),
            tables=(),
            table_text_overlap=False,
            normalization_warnings=page_warnings(status, ()),
        )
        chunks.extend(chunk_page(page, CHUNKING_V2_CONFIG, FakeCounter()))
    return ChunkCollection(tuple(chunks), "c" * 64, chunks[0].chunking_config_id, "d" * 64)


COLLECTION = _collection()


def _dense(
    collection: ChunkCollection = COLLECTION, encoder: FakeEncoder | None = None
) -> DenseRetriever:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(len(collection.chunks), DIM))
    unit: NDArray[np.float32] = (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(
        np.float32
    )
    manifest = {
        "embedding_config": EMBEDDING.semantic_dict(),
        "embedding_config_id": EMBEDDING.config_id,
        "source_chunks": {"sha256": collection.artifact_sha256},
        "matrix": {"sha256": hashlib.sha256(em.matrix_bytes(unit)).hexdigest()},
    }
    artifact = EmbeddingArtifact(unit, tuple(c.chunk_id for c in collection.chunks), manifest)
    return DenseRetriever(
        collection, artifact, encoder or FakeEncoder(), embedding_config=EMBEDDING
    )


def _hybrid(encoder: FakeEncoder | None = None) -> HybridRetriever:
    return HybridRetriever(COLLECTION, BM25Retriever(COLLECTION.chunks), _dense(encoder=encoder))


def _results(
    method: str, chunks: Sequence[ChunkRecord], scores: Sequence[float] | None = None
) -> tuple[RetrievalResult, ...]:
    values = scores or [10.0 - i for i in range(len(chunks))]
    return tuple(
        RetrievalResult(method, i, float(values[i - 1]), c) for i, c in enumerate(chunks, 1)
    )


C = COLLECTION.chunks
ORDER = {c.chunk_id: i for i, c in enumerate(C)}

# --- configuration -------------------------------------------------------------------


def test_fixed_configuration_and_identity() -> None:
    config = HybridConfig()
    assert (config.rrf_constant, config.bm25_depth, config.dense_depth, config.top_k) == (
        60,
        10,
        10,
        5,
    )
    first = config.config_id("bm25-id", "dense-id")
    assert first == HybridConfig(top_k=9).config_id("bm25-id", "dense-id")  # k is not identity
    assert first != config.config_id("bm25-other", "dense-id")  # components are
    semantic = config.semantic_dict("b", "d")
    assert semantic["candidate_depths"] == {"bm25": 10, "dense": 10}
    assert semantic["score_threshold"] is None
    assert semantic["page_deduplication"] is False
    with pytest.raises(ValueError, match="fixed"):
        HybridConfig(rrf_constant=30)


@pytest.mark.parametrize("bad", [0, -1, 11, True, 2.0])
def test_invalid_top_k(bad: object) -> None:
    with pytest.raises(ValueError, match="from 1 to 10"):
        HybridConfig(top_k=bad)  # type: ignore[arg-type]  # invalid on purpose
    with pytest.raises(ValueError, match="from 1 to 10"):
        _hybrid().run("revenue", bad)  # type: ignore[arg-type]  # invalid on purpose


# --- RRF mathematics --------------------------------------------------------------------


def test_hand_calculated_scores_and_contributions() -> None:
    a, b, c, d = C[0], C[1], C[2], C[3]
    fused = rrf_fuse(
        {"bm25": _results("bm25", [a, b, c]), "dense": _results("dense", [b, d])}, ORDER
    )
    scores = {f.chunk.chunk_id: f.rrf_score for f in fused}
    assert scores[b.chunk_id] == pytest.approx(1 / 62 + 1 / 61)  # in both
    assert scores[a.chunk_id] == pytest.approx(1 / 61)  # BM25 only
    assert scores[d.chunk_id] == pytest.approx(1 / 62)  # dense only
    assert scores[c.chunk_id] == pytest.approx(1 / 63)
    assert [f.chunk.chunk_id for f in fused] == [b.chunk_id, a.chunk_id, d.chunk_id, c.chunk_id]
    shared = fused[0]
    assert [(e.method, e.rank) for e in shared.components] == [("bm25", 2), ("dense", 1)]
    assert math.fsum(e.contribution for e in shared.components) == shared.rrf_score
    bm25_only = fused[1]
    assert bm25_only.components[1].rank is None
    assert bm25_only.components[1].score is None
    assert bm25_only.components[1].contribution == 0.0


def test_empty_rankings() -> None:
    assert rrf_fuse({"bm25": (), "dense": ()}, ORDER) == []
    only_dense = rrf_fuse({"bm25": (), "dense": _results("dense", [C[2]])}, ORDER)
    assert [f.chunk.chunk_id for f in only_dense] == [C[2].chunk_id]
    assert only_dense[0].rrf_score == pytest.approx(1 / 61)


def test_raw_scores_do_not_affect_fusion() -> None:
    rankings = {"bm25": _results("bm25", [C[0], C[1]]), "dense": _results("dense", [C[1], C[2]])}
    other = {
        "bm25": _results("bm25", [C[0], C[1]], [999.0, -5.0]),
        "dense": _results("dense", [C[1], C[2]], [-0.9, -0.95]),
    }
    first, second = rrf_fuse(rankings, ORDER), rrf_fuse(other, ORDER)
    assert [(f.chunk.chunk_id, f.rrf_score) for f in first] == [
        (f.chunk.chunk_id, f.rrf_score) for f in second
    ]


def test_shared_candidates_outrank_single_list_candidates_at_c60_depth10() -> None:
    # Worst shared case (rank 10 in both) still beats the best single-list case (rank 1).
    assert 2 / (60 + 10) > 1 / (60 + 1)
    bm25 = [C[0], C[3], C[1]]  # C[0] only here, at rank 1
    dense = [C[2], C[3], C[1]]  # C[2] only here, at rank 1
    fused = rrf_fuse({"bm25": _results("bm25", bm25), "dense": _results("dense", dense)}, ORDER)
    ranked = [f.chunk.chunk_id for f in fused]
    assert set(ranked[:2]) == {C[1].chunk_id, C[3].chunk_id}  # shared chunks first
    assert set(ranked[2:]) == {C[0].chunk_id, C[2].chunk_id}


def test_ties_use_collection_order_not_component_order() -> None:
    first = rrf_fuse({"bm25": _results("bm25", [C[3]]), "dense": _results("dense", [C[0]])}, ORDER)
    second = rrf_fuse({"dense": _results("dense", [C[0]]), "bm25": _results("bm25", [C[3]])}, ORDER)
    assert first[0].rrf_score == first[1].rrf_score  # exact tie: 1/61 each
    assert [f.chunk.chunk_id for f in first] == [C[0].chunk_id, C[3].chunk_id]
    assert [(f.chunk.chunk_id, f.rrf_score) for f in first] == [
        (f.chunk.chunk_id, f.rrf_score) for f in second
    ]
    swapped = rrf_fuse(
        {"bm25": _results("bm25", [C[0], C[3]]), "dense": _results("dense", [C[3], C[0]])}, ORDER
    )
    assert swapped[0].rrf_score == swapped[1].rrf_score  # 1/61 + 1/62 both ways (fsum)
    assert swapped[0].chunk.chunk_id == C[0].chunk_id


def test_distinct_chunks_on_one_page_stay_distinct() -> None:
    long_page = {(A, 1): " ".join(f"w{i}" for i in range(300))}  # > 120 words: several chunks
    collection = _collection(long_page)
    same_page = collection.chunks[:2]
    assert same_page[0].page_id == same_page[1].page_id
    order = {c.chunk_id: i for i, c in enumerate(collection.chunks)}
    fused = rrf_fuse({"bm25": _results("bm25", same_page), "dense": ()}, order)
    assert len(fused) == 2


# --- candidate validation ----------------------------------------------------------------

INDEX = {c.chunk_id: c for c in C}


@pytest.mark.parametrize(
    ("results", "message"),
    [
        (lambda: (*_results("bm25", [C[0]]), RetrievalResult("bm25", 2, 1.0, C[0])), "duplicate"),
        (lambda: (RetrievalResult("bm25", 2, 1.0, C[0]),), "ranks are not 1..n"),
        (lambda: (RetrievalResult("bm25", 1, 1.0, replace(C[0], company="Other")),), "differs"),
        (lambda: _results("dense", [C[0]]), "labelled 'dense'"),
        (lambda: _results("bm25", list(C) * 3)[:11], "exceed the candidate depth"),
    ],
)
def test_invalid_component_rankings(results: Any, message: str) -> None:
    built = results()
    problems = ranking_problems("bm25", built, 10, INDEX)
    assert any(message in p for p in problems), problems


def test_unknown_chunks_and_non_finite_scores_are_rejected() -> None:
    other = _collection({(B, 9): "unrelated words here"}).chunks[0]
    assert any(
        "unknown chunk" in p for p in ranking_problems("bm25", _results("bm25", [other]), 10, INDEX)
    )
    with pytest.raises(ValueError, match="finite"):
        RetrievalResult("bm25", 1, math.nan, C[0])  # cannot even be constructed


def test_incompatible_collections_are_rejected() -> None:
    v1 = ChunkCollection(
        tuple(
            replace(
                c,
                chunking_config_id=ChunkingConfig().config_id,
                chunk_id=c.chunk_id.replace(
                    CHUNKING_V2_CONFIG.config_id, ChunkingConfig().config_id
                ),
            )
            for c in C
        ),
        "c" * 64,
        ChunkingConfig().config_id,
        "d" * 64,
    )
    with pytest.raises(RetrievalError, match="fuses collection v2"):
        HybridRetriever(v1, BM25Retriever(v1.chunks), _dense(v1))
    with pytest.raises(RetrievalError, match="BM25 does not index exactly this collection"):
        HybridRetriever(COLLECTION, BM25Retriever(v1.chunks), _dense())  # BM25 v1 + dense v2
    other_hash = ChunkCollection(C, "e" * 64, COLLECTION.chunking_config_id, "d" * 64)
    with pytest.raises(RetrievalError, match="dense runs on collection"):
        HybridRetriever(COLLECTION, BM25Retriever(C), _dense(other_hash))
    with pytest.raises(RetrievalError, match="collection order"):
        HybridRetriever(COLLECTION, BM25Retriever(tuple(reversed(C))), _dense())


# --- retriever: ranking, top-k, lifecycle ----------------------------------------------


def _spy(
    retriever: HybridRetriever, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, str, int | None]]:
    calls: list[tuple[str, str, int | None]] = []
    for name, component in (("bm25", retriever.bm25), ("dense", retriever.dense)):
        original = component.search

        def wrapped(
            query: str, top_k: int | None = None, *, _o: Any = original, _n: str = name
        ) -> Any:
            calls.append((_n, query, top_k))
            return _o(query, top_k)

        monkeypatch.setattr(component, "search", wrapped)
    return calls


def test_components_called_once_with_fixed_depth_and_original_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = _hybrid()
    calls = _spy(retriever, monkeypatch)
    query = "  Revenue GROWTH, $39,331?  "
    retriever.run(query, 1)
    retriever.run(query, 10)
    assert calls == [("bm25", query, 10), ("dense", query, 10)] * 2  # depth independent of k


def test_top_k_prefix_and_small_unions() -> None:
    retriever = _hybrid()
    top10 = retriever.search_detailed("revenue growth cloud", 10)
    top5 = retriever.search_detailed("revenue growth cloud", 5)
    assert top5 == top10[:5]
    assert len(retriever.search("revenue growth cloud", 1)) == 1
    assert len(top10) == len(C)  # union of 4 chunks < 10: fewer results
    ids = [h.result.chunk.chunk_id for h in top10]
    assert len(set(ids)) == len(ids)
    scores = [h.result.score for h in top10]
    assert scores == sorted(scores, reverse=True)
    assert [h.result.rank for h in top10] == list(range(1, len(top10) + 1))


def test_empty_queries_skip_both_components_and_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = FakeEncoder()
    retriever = _hybrid(encoder)
    calls = _spy(retriever, monkeypatch)
    for query in ("", "   ", "$ % ( ) -"):
        run = retriever.run(query)
        assert run.fused == ()
        assert (run.bm25, run.dense) == ((), ())
    assert calls == []
    assert encoder.calls == {"describe": 0, "encode": 0}


def test_oversized_queries_fail_explicitly() -> None:
    retriever = _hybrid(FakeEncoder(limit=256))
    with pytest.raises(QueryTooLongError, match="never truncated"):
        retriever.run("revenue " * 300)


def test_empty_bm25_ranking_is_fused_from_dense_alone() -> None:
    run = _hybrid().run("zyzzyva quokka")  # no lexical match; dense still ranks
    assert run.bm25 == ()
    assert len(run.dense) == len(C)
    assert all(h.components[0].rank is None for h in run.fused)
    assert [h.result.chunk.chunk_id for h in run.fused] == [r.chunk.chunk_id for r in run.dense][:5]


def test_component_failures_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = _hybrid()

    def broken(*_: object, **__: object) -> Any:
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(retriever.dense, "search", broken)
    with pytest.raises(RuntimeError, match="simulated"):
        retriever.run("revenue growth")
    retriever = _hybrid()
    monkeypatch.setattr(
        retriever.bm25, "search", lambda *_, **__: (RetrievalResult("bm25", 2, 1.0, C[0]),)
    )
    with pytest.raises(RetrievalError, match=r"ranks are not 1\.\.n"):
        retriever.run("revenue growth")


def test_reuse_and_no_document_reembedding() -> None:
    encoder = FakeEncoder()
    retriever = _hybrid(encoder)
    for query in ("revenue", "cloud growth", "cash flow"):
        retriever.run(query)
    assert encoder.calls["describe"] == 1  # identity checked once, model reused
    assert encoder.encoded == [["revenue"], ["cloud growth"], ["cash flow"]]


def test_result_integrity_and_identity() -> None:
    retriever = _hybrid()
    run = retriever.run("revenue growth cloud", 10)
    bm25 = {r.chunk.chunk_id: r for r in run.bm25}
    dense = {r.chunk.chunk_id: r for r in run.dense}
    for h in run.fused:
        chunk = h.result.chunk
        assert chunk == INDEX[chunk.chunk_id]  # original text and provenance
        assert h.result.method == "hybrid_rrf"
        b, d = h.components
        assert (b.rank, b.score) == (
            (bm25[chunk.chunk_id].rank, bm25[chunk.chunk_id].score)
            if chunk.chunk_id in bm25
            else (None, None)
        )
        assert (d.rank, d.score) == (dense[chunk.chunk_id].rank, dense[chunk.chunk_id].score)
        assert math.fsum([b.contribution, d.contribution]) == h.result.score
        data = h.to_dict()
        assert data["text"] == chunk.text
        assert set(data["components"]) == {"bm25", "dense"}
    ident = retriever.identity()
    assert ident["hybrid_config_id"] == retriever.config_id
    assert ident["bm25_config_id"] == retriever.bm25.config.config_id
    assert ident["dense_config_id"] == retriever.dense.config.config_id
    assert ident["collection_sha256"] == "c" * 64
    assert ident["embedding_config_id"] == EMBEDDING.config_id
    assert len(ident["matrix_sha256"]) == 64


# --- CLI ------------------------------------------------------------------------------------


def test_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    retriever = _hybrid()
    monkeypatch.setattr(hy, "load_hybrid_retriever", lambda settings, **_: retriever)
    (tmp_path / "pyproject.toml").write_text("", "utf-8")
    base = ["--project-root", str(tmp_path)]
    assert hy.main([*base, "--query", "revenue growth"]) == 0
    out = capsys.readouterr().out
    assert "Hybrid RRF (config" in out
    assert "constant 60" in out
    assert "not relevance probabilities" in out
    assert hy.main([*base, "--query", "revenue growth", "--json", "--top-k", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "hybrid_rrf"
    assert [r["rank"] for r in payload["results"]] == [1, 2]
    assert set(payload["components"]) == {"bm25", "dense"}
    assert payload["results"][0]["components"]["dense"]["rank"] is not None
    assert hy.main([*base, "--query", "?!"]) == 0
    assert "No results" in capsys.readouterr().out
    assert hy.main([*base, "--collection", "v1", "--query", "x"]) == 1
    assert "fuses collection v2 only" in capsys.readouterr().err
    assert hy.main([*base, "--smoke"]) == 0
    report = json.loads(
        next((tmp_path / "reports").glob("hybrid_smoke_report.*.json")).read_text("utf-8")
    )
    assert report["hybrid"]["method"] == "hybrid_rrf"
    assert "candidate_union" in report["queries"][0]
    assert any("rejected" in q for q in report["queries"])  # oversized input
    with pytest.raises(SystemExit) as exc_info:
        hy.main([*base, "--query", "x", "--top-k", "11"])
    assert exc_info.value.code == 2
