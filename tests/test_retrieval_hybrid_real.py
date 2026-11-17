"""Explicit real-model test of RRF hybrid retrieval on the full V2 collection.

Skipped by default: needs the corpus PDFs (``FOUNDERLENS_REAL_CORPUS=1``) and
the pinned model in the local Hugging Face cache (``FOUNDERLENS_REAL_MODEL=1``).
Builds V2 and its embeddings under ``tmp_path`` with the existing pipeline,
downloads nothing, uses smoke queries only (no evaluation questions) and checks
function and invariants, not relevance. Run::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; $env:FOUNDERLENS_REAL_MODEL = "1"
    python -m pytest tests/test_retrieval_hybrid_real.py -q
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
from pathlib import Path

import pytest

from founderlens.config import Settings, load_settings
from founderlens.corpus.manifest import load_manifest
from founderlens.ingestion.chunk import run_chunking
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import run_normalization
from founderlens.retrieval.collections import collection_path, load_named_collection
from founderlens.retrieval.dense import QueryTooLongError
from founderlens.retrieval.embeddings import (
    EmbeddingConfig,
    SentenceTransformerEncoder,
    run_build,
    run_build_collection,
)
from founderlens.retrieval.hybrid import HYBRID_SMOKE_QUERIES, load_hybrid_retriever

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1"
    or os.environ.get("FOUNDERLENS_REAL_MODEL") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 and FOUNDERLENS_REAL_MODEL=1 (model must be cached)",
)

REPEAT_TOLERANCE = 1e-12  # RRF scores are sums of exact reciprocals of identical ranks


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_hybrid_retrieval_on_v2(tmp_path: Path) -> None:
    source = load_settings()
    manifest = load_manifest(source.corpus_manifest_path)
    originals = {d.filename: _sha(source.raw_data_dir / d.filename) for d in manifest.documents}
    (tmp_path / "data" / "raw").mkdir(parents=True)
    shutil.copyfile(source.corpus_manifest_path, tmp_path / "data" / "manifest.json")
    for filename in originals:
        shutil.copyfile(source.raw_data_dir / filename, tmp_path / "data" / "raw" / filename)
    settings = Settings(tmp_path)
    run_extraction(settings)
    run_normalization(settings)
    run_chunking(settings)
    encoder = SentenceTransformerEncoder(EmbeddingConfig())
    run_build_collection(settings, encoder)
    v2 = load_named_collection(settings, "v2")
    run_build(settings, encoder, EmbeddingConfig(), v2, collection_path(settings, "v2"))

    first = load_hybrid_retriever(settings)
    second = load_hybrid_retriever(settings)  # independently initialized
    ident = first.identity()
    assert ident == second.identity()
    assert ident["collection_sha256"] == v2.artifact_sha256
    assert first.dense.collection.artifact_sha256 == v2.artifact_sha256
    index = {c.chunk_id: c for c in v2.chunks}

    for _, query in HYBRID_SMOKE_QUERIES:
        if query.startswith("revenue revenue"):
            with pytest.raises(QueryTooLongError, match="never truncated"):
                first.run(query)
            continue
        run = first.run(query, 10)
        again = second.run(query, 10)
        if not run.fused:
            assert (run.bm25, run.dense) == ((), ())  # empty / punctuation-only input
            continue
        assert [r.chunk.chunk_id for r in run.bm25] == [r.chunk.chunk_id for r in again.bm25]
        assert [r.chunk.chunk_id for r in run.dense] == [r.chunk.chunk_id for r in again.dense]
        assert [h.result.chunk.chunk_id for h in run.fused] == [
            h.result.chunk.chunk_id for h in again.fused
        ]
        assert (
            max(
                abs(a.result.score - b.result.score)
                for a, b in zip(run.fused, again.fused, strict=True)
            )
            <= REPEAT_TOLERANCE
        )
        for ranking in (run.bm25, run.dense):
            assert [r.rank for r in ranking] == list(range(1, len(ranking) + 1))
            assert len(ranking) <= 10
        # Independent recomputation from the component lists.
        expected: dict[str, float] = {}
        for ranking in (run.bm25, run.dense):
            for r in ranking:
                expected[r.chunk.chunk_id] = expected.get(r.chunk.chunk_id, 0.0) + 1 / (60 + r.rank)
        order = {c.chunk_id: i for i, c in enumerate(v2.chunks)}
        top = sorted(expected, key=lambda cid: (-expected[cid], order[cid]))[:10]
        assert [h.result.chunk.chunk_id for h in run.fused] == top
        for h in run.fused:
            assert math.isclose(h.result.score, expected[h.result.chunk.chunk_id], abs_tol=1e-15)
            assert h.result.chunk == index[h.result.chunk.chunk_id]  # original text, provenance
        ids = [h.result.chunk.chunk_id for h in run.fused]
        assert len(set(ids)) == len(ids)
        assert first.run(query, 5).fused == run.fused[:5]  # top-k is a prefix

    for filename, digest in originals.items():
        assert _sha(source.raw_data_dir / filename) == digest
