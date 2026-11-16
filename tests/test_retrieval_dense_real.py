"""Explicit real-model test of exact dense retrieval on the full V2 collection.

Skipped by default: needs the corpus PDFs (``FOUNDERLENS_REAL_CORPUS=1``) and
the pinned model in the local Hugging Face cache (``FOUNDERLENS_REAL_MODEL=1``).
Builds V2 and its embeddings under ``tmp_path`` with the existing pipeline,
downloads nothing, and checks function and invariants, not relevance. Run::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; $env:FOUNDERLENS_REAL_MODEL = "1"
    python -m pytest tests/test_retrieval_dense_real.py -q
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from founderlens.config import Settings, load_settings
from founderlens.corpus.manifest import load_manifest
from founderlens.ingestion.chunk import run_chunking
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import run_normalization
from founderlens.retrieval import embeddings as em
from founderlens.retrieval.collections import collection_path, load_named_collection
from founderlens.retrieval.dense import (
    DENSE_SMOKE_QUERIES,
    SCORE_BOUND_TOLERANCE,
    QueryTooLongError,
    load_dense_retriever,
)
from founderlens.retrieval.embeddings import (
    EmbeddingConfig,
    SentenceTransformerEncoder,
    run_build,
    run_build_collection,
)
from founderlens.retrieval.schema import RetrievalError

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1"
    or os.environ.get("FOUNDERLENS_REAL_MODEL") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 and FOUNDERLENS_REAL_MODEL=1 (model must be cached)",
)

REPEAT_TOLERANCE = 1e-6


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_dense_retrieval_on_v2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = load_settings()
    manifest = load_manifest(source.corpus_manifest_path)
    originals = {d.filename: _sha(source.raw_data_dir / d.filename) for d in manifest.documents}
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    shutil.copyfile(source.corpus_manifest_path, tmp_path / "data" / "manifest.json")
    for filename in originals:
        shutil.copyfile(source.raw_data_dir / filename, raw_dir / filename)
    settings = Settings(tmp_path)
    run_extraction(settings)
    run_normalization(settings)
    run_chunking(settings)
    encoder = SentenceTransformerEncoder(EmbeddingConfig())
    run_build_collection(settings, encoder)
    v2 = load_named_collection(settings, "v2")
    built = run_build(settings, encoder, EmbeddingConfig(), v2, collection_path(settings, "v2"))
    repository_matrix = (
        source.artifacts_dir / "embeddings" / v2.chunking_config_id / "embeddings.npy"
    )
    if repository_matrix.is_file():  # same artifact as the local frozen one
        assert built.manifest["matrix"]["sha256"] == _sha(repository_matrix)

    retriever = load_dense_retriever(settings, "v2")
    assert retriever.identity()["chunks"] == len(v2.chunks) == 164
    batches: list[int] = []
    original_encode = retriever.encoder.encode

    def recording(texts: Any) -> Any:
        batches.append(len(texts))
        return original_encode(texts)

    monkeypatch.setattr(retriever.encoder, "encode", recording)
    monkeypatch.setattr(em, "write_embedding_artifact", lambda *_, **__: pytest.fail("rebuild"))

    by_id = {c.chunk_id: c for c in v2.chunks}
    pages = {(d.document_id, n) for d in manifest.documents for n in range(1, d.page_count + 1)}
    for _, query in DENSE_SMOKE_QUERIES[:-2]:  # all but the empty and oversized inputs
        vector = retriever.encode_query(query)
        if vector is None:
            assert retriever.search(query) == ()
            continue
        assert vector.shape == (384,)
        assert vector.dtype == np.float32
        assert abs(float(np.linalg.norm(vector.astype(np.float64))) - 1.0) <= 1e-5
        results = retriever.search(query, top_k=10)
        assert [r.rank for r in results] == list(range(1, 11))
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(abs(s) <= 1.0 + SCORE_BOUND_TOLERANCE for s in scores)
        for r in results:
            assert by_id[r.chunk.chunk_id] == r.chunk
            assert (r.chunk.document_id, r.chunk.page_number) in pages
        again = retriever.search(query, top_k=10)
        assert [r.chunk.chunk_id for r in again] == [r.chunk.chunk_id for r in results]
        assert (
            max(abs(a.score - b.score) for a, b in zip(again, results, strict=True))
            <= REPEAT_TOLERANCE
        )
    assert batches
    assert set(batches) == {1}  # only single query vectors, never documents

    fresh = load_dense_retriever(settings, "v2")  # independent model load and matrix
    query = "How quickly is the AI chip maker's main business growing?"
    first = retriever.search(query, top_k=164)
    second = fresh.search(query, top_k=164)
    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]
    assert (
        max(abs(a.score - b.score) for a, b in zip(first, second, strict=True)) <= REPEAT_TOLERANCE
    )

    with pytest.raises(QueryTooLongError, match="never truncated"):
        retriever.search("revenue " * 300)
    with pytest.raises(RetrievalError, match="cannot be embedded"):
        load_dense_retriever(settings, "v1")
    data = bytearray(built.directory.joinpath("embeddings.npy").read_bytes())
    data[-1] ^= 0xFF
    built.directory.joinpath("embeddings.npy").write_bytes(bytes(data))
    with pytest.raises(RetrievalError, match="SHA-256 does not match"):
        load_dense_retriever(settings, "v2")

    for filename, digest in originals.items():
        assert _sha(source.raw_data_dir / filename) == digest
