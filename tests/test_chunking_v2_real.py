"""Explicit real-model test of chunking V2, BM25 on V2 and full V2 embeddings.

Skipped by default: needs the corpus PDFs (``FOUNDERLENS_REAL_CORPUS=1``) and
the pinned model in the local Hugging Face cache (``FOUNDERLENS_REAL_MODEL=1``).
It downloads nothing and writes only under ``tmp_path``. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; $env:FOUNDERLENS_REAL_MODEL = "1"
    python -m pytest tests/test_chunking_v2_real.py -q
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from founderlens.config import Settings, load_settings
from founderlens.corpus.manifest import load_manifest
from founderlens.ingestion.chunk import run_chunking
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import run_normalization
from founderlens.retrieval.bm25 import SMOKE_QUERIES, BM25Retriever
from founderlens.retrieval.collections import (
    CHUNKING_V2_CONFIG,
    collection_path,
    load_named_collection,
)
from founderlens.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    SentenceTransformerEncoder,
    audit_token_lengths,
    load_embedding_artifact,
    run_build,
    run_build_collection,
    run_validate_collection,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1"
    or os.environ.get("FOUNDERLENS_REAL_MODEL") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 and FOUNDERLENS_REAL_MODEL=1 (model must be cached)",
)

REPEAT_TOLERANCE = 1e-6


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_chunking_v2_bm25_and_full_embeddings(tmp_path: Path) -> None:
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
    normalized = run_normalization(settings).artifact_path
    v1_path = run_chunking(settings).artifact_path
    v1_sha, normalized_sha = _sha(v1_path), _sha(normalized)

    encoder = SentenceTransformerEncoder(EmbeddingConfig())
    first = run_build_collection(settings, encoder)
    v2_path = collection_path(settings, "v2")
    v2_bytes = v2_path.read_bytes()
    second = run_build_collection(settings, SentenceTransformerEncoder(EmbeddingConfig()))
    assert v2_path.read_bytes() == v2_bytes  # byte-identical V2 builds
    assert second.artifact_sha256 == first.artifact_sha256
    assert _sha(v1_path) == v1_sha  # V1 untouched
    repository_v2 = collection_path(source, "v2")
    if repository_v2.is_file():
        assert _sha(repository_v2) == first.artifact_sha256  # same as the local V2 artifact

    v2 = run_validate_collection(settings, "v2", encoder)  # spans, coverage, words, tokens
    v1 = load_named_collection(settings, "v1")
    assert v2.chunking_config_id == CHUNKING_V2_CONFIG.config_id
    assert {c.page_id for c in v2.chunks} == {c.page_id for c in v1.chunks}  # every page kept
    audit = audit_token_lengths(v2.chunks, encoder)
    assert audit.ok
    assert max(audit.lengths) <= 256
    assert not audit_token_lengths(v1.chunks, encoder).ok  # V1 still has over-limit chunks
    with pytest.raises(EmbeddingError, match="exceed the 256-token limit"):
        run_build(settings, encoder, EmbeddingConfig(), v1, v1_path)

    bm25 = BM25Retriever(v2.chunks)
    rebuilt = BM25Retriever(load_named_collection(settings, "v2").chunks)
    assert bm25.indexed_chunk_count == len(v2.chunks)
    by_id = {c.chunk_id: c for c in v2.chunks}
    for _, query in SMOKE_QUERIES:
        results = bm25.search(query, top_k=10)
        assert rebuilt.search(query, top_k=10) == results
        assert all(by_id[r.chunk.chunk_id] == r.chunk for r in results)

    built = run_build(settings, encoder, EmbeddingConfig(), v2, v2_path)
    matrix = built.matrix
    assert matrix.shape == (len(v2.chunks), 384)
    assert matrix.dtype == np.float32
    assert np.all(np.isfinite(matrix))
    assert np.allclose(np.linalg.norm(matrix.astype(np.float64), axis=1), 1.0, atol=1e-5)
    loaded = load_embedding_artifact(built.directory, v2, EmbeddingConfig())
    assert loaded.chunk_ids == tuple(c.chunk_id for c in v2.chunks)
    assert np.array_equal(loaded.matrix, matrix)
    with pytest.raises(EmbeddingError, match="different chunk collection"):
        load_embedding_artifact(built.directory, v1, EmbeddingConfig())

    again = run_build(
        settings,
        SentenceTransformerEncoder(EmbeddingConfig()),
        EmbeddingConfig(),
        v2,
        v2_path,
        output_dir="artifacts/embeddings/repeat",
    )
    assert float(np.max(np.abs(again.matrix - matrix))) <= REPEAT_TOLERANCE

    assert _sha(normalized) == normalized_sha
    for filename, digest in originals.items():
        assert _sha(source.raw_data_dir / filename) == digest
