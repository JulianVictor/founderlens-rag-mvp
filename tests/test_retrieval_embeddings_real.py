"""Explicit real-model, real-corpus embedding test.

Skipped by default: needs the corpus PDFs (``FOUNDERLENS_REAL_CORPUS=1``) and
the pinned model in the local Hugging Face cache (``FOUNDERLENS_REAL_MODEL=1``;
fetch it once with ``python -m founderlens.retrieval.embeddings audit
--allow-download``). It never downloads anything itself. Everything is written
under ``tmp_path``. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; $env:FOUNDERLENS_REAL_MODEL = "1"
    python -m pytest tests/test_retrieval_embeddings_real.py -q
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
from founderlens.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    SentenceTransformerEncoder,
    audit_token_lengths,
    build_manifest,
    embed_texts,
    load_embedding_artifact,
    run_build,
    write_embedding_artifact,
)
from founderlens.retrieval.schema import ChunkCollection, load_chunk_collection

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1"
    or os.environ.get("FOUNDERLENS_REAL_MODEL") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 and FOUNDERLENS_REAL_MODEL=1 (model must be cached)",
)

REPEAT_TOLERANCE = 1e-6  # max absolute difference between repeated runs on one machine


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_model_audit_embedding_and_artifact(tmp_path: Path) -> None:
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
    chunks_path = run_chunking(settings).artifact_path
    chunks_sha = _sha(chunks_path)
    collection = load_chunk_collection(chunks_path, settings.corpus_manifest_path, normalized)

    config = EmbeddingConfig()
    encoder = SentenceTransformerEncoder(config)
    facts = encoder.describe()
    assert facts["model_revision"] == config.model_revision
    assert facts["effective_max_seq_length"] == 256
    assert facts["output_dimension"] == 384
    assert facts["tokenizer_class"] == "BertTokenizer"

    audit = audit_token_lengths(collection.chunks, encoder)
    assert len(audit.lengths) == len(collection.chunks)
    assert all(n >= 2 for n in audit.lengths)  # [CLS] and [SEP] are always counted
    over = {c.chunk_id for c, n in zip(collection.chunks, audit.lengths, strict=True) if n > 256}
    assert {o["chunk_id"] for o in audit.overflows} == over

    if not audit.ok:
        # The pipeline must refuse: no truncation and no artifact.
        with pytest.raises(EmbeddingError, match="exceed the 256-token limit"):
            run_build(settings, encoder, config, collection, chunks_path)
        assert not (tmp_path / "artifacts" / "embeddings" / collection.chunking_config_id).exists()
        too_long = next(c for c in collection.chunks if c.chunk_id in over)
        with pytest.raises(EmbeddingError, match="would be truncated"):
            embed_texts([too_long.text], encoder, config)

    # Real-model invariants and repeatability on every chunk that fits (test-only data).
    fitting = tuple(c for c in collection.chunks if c.chunk_id not in over)
    texts = [c.text for c in fitting]
    first = embed_texts(texts, encoder, config, labels=[c.chunk_id for c in fitting])
    second = embed_texts(texts, SentenceTransformerEncoder(config), config)
    assert first.shape == (len(fitting), 384)
    assert first.dtype == np.float32
    assert np.allclose(np.linalg.norm(first.astype(np.float64), axis=1), 1.0, atol=1e-5)
    assert float(np.max(np.abs(first - second))) <= REPEAT_TOLERANCE
    single = embed_texts(texts[:1], encoder, config)
    assert float(np.max(np.abs(single[0] - first[0]))) <= REPEAT_TOLERANCE  # batch-independent

    subset = ChunkCollection(
        chunks=fitting,
        artifact_sha256=hashlib.sha256("\n".join(c.chunk_id for c in fitting).encode()).hexdigest(),
        chunking_config_id=collection.chunking_config_id,
        normalized_sha256=collection.normalized_sha256,
    )
    directory = tmp_path / "artifacts" / "subset-test"
    meta = build_manifest(
        first, [c.chunk_id for c in fitting], subset, config, facts, chunks_path="test-subset"
    )
    write_embedding_artifact(directory, first, meta, subset, config)
    loaded = load_embedding_artifact(directory, subset, config)
    assert np.array_equal(loaded.matrix, first)
    assert loaded.chunk_ids == tuple(c.chunk_id for c in fitting)

    assert _sha(chunks_path) == chunks_sha
    for filename, digest in originals.items():
        assert _sha(source.raw_data_dir / filename) == digest
