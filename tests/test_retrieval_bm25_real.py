"""Explicit BM25 smoke test against the chunk collection built from the real PDFs.

Skipped by default (CI has no PDFs). PDFs and manifest are copied into
``tmp_path`` and the collection is rebuilt there, so nothing outside it is
written. Checks function and determinism, not relevance. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; python -m pytest tests/test_retrieval_bm25_real.py -q
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import pytest

from founderlens.config import Settings, load_settings
from founderlens.corpus.manifest import load_manifest
from founderlens.ingestion.chunk import run_chunking
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import run_normalization
from founderlens.retrieval.bm25 import SMOKE_QUERIES, BM25Retriever
from founderlens.retrieval.schema import load_chunk_collection

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 to search the real chunk collection",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_corpus_bm25(tmp_path: Path) -> None:
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
    inputs = {p: _sha(p) for p in (normalized, chunks_path, tmp_path / "data" / "manifest.json")}

    collection = load_chunk_collection(chunks_path, settings.corpus_manifest_path, normalized)
    assert collection.artifact_sha256 == _sha(chunks_path)
    repository_chunks = source.processed_data_dir / "chunks.jsonl"
    if repository_chunks.is_file():  # same collection as the local baseline artifact
        assert collection.artifact_sha256 == _sha(repository_chunks)
    assert {c.document_id for c in collection.chunks} == {d.document_id for d in manifest.documents}

    retriever = BM25Retriever(collection.chunks)
    rebuilt = BM25Retriever(
        load_chunk_collection(chunks_path, settings.corpus_manifest_path, normalized).chunks
    )
    assert retriever.indexed_chunk_count == len(collection.chunks)  # no chunk lacks tokens
    assert retriever.unindexed_chunk_ids == ()
    pages = {(d.document_id, n) for d in manifest.documents for n in range(1, d.page_count + 1)}
    originals_by_id = {c.chunk_id: c for c in collection.chunks}
    for _, query in SMOKE_QUERIES:
        results = retriever.search(query, top_k=10)
        assert rebuilt.search(query, top_k=10) == results  # deterministic across builds
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        for result in results:
            assert originals_by_id[result.chunk.chunk_id] == result.chunk
            assert (result.chunk.document_id, result.chunk.page_number) in pages
    assert retriever.search("zyzzyva quokka") == ()
    assert retriever.search("$ % ( ) -") == ()
    assert retriever.search("Blackwell")  # a term known to occur in the NVIDIA releases

    assert {p: _sha(p) for p in inputs} == inputs
    for filename, digest in originals.items():
        assert _sha(source.raw_data_dir / filename) == digest
        assert _sha(raw_dir / filename) == digest
