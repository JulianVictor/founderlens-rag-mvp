"""Explicit chunking smoke test against the four real corpus PDFs.

Skipped by default (CI has no PDFs). PDFs and manifest are copied into
``tmp_path``, so nothing outside it is written. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; python -m pytest tests/test_ingestion_chunk_real.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from itertools import pairwise
from pathlib import Path

import pytest

from founderlens.config import Settings, load_settings
from founderlens.corpus.manifest import load_manifest
from founderlens.ingestion.chunk import (
    DEFAULT_CHUNKING_CONFIG,
    count_words,
    page_overlaps,
    parse_chunks_jsonl,
    run_chunking,
    uncovered_characters,
    validate_chunks,
)
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import parse_normalized_jsonl, run_normalization

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 to chunk the real downloaded PDFs",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_corpus_chunking(tmp_path: Path) -> None:
    source = load_settings()
    manifest = load_manifest(source.corpus_manifest_path)
    originals = {d.filename: _sha(source.raw_data_dir / d.filename) for d in manifest.documents}
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    shutil.copyfile(source.corpus_manifest_path, tmp_path / "data" / "manifest.json")
    for filename in originals:
        shutil.copyfile(source.raw_data_dir / filename, raw_dir / filename)
    settings = Settings(tmp_path)

    raw_bytes = run_extraction(settings).artifact_path.read_bytes()
    normalized_path = run_normalization(settings).artifact_path
    normalized_bytes = normalized_path.read_bytes()
    first = run_chunking(settings)
    first_bytes = first.artifact_path.read_bytes()
    second = run_chunking(settings)
    assert second.artifact_path.read_bytes() == first_bytes  # byte-identical reruns
    assert normalized_path.read_bytes() == normalized_bytes  # normalized artifact untouched
    assert (settings.processed_data_dir / "pages.raw.jsonl").read_bytes() == raw_bytes

    pages = parse_normalized_jsonl(normalized_bytes.decode("utf-8"))
    chunks = parse_chunks_jsonl(first_bytes.decode("utf-8"))
    config = DEFAULT_CHUNKING_CONFIG
    assert len(pages) == sum(d.page_count for d in manifest.documents) == 40
    assert validate_chunks(chunks, pages, config) == []
    assert {c.document_id for c in chunks} == {d.document_id for d in manifest.documents}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.word_count <= config.chunk_size_words for c in chunks)
    by_id = {p.page_id: p for p in pages}
    for page in pages:
        page_chunks = [c for c in chunks if c.page_id == page.page_id]
        assert bool(page_chunks) == bool(page.normalized_text.strip())
        assert uncovered_characters(page.normalized_text, page_chunks) == 0
        assert all(
            o <= config.chunk_overlap_words
            for o in page_overlaps(page.normalized_text, page_chunks)
        )
        assert all(a.end_char < b.end_char for a, b in pairwise(page_chunks))  # new content
    for chunk in chunks:
        page = by_id[chunk.page_id]
        assert page.normalized_text[chunk.start_char : chunk.end_char] == chunk.text
        assert (chunk.document_id, chunk.document_sha256, chunk.page_number) == (
            page.document_id,
            page.document_sha256,
            page.page_number,
        )
        assert (chunk.company, chunk.fiscal_period, chunk.source_url) == (
            page.company,
            page.fiscal_period,
            page.source_url,
        )
        entry = manifest.get(chunk.document_id)
        assert entry is not None
        assert chunk.document_sha256 == entry.sha256

    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["chunk_artifact"]["sha256"] == first.artifact_sha256 == _sha(first.artifact_path)
    assert report["normalized_artifact"]["sha256"] == _sha(normalized_path)
    assert report["chunk_count"] == len(chunks)
    assert report["pages_processed"] == 40
    assert report["source_words"] == sum(count_words(p.normalized_text) for p in pages)
    assert report["overlap"]["emitted_equals_source_plus_repeated"] is True

    for filename, digest in originals.items():
        assert _sha(source.raw_data_dir / filename) == digest
        assert _sha(raw_dir / filename) == digest
