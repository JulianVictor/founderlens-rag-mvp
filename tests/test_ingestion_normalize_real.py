"""Explicit normalization smoke test against the four real corpus PDFs.

Skipped by default (CI has no PDFs). PDFs and manifest are copied into
``tmp_path``, so nothing outside it is written. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; python -m pytest tests/test_ingestion_normalize_real.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from founderlens.config import Settings, load_settings
from founderlens.corpus.manifest import load_manifest
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import parse_normalized_jsonl, run_normalization
from founderlens.ingestion.schema import parse_pages_jsonl
from founderlens.ingestion.tables import TableStatus

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 to normalize the real downloaded PDFs",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_corpus_normalization(tmp_path: Path) -> None:
    source = load_settings()
    manifest = load_manifest(source.corpus_manifest_path)
    originals = {d.filename: _sha(source.raw_data_dir / d.filename) for d in manifest.documents}
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    shutil.copyfile(source.corpus_manifest_path, tmp_path / "data" / "manifest.json")
    for filename in originals:
        shutil.copyfile(source.raw_data_dir / filename, raw_dir / filename)
    settings = Settings(tmp_path)

    extraction = run_extraction(settings)
    raw_bytes = extraction.artifact_path.read_bytes()
    first = run_normalization(settings)
    first_bytes = first.artifact_path.read_bytes()
    second = run_normalization(settings)
    assert second.artifact_path.read_bytes() == first_bytes  # byte-identical reruns
    assert extraction.artifact_path.read_bytes() == raw_bytes  # raw artifact untouched

    raw = parse_pages_jsonl(raw_bytes.decode("utf-8"))
    records = parse_normalized_jsonl(first_bytes.decode("utf-8"))
    assert len(records) == sum(d.page_count for d in manifest.documents) == 40
    assert [r.page_id for r in records] == [r.page_id for r in raw]
    for norm, src in zip(records, raw, strict=True):
        assert (norm.document_sha256, norm.company, norm.fiscal_period, norm.source_url) == (
            src.document_sha256,
            src.company,
            src.fiscal_period,
            src.source_url,
        )
        assert norm.raw_character_count == src.character_count
    tables = [t for r in records for t in r.tables]
    assert len({t.table_id for t in tables}) == len(tables)
    assert all(t.structural_problems() == [] for t in tables)

    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["raw_artifact"]["sha256"] == _sha(extraction.artifact_path)
    assert report["normalized_artifact"]["sha256"] == first.artifact_sha256
    assert report["pages"] == 40
    counts = report["tables"]
    assert counts["candidates"] == len(tables)
    assert counts["structured"] + counts["uncertain"] + counts["fallback"] == len(tables)
    # Every non-structured table is reported on its page, never hidden.
    flagged = {t.table_id for t in tables if t.structure_status is not TableStatus.STRUCTURED}
    reported = " ".join(w["message"] for w in report["warnings"])
    assert all(table_id in reported for table_id in flagged)

    for filename, digest in originals.items():
        assert _sha(source.raw_data_dir / filename) == digest
        assert _sha(raw_dir / filename) == digest
