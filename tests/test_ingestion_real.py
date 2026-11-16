"""Explicit extraction smoke test against the four real corpus PDFs.

Skipped by default (CI has no PDFs). The PDFs and manifest are copied into
``tmp_path`` so the test writes nothing outside it. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; python -m pytest tests/test_ingestion_real.py -q
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
from founderlens.ingestion.schema import make_page_id, parse_pages_jsonl

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 to extract the real downloaded PDFs",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_corpus_extraction(tmp_path: Path) -> None:
    source = load_settings()
    manifest = load_manifest(source.corpus_manifest_path)
    originals = {d.filename: _sha(source.raw_data_dir / d.filename) for d in manifest.documents}

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    shutil.copyfile(source.corpus_manifest_path, tmp_path / "data" / "manifest.json")
    for filename in originals:
        shutil.copyfile(source.raw_data_dir / filename, raw / filename)
    settings = Settings(tmp_path)

    first = run_extraction(settings)
    first_bytes = first.artifact_path.read_bytes()
    second = run_extraction(settings)
    assert second.artifact_path.read_bytes() == first_bytes  # byte-identical reruns
    assert (
        first.artifact_sha256 == second.artifact_sha256 == hashlib.sha256(first_bytes).hexdigest()
    )

    records = parse_pages_jsonl(first_bytes.decode("utf-8"))
    assert len(manifest.documents) == 4
    assert len(records) == sum(d.page_count for d in manifest.documents) == 40
    assert len({r.page_id for r in records}) == len(records)
    expected_ids = [
        make_page_id(d.document_id, n)
        for d in manifest.documents
        for n in range(1, d.page_count + 1)
    ]
    assert [r.page_id for r in records] == expected_ids
    for record in records:
        entry = manifest.get(record.document_id)
        assert entry is not None
        assert record.document_sha256 == entry.sha256
        assert record.company == entry.company
        assert record.fiscal_period == entry.fiscal_period
        assert record.source_url == entry.source_url

    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["documents_processed"] == 4
    assert report["pages_expected"] == report["pages_extracted"] == 40
    assert report["artifact"]["sha256"] == first.artifact_sha256
    assert [d["pdf_page_count"] for d in report["documents"]] == [10, 10, 10, 10]
    # Empty or suspicious pages are reported, never hidden: the report must
    # account for every page and list a warning for each flagged one.
    assert len(report["pages"]) == 40
    flagged = {p["page_id"] for p in report["pages"] if p["low_text"]} | {
        p["page_id"] for p in report["pages"] if p["extraction_status"] == "empty"
    }
    assert flagged <= {w["page_id"] for w in report["warnings"]}

    # The source PDFs, and the copies that were read, are unchanged.
    for filename, digest in originals.items():
        assert _sha(source.raw_data_dir / filename) == digest
        assert _sha(raw / filename) == digest
