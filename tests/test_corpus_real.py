"""Explicit smoke test against the four real corpus PDFs in data/raw.

Skipped by default (CI has no PDFs and must not download them). Run locally after
``python -m founderlens.corpus.download``::

    $env:FOUNDERLENS_REAL_CORPUS = "1"; python -m pytest tests/test_corpus_real.py -q
"""

from __future__ import annotations

import os

import pytest

from founderlens.config import load_settings
from founderlens.corpus.validate import validate_corpus

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_CORPUS=1 to validate the real downloaded PDFs",
)


def test_real_corpus_matches_manifest() -> None:
    settings = load_settings()
    report = validate_corpus(settings.corpus_manifest_path, settings.raw_data_dir)
    assert [str(i) for i in report.issues] == []
    assert len(report.verified) == 4
    assert all(d.page_count == 10 for d in report.verified)
