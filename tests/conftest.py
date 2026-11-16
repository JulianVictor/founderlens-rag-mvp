"""Shared fixtures: synthetic PDFs and a temporary corpus layout (no real documents)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf
import pytest

DOC_A = "aaa-2024-q1"
DOC_B = "bbb-2024-q1"


def make_pdf_bytes(pages: int = 2, *, password: str | None = None) -> bytes:
    """A small synthetic PDF with ``pages`` pages of text (optionally encrypted)."""
    with pymupdf.open() as doc:
        for number in range(1, pages + 1):
            page = doc.new_page()
            page.insert_text((72, 72), f"Synthetic test page {number}")
        if password is None:
            data: bytes = doc.tobytes()
        else:
            data = doc.tobytes(
                encryption=pymupdf.mupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password
            )
    return data


def make_text_pdf(pages: list[str], *, unicode_font: bool = False) -> bytes:
    """A synthetic PDF with one page per string; ``""`` produces a blank page.

    ``unicode_font`` embeds PyMuPDF's bundled ``cjk`` font (Droid Sans
    Fallback), which maps the euro sign, curly quotes, dashes, the trademark
    sign and CJK glyphs but has no U+2212 minus sign. The default Base-14
    Helvetica covers ASCII only.
    """
    font = pymupdf.Font("cjk") if unicode_font else None
    with pymupdf.open() as doc:
        for text in pages:
            page = doc.new_page()
            if not text:
                continue
            if font is None:
                page.insert_text((72, 72), text, fontsize=11)
            else:
                page.insert_font(fontname="F0", fontbuffer=font.buffer)
                page.insert_text((72, 72), text, fontname="F0", fontsize=11)
        data: bytes = doc.tobytes(garbage=3, deflate=True)
    return data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document_entry(document_id: str, data: bytes, pages: int, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "document_id": document_id,
        "company": "Example Corp",
        "ticker": "EXMP",
        "title": f"Example results {document_id}",
        "document_type": "earnings_release",
        "fiscal_period": "FY2024 Q1",
        "period_end": "2024-03-31",
        "publication_date": "2024-04-25",
        "source_url": "https://example.com/investors/results",
        "download_url": "https://s206.q4cdn.com/479360582/files/example.pdf",
        "acquisition_date": "2024-05-01",
        "filename": f"{document_id}.pdf",
        "sha256": sha256(data),
        "size_bytes": len(data),
        "page_count": pages,
    }
    entry.update(overrides)
    return entry


def manifest_dict(documents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus_id": "founderlens-financial-research",
        "corpus_version": "1.0.0",
        "description": "Synthetic test corpus.",
        "documents": documents,
    }


ProjectFactory = Callable[..., Path]


@pytest.fixture
def make_project(tmp_path: Path) -> ProjectFactory:
    """Create ``tmp_path`` as a project root with data/manifest.json and data/raw PDFs.

    Returns a factory: ``make_project(documents={id: (bytes, pages)}, write_files=True,
    entry_overrides={id: {...}})`` -> project root.
    """

    def factory(
        documents: dict[str, tuple[bytes, int]] | None = None,
        *,
        write_files: bool = True,
        entry_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> Path:
        if documents is None:
            documents = {DOC_A: (make_pdf_bytes(2), 2), DOC_B: (make_pdf_bytes(3), 3)}
        overrides = entry_overrides or {}
        raw = tmp_path / "data" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        entries = []
        for doc_id, (data, pages) in sorted(documents.items()):
            entries.append(document_entry(doc_id, data, pages, **overrides.get(doc_id, {})))
            if write_files:
                (raw / f"{doc_id}.pdf").write_bytes(data)
        (tmp_path / "data" / "manifest.json").write_text(
            json.dumps(manifest_dict(entries), indent=2), encoding="utf-8"
        )
        return tmp_path

    return factory
