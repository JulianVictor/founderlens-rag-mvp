"""Page-aware extraction, artifact publishing, report and CLI on synthetic PDFs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from founderlens.config import Settings
from founderlens.corpus.manifest import DocumentEntry, load_manifest
from founderlens.ingestion import extract as ex
from founderlens.ingestion.extract import (
    BASELINE_CONFIG,
    ExtractionError,
    diagnose_page,
    extract_corpus,
    extract_document,
    page_warnings,
    publish_atomic,
    run_extraction,
    validate_records,
)
from founderlens.ingestion.schema import ExtractionStatus, PageRecord, parse_pages_jsonl
from tests.conftest import DOC_A, DOC_B, make_pdf_bytes, make_text_pdf

Factory = Callable[..., Path]


def _entry(data: bytes, pages: int, document_id: str = "doc-x", **overrides: Any) -> DocumentEntry:
    values: dict[str, Any] = {
        "document_id": document_id,
        "company": "Example Corp",
        "ticker": "EXMP",
        "title": "Example results",
        "document_type": "earnings_release",
        "fiscal_period": "FY2024 Q1",
        "period_end": date(2024, 3, 31),
        "publication_date": date(2024, 4, 25),
        "source_url": "https://example.com/investors/results",
        "download_url": None,
        "acquisition_date": date(2024, 5, 1),
        "filename": f"{document_id}.pdf",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "page_count": pages,
    }
    values.update(overrides)
    return DocumentEntry(**values)


def _write(tmp_path: Path, data: bytes, name: str = "doc-x.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _fail_on_page(
    monkeypatch: pytest.MonkeyPatch, *, document_marker: str, page_index: int
) -> None:
    """Make Page.get_text raise for one page whose text contains ``document_marker``."""
    original: Callable[..., Any] = pymupdf.Page.get_text

    def failing(self: pymupdf.Page, *args: Any, **kwargs: Any) -> Any:
        text = original(self, *args, **kwargs)
        if self.number == page_index and document_marker in text:
            raise RuntimeError("simulated parser failure")
        return text

    monkeypatch.setattr(pymupdf.Page, "get_text", failing)


# --- single document ---------------------------------------------------------


def test_single_page_pdf(tmp_path: Path) -> None:
    data = make_text_pdf(["Revenue $35,082 million"])
    result = extract_document(_entry(data, 1), _write(tmp_path, data))
    assert result.pdf_page_count == 1
    [record] = result.records
    assert record.page_number == 1
    assert record.page_id == "doc-x:p0001"
    assert "Revenue $35,082 million" in record.text
    assert record.character_count == len(record.text)
    assert record.extraction_status is ExtractionStatus.OK


def test_multi_page_numbering_order_and_metadata(tmp_path: Path) -> None:
    data = make_text_pdf([f"Body of physical page {n}" for n in (1, 2, 3)])
    entry = _entry(data, 3, company="Alphabet Inc.", fiscal_period="FY2024 Q4")
    records = extract_document(entry, _write(tmp_path, data)).records
    assert [r.page_number for r in records] == [1, 2, 3]
    assert [r.page_id for r in records] == ["doc-x:p0001", "doc-x:p0002", "doc-x:p0003"]
    for n, record in enumerate(records, start=1):
        assert f"Body of physical page {n}" in record.text
        assert record.document_id == "doc-x"
        assert record.document_sha256 == entry.sha256
        assert record.company == "Alphabet Inc."
        assert record.fiscal_period == "FY2024 Q4"
        assert record.source_url == entry.source_url


def test_empty_pages_are_kept_not_dropped(tmp_path: Path) -> None:
    data = make_text_pdf(["First page", "", "Third page"])
    records = extract_document(_entry(data, 3), _write(tmp_path, data)).records
    assert len(records) == 3
    assert [r.extraction_status for r in records] == [
        ExtractionStatus.OK,
        ExtractionStatus.EMPTY,
        ExtractionStatus.OK,
    ]
    assert records[1].page_id == "doc-x:p0002"
    assert records[1].character_count == len(records[1].text) == 0


def test_unicode_and_financial_punctuation_preserved(tmp_path: Path) -> None:
    text = "Revenue € (1,234.5) \u2019quoted\u2019 — Net income $26,301 ™ 12.5% 日本\n(in millions)"
    data = make_text_pdf([text], unicode_font=True)
    [record] = extract_document(_entry(data, 1), _write(tmp_path, data)).records
    for fragment in ("€", "(1,234.5)", "\u2019quoted\u2019", "—", "$26,301", "™", "12.5%", "日本"):
        assert fragment in record.text
    assert "\n(in millions)" in record.text  # line boundaries are not collapsed


def test_unmapped_glyph_surfaces_as_replacement_character(tmp_path: Path) -> None:
    # The bundled cjk font has no U+2212. With TEXT_CID_FOR_UNKNOWN_UNICODE
    # off (baseline config), the glyph appears as U+FFFD and is diagnosed.
    data = make_text_pdf(["Change \u22125%"], unicode_font=True)
    [record] = extract_document(_entry(data, 1), _write(tmp_path, data)).records
    assert "\ufffd" in record.text
    diag = diagnose_page(record)
    assert diag.replacement_character_count == 1
    assert [w.code for w in page_warnings(diag)] == ["low_text", "replacement_characters"]


def test_missing_file(tmp_path: Path) -> None:
    data = make_pdf_bytes(1)
    with pytest.raises(ExtractionError, match=r"doc-x: cannot read doc-x\.pdf"):
        extract_document(_entry(data, 1), tmp_path / "doc-x.pdf")


@pytest.mark.parametrize(
    "corrupt", [b"%PDF-1.7\nnot a real pdf", make_pdf_bytes(2)[:400]], ids=["garbage", "truncated"]
)
def test_corrupted_pdf_fails_clearly(tmp_path: Path, corrupt: bytes) -> None:
    # The manifest hash matches the corrupt bytes, so only parsing can fail.
    with pytest.raises(ExtractionError, match="doc-x: cannot open PDF"):
        extract_document(_entry(corrupt, 1), _write(tmp_path, corrupt))


def test_hash_mismatch_at_extraction_time(tmp_path: Path) -> None:
    data = make_pdf_bytes(1)
    entry = _entry(data, 1, sha256="0" * 64)
    with pytest.raises(ExtractionError, match="SHA-256 mismatch"):
        extract_document(entry, _write(tmp_path, data))


def test_manifest_page_count_mismatch(tmp_path: Path) -> None:
    data = make_pdf_bytes(2)
    with pytest.raises(ExtractionError, match="PDF has 2 pages, manifest expects 3"):
        extract_document(_entry(data, 3), _write(tmp_path, data))


def test_parser_failure_names_document_and_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = make_pdf_bytes(3)
    _fail_on_page(monkeypatch, document_marker="Synthetic", page_index=1)
    with pytest.raises(ExtractionError, match="doc-x page 2: text extraction failed"):
        extract_document(_entry(data, 3), _write(tmp_path, data))


def test_original_pdf_is_not_modified(tmp_path: Path) -> None:
    data = make_text_pdf(["One", "Two"])
    path = _write(tmp_path, data)
    extract_document(_entry(data, 2), path)
    assert path.read_bytes() == data


def test_baseline_config_is_documented_and_explicit() -> None:
    cfg = BASELINE_CONFIG.to_dict()
    assert cfg["output"] == "text"
    assert cfg["sort"] is True
    assert cfg["ocr"] is False
    expected = (
        pymupdf.TEXT_PRESERVE_LIGATURES
        | pymupdf.TEXT_PRESERVE_WHITESPACE
        | pymupdf.TEXT_MEDIABOX_CLIP
    )
    assert cfg["flags"] == expected
    assert not cfg["flags"] & pymupdf.TEXT_CID_FOR_UNKNOWN_UNICODE
    assert not cfg["flags"] & pymupdf.TEXT_DEHYPHENATE


# --- corpus ------------------------------------------------------------------


def _paths(root: Path) -> tuple[Path, Path]:
    return root / "data" / "manifest.json", root / "data" / "raw"


def test_multiple_documents_in_deterministic_order(make_project: Factory) -> None:
    root = make_project(
        {
            DOC_B: (make_text_pdf(["B one", "B two", "B three"]), 3),
            DOC_A: (make_text_pdf(["A one", "A two"]), 2),
        }
    )
    extraction = extract_corpus(*_paths(root), expected_document_count=2)
    ids = [r.page_id for r in extraction.records]
    assert ids == [
        f"{DOC_A}:p0001",
        f"{DOC_A}:p0002",
        f"{DOC_B}:p0001",
        f"{DOC_B}:p0002",
        f"{DOC_B}:p0003",
    ]
    assert [d.pdf_page_count for d in extraction.documents] == [2, 3]


def test_corpus_validation_runs_before_extraction(make_project: Factory) -> None:
    root = make_project()
    (root / "data" / "raw" / f"{DOC_B}.pdf").unlink()
    with pytest.raises(ExtractionError) as info:
        extract_corpus(*_paths(root), expected_document_count=2)
    assert info.value.problems[0].startswith("corpus validation failed")
    assert any(p.startswith(f"{DOC_B}: missing file") for p in info.value.problems)


def test_manifest_page_count_mismatch_is_caught_for_the_corpus(make_project: Factory) -> None:
    root = make_project(entry_overrides={DOC_A: {"page_count": 5}})
    with pytest.raises(ExtractionError, match="page count is 2, manifest expects 5"):
        extract_corpus(*_paths(root), expected_document_count=2)


def test_validate_records_detects_gaps_duplicates_and_metadata(make_project: Factory) -> None:
    root = make_project()
    manifest_path, raw = _paths(root)
    manifest = load_manifest(manifest_path)
    records = list(extract_corpus(manifest_path, raw, expected_document_count=2).records)
    assert validate_records(records, manifest) == []

    assert any(
        "expected 5 page records, got 4" in p for p in validate_records(records[:-1], manifest)
    )
    swapped = [records[1], records[0], *records[2:]]
    assert any("not in manifest document order" in p for p in validate_records(swapped, manifest))
    duplicated = [*records[:-1], records[0]]
    assert any("duplicate page_id" in p for p in validate_records(duplicated, manifest))
    wrong = [replace(records[0], company="Other Corp"), *records[1:]]
    assert any("company does not match" in p for p in validate_records(wrong, manifest))


# --- artifact and report -----------------------------------------------------


def _artifact(root: Path) -> Path:
    return root / "data" / "processed" / "pages.raw.jsonl"


def _leftovers(root: Path) -> list[str]:
    folder = root / "data" / "processed"
    return (
        sorted(p.name for p in folder.iterdir() if p.name.endswith(".tmp"))
        if folder.exists()
        else []
    )


def test_artifact_format(make_project: Factory) -> None:
    root = make_project({DOC_A: (make_text_pdf(["Alpha", "", "Gamma"]), 3)})
    outcome = run_extraction(Settings(root), expected_document_count=1)
    raw = _artifact(root).read_bytes()
    text = raw.decode("utf-8")
    assert b"\r" not in raw
    lines = text.split("\n")
    assert lines[-1] == ""  # ends with a single LF
    assert len(lines[:-1]) == 3
    for line in lines[:-1]:
        obj = json.loads(line)
        assert list(obj) == sorted(obj)
    records = parse_pages_jsonl(text)
    assert len({r.page_id for r in records}) == 3
    assert records[1].extraction_status is ExtractionStatus.EMPTY
    assert str(root) not in text  # no machine-specific paths
    assert str(root).replace("\\", "/") not in text
    assert outcome.artifact_sha256 == hashlib.sha256(raw).hexdigest()
    assert not _leftovers(root)


def test_repeated_runs_are_byte_identical(make_project: Factory) -> None:
    root = make_project()
    first = run_extraction(Settings(root), expected_document_count=2)
    first_bytes = _artifact(root).read_bytes()
    second = run_extraction(Settings(root), expected_document_count=2)
    assert _artifact(root).read_bytes() == first_bytes
    assert first.artifact_sha256 == second.artifact_sha256
    deterministic = {k: v for k, v in first.report.items() if k != "operational"}
    assert deterministic == {k: v for k, v in second.report.items() if k != "operational"}


def test_failed_run_writes_no_partial_output(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(
        {DOC_A: (make_text_pdf(["A one"]), 1), DOC_B: (make_text_pdf(["B one", "B two"]), 2)}
    )
    _fail_on_page(monkeypatch, document_marker="B two", page_index=1)
    with pytest.raises(ExtractionError, match=f"{DOC_B} page 2"):
        run_extraction(Settings(root), expected_document_count=2)
    assert not _artifact(root).exists()
    assert not (root / "reports" / "extraction_report.json").exists()
    assert not (root / "data" / "processed").exists()  # nothing created on failure


def test_failed_run_preserves_previous_artifact(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project(
        {DOC_A: (make_text_pdf(["A one"]), 1), DOC_B: (make_text_pdf(["B one", "B two"]), 2)}
    )
    run_extraction(Settings(root), expected_document_count=2)
    before = _artifact(root).read_bytes()
    report_before = (root / "reports" / "extraction_report.json").read_bytes()

    _fail_on_page(monkeypatch, document_marker="B two", page_index=1)
    with pytest.raises(ExtractionError):
        run_extraction(Settings(root), expected_document_count=2)
    assert _artifact(root).read_bytes() == before
    assert (root / "reports" / "extraction_report.json").read_bytes() == report_before
    assert not _leftovers(root)


def test_failed_verification_keeps_destination_and_removes_temp(tmp_path: Path) -> None:
    dest = tmp_path / "out" / "pages.raw.jsonl"
    publish_atomic(dest, b"previous\n")

    def reject(_: Path) -> None:
        raise ExtractionError(["verification failed"])

    with pytest.raises(ExtractionError, match="verification failed"):
        publish_atomic(dest, b"new partial", reject)
    assert dest.read_bytes() == b"previous\n"
    assert sorted(p.name for p in dest.parent.iterdir()) == ["pages.raw.jsonl"]


def test_artifact_verification_rejects_inconsistent_bytes(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_project()
    run_extraction(Settings(root), expected_document_count=2)
    before = _artifact(root).read_bytes()
    # Simulate a serializer bug that drops the last record.
    monkeypatch.setattr(
        ex,
        "serialize_records",
        lambda records: b"".join(r.to_json_line().encode() for r in records[:-1]),
    )
    with pytest.raises(ExtractionError, match="written artifact invalid"):
        run_extraction(Settings(root), expected_document_count=2)
    assert _artifact(root).read_bytes() == before


def test_report_contents(make_project: Factory) -> None:
    # insert_text does not wrap, so long text is given as several short lines.
    long_page = "\n".join(["Alpha beta gamma delta epsilon"] * 8)
    root = make_project({DOC_A: (make_text_pdf([long_page, "", "short"]), 3)})
    outcome = run_extraction(Settings(root), expected_document_count=1)
    report = json.loads((root / "reports" / "extraction_report.json").read_text(encoding="utf-8"))
    assert report == json.loads(json.dumps(outcome.report))
    manifest_bytes = (root / "data" / "manifest.json").read_bytes()
    assert report["corpus_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert report["parser"] == {
        "name": "PyMuPDF",
        "module": "pymupdf",
        "version": pymupdf.VersionBind,
    }
    assert report["extraction_config"] == BASELINE_CONFIG.to_dict()
    assert report["documents_processed"] == 1
    assert report["pages_expected"] == report["pages_extracted"] == 3
    assert report["empty_pages"] == 1
    records = parse_pages_jsonl(_artifact(root).read_text(encoding="utf-8"))
    assert report["total_characters"] == sum(r.character_count for r in records)
    [doc] = report["documents"]
    assert doc["pdf_page_count"] == doc["manifest_page_count"] == 3
    assert doc["mean_characters_per_page"] == round(report["total_characters"] / 3, 1)
    assert doc["low_text_pages"] == [f"{DOC_A}:p0003"]
    assert [p["page_id"] for p in report["pages"]] == [r.page_id for r in records]
    codes = {(w["code"], w["page_id"]) for w in report["warnings"]}
    assert codes == {("empty_page", f"{DOC_A}:p0002"), ("low_text", f"{DOC_A}:p0003")}
    assert all(w["source"] == "automatic_heuristic" for w in report["warnings"])
    assert report["manual_observations"]["documented_in"].startswith("docs/ingestion.md")
    assert report["artifact"]["path"] == "data/processed/pages.raw.jsonl"
    assert report["artifact"]["records"] == 3
    assert set(report["operational"]) == {"note", "extraction_duration_seconds"}


def test_page_records_round_trip_from_artifact(make_project: Factory) -> None:
    root = make_project()
    outcome = run_extraction(Settings(root), expected_document_count=2)
    records = parse_pages_jsonl(_artifact(root).read_text(encoding="utf-8"))
    assert records == list(outcome.extraction.records)
    assert all(isinstance(r, PageRecord) for r in records)


# --- CLI ---------------------------------------------------------------------


def _four_documents() -> dict[str, tuple[bytes, int]]:
    ids = ["aaa-2024-q1", "bbb-2024-q1", "ccc-2024-q1", "ddd-2024-q1"]
    return {doc_id: (make_text_pdf([f"{doc_id} SECRET-BODY page 1", ""]), 2) for doc_id in ids}


def test_cli_success(make_project: Factory, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_project(_four_documents())
    assert ex.main(["--project-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "Pages: 8 extracted of 8 expected; empty: 4" in out
    assert "Artifact: data/processed/pages.raw.jsonl" in out
    assert "Report:   reports/extraction_report.json" in out
    assert "[empty_page] aaa-2024-q1:p0002" in out
    assert "SECRET-BODY" not in out  # page text is never printed
    assert _artifact(root).is_file()


def test_cli_failure_is_actionable(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    documents = _four_documents()
    root = make_project(documents)
    (root / "data" / "raw" / "ccc-2024-q1.pdf").unlink()
    assert ex.main(["--project-root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "Extraction FAILED" in err
    assert "corpus validation failed" in err
    assert "ccc-2024-q1: missing file" in err
    assert "No artifact was published" in err
    assert not _artifact(root).exists()


def test_cli_rejects_invalid_project_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ex.main(["--project-root", str(tmp_path / "missing")]) == 1
    assert capsys.readouterr().err.startswith("error:")
