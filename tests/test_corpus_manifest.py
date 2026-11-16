"""Manifest parsing: structure, types, dates, URLs, filenames, uniqueness, ordering."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.corpus.manifest import (
    EXPECTED_DOCUMENT_COUNT,
    ManifestError,
    is_safe_filename,
    load_manifest,
    parse_manifest,
)
from tests.conftest import DOC_A, DOC_B, document_entry, make_pdf_bytes, manifest_dict


@pytest.fixture
def valid() -> dict[str, Any]:
    data = make_pdf_bytes(1)
    return manifest_dict([document_entry(DOC_A, data, 1), document_entry(DOC_B, data, 1)])


def _problems(data: object) -> str:
    with pytest.raises(ManifestError) as info:
        parse_manifest(data)
    return " | ".join(info.value.problems)


def test_valid_manifest_parses_to_typed_entries(valid: dict[str, Any]) -> None:
    manifest = parse_manifest(valid)
    assert manifest.corpus_version == "1.0.0"
    assert [d.document_id for d in manifest.documents] == [DOC_A, DOC_B]
    first = manifest.documents[0]
    assert first.period_end.isoformat() == "2024-03-31"
    assert first.size_bytes > 0
    assert manifest.page_counts == {DOC_A: 1, DOC_B: 1}
    assert manifest.get(DOC_B) is manifest.documents[1]
    assert manifest.get("missing") is None


def test_download_url_is_optional(valid: dict[str, Any]) -> None:
    del valid["documents"][0]["download_url"]
    assert parse_manifest(valid).documents[0].download_url is None


def test_duplicate_document_ids_rejected(valid: dict[str, Any]) -> None:
    valid["documents"][1]["document_id"] = DOC_A
    assert "duplicate document_id" in _problems(valid)


def test_duplicate_filenames_rejected(valid: dict[str, Any]) -> None:
    valid["documents"][1]["filename"] = valid["documents"][0]["filename"]
    assert "duplicate filename" in _problems(valid)


def test_documents_must_be_sorted(valid: dict[str, Any]) -> None:
    valid["documents"].reverse()
    assert "sorted by document_id" in _problems(valid)


@pytest.mark.parametrize("field", ["sha256", "page_count", "source_url", "title"])
def test_missing_required_field_rejected(valid: dict[str, Any], field: str) -> None:
    del valid["documents"][0][field]
    assert f"missing field(s) ['{field}']" in _problems(valid)


def test_unknown_field_rejected(valid: dict[str, Any]) -> None:
    valid["documents"][0]["sha265"] = "typo"
    assert "unknown field(s) ['sha265']" in _problems(valid)


@pytest.mark.parametrize("value", ["2024-13-01", "2024/03/31", "20240331", "31-03-2024", ""])
def test_invalid_dates_rejected(valid: dict[str, Any], value: str) -> None:
    valid["documents"][0]["period_end"] = value
    assert "period_end" in _problems(valid)


def test_publication_before_period_end_rejected(valid: dict[str, Any]) -> None:
    valid["documents"][0]["publication_date"] = "2024-01-01"
    assert "publication_date precedes period_end" in _problems(valid)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/x",
        "ftp://example.com/x",
        "example.com/x",
        "https://user:pw@example.com/",
    ],
)
def test_invalid_source_url_rejected(valid: dict[str, Any], url: str) -> None:
    valid["documents"][0]["source_url"] = url
    assert "source_url" in _problems(valid)


@pytest.mark.parametrize(
    "url",
    [
        "http://s206.q4cdn.com/479360582/files/x.pdf",
        "https://evil.example.com/x.pdf",
        "https://s206.q4cdn.com.evil.example/x.pdf",
    ],
)
def test_download_url_must_be_https_on_approved_host(valid: dict[str, Any], url: str) -> None:
    valid["documents"][0]["download_url"] = url
    assert "download_url" in _problems(valid)


@pytest.mark.parametrize(
    "filename",
    [
        "../escape.pdf",
        "sub/dir.pdf",
        "..\\escape.pdf",
        "C:\\abs.pdf",
        "/abs.pdf",
        "Upper.pdf",
        "x.txt",
        "a..b.pdf",
    ],
)
def test_unsafe_filenames_rejected(valid: dict[str, Any], filename: str) -> None:
    assert not is_safe_filename(filename)
    valid["documents"][0]["filename"] = filename
    assert "unsafe filename" in _problems(valid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", "0" * 63),
        ("sha256", "G" * 64),
        ("size_bytes", 0),
        ("size_bytes", -1),
        ("page_count", True),
        ("page_count", "10"),
    ],
)
def test_invalid_integrity_fields_rejected(
    valid: dict[str, Any], field: str, value: object
) -> None:
    valid["documents"][0][field] = value
    assert field in _problems(valid)


def test_wrong_schema_version_and_corpus_id_rejected(valid: dict[str, Any]) -> None:
    bad = copy.deepcopy(valid)
    bad["schema_version"] = 2
    bad["corpus_id"] = "other"
    problems = _problems(bad)
    assert "schema_version" in problems
    assert "corpus_id" in problems


def test_non_object_root_rejected() -> None:
    assert "root must be a JSON object" in _problems([])


def test_load_manifest_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(path)


def test_load_manifest_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="cannot read manifest"):
        load_manifest(tmp_path / "absent.json")


def test_committed_manifest_is_valid_and_complete() -> None:
    manifest = load_manifest(load_settings().corpus_manifest_path)
    assert len(manifest.documents) == EXPECTED_DOCUMENT_COUNT
    assert [d.document_id for d in manifest.documents] == [
        "goog-2024-q3",
        "goog-2024-q4",
        "nvda-fy2025-q3",
        "nvda-fy2025-q4",
    ]
    for doc in manifest.documents:
        assert doc.download_url is not None
        assert doc.filename == f"{doc.document_id}.pdf"


def test_committed_manifest_has_no_absolute_paths() -> None:
    text = load_settings().corpus_manifest_path.read_text(encoding="utf-8")
    data = json.loads(text)
    for doc in data["documents"]:
        assert not Path(doc["filename"]).is_absolute()
    assert "C:\\" not in text
    assert "/Users/" not in text
