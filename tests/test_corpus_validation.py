"""Corpus validation against synthetic PDFs: integrity, readability and stray files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from founderlens.corpus import validate as cv
from founderlens.corpus.download import sha256_file
from founderlens.corpus.validate import inspect_pdf, validate_corpus
from tests.conftest import DOC_A, DOC_B, make_pdf_bytes

Factory = Callable[..., Path]


def _run(root: Path, expected_count: int = 2) -> list[str]:
    report = validate_corpus(
        root / "data" / "manifest.json", root / "data" / "raw", expected_count=expected_count
    )
    return [str(issue) for issue in report.issues]


def _raw(root: Path, doc_id: str) -> Path:
    return root / "data" / "raw" / f"{doc_id}.pdf"


def test_valid_corpus_passes(make_project: Factory) -> None:
    root = make_project()
    report = validate_corpus(
        root / "data" / "manifest.json", root / "data" / "raw", expected_count=2
    )
    assert report.ok
    assert [d.document_id for d in report.verified] == [DOC_A, DOC_B]


def test_expected_document_count_enforced(make_project: Factory) -> None:
    assert any("expected 4 documents, found 2" in i for i in _run(make_project(), expected_count=4))


def test_missing_pdf_reported_with_remedy(make_project: Factory) -> None:
    root = make_project()
    _raw(root, DOC_B).unlink()
    issues = _run(root)
    assert len(issues) == 1
    assert issues[0].startswith(f"{DOC_B}: missing file")
    assert "founderlens.corpus.download" in issues[0]


def test_empty_file_reported(make_project: Factory) -> None:
    root = make_project()
    _raw(root, DOC_A).write_bytes(b"")
    assert any("is empty" in i for i in _run(root))


def test_invalid_pdf_signature_reported(make_project: Factory) -> None:
    root = make_project()
    _raw(root, DOC_A).write_bytes(b"<html>Just a moment...</html>")
    assert any("does not start with the PDF signature" in i for i in _run(root))


def test_corrupted_pdf_reported(make_project: Factory) -> None:
    corrupt = b"%PDF-1.7\n" + b"\x00garbage" * 50
    root = make_project({DOC_A: (corrupt, 1)})
    issues = _run(root, expected_count=1)
    assert any("cannot be parsed" in i for i in issues)


def test_truncated_pdf_reported(make_project: Factory) -> None:
    data = make_pdf_bytes(2)
    root = make_project({DOC_A: (data[: len(data) // 2], 2)})
    assert any("cannot be parsed" in i or "damaged" in i for i in _run(root, expected_count=1))


def test_password_protected_pdf_reported(make_project: Factory) -> None:
    root = make_project({DOC_A: (make_pdf_bytes(1, password="secret"), 1)})
    assert any("password-protected" in i for i in _run(root, expected_count=1))


def test_incorrect_hash_reported(make_project: Factory) -> None:
    root = make_project(entry_overrides={DOC_A: {"sha256": "0" * 64}})
    issues = _run(root)
    assert issues == [
        f"{DOC_A}: SHA-256 mismatch: got {sha256_file(_raw(root, DOC_A))}, expected {'0' * 64}"
    ]


def test_incorrect_size_reported(make_project: Factory) -> None:
    root = make_project(entry_overrides={DOC_B: {"size_bytes": 1}})
    assert any("manifest expects 1" in i for i in _run(root))


def test_incorrect_page_count_reported(make_project: Factory) -> None:
    root = make_project(entry_overrides={DOC_A: {"page_count": 9}})
    assert _run(root) == [f"{DOC_A}: page count is 2, manifest expects 9"]


def test_unexpected_files_in_corpus_directory_reported(make_project: Factory) -> None:
    root = make_project()
    raw = root / "data" / "raw"
    (raw / "extra.pdf").write_bytes(make_pdf_bytes(1))
    (raw / ".aaa-2024-q1.pdf.123.part").write_bytes(b"partial")
    (raw / "desktop.ini").write_text("ignored OS metadata", encoding="utf-8")
    issues = _run(root)
    assert any("unexpected file 'extra.pdf'" in i for i in issues)
    assert any(".part" in i for i in issues)
    assert not any("desktop.ini" in i for i in issues)


def test_manifest_errors_are_reported_not_raised(make_project: Factory) -> None:
    root = make_project()
    (root / "data" / "manifest.json").write_text("[]", encoding="utf-8")
    assert _run(root) == ["manifest: manifest root must be a JSON object"]


def test_validation_is_read_only(make_project: Factory) -> None:
    root = make_project()
    snapshot = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    _run(root)
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == snapshot


def test_inspect_pdf_counts_pages(tmp_path: Path) -> None:
    path = tmp_path / "x.pdf"
    path.write_bytes(make_pdf_bytes(3))
    assert inspect_pdf(path) == (3, None)


def test_cli_succeeds_on_valid_four_document_corpus(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = {f"{c * 3}-2024-q1": (make_pdf_bytes(n), n) for n, c in enumerate("abcd", start=1)}
    root = make_project(docs)
    assert cv.main(["--project-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "Corpus valid" in out
    assert out.count("OK  ") == 4


def test_cli_fails_with_actionable_diagnostics(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_project()  # only two documents; the CLI requires four
    _raw(root, DOC_A).unlink()
    assert cv.main(["--project-root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "Corpus INVALID" in err
    assert "expected 4 documents" in err
    assert "Run: python -m founderlens.corpus.download" in err


def test_cli_reports_config_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cv.main(["--project-root", "relative"]) == 1
    assert "error:" in capsys.readouterr().err
