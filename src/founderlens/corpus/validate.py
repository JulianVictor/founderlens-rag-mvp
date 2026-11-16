"""Verify that the local corpus exactly matches ``data/manifest.json``.

Usage::

    python -m founderlens.corpus.validate

Checks the manifest structure, then for every document: the file exists, is
non-empty, starts with the PDF signature, matches the manifest size and
SHA-256, opens in PyMuPDF without repair, is not password-protected, and has
the expected page count. Also reports unexpected files in ``data/raw``.

Validation is read-only: it never downloads, rewrites the manifest, or
regenerates metadata. PyMuPDF is used only to confirm the files are readable
PDFs and to count pages; no text is extracted or persisted.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from founderlens.config import ConfigError, load_settings
from founderlens.corpus.download import sha256_file
from founderlens.corpus.manifest import (
    EXPECTED_DOCUMENT_COUNT,
    CorpusManifest,
    DocumentEntry,
    ManifestError,
    is_safe_filename,
    load_manifest,
)

PDF_SIGNATURE = b"%PDF-"
# OS metadata files that may appear in any folder and are not corpus content.
_IGNORED_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


@dataclass(frozen=True, slots=True)
class Issue:
    subject: str
    message: str

    def __str__(self) -> str:
        return f"{self.subject}: {self.message}"


@dataclass(slots=True)
class CorpusReport:
    manifest: CorpusManifest | None
    issues: list[Issue] = field(default_factory=list)
    verified: list[DocumentEntry] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


def inspect_pdf(path: Path) -> tuple[int | None, str | None]:
    """Return ``(page_count, problem)`` for a PDF, using PyMuPDF only for validity checks."""
    pymupdf.TOOLS.mupdf_display_errors(False)
    try:
        with pymupdf.open(path, filetype="pdf") as doc:
            if doc.needs_pass or doc.is_encrypted:
                return None, "PDF is password-protected or encrypted"
            if doc.is_repaired:
                return None, "PDF structure is damaged (PyMuPDF had to repair it)"
            if doc.page_count < 1:
                return None, "PDF has no pages"
            for page in doc:
                page.bound()  # forces each page object to load
            return int(doc.page_count), None
    except (RuntimeError, ValueError) as exc:  # pymupdf.FileDataError subclasses RuntimeError
        return None, f"PDF cannot be parsed: {exc}"
    finally:
        pymupdf.TOOLS.reset_mupdf_warnings()


def _check_document(entry: DocumentEntry, raw_dir: Path) -> list[Issue]:
    subject = entry.document_id
    if not is_safe_filename(entry.filename):
        return [Issue(subject, f"unsafe filename {entry.filename!r}")]
    path = raw_dir / entry.filename
    if not path.exists():
        return [Issue(subject, f"missing file {path}. Run: python -m founderlens.corpus.download")]
    if not path.is_file():
        return [Issue(subject, f"{path} is not a regular file")]

    size = path.stat().st_size
    if size == 0:
        return [Issue(subject, f"{path} is empty")]
    with path.open("rb") as handle:
        header = handle.read(len(PDF_SIGNATURE))
    if header != PDF_SIGNATURE:
        return [Issue(subject, f"{path} does not start with the PDF signature {PDF_SIGNATURE!r}")]

    issues: list[Issue] = []
    if size != entry.size_bytes:
        issues.append(Issue(subject, f"size is {size} bytes, manifest expects {entry.size_bytes}"))
    digest = sha256_file(path)
    if digest != entry.sha256:
        issues.append(Issue(subject, f"SHA-256 mismatch: got {digest}, expected {entry.sha256}"))

    page_count, problem = inspect_pdf(path)
    if problem:
        issues.append(Issue(subject, problem))
    elif page_count != entry.page_count:
        issues.append(
            Issue(subject, f"page count is {page_count}, manifest expects {entry.page_count}")
        )
    return issues


def validate_corpus(
    manifest_path: Path,
    raw_dir: Path,
    *,
    expected_count: int = EXPECTED_DOCUMENT_COUNT,
) -> CorpusReport:
    """Validate the manifest and every corpus file. Never modifies anything."""
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        return CorpusReport(None, [Issue("manifest", p) for p in exc.problems])

    report = CorpusReport(manifest)
    if len(manifest.documents) != expected_count:
        report.issues.append(
            Issue(
                "manifest",
                f"expected {expected_count} documents, found {len(manifest.documents)}",
            )
        )

    for entry in manifest.documents:
        issues = _check_document(entry, raw_dir)
        report.issues.extend(issues)
        if not issues:
            report.verified.append(entry)

    if raw_dir.is_dir():
        expected = {d.filename for d in manifest.documents}
        for item in sorted(raw_dir.iterdir()):
            if item.name in expected or item.name in _IGNORED_NAMES:
                continue
            kind = "directory" if item.is_dir() else "file"
            report.issues.append(
                Issue("data/raw", f"unexpected {kind} {item.name!r} (not in manifest)")
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.corpus.validate",
        description="Verify local corpus PDFs against data/manifest.json.",
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = validate_corpus(settings.corpus_manifest_path, settings.raw_data_dir)
    if report.manifest is not None:
        m = report.manifest
        print(f"Corpus {m.corpus_id} v{m.corpus_version}: {len(m.documents)} document(s)")
        verified = {d.document_id for d in report.verified}
        for d in m.documents:
            status = "OK  " if d.document_id in verified else "FAIL"
            print(
                f"  {status} {d.document_id:<16} {d.page_count:>3} pages "
                f"{d.size_bytes:>9} bytes  sha256 {d.sha256[:12]}..."
            )
    if report.ok:
        print("Corpus valid: all files match manifest size, SHA-256 and page count.")
        return 0
    print(f"Corpus INVALID ({len(report.issues)} issue(s)):", file=sys.stderr)
    for issue in report.issues:
        print(f"  - {issue}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
