"""Page-aware raw text extraction from the verified corpus.

Usage::

    python -m founderlens.ingestion.extract

Pipeline:

1. Validate the corpus with :func:`founderlens.corpus.validate.validate_corpus`
   (manifest structure, file presence, size, SHA-256, readability, page count).
2. For each manifest document, in manifest order: read the bytes once, check
   their SHA-256 again, open them from memory with PyMuPDF (the file on disk is
   never opened for writing), confirm the page count, and extract every
   physical page in order. A parser failure aborts the run, naming the
   document and page; it is never turned into an empty page.
3. Validate the records against the manifest (count, order, identity,
   metadata, unique page IDs).
4. Publish ``data/processed/pages.raw.jsonl`` atomically: write a temporary
   file in the destination directory, re-read and re-validate it, then rename
   it over the old artifact. A failed run leaves any previous artifact intact.
5. Write ``reports/extraction_report.json`` (diagnostics and run timing).

Text is the extractor's raw output. No cleaning, normalization or chunking
happens here (Commit 2.2 and later). No network access, no LLM.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import operator
import os
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.corpus.manifest import EXPECTED_DOCUMENT_COUNT, CorpusManifest, DocumentEntry
from founderlens.corpus.validate import validate_corpus
from founderlens.ingestion.schema import (
    PAGE_SCHEMA_VERSION,
    ExtractionStatus,
    PageRecord,
    PageRecordError,
    parse_pages_jsonl,
)

PAGES_RAW_FILENAME = "pages.raw.jsonl"
REPORT_FILENAME = "extraction_report.json"

# Heuristic threshold for "suspiciously little text" on a non-empty page. It is
# a diagnostic signal only: a short page (for example a closing page with a few
# contact lines) can be perfectly extracted.
LOW_TEXT_MIN_NON_WHITESPACE = 100
REPLACEMENT_CHARACTER = "�"

MANUAL_OBSERVATIONS_REF = "docs/ingestion.md#manual-extraction-observations"


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """PyMuPDF text-extraction settings. Changing any field changes the artifact.

    Baseline (Commit 2.1):

    * ``Page.get_text("text")``: plain text, one output line per text line,
      with no HTML/JSON structure.
    * ``sort=True``: lines are ordered by position (top-to-bottom, then
      left-to-right) instead of content-stream order. On these releases
      content-stream order puts every table cell on its own line and
      separates ``$`` from its value; position order keeps each table row's
      label and values on one line, padded with spaces.
    * flags ``TEXT_PRESERVE_LIGATURES | TEXT_PRESERVE_WHITESPACE |
      TEXT_MEDIABOX_CLIP``: PyMuPDF's default ``text`` flags *minus*
      ``TEXT_CID_FOR_UNKNOWN_UNICODE``, so a glyph with no Unicode mapping
      appears as U+FFFD (counted by the diagnostics) rather than as an
      arbitrary character code. No dehyphenation, no image placeholders.
    """

    output: str = "text"
    sort: bool = True
    flag_names: tuple[str, ...] = (
        "TEXT_PRESERVE_LIGATURES",
        "TEXT_PRESERVE_WHITESPACE",
        "TEXT_MEDIABOX_CLIP",
    )

    @property
    def flags(self) -> int:
        return functools.reduce(operator.or_, (int(getattr(pymupdf, n)) for n in self.flag_names))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "pymupdf.Page.get_text",
            "output": self.output,
            "sort": self.sort,
            "flags": self.flags,
            "flag_names": list(self.flag_names),
            "ocr": False,
            "text_modifications": "none (raw extractor output)",
        }


BASELINE_CONFIG = ExtractionConfig()


class ExtractionError(RuntimeError):
    """Extraction run failed. ``problems`` lists every actionable reason."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PageDiagnostics:
    page_id: str
    extraction_status: ExtractionStatus
    character_count: int
    non_whitespace_count: int
    line_count: int
    replacement_character_count: int
    low_text: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "extraction_status": self.extraction_status.value,
            "character_count": self.character_count,
            "non_whitespace_count": self.non_whitespace_count,
            "line_count": self.line_count,
            "replacement_character_count": self.replacement_character_count,
            "low_text": self.low_text,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticWarning:
    """An automatically detected heuristic signal; not proof of a wrong extraction."""

    code: str
    page_id: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": "automatic_heuristic",
            "code": self.code,
            "page_id": self.page_id,
            "message": self.message,
        }


def diagnose_page(record: PageRecord) -> PageDiagnostics:
    non_whitespace = sum(1 for ch in record.text if not ch.isspace())
    return PageDiagnostics(
        page_id=record.page_id,
        extraction_status=record.extraction_status,
        character_count=record.character_count,
        non_whitespace_count=non_whitespace,
        line_count=len(record.text.splitlines()),
        replacement_character_count=record.text.count(REPLACEMENT_CHARACTER),
        low_text=(
            record.extraction_status is ExtractionStatus.OK
            and non_whitespace < LOW_TEXT_MIN_NON_WHITESPACE
        ),
    )


def page_warnings(diag: PageDiagnostics) -> list[DiagnosticWarning]:
    warnings: list[DiagnosticWarning] = []
    if diag.extraction_status is ExtractionStatus.EMPTY:
        warnings.append(
            DiagnosticWarning(
                "empty_page",
                diag.page_id,
                "No text extracted; the page may be blank or image-only (no OCR is applied).",
            )
        )
    if diag.low_text:
        warnings.append(
            DiagnosticWarning(
                "low_text",
                diag.page_id,
                f"Only {diag.non_whitespace_count} non-whitespace characters "
                f"(threshold {LOW_TEXT_MIN_NON_WHITESPACE}); check the page visually.",
            )
        )
    if diag.replacement_character_count:
        warnings.append(
            DiagnosticWarning(
                "replacement_characters",
                diag.page_id,
                f"{diag.replacement_character_count} U+FFFD replacement character(s): "
                "some glyphs had no Unicode mapping.",
            )
        )
    return warnings


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DocumentExtraction:
    entry: DocumentEntry
    pdf_page_count: int
    records: tuple[PageRecord, ...]


@dataclass(frozen=True, slots=True)
class CorpusExtraction:
    manifest: CorpusManifest
    manifest_sha256: str
    config: ExtractionConfig
    documents: tuple[DocumentExtraction, ...]
    duration_seconds: float

    @property
    def records(self) -> tuple[PageRecord, ...]:
        return tuple(r for d in self.documents for r in d.records)


def extract_document(
    entry: DocumentEntry, pdf_path: Path, config: ExtractionConfig = BASELINE_CONFIG
) -> DocumentExtraction:
    """Extract every physical page of one verified PDF, in order.

    Raises :class:`ExtractionError` naming the document (and page, where
    relevant) on any read, integrity, parse or page-count failure.
    """
    doc_id = entry.document_id
    try:
        data = pdf_path.read_bytes()
    except OSError as exc:
        raise ExtractionError([f"{doc_id}: cannot read {pdf_path.name}: {exc}"]) from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry.sha256:
        raise ExtractionError(
            [f"{doc_id}: SHA-256 mismatch (got {digest}, manifest {entry.sha256})"]
        )

    pymupdf.TOOLS.mupdf_display_errors(False)
    try:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except (RuntimeError, ValueError) as exc:  # pymupdf.FileDataError subclasses RuntimeError
            raise ExtractionError([f"{doc_id}: cannot open PDF: {exc}"]) from exc
        with doc:
            if doc.needs_pass or doc.is_encrypted:
                raise ExtractionError([f"{doc_id}: PDF is password-protected or encrypted"])
            if doc.is_repaired:
                raise ExtractionError([f"{doc_id}: PDF structure is damaged (needed repair)"])
            pdf_page_count = int(doc.page_count)
            if pdf_page_count != entry.page_count:
                raise ExtractionError(
                    [
                        f"{doc_id}: PDF has {pdf_page_count} pages, "
                        f"manifest expects {entry.page_count}"
                    ]
                )
            records = tuple(
                _extract_page(doc, entry, number, config) for number in range(1, pdf_page_count + 1)
            )
    finally:
        pymupdf.TOOLS.reset_mupdf_warnings()
    return DocumentExtraction(entry=entry, pdf_page_count=pdf_page_count, records=records)


def _extract_page(
    doc: pymupdf.Document, entry: DocumentEntry, page_number: int, config: ExtractionConfig
) -> PageRecord:
    where = f"{entry.document_id} page {page_number}"
    try:
        page = doc.load_page(page_number - 1)
        text: object = page.get_text(config.output, flags=config.flags, sort=config.sort)
    except (RuntimeError, ValueError) as exc:
        raise ExtractionError([f"{where}: text extraction failed: {exc}"]) from exc
    if not isinstance(text, str):
        raise ExtractionError([f"{where}: parser returned {type(text).__name__}, not text"])
    try:
        return PageRecord.build(
            document_id=entry.document_id,
            document_sha256=entry.sha256,
            company=entry.company,
            fiscal_period=entry.fiscal_period,
            source_url=entry.source_url,
            page_number=page_number,
            text=text,
        )
    except PageRecordError as exc:
        raise ExtractionError([f"{where}: invalid page record: {exc}"]) from exc


def validate_records(records: Sequence[PageRecord], manifest: CorpusManifest) -> list[str]:
    """Check records against the manifest: complete, ordered, unique, correct metadata."""
    problems: list[str] = []
    expected_total = sum(d.page_count for d in manifest.documents)
    if len(records) != expected_total:
        problems.append(f"expected {expected_total} page records, got {len(records)}")
    expected_order = [
        (d.document_id, n) for d in manifest.documents for n in range(1, d.page_count + 1)
    ]
    if [(r.document_id, r.page_number) for r in records] != expected_order:
        problems.append("page records are not in manifest document order and physical page order")
    seen: set[str] = set()
    for record in records:
        if record.page_id in seen:
            problems.append(f"duplicate page_id {record.page_id}")
        seen.add(record.page_id)
        entry = manifest.get(record.document_id)
        if entry is None:
            problems.append(f"{record.page_id}: document not in manifest")
            continue
        for name, expected in (
            ("document_sha256", entry.sha256),
            ("company", entry.company),
            ("fiscal_period", entry.fiscal_period),
            ("source_url", entry.source_url),
        ):
            if getattr(record, name) != expected:
                problems.append(f"{record.page_id}: {name} does not match the manifest")
    return problems


def extract_corpus(
    manifest_path: Path,
    raw_dir: Path,
    *,
    config: ExtractionConfig = BASELINE_CONFIG,
    expected_document_count: int = EXPECTED_DOCUMENT_COUNT,
) -> CorpusExtraction:
    """Validate the corpus, then extract every manifest page. Writes nothing."""
    report = validate_corpus(manifest_path, raw_dir, expected_count=expected_document_count)
    if not report.ok or report.manifest is None:
        raise ExtractionError(
            ["corpus validation failed; run python -m founderlens.corpus.validate"]
            + [str(issue) for issue in report.issues]
        )
    manifest = report.manifest
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    started = time.perf_counter()
    documents = tuple(
        extract_document(entry, raw_dir / entry.filename, config) for entry in manifest.documents
    )
    duration = time.perf_counter() - started

    problems = validate_records([r for d in documents for r in d.records], manifest)
    if problems:
        raise ExtractionError(problems)
    return CorpusExtraction(manifest, manifest_sha256, config, documents, duration)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def serialize_records(records: Sequence[PageRecord]) -> bytes:
    """Deterministic UTF-8 JSONL: one record per line, LF endings, input order."""
    return "".join(r.to_json_line() for r in records).encode("utf-8")


def publish_atomic(dest: Path, data: bytes, verify: Callable[[Path], None] | None = None) -> None:
    """Write ``data`` to a temp file next to ``dest``, verify it, then rename over ``dest``.

    If writing or verification fails, the temp file is removed and any existing
    ``dest`` is left untouched.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if verify is not None:
            verify(tmp)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_pages_artifact(
    records: Sequence[PageRecord], dest: Path, manifest: CorpusManifest
) -> tuple[str, int]:
    """Publish the JSONL artifact after re-reading and validating it. Returns (sha256, bytes)."""
    data = serialize_records(records)

    def verify(path: Path) -> None:
        reread = path.read_bytes()
        try:
            parsed = parse_pages_jsonl(reread.decode("utf-8"))
        except (UnicodeDecodeError, PageRecordError) as exc:
            raise ExtractionError([f"written artifact does not parse: {exc}"]) from exc
        problems = validate_records(parsed, manifest)
        if reread != data or parsed != list(records):
            problems.append("written artifact differs from the extracted records")
        if problems:
            raise ExtractionError([f"written artifact invalid: {p}" for p in problems])

    publish_atomic(dest, data, verify)
    return hashlib.sha256(data).hexdigest(), len(data)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def build_report(
    extraction: CorpusExtraction, *, artifact_path: str, artifact_sha256: str, artifact_bytes: int
) -> dict[str, Any]:
    """Diagnostics report. Everything except ``operational`` is deterministic."""
    page_diags: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    documents: list[dict[str, Any]] = []
    for doc in extraction.documents:
        diags = [diagnose_page(r) for r in doc.records]
        for d in diags:
            page_diags.append(d.to_dict())
            warnings.extend(w.to_dict() for w in page_warnings(d))
        total = sum(d.character_count for d in diags)
        documents.append(
            {
                "document_id": doc.entry.document_id,
                "document_sha256": doc.entry.sha256,
                "manifest_page_count": doc.entry.page_count,
                "pdf_page_count": doc.pdf_page_count,
                "pages_extracted": len(doc.records),
                "empty_pages": sum(
                    1 for d in diags if d.extraction_status is ExtractionStatus.EMPTY
                ),
                "total_characters": total,
                "mean_characters_per_page": round(total / len(diags), 1) if diags else 0.0,
                "replacement_character_pages": [
                    d.page_id for d in diags if d.replacement_character_count
                ],
                "low_text_pages": [d.page_id for d in diags if d.low_text],
            }
        )
    records = extraction.records
    return {
        "report_type": "founderlens-extraction-report",
        "corpus_id": extraction.manifest.corpus_id,
        "corpus_version": extraction.manifest.corpus_version,
        "corpus_manifest_sha256": extraction.manifest_sha256,
        "extraction_schema_version": PAGE_SCHEMA_VERSION,
        "parser": {"name": "PyMuPDF", "module": "pymupdf", "version": str(pymupdf.VersionBind)},
        "extraction_config": extraction.config.to_dict(),
        "documents_processed": len(extraction.documents),
        "pages_expected": sum(d.page_count for d in extraction.manifest.documents),
        "pages_extracted": len(records),
        "empty_pages": sum(1 for r in records if r.extraction_status is ExtractionStatus.EMPTY),
        "total_characters": sum(r.character_count for r in records),
        "heuristics": {
            "low_text_min_non_whitespace": LOW_TEXT_MIN_NON_WHITESPACE,
            "note": "Warnings are automatic diagnostic signals, not proof of incorrect extraction.",
        },
        "documents": documents,
        "pages": page_diags,
        "warnings": warnings,
        "manual_observations": {
            "documented_in": MANUAL_OBSERVATIONS_REF,
            "note": "Layout and table problems observed by manual inspection are documented "
            "separately; they are not detected by this run.",
        },
        "artifact": {
            "path": artifact_path,
            "sha256": artifact_sha256,
            "bytes": artifact_bytes,
            "records": len(records),
        },
        "operational": {
            "note": "Varies between runs; not part of the deterministic artifact.",
            "extraction_duration_seconds": round(extraction.duration_seconds, 3),
        },
    }


def serialize_report(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


# --------------------------------------------------------------------------- #
# Orchestration and CLI
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    extraction: CorpusExtraction
    artifact_path: Path
    artifact_sha256: str
    report_path: Path
    report: dict[str, Any] = field(repr=False)


def run_extraction(
    settings: Settings,
    *,
    config: ExtractionConfig = BASELINE_CONFIG,
    expected_document_count: int = EXPECTED_DOCUMENT_COUNT,
) -> ExtractionOutcome:
    """Extract, publish the page artifact, then write the report."""
    extraction = extract_corpus(
        settings.corpus_manifest_path,
        settings.raw_data_dir,
        config=config,
        expected_document_count=expected_document_count,
    )
    artifact_path = settings.processed_data_dir / PAGES_RAW_FILENAME
    artifact_sha256, artifact_bytes = write_pages_artifact(
        extraction.records, artifact_path, extraction.manifest
    )
    report = build_report(
        extraction,
        artifact_path=artifact_path.relative_to(settings.project_root).as_posix(),
        artifact_sha256=artifact_sha256,
        artifact_bytes=artifact_bytes,
    )
    report_path = settings.reports_dir / REPORT_FILENAME
    publish_atomic(report_path, serialize_report(report))
    return ExtractionOutcome(extraction, artifact_path, artifact_sha256, report_path, report)


def _print_summary(outcome: ExtractionOutcome, root: Path) -> None:
    r = outcome.report
    cfg = r["extraction_config"]
    print(
        f"Extracted corpus {r['corpus_id']} v{r['corpus_version']} with "
        f"PyMuPDF {r['parser']['version']} (output={cfg['output']}, sort={cfg['sort']}, "
        f"flags={cfg['flags']})"
    )
    for d in r["documents"]:
        print(
            f"  {d['document_id']:<16} {d['pages_extracted']:>3}/{d['manifest_page_count']} pages"
            f"  empty {d['empty_pages']}  chars {d['total_characters']:>7}"
            f"  mean/page {d['mean_characters_per_page']:>8}"
        )
    print(
        f"Pages: {r['pages_extracted']} extracted of {r['pages_expected']} expected; "
        f"empty: {r['empty_pages']}; characters: {r['total_characters']}"
    )
    print(
        f"Automatic warnings: {len(r['warnings'])} "
        "(diagnostic signals, not proof of extraction errors)"
    )
    for w in r["warnings"]:
        print(f"  - [{w['code']}] {w['page_id']}: {w['message']}")
    print(f"Artifact: {r['artifact']['path']}  sha256 {r['artifact']['sha256']}")
    print(f"Report:   {outcome.report_path.relative_to(root).as_posix()}")
    print(f"Duration: {r['operational']['extraction_duration_seconds']} s (extraction only)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.ingestion.extract",
        description="Extract raw page-level text from the verified corpus PDFs.",
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.project_root)
        outcome = run_extraction(settings)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ExtractionError as exc:
        print(f"Extraction FAILED ({len(exc.problems)} problem(s)):", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        print("No artifact was published; any previous artifact is unchanged.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Extraction FAILED: cannot write output: {exc}", file=sys.stderr)
        return 1
    _print_summary(outcome, settings.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
