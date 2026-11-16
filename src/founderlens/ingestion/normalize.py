"""Conservative text normalization plus word-coordinate table candidates.

Usage::

    python -m founderlens.ingestion.normalize

Pipeline:

1. Validate the corpus (manifest, sizes, SHA-256, readability, page counts).
2. Load ``data/processed/pages.raw.jsonl`` and validate it against the
   manifest (count, order, identity, metadata).
3. Reopen each PDF from memory after re-checking its SHA-256. For every page,
   re-extract the raw text with the Commit 2.1 baseline configuration and
   require it to equal the raw record, so a stale raw artifact is rejected.
   Then detect table candidates (``founderlens.ingestion.tables``).
4. Normalize each page's raw text (rules below) into a separate field. The
   raw artifact is never modified.
5. Validate the normalized records, publish ``pages.normalized.jsonl``
   atomically, then write ``reports/normalization_report.json``.

Text rules (``normalize_text``): CR/CRLF to LF; Unicode NFC (canonical
composition only, no compatibility folding, so ``™``, ligatures and full-width
forms are untouched); trailing whitespace removed from each line; runs of
blank lines reduced to one; leading and trailing blank lines removed; a final
LF on non-empty text. Leading and internal spaces are kept because the
layout-sorted text uses them for column alignment. Nothing is lowercased,
dehyphenated, removed as header/footer, or rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pymupdf

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.corpus.manifest import EXPECTED_DOCUMENT_COUNT, CorpusManifest
from founderlens.corpus.validate import validate_corpus
from founderlens.ingestion.extract import (
    BASELINE_CONFIG,
    PAGES_RAW_FILENAME,
    ExtractionConfig,
    ExtractionError,
    extract_page_text,
    open_verified_pdf,
    publish_atomic,
    validate_records,
)
from founderlens.ingestion.schema import (
    ExtractionStatus,
    PageRecord,
    PageRecordError,
    make_page_id,
    parse_pages_jsonl,
)
from founderlens.ingestion.tables import (
    BASELINE_TABLE_CONFIG,
    TableConfig,
    TableRecord,
    TableStatus,
    extract_tables,
)

NORMALIZED_SCHEMA_VERSION = 1
PAGES_NORMALIZED_FILENAME = "pages.normalized.jsonl"
REPORT_FILENAME = "normalization_report.json"
MANUAL_INSPECTION_REF = "docs/ingestion.md#commit-22-before-and-after-inspection"

NORMALIZED_FIELDS: tuple[str, ...] = (
    "document_id",
    "document_sha256",
    "company",
    "fiscal_period",
    "source_url",
    "page_number",
    "page_id",
    "extraction_status",
    "raw_character_count",
    "normalized_text",
    "normalized_character_count",
    "tables",
    "table_text_overlap",
    "normalization_warnings",
)


class NormalizationError(RuntimeError):
    """Normalization run failed. ``problems`` lists every actionable reason."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    # Canonical forms only; compatibility forms (NFKC/NFKD) would rewrite ™, ligatures, etc.
    unicode_form: Literal["NFC", "NFD"] = "NFC"
    max_consecutive_blank_lines: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "unicode_form": self.unicode_form,
            "line_endings": "LF (CR and CRLF converted)",
            "trailing_whitespace": "removed per line",
            "max_consecutive_blank_lines": self.max_consecutive_blank_lines,
            "leading_and_trailing_blank_lines": "removed",
            "leading_and_internal_spaces": "preserved (column alignment)",
            "not_applied": [
                "lowercasing",
                "whitespace collapse within lines",
                "compatibility normalization (NFKC/NFKD)",
                "dehyphenation",
                "header/footer removal",
                "value conversion",
            ],
        }


BASELINE_NORMALIZATION_CONFIG = NormalizationConfig()


def normalize_text(text: str, config: NormalizationConfig = BASELINE_NORMALIZATION_CONFIG) -> str:
    """Apply the conservative rules in the module docstring. Idempotent."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize(config.unicode_form, text)
    out: list[str] = []
    blank_run = 0
    for line in text.split("\n"):
        line = line.rstrip()
        if line:
            blank_run = 0
            out.append(line)
        elif out:
            blank_run += 1
            if blank_run <= config.max_consecutive_blank_lines:
                out.append("")
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n" if out else ""


# --------------------------------------------------------------------------- #
# Normalized page record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PageWarning:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class NormalizedPageRecord:
    """One physical page: normalized text and, separately, table candidates.

    ``normalized_text`` always contains the whole page, including the text
    of any table. ``tables`` is an additional structured view, so
    ``table_text_overlap`` is true whenever a table carries rows. Consumers
    must not concatenate the two without deduplicating (Commit 2.3).
    """

    document_id: str
    document_sha256: str
    company: str
    fiscal_period: str
    source_url: str
    page_number: int
    page_id: str
    extraction_status: ExtractionStatus
    raw_character_count: int
    normalized_text: str
    normalized_character_count: int
    tables: tuple[TableRecord, ...]
    table_text_overlap: bool
    normalization_warnings: tuple[PageWarning, ...]

    def __post_init__(self) -> None:
        if self.page_id != make_page_id(self.document_id, self.page_number):
            raise PageRecordError(f"page_id {self.page_id!r} does not match document and page")
        if self.normalized_character_count != len(self.normalized_text):
            raise PageRecordError(f"{self.page_id}: normalized_character_count mismatch")
        empty = self.extraction_status is ExtractionStatus.EMPTY
        if empty != (self.normalized_text == ""):
            raise PageRecordError(f"{self.page_id}: empty status and normalized text disagree")
        expected_overlap = any(t.rows for t in self.tables)
        if self.table_text_overlap != expected_overlap:
            raise PageRecordError(f"{self.page_id}: table_text_overlap is inconsistent")
        for index, table in enumerate(self.tables, start=1):
            if table.table_id != f"{self.page_id}:t{index:02d}":
                raise PageRecordError(f"{self.page_id}: table ids must be sequential")
            if (table.page_id, table.document_id, table.page_number) != (
                self.page_id,
                self.document_id,
                self.page_number,
            ):
                raise PageRecordError(f"{table.table_id}: table provenance does not match page")
            problems = table.structural_problems()
            if problems:
                raise PageRecordError("; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_sha256": self.document_sha256,
            "company": self.company,
            "fiscal_period": self.fiscal_period,
            "source_url": self.source_url,
            "page_number": self.page_number,
            "page_id": self.page_id,
            "extraction_status": self.extraction_status.value,
            "raw_character_count": self.raw_character_count,
            "normalized_text": self.normalized_text,
            "normalized_character_count": self.normalized_character_count,
            "tables": [t.to_dict() for t in self.tables],
            "table_text_overlap": self.table_text_overlap,
            "normalization_warnings": [w.to_dict() for w in self.normalization_warnings],
        }

    def to_json_line(self) -> str:
        return (
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )

    @classmethod
    def from_dict(cls, data: object) -> NormalizedPageRecord:
        if not isinstance(data, dict):
            raise PageRecordError("normalized record must be a JSON object")
        missing = sorted(set(NORMALIZED_FIELDS) - set(data))
        unknown = sorted(set(data) - set(NORMALIZED_FIELDS))
        if missing or unknown:
            raise PageRecordError(f"normalized record fields: missing {missing}, unknown {unknown}")
        try:
            return cls(
                document_id=data["document_id"],
                document_sha256=data["document_sha256"],
                company=data["company"],
                fiscal_period=data["fiscal_period"],
                source_url=data["source_url"],
                page_number=data["page_number"],
                page_id=data["page_id"],
                extraction_status=ExtractionStatus(data["extraction_status"]),
                raw_character_count=data["raw_character_count"],
                normalized_text=data["normalized_text"],
                normalized_character_count=data["normalized_character_count"],
                tables=tuple(TableRecord.from_dict(t) for t in data["tables"]),
                table_text_overlap=data["table_text_overlap"],
                normalization_warnings=tuple(
                    PageWarning(w["code"], w["message"]) for w in data["normalization_warnings"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, PageRecordError):
                raise
            raise PageRecordError(f"invalid normalized record: {exc!r}") from exc


def parse_normalized_jsonl(text: str) -> list[NormalizedPageRecord]:
    """Strict JSONL: LF only, no blank lines, final LF. Errors name the line."""
    if "\r" in text:
        raise PageRecordError("JSONL must use LF line endings (found CR)")
    if text and not text.endswith("\n"):
        raise PageRecordError("JSONL must end with a line feed")
    records: list[NormalizedPageRecord] = []
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        if not line:
            raise PageRecordError(f"line {number}: blank line")
        try:
            records.append(NormalizedPageRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, PageRecordError) as exc:
            raise PageRecordError(f"line {number}: {exc}") from exc
    return records


def page_warnings(
    status: ExtractionStatus, tables: Sequence[TableRecord]
) -> tuple[PageWarning, ...]:
    warnings: list[PageWarning] = []
    if status is ExtractionStatus.EMPTY:
        warnings.append(PageWarning("empty_page", "no text on this page (no OCR applied)"))
    for label, wanted in (("uncertain", TableStatus.UNCERTAIN), ("fallback", TableStatus.FALLBACK)):
        ids = [t.table_id for t in tables if t.structure_status is wanted]
        if ids:
            warnings.append(
                PageWarning(
                    f"{label}_tables",
                    f"{len(ids)} {label} table candidate(s): {', '.join(ids)}; "
                    "use normalized_text as the source for these regions",
                )
            )
    return tuple(warnings)


def normalize_page(
    raw: PageRecord,
    tables: tuple[TableRecord, ...],
    config: NormalizationConfig = BASELINE_NORMALIZATION_CONFIG,
) -> NormalizedPageRecord:
    text = normalize_text(raw.text, config)
    return NormalizedPageRecord(
        document_id=raw.document_id,
        document_sha256=raw.document_sha256,
        company=raw.company,
        fiscal_period=raw.fiscal_period,
        source_url=raw.source_url,
        page_number=raw.page_number,
        page_id=raw.page_id,
        extraction_status=raw.extraction_status,
        raw_character_count=raw.character_count,
        normalized_text=text,
        normalized_character_count=len(text),
        tables=tables,
        table_text_overlap=any(t.rows for t in tables),
        normalization_warnings=page_warnings(raw.extraction_status, tables),
    )


def validate_normalized(
    records: Sequence[NormalizedPageRecord],
    manifest: CorpusManifest,
    raw_records: Sequence[PageRecord],
) -> list[str]:
    """Check normalized records against the manifest and the raw records."""
    problems: list[str] = []
    if [r.page_id for r in records] != [r.page_id for r in raw_records]:
        problems.append("normalized page ids do not match the raw artifact's page ids")
    if len(records) != sum(d.page_count for d in manifest.documents):
        problems.append("normalized record count does not match the manifest page count")
    if len({r.page_id for r in records}) != len(records):
        problems.append("duplicate page_id in normalized records")
    table_ids = [t.table_id for r in records for t in r.tables]
    if len(set(table_ids)) != len(table_ids):
        problems.append("duplicate table_id in normalized records")
    raw_by_id = {r.page_id: r for r in raw_records}
    for record in records:
        raw = raw_by_id.get(record.page_id)
        if raw is None:
            continue
        for name in ("document_sha256", "company", "fiscal_period", "source_url"):
            if getattr(record, name) != getattr(raw, name):
                problems.append(f"{record.page_id}: {name} differs from the raw record")
        if record.raw_character_count != raw.character_count:
            problems.append(f"{record.page_id}: raw_character_count differs from the raw record")
        if record.extraction_status is not raw.extraction_status:
            problems.append(f"{record.page_id}: extraction_status differs from the raw record")
    return problems


# --------------------------------------------------------------------------- #
# Corpus run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CorpusNormalization:
    manifest: CorpusManifest
    manifest_sha256: str
    raw_artifact_sha256: str
    raw_records: tuple[PageRecord, ...]
    records: tuple[NormalizedPageRecord, ...]
    config: NormalizationConfig
    table_config: TableConfig
    duration_seconds: float


def load_raw_artifact(path: Path, manifest: CorpusManifest) -> tuple[list[PageRecord], str]:
    """Read and validate the raw page artifact. Returns (records, sha256)."""
    if not path.is_file():
        raise NormalizationError(
            [f"raw artifact not found: {path.name}; run python -m founderlens.ingestion.extract"]
        )
    data = path.read_bytes()
    try:
        records = parse_pages_jsonl(data.decode("utf-8"))
    except (UnicodeDecodeError, PageRecordError) as exc:
        raise NormalizationError([f"raw artifact {path.name} is invalid: {exc}"]) from exc
    problems = validate_records(records, manifest)
    if problems:
        raise NormalizationError(
            [f"raw artifact {path.name} does not match the manifest: {p}" for p in problems]
            + ["re-run python -m founderlens.ingestion.extract"]
        )
    return records, hashlib.sha256(data).hexdigest()


def normalize_corpus(
    manifest_path: Path,
    raw_dir: Path,
    raw_artifact_path: Path,
    *,
    config: NormalizationConfig = BASELINE_NORMALIZATION_CONFIG,
    table_config: TableConfig = BASELINE_TABLE_CONFIG,
    extraction_config: ExtractionConfig = BASELINE_CONFIG,
    expected_document_count: int = EXPECTED_DOCUMENT_COUNT,
) -> CorpusNormalization:
    """Validate inputs, then normalize every page and detect tables. Writes nothing."""
    report = validate_corpus(manifest_path, raw_dir, expected_count=expected_document_count)
    if not report.ok or report.manifest is None:
        raise NormalizationError(
            ["corpus validation failed; run python -m founderlens.corpus.validate"]
            + [str(issue) for issue in report.issues]
        )
    manifest = report.manifest
    raw_records, raw_sha256 = load_raw_artifact(raw_artifact_path, manifest)
    raw_by_id = {r.page_id: r for r in raw_records}

    started = time.perf_counter()
    records: list[NormalizedPageRecord] = []
    try:
        for entry in manifest.documents:
            with open_verified_pdf(entry, raw_dir / entry.filename) as doc:
                for number in range(1, entry.page_count + 1):
                    raw = raw_by_id[make_page_id(entry.document_id, number)]
                    fresh = extract_page_text(doc, entry.document_id, number, extraction_config)
                    if fresh != raw.text:
                        raise NormalizationError(
                            [
                                f"{raw.page_id}: raw artifact text differs from a fresh "
                                "extraction (stale artifact or changed extraction settings); "
                                "re-run python -m founderlens.ingestion.extract"
                            ]
                        )
                    try:
                        tables = extract_tables(
                            doc.load_page(number - 1),
                            page_id=raw.page_id,
                            document_id=entry.document_id,
                            page_number=number,
                            config=table_config,
                        )
                    except (RuntimeError, ValueError) as exc:
                        raise NormalizationError(
                            [f"{entry.document_id} page {number}: table detection failed: {exc}"]
                        ) from exc
                    records.append(normalize_page(raw, tables, config))
    except ExtractionError as exc:
        raise NormalizationError(exc.problems) from exc
    except PageRecordError as exc:
        raise NormalizationError([f"invalid normalized record: {exc}"]) from exc
    duration = time.perf_counter() - started

    problems = validate_normalized(records, manifest, raw_records)
    if problems:
        raise NormalizationError(problems)
    return CorpusNormalization(
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        raw_artifact_sha256=raw_sha256,
        raw_records=tuple(raw_records),
        records=tuple(records),
        config=config,
        table_config=table_config,
        duration_seconds=duration,
    )


def serialize_normalized(records: Sequence[NormalizedPageRecord]) -> bytes:
    return "".join(r.to_json_line() for r in records).encode("utf-8")


def write_normalized_artifact(result: CorpusNormalization, dest: Path) -> tuple[str, int]:
    """Publish atomically after re-reading and re-validating. Returns (sha256, bytes)."""
    data = serialize_normalized(result.records)

    def verify(path: Path) -> None:
        reread = path.read_bytes()
        try:
            parsed = parse_normalized_jsonl(reread.decode("utf-8"))
        except (UnicodeDecodeError, PageRecordError) as exc:
            raise NormalizationError([f"written artifact does not parse: {exc}"]) from exc
        problems = validate_normalized(parsed, result.manifest, result.raw_records)
        if reread != data or parsed != list(result.records):
            problems.append("written artifact differs from the normalized records")
        if problems:
            raise NormalizationError([f"written artifact invalid: {p}" for p in problems])

    publish_atomic(dest, data, verify)
    return hashlib.sha256(data).hexdigest(), len(data)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _status_counts(tables: Sequence[TableRecord]) -> dict[str, int]:
    counts = Counter(t.structure_status.value for t in tables)
    return {
        "candidates": len(tables),
        **{status.value: counts.get(status.value, 0) for status in TableStatus},
    }


def build_report(
    result: CorpusNormalization,
    *,
    raw_artifact_path: str,
    artifact_path: str,
    artifact_sha256: str,
    artifact_bytes: int,
) -> dict[str, Any]:
    """Diagnostics. Everything except ``operational`` is deterministic."""
    records = result.records
    all_tables = [t for r in records for t in r.tables]
    documents = []
    for entry in result.manifest.documents:
        pages = [r for r in records if r.document_id == entry.document_id]
        documents.append(
            {
                "document_id": entry.document_id,
                "pages": len(pages),
                "empty_pages": sum(
                    1 for p in pages if p.extraction_status is ExtractionStatus.EMPTY
                ),
                "raw_characters": sum(p.raw_character_count for p in pages),
                "normalized_characters": sum(p.normalized_character_count for p in pages),
                "tables": _status_counts([t for p in pages for t in p.tables]),
            }
        )
    return {
        "report_type": "founderlens-normalization-report",
        "corpus_id": result.manifest.corpus_id,
        "corpus_version": result.manifest.corpus_version,
        "corpus_manifest_sha256": result.manifest_sha256,
        "normalized_schema_version": NORMALIZED_SCHEMA_VERSION,
        "parser": {"name": "PyMuPDF", "module": "pymupdf", "version": str(pymupdf.VersionBind)},
        "raw_artifact": {"path": raw_artifact_path, "sha256": result.raw_artifact_sha256},
        "normalized_artifact": {
            "path": artifact_path,
            "sha256": artifact_sha256,
            "bytes": artifact_bytes,
            "records": len(records),
        },
        "normalization_config": result.config.to_dict(),
        "table_config": result.table_config.to_dict(),
        "table_status_definitions": {
            "structured": "passed all automatic structural checks; financial associations "
            "are not independently verified",
            "uncertain": "rows and columns assigned without collisions, but at least one "
            "check failed (see table warnings); not verified evidence",
            "fallback": "no consistent column structure; only the bounding box is kept and "
            "normalized_text is the source",
        },
        "documents_processed": len(result.manifest.documents),
        "pages": len(records),
        "empty_pages": sum(1 for r in records if r.extraction_status is ExtractionStatus.EMPTY),
        "raw_characters": sum(r.raw_character_count for r in records),
        "normalized_characters": sum(r.normalized_character_count for r in records),
        "tables": _status_counts(all_tables),
        "table_warning_counts": dict(
            sorted(Counter(w.code for t in all_tables for w in t.warnings).items())
        ),
        "documents": documents,
        "pages_detail": [
            {
                "page_id": r.page_id,
                "raw_characters": r.raw_character_count,
                "normalized_characters": r.normalized_character_count,
                "tables": [
                    {"table_id": t.table_id, "structure_status": t.structure_status.value}
                    for t in r.tables
                ],
                "warnings": [w.code for w in r.normalization_warnings],
            }
            for r in records
        ],
        "warnings": [
            {"source": "automatic", "page_id": r.page_id, **w.to_dict()}
            for r in records
            for w in r.normalization_warnings
        ],
        "manual_inspection": {
            "documented_in": MANUAL_INSPECTION_REF,
            "note": "Implementer-performed before/after inspection of selected pages; "
            "not produced by this run and not an independent review.",
        },
        "operational": {
            "note": "Varies between runs; not part of the deterministic artifact.",
            "normalization_duration_seconds": round(result.duration_seconds, 3),
        },
    }


def serialize_report(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


# --------------------------------------------------------------------------- #
# Orchestration and CLI
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NormalizationOutcome:
    result: CorpusNormalization
    artifact_path: Path
    artifact_sha256: str
    report_path: Path
    report: dict[str, Any] = field(repr=False)


def run_normalization(
    settings: Settings, *, expected_document_count: int = EXPECTED_DOCUMENT_COUNT
) -> NormalizationOutcome:
    raw_path = settings.processed_data_dir / PAGES_RAW_FILENAME
    result = normalize_corpus(
        settings.corpus_manifest_path,
        settings.raw_data_dir,
        raw_path,
        expected_document_count=expected_document_count,
    )
    artifact_path = settings.processed_data_dir / PAGES_NORMALIZED_FILENAME
    sha256, size = write_normalized_artifact(result, artifact_path)
    root = settings.project_root
    report = build_report(
        result,
        raw_artifact_path=raw_path.relative_to(root).as_posix(),
        artifact_path=artifact_path.relative_to(root).as_posix(),
        artifact_sha256=sha256,
        artifact_bytes=size,
    )
    report_path = settings.reports_dir / REPORT_FILENAME
    publish_atomic(report_path, serialize_report(report))
    return NormalizationOutcome(result, artifact_path, sha256, report_path, report)


def _print_summary(outcome: NormalizationOutcome, root: Path) -> None:
    r = outcome.report
    t = r["tables"]
    print(
        f"Normalized corpus {r['corpus_id']} v{r['corpus_version']} "
        f"(PyMuPDF {r['parser']['version']})"
    )
    for d in r["documents"]:
        dt = d["tables"]
        print(
            f"  {d['document_id']:<16} {d['pages']:>3} pages  chars {d['raw_characters']:>7}"
            f" -> {d['normalized_characters']:>7}  tables {dt['candidates']:>2} "
            f"(structured {dt['structured']}, uncertain {dt['uncertain']}, "
            f"fallback {dt['fallback']})"
        )
    print(
        f"Pages: {r['pages']}; empty: {r['empty_pages']}; characters: "
        f"{r['raw_characters']} raw -> {r['normalized_characters']} normalized"
    )
    print(
        f"Table candidates: {t['candidates']} (structured {t['structured']}, "
        f"uncertain {t['uncertain']}, fallback {t['fallback']}); structured means "
        "automatic checks passed, not verified"
    )
    print(f"Raw artifact:        {r['raw_artifact']['path']}  sha256 {r['raw_artifact']['sha256']}")
    print(
        f"Normalized artifact: {r['normalized_artifact']['path']}  "
        f"sha256 {r['normalized_artifact']['sha256']}"
    )
    print(f"Report:              {outcome.report_path.relative_to(root).as_posix()}")
    print(f"Duration: {r['operational']['normalization_duration_seconds']} s")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.ingestion.normalize",
        description="Normalize raw page text and detect conservative table candidates.",
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        outcome = run_normalization(settings)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except NormalizationError as exc:
        print(f"Normalization FAILED ({len(exc.problems)} problem(s)):", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        print("No artifact was published; any previous artifact is unchanged.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Normalization FAILED: cannot read or write output: {exc}", file=sys.stderr)
        return 1
    _print_summary(outcome, settings.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
