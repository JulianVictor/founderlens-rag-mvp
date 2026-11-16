"""Typed page-level extraction record and its deterministic JSON form.

One :class:`PageRecord` represents exactly one physical PDF page. Records are
validated on construction, so an inconsistent record (wrong page ID, wrong
character count, status that contradicts the text) cannot exist in memory or
be written to disk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

# Bump when the persisted record fields or their meaning change.
PAGE_SCHEMA_VERSION = 1

_DOCUMENT_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

RECORD_FIELDS: tuple[str, ...] = (
    "document_id",
    "document_sha256",
    "company",
    "fiscal_period",
    "source_url",
    "page_number",
    "page_id",
    "text",
    "character_count",
    "extraction_status",
)


class PageRecordError(ValueError):
    """Raised when page record fields are missing, mistyped or inconsistent."""


class ExtractionStatus(StrEnum):
    """Outcome of extracting one page.

    ``empty`` means the parser succeeded but returned no non-whitespace text
    (for example a blank or image-only page). Parser failures never become
    ``empty``; they abort the extraction run instead.
    """

    OK = "ok"
    EMPTY = "empty"


class PageIdentity(Protocol):
    """Page identity and source metadata shared by raw and normalized page records."""

    @property
    def document_id(self) -> str: ...
    @property
    def document_sha256(self) -> str: ...
    @property
    def company(self) -> str: ...
    @property
    def fiscal_period(self) -> str: ...
    @property
    def source_url(self) -> str: ...
    @property
    def page_number(self) -> int: ...
    @property
    def page_id(self) -> str: ...


def make_page_id(document_id: str, page_number: int) -> str:
    """Stable page identifier: ``{document_id}:p{page_number:04d}`` (one-based)."""
    _check_page_number(page_number)
    return f"{document_id}:p{page_number:04d}"


def status_for_text(text: str) -> ExtractionStatus:
    """``empty`` when the text has no non-whitespace character, otherwise ``ok``."""
    return ExtractionStatus.OK if text.strip() else ExtractionStatus.EMPTY


def _is_strict_int(value: object) -> bool:
    """True for ``int`` values but not ``bool`` (records may come from untyped JSON)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_page_number(page_number: object) -> None:
    valid = isinstance(page_number, int) and not isinstance(page_number, bool) and page_number >= 1
    if not valid:
        raise PageRecordError(f"page_number must be a positive integer, got {page_number!r}")


@dataclass(frozen=True, slots=True)
class PageRecord:
    """Raw extracted text of one physical PDF page plus its source metadata.

    ``text`` is the extractor's output, unmodified. ``character_count`` is
    ``len(text)`` in Unicode code points, whitespace included.
    """

    document_id: str
    document_sha256: str
    company: str
    fiscal_period: str
    source_url: str
    page_number: int
    page_id: str
    text: str
    character_count: int
    extraction_status: ExtractionStatus

    def __post_init__(self) -> None:
        for name in ("document_id", "document_sha256", "company", "fiscal_period", "source_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise PageRecordError(f"{name} must be a non-empty string")
        if not _DOCUMENT_ID_RE.fullmatch(self.document_id):
            raise PageRecordError(f"document_id {self.document_id!r} is not lowercase kebab-case")
        if not _SHA256_RE.fullmatch(self.document_sha256):
            raise PageRecordError("document_sha256 must be 64 lowercase hex characters")
        _check_page_number(self.page_number)
        expected_id = make_page_id(self.document_id, self.page_number)
        if self.page_id != expected_id:
            raise PageRecordError(f"page_id {self.page_id!r} must be {expected_id!r}")
        if not isinstance(self.text, str):
            raise PageRecordError("text must be a string")
        if not _is_strict_int(self.character_count) or self.character_count != len(self.text):
            raise PageRecordError(
                f"{self.page_id}: character_count {self.character_count!r} "
                f"does not match len(text) = {len(self.text)}"
            )
        if not isinstance(self.extraction_status, ExtractionStatus):
            raise PageRecordError("extraction_status must be an ExtractionStatus")
        if self.extraction_status is not status_for_text(self.text):
            raise PageRecordError(
                f"{self.page_id}: extraction_status {self.extraction_status.value!r} "
                f"contradicts the text (expected {status_for_text(self.text).value!r})"
            )

    @classmethod
    def build(
        cls,
        *,
        document_id: str,
        document_sha256: str,
        company: str,
        fiscal_period: str,
        source_url: str,
        page_number: int,
        text: str,
    ) -> PageRecord:
        """Create a record, deriving ``page_id``, ``character_count`` and status."""
        return cls(
            document_id=document_id,
            document_sha256=document_sha256,
            company=company,
            fiscal_period=fiscal_period,
            source_url=source_url,
            page_number=page_number,
            page_id=make_page_id(document_id, page_number),
            text=text,
            character_count=len(text),
            extraction_status=status_for_text(text),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_sha256": self.document_sha256,
            "company": self.company,
            "fiscal_period": self.fiscal_period,
            "source_url": self.source_url,
            "page_number": self.page_number,
            "page_id": self.page_id,
            "text": self.text,
            "character_count": self.character_count,
            "extraction_status": self.extraction_status.value,
        }

    def to_json_line(self) -> str:
        """Deterministic one-line JSON (sorted keys, no extra spaces, UTF-8 text) plus LF.

        ``json.dumps`` escapes the text's own newlines, so a record never
        spans more than one physical line.
        """
        return (
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )

    @classmethod
    def from_dict(cls, data: object) -> PageRecord:
        """Parse a decoded JSON object, rejecting missing, unknown or invalid fields."""
        if not isinstance(data, dict):
            raise PageRecordError("page record must be a JSON object")
        missing = sorted(set(RECORD_FIELDS) - set(data))
        unknown = sorted(set(data) - set(RECORD_FIELDS))
        if missing or unknown:
            raise PageRecordError(f"page record fields: missing {missing}, unknown {unknown}")
        try:
            status = ExtractionStatus(data["extraction_status"])
        except ValueError as exc:
            raise PageRecordError(
                f"unknown extraction_status {data['extraction_status']!r}"
            ) from exc
        return cls(
            document_id=data["document_id"],
            document_sha256=data["document_sha256"],
            company=data["company"],
            fiscal_period=data["fiscal_period"],
            source_url=data["source_url"],
            page_number=data["page_number"],
            page_id=data["page_id"],
            text=data["text"],
            character_count=data["character_count"],
            extraction_status=status,
        )


def parse_pages_jsonl(text: str) -> list[PageRecord]:
    """Parse a ``pages.raw.jsonl`` body. Raises :class:`PageRecordError` naming the line.

    The format is strict: LF line endings only, one record per line, no blank
    lines, and a final LF after the last record.
    """
    if "\r" in text:
        raise PageRecordError("JSONL must use LF line endings (found CR)")
    if text and not text.endswith("\n"):
        raise PageRecordError("JSONL must end with a line feed")
    records: list[PageRecord] = []
    for line_number, line in enumerate(text.split("\n")[:-1], start=1):
        if not line:
            raise PageRecordError(f"line {line_number}: blank line")
        try:
            records.append(PageRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, PageRecordError) as exc:
            raise PageRecordError(f"line {line_number}: {exc}") from exc
    return records
