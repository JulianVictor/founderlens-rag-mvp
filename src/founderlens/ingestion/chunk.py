"""Configurable, page-aware chunking of normalized page text (Commit 2.3).

Usage::

    python -m founderlens.ingestion.chunk
    python -m founderlens.ingestion.chunk --chunk-size-words 160 --chunk-overlap-words 32 \
        --output data/processed/chunks.160w32o.jsonl --report-output reports/chunking.160w32o.json

Pipeline: validate the corpus; load and validate ``pages.normalized.jsonl``
against the manifest (no PDF text is re-extracted); chunk every page's
``normalized_text`` independently; validate the whole collection (exact source
spans, word budget, overlap, coverage); publish ``chunks.jsonl`` atomically;
write ``reports/chunking_report.json``.

Words. A word is a maximal run of non-whitespace characters (``\\S+``,
Unicode-aware, the same whitespace as ``str.isspace``). ``"Revenue was $39,331
million."`` is four words. Sizes and overlaps count these words; they are not
embedding-model tokens.

Boundaries. The boundary before word ``i`` (``i >= 1``) is classified by the
whitespace gap separating it from word ``i - 1``: two or more line feeds is a
**paragraph** boundary, exactly one is a **line** boundary, none is a **word**
boundary.

Window rule (algorithm version 1), per page, with ``n`` words, size ``S``,
overlap ``O`` and minimum fill ``F = ceil(S / 2)``:

1. A chunk starting at word ``s`` ends at ``n`` when ``n - s <= S``; that is
   the page's final chunk.
2. Otherwise it ends at the candidate end ``e`` in
   ``[max(s + F, previous_end + 1), s + S]`` whose boundary is strongest
   (paragraph > line > word). Ties go to the latest ``e``. So a chunk is cut
   at the latest paragraph break in the second half of its budget, else the
   latest line break there, else exactly at the word budget (a forced split
   inside a physical line).
3. The next chunk starts at the earliest physical-line start in
   ``[max(e - O, s + 1), e - 1]``; if there is none, at ``max(e - O, s + 1)``
   (mid-line). Overlap therefore never exceeds ``O`` and is smaller when a
   line start requires it.

Ends strictly increase, so every chunk adds new words and the loop
terminates. Chunks never cross pages or documents and are exact slices of
``normalized_text``. Table candidates are not indexed or appended.

Algorithm version 2 (``ChunkingConfig.token_budget`` set, Commit 3.2) adds a
hard limit in embedding-model tokens, counted by an injected
:class:`TokenCounter` (special tokens included, no truncation or padding) on
the exact chunk text. For each chunk:

a. Try the version-1 end from rule 1 or 2. If it fits, use it, so pages
   without long chunks are chunked exactly as in version 1.
b. Otherwise count every candidate end from ``max(s + 1, previous_end + 1)``
   to ``min(s + S, n)``. Among those that fit, let ``e_max`` be the longest;
   choose the strongest boundary (then the latest end) among fitting ends at
   or after ``s + ceil((e_max - s) / 2)``. This is the version-1 rule applied
   to the token-feasible range: a table row is kept whole when a line break
   fits, without producing tiny chunks. Every candidate is counted, so no
   monotonic token count is assumed.
c. If nothing fits from ``s``, retry from ``s + 1``, ``s + 2``, ... up to
   ``previous_end`` (less overlap, never a gap). If even the single next word
   does not fit, the run fails and names the page and character span.

Ends still strictly increase, so termination and coverage are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from bisect import bisect_left
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.corpus.manifest import EXPECTED_DOCUMENT_COUNT, CorpusManifest
from founderlens.corpus.validate import validate_corpus
from founderlens.ingestion.extract import PAGES_RAW_FILENAME, publish_atomic, validate_records
from founderlens.ingestion.extract import REPORT_FILENAME as EXTRACTION_REPORT_FILENAME
from founderlens.ingestion.normalize import (
    PAGES_NORMALIZED_FILENAME,
    NormalizedPageRecord,
    parse_normalized_jsonl,
)
from founderlens.ingestion.normalize import REPORT_FILENAME as NORMALIZATION_REPORT_FILENAME
from founderlens.ingestion.schema import ExtractionStatus, PageRecordError, make_page_id
from founderlens.ingestion.tables import TableRecord, TableStatus

CHUNK_SCHEMA_VERSION = 1
# Bump whenever the word definition, boundary rule or overlap rule changes.
CHUNKING_ALGORITHM_VERSION = 1
# Version 1 plus a hard model-token budget (used when a TokenBudget is set).
TOKEN_BUDGET_ALGORITHM_VERSION = 2
CHUNKS_FILENAME = "chunks.jsonl"
REPORT_FILENAME = "chunking_report.json"
MANUAL_INSPECTION_REF = "docs/ingestion.md#commit-23-chunk-quality-inspection"

# Initial engineering defaults, not tuned values (see docs/ingestion.md).
DEFAULT_CHUNK_SIZE_WORDS = 120
DEFAULT_CHUNK_OVERLAP_WORDS = 24

_WORD_RE = re.compile(r"\S+")
_CONFIG_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CHUNK_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "document_id",
    "document_sha256",
    "page_id",
    "page_number",
    "company",
    "fiscal_period",
    "source_url",
    "chunk_index",
    "text",
    "start_char",
    "end_char",
    "character_count",
    "word_count",
    "chunking_config_id",
    "page_table_status_summary",
    "page_normalization_warning_codes",
)


class ChunkingConfigError(ValueError):
    """Invalid chunking parameters."""


class ChunkRecordError(ValueError):
    """A chunk record is missing fields, mistyped or internally inconsistent."""


class ChunkingError(RuntimeError):
    """Chunking run failed. ``problems`` lists every actionable reason."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def _is_strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------- #
# Words and boundaries
# --------------------------------------------------------------------------- #


def word_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` character offsets of every word (maximal non-whitespace run)."""
    return [m.span() for m in _WORD_RE.finditer(text)]


def count_words(text: str) -> int:
    return sum(1 for _ in _WORD_RE.finditer(text))


class Boundary(IntEnum):
    WORD = 0
    LINE = 1
    PARAGRAPH = 2


def classify_gap(gap: str) -> Boundary:
    """Boundary type of the whitespace between two consecutive words."""
    newlines = gap.count("\n")
    if newlines >= 2:
        return Boundary.PARAGRAPH
    return Boundary.LINE if newlines == 1 else Boundary.WORD


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TokenCounter(Protocol):
    """Counts model tokens; implemented by the embedding encoder (or a test fake)."""

    @property
    def max_seq_length(self) -> int:
        """Effective model input limit in tokens, special tokens included."""
        ...

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Token counts with special tokens, no truncation and no padding."""
        ...


_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Hard per-chunk limit in a specific model's tokens (algorithm version 2)."""

    max_tokens: int
    model_id: str
    model_revision: str
    tokenizer: str

    def __post_init__(self) -> None:
        if not _is_strict_int(self.max_tokens) or self.max_tokens < 3:
            raise ChunkingConfigError(
                f"max_tokens must be an integer >= 3, got {self.max_tokens!r}"
            )
        for name in ("model_id", "tokenizer"):
            value: object = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ChunkingConfigError(f"token budget {name} must be a non-empty string")
        revision: object = self.model_revision
        if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
            raise ChunkingConfigError(
                "token budget model_revision must be a 40-character commit SHA"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer": self.tokenizer,
            "special_tokens": "included",
            "truncation": "disabled",
            "padding": "disabled",
            "counted_text": "exact chunk text",
        }


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Window size and overlap in whitespace-delimited words, plus an optional token budget."""

    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS
    token_budget: TokenBudget | None = None

    def __post_init__(self) -> None:
        budget: object = self.token_budget
        if budget is not None and not isinstance(budget, TokenBudget):
            raise ChunkingConfigError("token_budget must be a TokenBudget or None")
        size, overlap = self.chunk_size_words, self.chunk_overlap_words
        if not _is_strict_int(size) or size < 1:
            raise ChunkingConfigError(f"chunk_size_words must be a positive integer, got {size!r}")
        if not _is_strict_int(overlap) or overlap < 0:
            raise ChunkingConfigError(
                f"chunk_overlap_words must be a non-negative integer, got {overlap!r}"
            )
        if overlap >= size:
            raise ChunkingConfigError(
                f"chunk_overlap_words ({overlap}) must be smaller than chunk_size_words ({size})"
            )

    @property
    def min_fill_words(self) -> int:
        """Earliest allowed end of a non-final chunk, counted from its start: ceil(S / 2)."""
        return (self.chunk_size_words + 1) // 2

    @property
    def algorithm_version(self) -> int:
        return (
            CHUNKING_ALGORITHM_VERSION
            if self.token_budget is None
            else TOKEN_BUDGET_ALGORITHM_VERSION
        )

    def to_dict(self) -> dict[str, Any]:
        """Every setting that affects chunk content. Hashed into the configuration id.

        Without a token budget this is exactly the version-1 dictionary, so the
        version-1 configuration id is unchanged.
        """
        settings: dict[str, Any] = {
            "algorithm_version": self.algorithm_version,
            "chunk_size_words": self.chunk_size_words,
            "chunk_overlap_words": self.chunk_overlap_words,
            "min_fill_words": self.min_fill_words,
            "word_definition": "maximal run of non-whitespace characters (regex \\S+)",
            "source_field": "normalized_text",
            "table_candidates_indexed": False,
        }
        if self.token_budget is not None:
            settings["token_budget"] = self.token_budget.to_dict()
        return settings

    @property
    def config_sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def config_id(self) -> str:
        """First 12 hex characters of :attr:`config_sha256`."""
        return self.config_sha256[:12]


DEFAULT_CHUNKING_CONFIG = ChunkingConfig()


# --------------------------------------------------------------------------- #
# Window planning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Window:
    """Words ``[first_word, end_word)`` of a page and their character span."""

    first_word: int
    end_word: int
    start_char: int
    end_char: int
    token_limited: bool = False  # version 2 chose a different start or end than version 1


def check_counter(config: ChunkingConfig, counter: TokenCounter | None) -> None:
    """A token budget needs a counter for the same model limit; version 1 needs none."""
    budget = config.token_budget
    if budget is None:
        return
    if counter is None:
        raise ChunkingConfigError("a token budget requires a token counter (the model tokenizer)")
    if counter.max_seq_length != budget.max_tokens:
        raise ChunkingConfigError(
            f"token counter limit {counter.max_seq_length} != token budget {budget.max_tokens}"
        )


def plan_windows(
    text: str,
    config: ChunkingConfig,
    counter: TokenCounter | None = None,
    *,
    label: str = "text",
) -> list[Window]:
    """Apply the window rule from the module docstring to one page's text."""
    check_counter(config, counter)
    spans = word_spans(text)
    n = len(spans)
    if n == 0:
        return []
    size, overlap = config.chunk_size_words, config.chunk_overlap_words
    # bounds[i]: boundary before word i; the page start counts as a paragraph start.
    bounds = [Boundary.PARAGRAPH] + [
        classify_gap(text[spans[i - 1][1] : spans[i][0]]) for i in range(1, n)
    ]

    def preferred_end(start: int, previous_end: int) -> int:
        if n - start <= size:
            return n
        low = max(start + config.min_fill_words, previous_end + 1)
        return max(range(low, start + size + 1), key=lambda e: (bounds[e], e))

    max_tokens = config.token_budget.max_tokens if config.token_budget is not None else 0

    def fitting(start: int, ends: Sequence[int]) -> list[bool]:
        if counter is None:  # unreachable: check_counter() enforces it
            raise ChunkingConfigError("a token budget requires a token counter")
        texts = [text[spans[start][0] : spans[e - 1][1]] for e in ends]
        lengths = counter.token_lengths(texts)
        if len(lengths) != len(texts):
            raise ChunkingError([f"{label}: token counter returned the wrong number of lengths"])
        return [length <= max_tokens for length in lengths]

    def budgeted(start: int, previous_end: int) -> tuple[int, int, bool]:
        for first in range(start, max(start, previous_end) + 1):
            last = min(first + size, n)
            progress = max(first + 1, previous_end + 1)
            if progress > last:
                continue
            if first == start:
                default = preferred_end(first, previous_end)
                if fitting(first, [default])[0]:
                    return first, default, False
            candidates = list(range(progress, last + 1))
            fits = [e for e, ok in zip(candidates, fitting(first, candidates), strict=True) if ok]
            if not fits:
                continue
            # Minimum fill relative to the longest window that fits, then the strongest
            # boundary, then the latest end (the version-1 rule applied to what fits).
            low = first + math.ceil((max(fits) - first) / 2)
            end = max(
                (e for e in fits if e >= low),
                key=lambda e: (bounds[e] if e < n else Boundary.PARAGRAPH, e),
            )
            return first, end, True
        word_start, word_end = spans[previous_end]
        raise ChunkingError(
            [
                f"{label}: the word at characters [{word_start}, {word_end}) does not fit the "
                f"{max_tokens}-token budget even on its own; it cannot be chunked without "
                "truncation"
            ]
        )

    windows: list[Window] = []
    start, previous_end = 0, 0
    while True:
        limited = False
        if config.token_budget is None:
            end = preferred_end(start, previous_end)
        else:
            start, end, limited = budgeted(start, previous_end)
        windows.append(Window(start, end, spans[start][0], spans[end - 1][1], limited))
        if end == n:
            return windows
        earliest = max(end - overlap, start + 1)
        line_starts = [j for j in range(earliest, end) if bounds[j] >= Boundary.LINE]
        start = line_starts[0] if line_starts else earliest
        previous_end = end


# --------------------------------------------------------------------------- #
# Chunk record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TableStatusSummary:
    """Page-level table candidate counts. Says nothing about which chunk holds a table."""

    structured: int
    uncertain: int
    fallback: int

    @classmethod
    def of(cls, tables: Sequence[TableRecord]) -> TableStatusSummary:
        counts = Counter(t.structure_status for t in tables)
        return cls(
            counts[TableStatus.STRUCTURED],
            counts[TableStatus.UNCERTAIN],
            counts[TableStatus.FALLBACK],
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "structured": self.structured,
            "uncertain": self.uncertain,
            "fallback": self.fallback,
        }


def make_chunk_id(page_id: str, chunk_index: int, config_id: str) -> str:
    """``{page_id}:c{chunk_index:04d}:{config_id}``; ordinal within the page, zero-based."""
    return f"{page_id}:c{chunk_index:04d}:{config_id}"


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One chunk: ``text == normalized_text[start_char:end_char]`` of exactly one page.

    Offsets are Python string (code point) positions, start inclusive and end
    exclusive. ``page_table_status_summary`` and
    ``page_normalization_warning_codes`` describe the whole page, not this chunk.
    """

    chunk_id: str
    document_id: str
    document_sha256: str
    page_id: str
    page_number: int
    company: str
    fiscal_period: str
    source_url: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    character_count: int
    word_count: int
    chunking_config_id: str
    page_table_status_summary: TableStatusSummary
    page_normalization_warning_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("document_id", "document_sha256", "company", "fiscal_period", "source_url"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ChunkRecordError(f"{name} must be a non-empty string")
        if not _SHA256_RE.fullmatch(self.document_sha256):
            raise ChunkRecordError("document_sha256 must be 64 lowercase hex characters")
        for name in ("page_number", "chunk_index", "start_char", "end_char"):
            if not _is_strict_int(getattr(self, name)):
                raise ChunkRecordError(f"{name} must be an integer")
        try:
            expected_page_id = make_page_id(self.document_id, self.page_number)
        except PageRecordError as exc:
            raise ChunkRecordError(str(exc)) from exc
        if self.page_id != expected_page_id:
            raise ChunkRecordError(f"page_id {self.page_id!r} must be {expected_page_id!r}")
        if self.chunk_index < 0:
            raise ChunkRecordError("chunk_index must be non-negative")
        # Records may come from untyped JSON, so re-check declared types at runtime.
        config_id: object = self.chunking_config_id
        text: object = self.text
        summary: object = self.page_table_status_summary
        codes: object = self.page_normalization_warning_codes
        if not isinstance(config_id, str) or not _CONFIG_ID_RE.fullmatch(config_id):
            raise ChunkRecordError("chunking_config_id must be 12 lowercase hex characters")
        expected_id = make_chunk_id(self.page_id, self.chunk_index, self.chunking_config_id)
        if self.chunk_id != expected_id:
            raise ChunkRecordError(f"chunk_id {self.chunk_id!r} must be {expected_id!r}")
        if not isinstance(text, str) or not text.strip():
            raise ChunkRecordError(f"{self.chunk_id}: text must contain non-whitespace characters")
        if not 0 <= self.start_char < self.end_char:
            raise ChunkRecordError(f"{self.chunk_id}: offsets must satisfy 0 <= start < end")
        if not (self.character_count == len(self.text) == self.end_char - self.start_char):
            raise ChunkRecordError(
                f"{self.chunk_id}: character_count must equal len(text) and end_char - start_char"
            )
        if self.word_count != count_words(self.text):
            raise ChunkRecordError(f"{self.chunk_id}: word_count does not match the text")
        if not isinstance(summary, TableStatusSummary) or not all(
            _is_strict_int(v) and v >= 0 for v in summary.to_dict().values()
        ):
            raise ChunkRecordError(f"{self.chunk_id}: invalid page_table_status_summary")
        if not isinstance(codes, tuple) or not all(isinstance(c, str) and c for c in codes):
            raise ChunkRecordError(f"{self.chunk_id}: invalid page_normalization_warning_codes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_sha256": self.document_sha256,
            "page_id": self.page_id,
            "page_number": self.page_number,
            "company": self.company,
            "fiscal_period": self.fiscal_period,
            "source_url": self.source_url,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "character_count": self.character_count,
            "word_count": self.word_count,
            "chunking_config_id": self.chunking_config_id,
            "page_table_status_summary": self.page_table_status_summary.to_dict(),
            "page_normalization_warning_codes": list(self.page_normalization_warning_codes),
        }

    def to_json_line(self) -> str:
        return (
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )

    @classmethod
    def from_dict(cls, data: object) -> ChunkRecord:
        if not isinstance(data, dict):
            raise ChunkRecordError("chunk record must be a JSON object")
        missing = sorted(set(CHUNK_FIELDS) - set(data))
        unknown = sorted(set(data) - set(CHUNK_FIELDS))
        if missing or unknown:
            raise ChunkRecordError(f"chunk record fields: missing {missing}, unknown {unknown}")
        try:
            summary = data["page_table_status_summary"]
            if not isinstance(summary, dict) or set(summary) != {
                "structured",
                "uncertain",
                "fallback",
            }:
                raise ChunkRecordError("page_table_status_summary has the wrong keys")
            codes = data["page_normalization_warning_codes"]
            if not isinstance(codes, list):
                raise ChunkRecordError("page_normalization_warning_codes must be a list")
            return cls(
                **{name: data[name] for name in CHUNK_FIELDS[:15]},
                page_table_status_summary=TableStatusSummary(**summary),
                page_normalization_warning_codes=tuple(codes),
            )
        except (KeyError, TypeError) as exc:
            raise ChunkRecordError(f"invalid chunk record: {exc!r}") from exc


def parse_chunks_jsonl(text: str) -> list[ChunkRecord]:
    """Strict JSONL: LF only, no blank lines, final LF. Errors name the line."""
    if "\r" in text:
        raise ChunkRecordError("JSONL must use LF line endings (found CR)")
    if text and not text.endswith("\n"):
        raise ChunkRecordError("JSONL must end with a line feed")
    records: list[ChunkRecord] = []
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        if not line:
            raise ChunkRecordError(f"line {number}: blank line")
        try:
            records.append(ChunkRecord.from_dict(json.loads(line)))
        except (json.JSONDecodeError, ChunkRecordError) as exc:
            raise ChunkRecordError(f"line {number}: {exc}") from exc
    return records


def chunk_page(
    page: NormalizedPageRecord, config: ChunkingConfig, counter: TokenCounter | None = None
) -> tuple[ChunkRecord, ...]:
    """Chunks of one page's ``normalized_text``. Empty pages yield none."""
    return chunk_page_windows(page, config, counter)[0]


def chunk_page_windows(
    page: NormalizedPageRecord, config: ChunkingConfig, counter: TokenCounter | None = None
) -> tuple[tuple[ChunkRecord, ...], tuple[Window, ...]]:
    """Chunks of one page plus the windows they came from (for diagnostics)."""
    text = page.normalized_text
    summary = TableStatusSummary.of(page.tables)
    codes = tuple(w.code for w in page.normalization_warnings)
    chunks = []
    windows = tuple(plan_windows(text, config, counter, label=page.page_id))
    for index, window in enumerate(windows):
        chunk_text = text[window.start_char : window.end_char]
        chunks.append(
            ChunkRecord(
                chunk_id=make_chunk_id(page.page_id, index, config.config_id),
                document_id=page.document_id,
                document_sha256=page.document_sha256,
                page_id=page.page_id,
                page_number=page.page_number,
                company=page.company,
                fiscal_period=page.fiscal_period,
                source_url=page.source_url,
                chunk_index=index,
                text=chunk_text,
                start_char=window.start_char,
                end_char=window.end_char,
                character_count=len(chunk_text),
                word_count=window.end_word - window.first_word,
                chunking_config_id=config.config_id,
                page_table_status_summary=summary,
                page_normalization_warning_codes=codes,
            )
        )
    return tuple(chunks), windows


# --------------------------------------------------------------------------- #
# Collection validation and overlap
# --------------------------------------------------------------------------- #


def _word_range(spans: Sequence[tuple[int, int]], chunk: ChunkRecord) -> tuple[int, int] | None:
    """Word indices ``[first, end)`` of a chunk, or None if it is not word-aligned."""
    first = bisect_left(spans, chunk.start_char, key=lambda span: span[0])
    last = bisect_left(spans, chunk.end_char, key=lambda span: span[1])
    if first >= len(spans) or last >= len(spans):
        return None
    if spans[first][0] != chunk.start_char or spans[last][1] != chunk.end_char:
        return None
    return first, last + 1


def page_overlaps(text: str, chunks: Sequence[ChunkRecord]) -> list[int]:
    """Shared source word positions between each pair of consecutive chunks of one page.

    Two separate occurrences of the same value do not count; only words whose
    source position lies inside both chunks do.
    """
    spans = word_spans(text)
    ranges = [_word_range(spans, c) for c in chunks]
    overlaps = []
    for previous, current in pairwise(ranges):
        if previous is None or current is None:
            raise ChunkingError(["chunk offsets are not aligned to word boundaries"])
        overlaps.append(max(0, previous[1] - current[0]))
    return overlaps


def uncovered_characters(text: str, chunks: Sequence[ChunkRecord]) -> int:
    """Non-whitespace characters of ``text`` outside every chunk span."""
    covered = bytearray(len(text))
    for chunk in chunks:
        covered[chunk.start_char : chunk.end_char] = b"\x01" * (chunk.end_char - chunk.start_char)
    return sum(1 for i, ch in enumerate(text) if not covered[i] and not ch.isspace())


def _check_page(
    page: NormalizedPageRecord, chunks: Sequence[ChunkRecord], config: ChunkingConfig
) -> list[str]:
    text = page.normalized_text
    spans = word_spans(text)
    if not spans:
        return [f"{page.page_id}: empty page must have no chunks"] if chunks else []
    if not chunks:
        return [f"{page.page_id}: non-empty page has no chunks"]
    problems: list[str] = []
    summary = TableStatusSummary.of(page.tables)
    codes = tuple(w.code for w in page.normalization_warnings)
    ranges: list[tuple[int, int]] = []
    for chunk in chunks:
        where = chunk.chunk_id
        if chunk.chunking_config_id != config.config_id:
            problems.append(f"{where}: chunking_config_id does not match the configuration")
        for name in ("document_id", "document_sha256", "company", "fiscal_period", "source_url"):
            if getattr(chunk, name) != getattr(page, name):
                problems.append(f"{where}: {name} differs from the source page")
        if chunk.page_number != page.page_number:
            problems.append(f"{where}: page_number differs from the source page")
        if text[chunk.start_char : chunk.end_char] != chunk.text:
            problems.append(f"{where}: text is not the source span [start_char, end_char)")
        if chunk.word_count > config.chunk_size_words:
            problems.append(f"{where}: {chunk.word_count} words exceed the word budget")
        if chunk.page_table_status_summary != summary or chunk.page_normalization_warning_codes != (
            codes
        ):
            problems.append(f"{where}: page-level metadata differs from the source page")
        word_range = _word_range(spans, chunk)
        if word_range is None:
            problems.append(f"{where}: offsets are not aligned to word boundaries")
        else:
            ranges.append(word_range)
    if problems:
        return problems
    for ((first, end), (next_first, next_end)), chunk in zip(
        pairwise(ranges), chunks[1:], strict=True
    ):
        if not (first < next_first and end < next_end):
            problems.append(f"{chunk.chunk_id}: does not advance (no new source words)")
        if next_first > end:
            problems.append(f"{chunk.chunk_id}: words between chunks are not covered")
        if end - next_first > config.chunk_overlap_words:
            problems.append(f"{chunk.chunk_id}: overlap exceeds {config.chunk_overlap_words}")
    if uncovered_characters(text, chunks):
        problems.append(f"{page.page_id}: non-whitespace characters are not covered by chunks")
    return problems


def token_budget_problems(
    chunks: Sequence[ChunkRecord], config: ChunkingConfig, counter: TokenCounter | None
) -> list[str]:
    """Every chunk independently re-counted against the token budget (version 2 only)."""
    if config.token_budget is None:
        return []
    check_counter(config, counter)
    if counter is None or not chunks:
        return []
    lengths = counter.token_lengths([c.text for c in chunks])
    limit = config.token_budget.max_tokens
    return [
        f"{c.chunk_id}: {n} tokens exceed the {limit}-token budget"
        for c, n in zip(chunks, lengths, strict=True)
        if n > limit
    ]


def validate_chunks(
    chunks: Sequence[ChunkRecord],
    pages: Sequence[NormalizedPageRecord],
    config: ChunkingConfig,
    counter: TokenCounter | None = None,
) -> list[str]:
    """Whole-collection checks: identity, order, exact spans, budgets, overlap, coverage."""
    problems: list[str] = token_budget_problems(chunks, config, counter)
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        problems.append("duplicate chunk_id in the chunk collection")
    known = {p.page_id for p in pages}
    unknown = sorted({c.page_id for c in chunks} - known)
    if unknown:
        problems.append(f"chunks reference pages not in the normalized artifact: {unknown}")
    by_page: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_page.setdefault(chunk.page_id, []).append(chunk)
    expected_order = [(p.page_id, i) for p in pages for i in range(len(by_page.get(p.page_id, [])))]
    if [(c.page_id, c.chunk_index) for c in chunks] != expected_order:
        problems.append("chunks are not in page order with sequential zero-based chunk_index")
    for page in pages:
        problems.extend(_check_page(page, by_page.get(page.page_id, []), config))
    return problems


# --------------------------------------------------------------------------- #
# Corpus run
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CorpusChunking:
    manifest: CorpusManifest
    manifest_sha256: str
    normalized_sha256: str
    pages: tuple[NormalizedPageRecord, ...]
    chunks: tuple[ChunkRecord, ...]
    config: ChunkingConfig
    duration_seconds: float
    token_limited_chunk_ids: tuple[str, ...] = ()
    counter: TokenCounter | None = field(default=None, repr=False, compare=False)


def load_normalized_artifact(
    path: Path, manifest: CorpusManifest
) -> tuple[list[NormalizedPageRecord], str]:
    """Read and validate the normalized page artifact. Returns (records, sha256)."""
    if not path.is_file():
        raise ChunkingError(
            [
                f"normalized artifact not found: {path.name}; run "
                "python -m founderlens.ingestion.extract, then "
                "python -m founderlens.ingestion.normalize"
            ]
        )
    data = path.read_bytes()
    try:
        records = parse_normalized_jsonl(data.decode("utf-8"))
    except (UnicodeDecodeError, PageRecordError) as exc:
        raise ChunkingError([f"normalized artifact {path.name} is invalid: {exc}"]) from exc
    problems = validate_records(records, manifest)
    if problems:
        raise ChunkingError(
            [f"normalized artifact {path.name} does not match the manifest: {p}" for p in problems]
            + ["re-run python -m founderlens.ingestion.normalize"]
        )
    return records, hashlib.sha256(data).hexdigest()


def chunk_corpus(
    manifest_path: Path,
    raw_dir: Path,
    normalized_path: Path,
    *,
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
    expected_document_count: int = EXPECTED_DOCUMENT_COUNT,
    counter: TokenCounter | None = None,
) -> CorpusChunking:
    """Validate inputs, then chunk every page and validate the collection. Writes nothing."""
    try:
        check_counter(config, counter)
    except ChunkingConfigError as exc:
        raise ChunkingError([str(exc)]) from exc
    report = validate_corpus(manifest_path, raw_dir, expected_count=expected_document_count)
    if not report.ok or report.manifest is None:
        raise ChunkingError(
            ["corpus validation failed; run python -m founderlens.corpus.validate"]
            + [str(issue) for issue in report.issues]
        )
    manifest = report.manifest
    pages, normalized_sha256 = load_normalized_artifact(normalized_path, manifest)
    started = time.perf_counter()
    chunks: list[ChunkRecord] = []
    limited: list[str] = []
    try:
        for page in pages:
            page_chunks, windows = chunk_page_windows(page, config, counter)
            chunks.extend(page_chunks)
            limited.extend(
                c.chunk_id for c, w in zip(page_chunks, windows, strict=True) if w.token_limited
            )
    except ChunkRecordError as exc:
        raise ChunkingError([f"invalid chunk: {exc}"]) from exc
    problems = validate_chunks(chunks, pages, config, counter)
    duration = time.perf_counter() - started
    if problems:
        raise ChunkingError(problems)
    return CorpusChunking(
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        normalized_sha256=normalized_sha256,
        pages=tuple(pages),
        chunks=tuple(chunks),
        config=config,
        duration_seconds=duration,
        token_limited_chunk_ids=tuple(limited),
        counter=counter,
    )


def serialize_chunks(chunks: Sequence[ChunkRecord]) -> bytes:
    return "".join(c.to_json_line() for c in chunks).encode("utf-8")


def write_chunks_artifact(result: CorpusChunking, dest: Path) -> tuple[str, int]:
    """Publish atomically after re-reading and re-validating. Returns (sha256, bytes)."""
    data = serialize_chunks(result.chunks)

    def verify(path: Path) -> None:
        reread = path.read_bytes()
        try:
            parsed = parse_chunks_jsonl(reread.decode("utf-8"))
        except (UnicodeDecodeError, ChunkRecordError) as exc:
            raise ChunkingError([f"written artifact does not parse: {exc}"]) from exc
        problems = validate_chunks(parsed, result.pages, result.config, result.counter)
        if reread != data or parsed != list(result.chunks):
            problems.append("written artifact differs from the chunk records")
        if problems:
            raise ChunkingError([f"written artifact invalid: {p}" for p in problems])

    publish_atomic(dest, data, verify)
    return hashlib.sha256(data).hexdigest(), len(data)


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #


def resolve_output(settings: Settings, value: str | Path | None, default: Path) -> Path:
    """Relative paths are resolved against the project root, never the working directory."""
    if value is None:
        return default.resolve()
    path = Path(value)
    return (path if path.is_absolute() else settings.project_root / path).resolve()


def output_path_problems(settings: Settings, chunks_path: Path, report_path: Path) -> list[str]:
    """Reject outputs outside the project or colliding with inputs or other stages' outputs."""
    root = settings.project_root.resolve()
    protected_files = {
        settings.corpus_manifest_path.resolve(): "the corpus manifest",
        (settings.processed_data_dir / PAGES_RAW_FILENAME).resolve(): "the raw page artifact",
        (settings.processed_data_dir / PAGES_NORMALIZED_FILENAME).resolve(): (
            "the normalized page artifact"
        ),
        (settings.reports_dir / EXTRACTION_REPORT_FILENAME).resolve(): "the extraction report",
        (settings.reports_dir / NORMALIZATION_REPORT_FILENAME).resolve(): (
            "the normalization report"
        ),
    }
    protected_dirs = {
        settings.raw_data_dir.resolve(): "the original PDF directory",
        settings.evaluation_dir.resolve(): "the evaluation dataset directory",
    }
    problems: list[str] = []
    for label, path, suffix in (
        ("--output", chunks_path, ".jsonl"),
        ("--report-output", report_path, ".json"),
    ):
        if not path.is_relative_to(root):
            problems.append(f"{label} must be inside the project root")
            continue
        shown = path.relative_to(root).as_posix()
        if path.suffix != suffix:
            problems.append(f"{label} {shown} must end with {suffix}")
        if path in protected_files:
            problems.append(f"{label} {shown} would overwrite {protected_files[path]}")
        for directory, name in protected_dirs.items():
            if path.is_relative_to(directory):
                problems.append(f"{label} {shown} is inside {name}")
        if path.is_dir():
            problems.append(f"{label} {shown} is a directory")
    if chunks_path == report_path:
        problems.append("--output and --report-output must be different files")
    return problems


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _distribution(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 3),
        "median": statistics.median(values),
    }


def _end_boundary(text: str, end_char: int) -> Boundary | None:
    following = _WORD_RE.search(text, end_char)
    return None if following is None else classify_gap(text[end_char : following.start()])


def _start_boundary(text: str, start_char: int) -> Boundary | None:
    before = text[:start_char].rstrip()
    return None if not before else classify_gap(text[len(before) : start_char])


def build_report(
    result: CorpusChunking,
    *,
    normalized_path: str,
    artifact_path: str,
    artifact_sha256: str,
    artifact_bytes: int,
) -> dict[str, Any]:
    """Diagnostics. Everything except ``operational`` is deterministic."""
    config = result.config
    chunks = result.chunks
    by_page: dict[str, list[ChunkRecord]] = {p.page_id: [] for p in result.pages}
    for chunk in chunks:
        by_page[chunk.page_id].append(chunk)

    overlaps: list[int] = []
    forced_splits: list[str] = []
    mid_line_starts: list[str] = []
    lines_over_budget = 0
    uncovered = 0
    for page in result.pages:
        text, page_chunks = page.normalized_text, by_page[page.page_id]
        overlaps.extend(page_overlaps(text, page_chunks))
        uncovered += uncovered_characters(text, page_chunks)
        lines_over_budget += sum(
            1 for line in text.split("\n") if count_words(line) > config.chunk_size_words
        )
        for chunk in page_chunks[:-1]:
            if _end_boundary(text, chunk.end_char) is Boundary.WORD:
                forced_splits.append(chunk.chunk_id)
        for chunk in page_chunks[1:]:
            if _start_boundary(text, chunk.start_char) is Boundary.WORD:
                mid_line_starts.append(chunk.chunk_id)

    source_words = sum(count_words(p.normalized_text) for p in result.pages)
    emitted_words = sum(c.word_count for c in chunks)
    short = [c.chunk_id for c in chunks if c.word_count < config.min_fill_words]
    empty_pages = [p.page_id for p in result.pages if p.extraction_status is ExtractionStatus.EMPTY]
    table_pages = [
        p.page_id
        for p in result.pages
        if any(t.structure_status is not TableStatus.STRUCTURED for t in p.tables)
    ]
    return {
        "report_type": "founderlens-chunking-report",
        "corpus_id": result.manifest.corpus_id,
        "corpus_version": result.manifest.corpus_version,
        "corpus_manifest_sha256": result.manifest_sha256,
        "normalized_artifact": {"path": normalized_path, "sha256": result.normalized_sha256},
        "chunk_artifact": {
            "path": artifact_path,
            "sha256": artifact_sha256,
            "bytes": artifact_bytes,
            "records": len(chunks),
        },
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "chunking_algorithm_version": CHUNKING_ALGORITHM_VERSION,
        "chunking_config": config.to_dict(),
        "chunking_config_id": config.config_id,
        "chunking_config_sha256": config.config_sha256,
        "documents_processed": len(result.manifest.documents),
        "pages_processed": len(result.pages),
        "empty_pages": empty_pages,
        "chunk_count": len(chunks),
        "chunks_per_document": {
            d.document_id: sum(1 for c in chunks if c.document_id == d.document_id)
            for d in result.manifest.documents
        },
        "chunks_per_page": {page_id: len(cs) for page_id, cs in by_page.items()},
        "source_words": source_words,
        "emitted_chunk_words": emitted_words,
        "chunk_word_count": _distribution([c.word_count for c in chunks]),
        "chunk_character_count": _distribution([c.character_count for c in chunks]),
        "overlap": {
            "definition": "source word positions shared by consecutive chunks of one page",
            "configured_maximum_words": config.chunk_overlap_words,
            "consecutive_pairs": len(overlaps),
            "min_words": min(overlaps) if overlaps else None,
            "max_words": max(overlaps) if overlaps else None,
            "mean_words": round(statistics.fmean(overlaps), 3) if overlaps else None,
            "repeated_word_occurrences": sum(overlaps),
            "emitted_equals_source_plus_repeated": emitted_words == source_words + sum(overlaps),
        },
        "diagnostics": {
            "source_coverage": {
                "validated": True,
                "uncovered_non_whitespace_characters": uncovered,
            },
            "forced_line_splits": {
                "definition": "non-final chunks that end inside a physical line",
                "count": len(forced_splits),
                "chunk_ids": forced_splits,
            },
            "mid_line_chunk_starts": {
                "definition": "chunks after the first on a page that start inside a physical line",
                "count": len(mid_line_starts),
                "chunk_ids": mid_line_starts,
            },
            "physical_lines_over_word_budget": lines_over_budget,
            "short_chunks": {
                "definition": f"chunks with fewer than min_fill_words ({config.min_fill_words})",
                "count": len(short),
                "chunk_ids": short,
            },
            "page_table_limitations": {
                "note": "Table candidates are not indexed or appended; chunks are normalized "
                "text only. Page-level table status says nothing about which chunk holds a table.",
                "pages_with_uncertain_or_fallback_tables": table_pages,
            },
            **_token_budget_diagnostics(result),
        },
        "manual_inspection": {
            "documented_in": MANUAL_INSPECTION_REF,
            "note": "Implementer-performed inspection of selected chunks; not produced by this "
            "run and not an independent review.",
        },
        "operational": {
            "note": "Varies between runs; not part of the deterministic artifact.",
            "chunking_duration_seconds": round(result.duration_seconds, 3),
        },
    }


def _token_budget_diagnostics(result: CorpusChunking) -> dict[str, Any]:
    budget, counter = result.config.token_budget, result.counter
    if budget is None or counter is None:
        return {}
    lengths = counter.token_lengths([c.text for c in result.chunks]) if result.chunks else []
    return {
        "token_budget": {
            "max_tokens": budget.max_tokens,
            "chunk_token_count": _distribution(lengths),
            "over_budget": sum(1 for n in lengths if n > budget.max_tokens),
            "token_limited_chunks": {
                "definition": "chunks whose start or end differs from the version-1 choice "
                "because the version-1 window exceeded the token budget",
                "count": len(result.token_limited_chunk_ids),
                "chunk_ids": list(result.token_limited_chunk_ids),
            },
        }
    }


def serialize_report(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


# --------------------------------------------------------------------------- #
# Orchestration and CLI
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ChunkingOutcome:
    result: CorpusChunking
    artifact_path: Path
    artifact_sha256: str
    report_path: Path
    report: dict[str, Any] = field(repr=False)


def run_chunking(
    settings: Settings,
    *,
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
    output: str | Path | None = None,
    report_output: str | Path | None = None,
    expected_document_count: int = EXPECTED_DOCUMENT_COUNT,
    counter: TokenCounter | None = None,
) -> ChunkingOutcome:
    chunks_path = resolve_output(settings, output, settings.processed_data_dir / CHUNKS_FILENAME)
    report_path = resolve_output(settings, report_output, settings.reports_dir / REPORT_FILENAME)
    problems = output_path_problems(settings, chunks_path, report_path)
    if config.token_budget is not None:
        # The version-1 baseline collection and report are frozen; never overwrite them.
        for path, default in (
            (chunks_path, settings.processed_data_dir / CHUNKS_FILENAME),
            (report_path, settings.reports_dir / REPORT_FILENAME),
        ):
            if path == default.resolve():
                problems.append(
                    f"{path.name} is the version-1 baseline; write token-budgeted "
                    "collections to their own --output and --report-output"
                )
    if problems:
        raise ChunkingError(problems)
    normalized_path = settings.processed_data_dir / PAGES_NORMALIZED_FILENAME
    result = chunk_corpus(
        settings.corpus_manifest_path,
        settings.raw_data_dir,
        normalized_path,
        config=config,
        expected_document_count=expected_document_count,
        counter=counter,
    )
    sha256, size = write_chunks_artifact(result, chunks_path)
    root = settings.project_root.resolve()
    report = build_report(
        result,
        normalized_path=normalized_path.resolve().relative_to(root).as_posix(),
        artifact_path=chunks_path.relative_to(root).as_posix(),
        artifact_sha256=sha256,
        artifact_bytes=size,
    )
    publish_atomic(report_path, serialize_report(report))
    return ChunkingOutcome(result, chunks_path, sha256, report_path, report)


def _print_summary(outcome: ChunkingOutcome, root: Path) -> None:
    r = outcome.report
    cfg = r["chunking_config"]
    words = r["chunk_word_count"]
    ov = r["overlap"]
    diag = r["diagnostics"]
    print(
        f"Chunked corpus {r['corpus_id']} v{r['corpus_version']}: "
        f"{cfg['chunk_size_words']} words, overlap {cfg['chunk_overlap_words']} "
        f"(config {r['chunking_config_id']}; words, not model tokens)"
    )
    for document_id, count in r["chunks_per_document"].items():
        print(f"  {document_id:<16} {count:>4} chunks")
    print(
        f"Pages: {r['pages_processed']}; empty: {len(r['empty_pages'])}; chunks: {r['chunk_count']}"
    )
    print(
        f"Chunk words: min {words['min']}, median {words['median']}, mean {words['mean']}, "
        f"max {words['max']}; source words {r['source_words']}, emitted {r['emitted_chunk_words']}"
    )
    print(
        f"Overlap words: min {ov['min_words']}, mean {ov['mean_words']}, max {ov['max_words']}; "
        f"repeated occurrences {ov['repeated_word_occurrences']}"
    )
    print(
        f"Forced line splits: {diag['forced_line_splits']['count']}; "
        f"short chunks: {diag['short_chunks']['count']}; source coverage validated"
    )
    print(f"Chunk artifact: {r['chunk_artifact']['path']}  sha256 {r['chunk_artifact']['sha256']}")
    print(f"Report:         {outcome.report_path.resolve().relative_to(root).as_posix()}")
    print(f"Duration: {r['operational']['chunking_duration_seconds']} s")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.ingestion.chunk",
        description="Split normalized page text into page-aware, overlapping word windows.",
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    parser.add_argument(
        "--chunk-size-words",
        type=int,
        default=DEFAULT_CHUNK_SIZE_WORDS,
        help=f"Maximum words per chunk (default {DEFAULT_CHUNK_SIZE_WORDS}).",
    )
    parser.add_argument(
        "--chunk-overlap-words",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_WORDS,
        help=f"Maximum words shared by consecutive chunks (default {DEFAULT_CHUNK_OVERLAP_WORDS}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"Chunk artifact path (default data/processed/{CHUNKS_FILENAME}).",
    )
    parser.add_argument(
        "--report-output", default=None, help=f"Report path (default reports/{REPORT_FILENAME})."
    )
    args = parser.parse_args(argv)
    try:
        config = ChunkingConfig(args.chunk_size_words, args.chunk_overlap_words)
    except ChunkingConfigError as exc:
        parser.error(str(exc))
    try:
        settings = load_settings(args.project_root)
        outcome = run_chunking(
            settings, config=config, output=args.output, report_output=args.report_output
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ChunkingError as exc:
        print(f"Chunking FAILED ({len(exc.problems)} problem(s)):", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        print("No chunk artifact was published; any previous one is unchanged.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Chunking FAILED: cannot read or write output: {exc}", file=sys.stderr)
        return 1
    _print_summary(outcome, settings.project_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
