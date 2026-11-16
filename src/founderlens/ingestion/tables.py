"""Conservative, word-coordinate table candidates for one PDF page.

Why not ``Page.find_tables()``: on this corpus (PyMuPDF 1.28.2) the default
``lines`` strategy merges each statement row's label and values into one cell
(Alphabet) or treats page-level decorative boxes as tables (NVIDIA). The
``text`` strategy returns one grid per page that includes prose and cuts words
across columns. See ``docs/ingestion.md``.

Method (bounded, not a general table parser):

1. Words from ``Page.get_text("words")`` are grouped into visual lines by
   vertical overlap, so a label printed a few points below its values stays
   on the values' line.
2. Each line is split into segments at horizontal gaps wider than
   ``segment_gap``. Wide lines with only small gaps are prose and end a table.
3. A table candidate starts at a line with a stub label (a segment in the
   left part of the page) and at least one numeric cell, and grows downward
   over non-prose lines with small vertical gaps.
4. Columns are the union of overlapping x-intervals of the value cells. A
   row with two cells in one column makes the candidate ``fallback``.
5. Header lines directly above the first row are assigned to columns by
   x-overlap. A segment covering several columns becomes a spanning header.
   Nothing is inferred: a column whose header cannot be found has ``None``.

Status values describe automatic structural checks only:

* ``structured``: every check passed: each column has a header, every row
  with values has a label, a unit statement appears on the page above the
  table. This does **not** mean every financial association was verified.
* ``uncertain``: rows and columns were assigned without collisions, but at
  least one check failed (see ``warnings``). Do not treat as verified.
* ``fallback``: no consistent column structure. Only the bounding box and
  warnings are kept; the page text remains the source of truth.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pymupdf

_DIGIT_RE = re.compile(r"\d")
_CURRENCY_SYMBOLS = frozenset({"$", "€", "£", "¥"})
_DASHES = frozenset({"-", "--", "\u2013", "\u2014"})
_TOUCHING_GAP = 0.5  # points; words closer than this had no space between them
_DATA_RE = re.compile(r"[$€£¥]\s*\(?\d|\d\s*%|\(\d|\d,\d{3}|\d\.\d")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_UNIT_RE = re.compile(r"\bin (?:millions|billions|thousands)\b", re.IGNORECASE)


class TableStatus(StrEnum):
    STRUCTURED = "structured"
    UNCERTAIN = "uncertain"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class TableConfig:
    """Geometric thresholds in PDF points; derived from this corpus's layout."""

    word_flags: int = (
        pymupdf.TEXT_PRESERVE_LIGATURES
        | pymupdf.TEXT_PRESERVE_WHITESPACE
        | pymupdf.TEXT_MEDIABOX_CLIP
    )
    line_overlap_ratio: float = 0.5
    segment_gap: float = 5.0
    prose_min_width_fraction: float = 0.45
    prose_max_gap: float = 12.0
    stub_max_x_fraction: float = 0.30
    max_row_gap: float = 36.0
    max_header_gap: float = 30.0
    max_header_lines: int = 8
    max_lines_without_values: int = 2
    title_center_tolerance: float = 0.03
    max_unit_distance: float = 110.0
    min_value_rows: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "word-coordinate table candidates (pymupdf.Page.get_text('words'))",
            "find_tables_used": False,
            "word_flags": self.word_flags,
            "line_overlap_ratio": self.line_overlap_ratio,
            "segment_gap_pt": self.segment_gap,
            "prose_min_width_fraction": self.prose_min_width_fraction,
            "prose_max_gap_pt": self.prose_max_gap,
            "stub_max_x_fraction": self.stub_max_x_fraction,
            "max_row_gap_pt": self.max_row_gap,
            "max_header_gap_pt": self.max_header_gap,
            "max_header_lines": self.max_header_lines,
            "max_lines_without_values": self.max_lines_without_values,
            "title_center_tolerance": self.title_center_tolerance,
            "max_unit_distance_pt": self.max_unit_distance,
            "min_value_rows": self.min_value_rows,
        }


BASELINE_TABLE_CONFIG = TableConfig()


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TableWarning:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True, slots=True)
class SpanningHeader:
    text: str
    first_column: int
    last_column: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "first_column": self.first_column,
            "last_column": self.last_column,
        }


@dataclass(frozen=True, slots=True)
class TableRow:
    """``label`` is ``""`` when no stub text was found. ``None`` cells are empty
    positions (no text detected in that column on this row), never zero."""

    label: str
    cells: tuple[str | None, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "cells": list(self.cells)}


@dataclass(frozen=True, slots=True)
class TableRecord:
    table_id: str
    page_id: str
    document_id: str
    page_number: int
    bounding_box: tuple[float, float, float, float]
    structure_status: TableStatus
    column_count: int
    column_headers: tuple[str | None, ...]
    spanning_headers: tuple[SpanningHeader, ...]
    stub_header: str | None
    unit_statements: tuple[str, ...]
    rows: tuple[TableRow, ...]
    warnings: tuple[TableWarning, ...]

    def structural_problems(self) -> list[str]:
        """Internal consistency checks (independent of how the table was built)."""
        problems: list[str] = []
        if not self.table_id.startswith(f"{self.page_id}:t"):
            problems.append(f"{self.table_id}: id must start with {self.page_id}:t")
        if self.structure_status is TableStatus.FALLBACK:
            if self.rows or self.column_count or self.column_headers or self.spanning_headers:
                problems.append(f"{self.table_id}: fallback tables carry no structure")
            if not self.warnings:
                problems.append(f"{self.table_id}: fallback tables must explain why")
            return problems
        if self.column_count < 1:
            problems.append(f"{self.table_id}: needs at least one column")
        if len(self.column_headers) != self.column_count:
            problems.append(f"{self.table_id}: column_headers length != column_count")
        for row in self.rows:
            if len(row.cells) != self.column_count:
                problems.append(f"{self.table_id}: row cell count != column_count")
        for span in self.spanning_headers:
            if not 0 <= span.first_column < span.last_column < self.column_count:
                problems.append(f"{self.table_id}: invalid spanning header columns")
        if self.structure_status is TableStatus.STRUCTURED and self.warnings:
            problems.append(f"{self.table_id}: structured tables cannot carry warnings")
        if self.structure_status is TableStatus.UNCERTAIN and not self.warnings:
            problems.append(f"{self.table_id}: uncertain tables must explain why")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_id": self.page_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "bounding_box": list(self.bounding_box),
            "structure_status": self.structure_status.value,
            "column_count": self.column_count,
            "column_headers": list(self.column_headers),
            "spanning_headers": [s.to_dict() for s in self.spanning_headers],
            "stub_header": self.stub_header,
            "unit_statements": list(self.unit_statements),
            "rows": [r.to_dict() for r in self.rows],
            "warnings": [w.to_dict() for w in self.warnings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableRecord:
        box = data["bounding_box"]
        return cls(
            table_id=data["table_id"],
            page_id=data["page_id"],
            document_id=data["document_id"],
            page_number=data["page_number"],
            bounding_box=(box[0], box[1], box[2], box[3]),
            structure_status=TableStatus(data["structure_status"]),
            column_count=data["column_count"],
            column_headers=tuple(data["column_headers"]),
            spanning_headers=tuple(
                SpanningHeader(s["text"], s["first_column"], s["last_column"])
                for s in data["spanning_headers"]
            ),
            stub_header=data["stub_header"],
            unit_statements=tuple(data["unit_statements"]),
            rows=tuple(TableRow(r["label"], tuple(r["cells"])) for r in data["rows"]),
            warnings=tuple(TableWarning(w["code"], w["message"]) for w in data["warnings"]),
        )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(frozen=True, slots=True)
class Segment:
    x0: float
    x1: float
    text: str

    @property
    def has_digit(self) -> bool:
        return bool(_DIGIT_RE.search(self.text))


@dataclass(frozen=True, slots=True)
class Line:
    y0: float
    y1: float
    words: tuple[Word, ...]
    segments: tuple[Segment, ...]

    @property
    def x0(self) -> float:
        return self.words[0].x0

    @property
    def x1(self) -> float:
        return max(w.x1 for w in self.words)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def max_gap(self) -> float:
        gaps = [b.x0 - a.x1 for a, b in itertools.pairwise(self.words)]
        return max(gaps, default=0.0)


def page_words(page: pymupdf.Page, config: TableConfig) -> list[Word]:
    raw: list[Any] = page.get_text("words", flags=config.word_flags)
    return [Word(float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])) for w in raw]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def group_lines(words: list[Word], config: TableConfig) -> list[Line]:
    """Group words into visual lines by vertical overlap; order lines top-down.

    Each word is compared with the line's first (anchor) word, not with the
    growing line band, so closely stacked header lines do not chain together.
    """
    ordered = sorted(words, key=lambda w: ((w.y0 + w.y1) / 2, w.x0, w.text))
    groups: list[list[Word]] = []
    for word in ordered:
        if groups:
            anchor = groups[-1][0]
            height = min(word.y1 - word.y0, anchor.y1 - anchor.y0) or 1.0
            if (
                _overlap(word.y0, word.y1, anchor.y0, anchor.y1)
                >= config.line_overlap_ratio * height
            ):
                groups[-1].append(word)
                continue
        groups.append([word])
    lines = []
    for group in groups:
        ws = tuple(sorted(group, key=lambda w: (w.x0, w.y0, w.text)))
        lines.append(
            Line(
                y0=min(w.y0 for w in ws),
                y1=max(w.y1 for w in ws),
                words=ws,
                segments=_segments(ws, config),
            )
        )
    return lines


def _segments(words: tuple[Word, ...], config: TableConfig) -> tuple[Segment, ...]:
    """Split a line at wide gaps; a lone currency symbol joins the next segment."""
    groups: list[list[Word]] = [[words[0]]]
    for prev, word in itertools.pairwise(words):
        if word.x0 - prev.x1 > config.segment_gap:
            groups.append([word])
        else:
            groups[-1].append(word)
    merged: list[list[Word]] = []
    for group in groups:
        if merged and len(merged[-1]) == 1 and merged[-1][0].text in _CURRENCY_SYMBOLS:
            merged[-1].extend(group)
        else:
            merged.append(group)
    return tuple(Segment(g[0].x0, max(w.x1 for w in g), _join(g)) for g in merged)


def _join(words: list[Word]) -> str:
    """Join words with one space, or none where they touch (``(1)`` + ``%`` -> ``(1)%``)."""
    parts = [words[0].text]
    for prev, word in itertools.pairwise(words):
        parts.append(word.text if word.x0 - prev.x1 < _TOUCHING_GAP else " " + word.text)
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Table candidates
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Candidate:
    rows: list[Line]
    headers: list[Line]
    values_above: bool


def _split_stub(line: Line, stub_limit: float) -> tuple[str, list[Segment]]:
    segs = list(line.segments)
    if segs and segs[0].x0 < stub_limit and not _is_value(segs[0]):
        return segs[0].text, segs[1:]
    return "", segs


def _is_value(segment: Segment) -> bool:
    """A segment that starts like a financial value: a digit, a currency symbol,
    a dash placeholder, or a sign/parenthesis/point followed by a digit or ``$``.
    ``(A)`` or ``(1)`` style footnote markers followed by words are not values."""
    text = segment.text
    if text[0].isdigit() or text[0] in _CURRENCY_SYMBOLS or text in _DASHES:
        return True
    return len(text) > 1 and text[0] in "(-.\u2013\u2014" and (text[1].isdigit() or text[1] in "$.")


def _is_prose(line: Line, page_width: float, config: TableConfig) -> bool:
    """True if the line contains a long run of closely spaced words.

    A run is a sequence of words separated by gaps below ``prose_max_gap``
    (justified prose has gaps of up to about 8 pt here). Table rows have
    wide gaps between label and values, so no run spans a large part of
    the page. Checking runs rather than the whole line also catches a
    footnote marker followed by footnote prose on the same visual line.
    """
    limit = config.prose_min_width_fraction * page_width
    start = line.words[0].x0
    for prev, word in itertools.pairwise(line.words):
        if word.x0 - prev.x1 >= config.prose_max_gap:
            start = word.x0
        elif word.x1 - start >= limit:
            return True
    return line.words[-1].x1 - start >= limit


def _is_page_title(line: Line, page_width: float, config: TableConfig) -> bool:
    """A single segment centred on the page (statement titles, unit lines).

    Such lines sit above a table's column headers; treating them as headers
    would attach, for example, the company name to the first value column.
    """
    if len(line.segments) != 1:
        return False
    center = (line.x0 + line.x1) / 2
    return abs(center - page_width / 2) <= config.title_center_tolerance * page_width


def _looks_like_data(text: str) -> bool:
    """Value-like header text: amounts, percentages, decimals or bare non-year numbers.

    Years (``2024``), quarter labels (``Q3 FY25``) and dates (``October 27,``)
    are header-like; ``$ 19,309``, ``12%``, ``0.5`` or ``1`` are not.
    """
    if text in _DASHES or _DATA_RE.search(text):
        return True
    return any(t.isdigit() and not _YEAR_RE.fullmatch(t) for t in text.split())


def _header_lines(
    lines: list[Line],
    first_row: int,
    found: list[_Candidate],
    page_width: float,
    config: TableConfig,
) -> tuple[list[Line], bool]:
    """Collect header lines bottom-up above ``lines[first_row]``.

    Stops at prose, a large vertical gap, a page-centred title, the previous
    table's rows, a stub-only line above the column-header block, or a line
    with value-like text in the column area. The last case is reported
    (``values_above``) because it usually means a wrapped row label whose
    values sit above the first labelled row.
    """
    stub_limit = config.stub_max_x_fraction * page_width
    headers: list[Line] = []
    seen_column_text = False
    top = lines[first_row].y0
    for k in range(first_row - 1, -1, -1):
        line = lines[k]
        if (
            len(headers) >= config.max_header_lines
            or _is_prose(line, page_width, config)
            or top - line.y1 > config.max_header_gap
            or _is_page_title(line, page_width, config)
            or (found and line in found[-1].rows)
        ):
            break
        stub_text, column_segments = _split_stub(line, stub_limit)
        if any(_looks_like_data(s.text) for s in column_segments):
            return headers, True
        stub_only = bool(stub_text) and not column_segments
        if stub_only and seen_column_text:
            break
        seen_column_text = seen_column_text or bool(column_segments)
        headers.insert(0, line)
        top = line.y0
    return headers, False


def _candidates(lines: list[Line], page_width: float, config: TableConfig) -> list[_Candidate]:
    stub_limit = config.stub_max_x_fraction * page_width
    found: list[_Candidate] = []
    i = 0
    while i < len(lines):
        label, cells = _split_stub(lines[i], stub_limit)
        if (
            _is_prose(lines[i], page_width, config)
            or not label
            or not any(c.has_digit for c in cells)
        ):
            i += 1
            continue
        rows = [lines[i]]
        j = i + 1
        without_values = 0
        while j < len(lines):
            line = lines[j]
            if _is_prose(line, page_width, config) or line.y0 - rows[-1].y1 > config.max_row_gap:
                break
            has_values = any(c.has_digit for c in _split_stub(line, stub_limit)[1])
            without_values = 0 if has_values else without_values + 1
            if without_values > config.max_lines_without_values:
                break
            rows.append(line)
            j += 1
        # Trailing label-only lines (footnote markers, section titles) are not rows.
        while rows and not any(c.has_digit for c in _split_stub(rows[-1], stub_limit)[1]):
            rows.pop()
        value_rows = sum(1 for r in rows if any(c.has_digit for c in _split_stub(r, stub_limit)[1]))
        if value_rows >= config.min_value_rows:
            headers, values_above = _header_lines(lines, i, found, page_width, config)
            found.append(_Candidate(rows=rows, headers=headers, values_above=values_above))
            i = j
        else:
            i += 1
    return found


def _columns(cells: list[Segment]) -> list[tuple[float, float]]:
    """Union of overlapping x-intervals, left to right."""
    columns: list[list[float]] = []
    for cell in sorted(cells, key=lambda c: (c.x0, c.x1)):
        if columns and cell.x0 <= columns[-1][1]:
            columns[-1][1] = max(columns[-1][1], cell.x1)
        else:
            columns.append([cell.x0, cell.x1])
    return [(c[0], c[1]) for c in columns]


def _column_index(segment: Segment, columns: list[tuple[float, float]]) -> list[int]:
    return [i for i, (c0, c1) in enumerate(columns) if _overlap(segment.x0, segment.x1, c0, c1) > 0]


def _build_table(
    candidate: _Candidate,
    *,
    table_id: str,
    page_id: str,
    document_id: str,
    page_number: int,
    page_width: float,
    unit_lines: list[Line],
    config: TableConfig,
) -> TableRecord:
    stub_limit = config.stub_max_x_fraction * page_width
    lines = candidate.headers + candidate.rows
    top = min(line.y0 for line in lines)
    bottom = max(line.y1 for line in lines)
    box = (
        round(min(line.x0 for line in lines), 2),
        round(top, 2),
        round(max(line.x1 for line in lines), 2),
        round(bottom, 2),
    )
    split = [_split_stub(line, stub_limit) for line in candidate.rows]
    columns = _columns([c for _, cells in split for c in cells])
    warnings: list[TableWarning] = []

    rows: list[TableRow] = []
    for label, cells in split:
        slots: list[str | None] = [None] * len(columns)
        for cell in cells:
            hits = _column_index(cell, columns)
            if len(hits) != 1 or slots[hits[0]] is not None:
                return _fallback(
                    table_id,
                    page_id,
                    document_id,
                    page_number,
                    box,
                    TableWarning(
                        "column_collision",
                        f"cell {cell.text!r} cannot be assigned to exactly one free column",
                    ),
                )
            slots[hits[0]] = cell.text
        rows.append(TableRow(label=label, cells=tuple(slots)))

    header_parts: list[list[str]] = [[] for _ in columns]
    spanning: list[SpanningHeader] = []
    stub_parts: list[str] = []
    for line in candidate.headers:
        for seg in line.segments:
            hits = _column_index(seg, columns)
            if seg.x1 < (columns[0][0] if columns else stub_limit) and not hits:
                stub_parts.append(seg.text)
            elif len(hits) == 1:
                header_parts[hits[0]].append(seg.text)
            elif len(hits) > 1 and hits == list(range(hits[0], hits[-1] + 1)):
                spanning.append(SpanningHeader(seg.text, hits[0], hits[-1]))
            else:
                warnings.append(
                    TableWarning(
                        "header_text_between_columns",
                        f"header text {seg.text!r} lies between columns and was not assigned",
                    )
                )
    column_headers = tuple("\n".join(parts) if parts else None for parts in header_parts)

    if candidate.values_above:
        warnings.append(
            TableWarning(
                "values_above_first_row",
                "value-like text directly above the first labelled row was not used as a "
                "header and is not in the table (possible wrapped row label)",
            )
        )
    ranges = sorted({(s.first_column, s.last_column) for s in spanning})
    if any(a != b and a[0] <= b[1] and b[0] <= a[1] for a in ranges for b in ranges if a < b):
        warnings.append(
            TableWarning(
                "spanning_header_ranges_ambiguous",
                "spanning headers overlap different column ranges; their column ranges are "
                "positional estimates and cannot be relied on",
            )
        )
    if not candidate.headers:
        warnings.append(
            TableWarning(
                "no_header_rows",
                "no header lines above the first row on this page "
                "(possible continuation of a table from a previous page)",
            )
        )
    missing = [i for i, h in enumerate(column_headers) if h is None]
    if missing and candidate.headers:
        warnings.append(
            TableWarning("columns_without_header", f"no header text for column(s) {missing}")
        )
    unlabeled = sum(1 for r in rows if not r.label and any(c is not None for c in r.cells))
    if unlabeled:
        warnings.append(
            TableWarning(
                "rows_without_label",
                f"{unlabeled} row(s) with values but no stub label "
                "(wrapped label or wrapped cell; not merged)",
            )
        )
    # Unit statements inside the table box, else the nearest one shortly above it. A
    # statement further up the page usually belongs to another section and is not used.
    # Only the matching segment is kept, so column headers sharing the line are not mixed in.
    # Unrounded edges: the rounded bounding box can exclude a line on the table's edge.
    inside = [line for line in unit_lines if line.y0 >= top and line.y1 <= bottom]
    above = [line for line in unit_lines if top - config.max_unit_distance <= line.y1 < top]
    chosen = inside or above[-1:]
    units = tuple(seg.text for line in chosen for seg in line.segments if _UNIT_RE.search(seg.text))
    if not units:
        warnings.append(
            TableWarning(
                "no_unit_statement_on_page",
                "no 'in millions/billions/thousands' statement above the table on this page",
            )
        )
    status = TableStatus.UNCERTAIN if warnings else TableStatus.STRUCTURED
    return TableRecord(
        table_id=table_id,
        page_id=page_id,
        document_id=document_id,
        page_number=page_number,
        bounding_box=box,
        structure_status=status,
        column_count=len(columns),
        column_headers=column_headers,
        spanning_headers=tuple(spanning),
        stub_header=" ".join(stub_parts) if stub_parts else None,
        unit_statements=units,
        rows=tuple(rows),
        warnings=tuple(warnings),
    )


def _fallback(
    table_id: str,
    page_id: str,
    document_id: str,
    page_number: int,
    box: tuple[float, float, float, float],
    warning: TableWarning,
) -> TableRecord:
    return TableRecord(
        table_id=table_id,
        page_id=page_id,
        document_id=document_id,
        page_number=page_number,
        bounding_box=box,
        structure_status=TableStatus.FALLBACK,
        column_count=0,
        column_headers=(),
        spanning_headers=(),
        stub_header=None,
        unit_statements=(),
        rows=(),
        warnings=(warning,),
    )


def extract_tables(
    page: pymupdf.Page,
    *,
    page_id: str,
    document_id: str,
    page_number: int,
    config: TableConfig = BASELINE_TABLE_CONFIG,
) -> tuple[TableRecord, ...]:
    """Table candidates on one page, top to bottom, with ids ``{page_id}:tNN``."""
    words = page_words(page, config)
    if not words:
        return ()
    lines = group_lines(words, config)
    width = float(page.rect.width)
    unit_lines = [line for line in lines if _UNIT_RE.search(line.text)]
    return tuple(
        _build_table(
            candidate,
            table_id=f"{page_id}:t{index:02d}",
            page_id=page_id,
            document_id=document_id,
            page_number=page_number,
            page_width=width,
            unit_lines=unit_lines,
            config=config,
        )
        for index, candidate in enumerate(_candidates(lines, width, config), start=1)
    )
