"""Table candidates on synthetic layout PDFs, through the real ``extract_tables`` path."""

from __future__ import annotations

import json

import pymupdf
import pytest

from founderlens.ingestion.tables import (
    TableRecord,
    TableRow,
    TableStatus,
    TableWarning,
    extract_tables,
)
from tests.conftest import PlacedText, make_layout_pdf, place_center, place_right

COL_A, COL_B = 420.0, 500.0  # right edges of two value columns


def _tables(items: list[PlacedText], page_id: str = "doc-x:p0001") -> tuple[TableRecord, ...]:
    with pymupdf.open(stream=make_layout_pdf([items]), filetype="pdf") as doc:
        return extract_tables(doc[0], page_id=page_id, document_id="doc-x", page_number=1)


def _row(
    y: float, label: str, a: str | None, b: str | None, *, dollar: bool = False
) -> list[PlacedText]:
    items: list[PlacedText] = [(50.0, y, label)]
    for right, value in ((COL_A, a), (COL_B, b)):
        if value is None:
            continue
        if dollar:
            items.append((right - 45, y, "$"))
        items.append(place_right(right, y, value))
    return items


def _header(y_span: float = 100.0, y_years: float = 112.0) -> list[PlacedText]:
    return [
        place_center(306, 60, "Example Corp"),
        place_center(306, 72, "(In millions)"),
        place_center((COL_A + COL_B) / 2 - 20, y_span, "Quarter Ended June 30,"),
        place_right(COL_A, y_years, "2023"),
        place_right(COL_B, y_years, "2024"),
    ]


def _basic_table() -> list[PlacedText]:
    return [
        *_header(),
        *_row(130, "Revenue", "1,234", "2,345", dollar=True),
        *_row(144, "Net loss", "(56)", "(78)"),
        *_row(158, "Operating margin(1)", "12.5%", "14.0%"),
    ]


def test_structured_table_keeps_rows_columns_and_headers() -> None:
    [table] = _tables(_basic_table())
    assert table.structure_status is TableStatus.STRUCTURED
    assert table.warnings == ()
    assert table.table_id == "doc-x:p0001:t01"
    assert table.column_count == 2
    assert table.column_headers == ("2023", "2024")
    assert [(s.text, s.first_column, s.last_column) for s in table.spanning_headers] == [
        ("Quarter Ended June 30,", 0, 1)
    ]
    assert table.unit_statements == ("(In millions)",)
    assert table.rows == (
        TableRow("Revenue", ("$ 1,234", "$ 2,345")),
        TableRow("Net loss", ("(56)", "(78)")),
        TableRow("Operating margin(1)", ("12.5%", "14.0%")),
    )


def test_page_title_is_not_absorbed_into_column_headers() -> None:
    [table] = _tables(_basic_table())
    joined = " ".join(h or "" for h in table.column_headers)
    assert "Example" not in joined
    assert "millions" not in joined


def test_empty_cell_does_not_shift_columns() -> None:
    items = [
        *_basic_table(),
        *_row(172, "Hedging gains", None, "17"),
        *_row(186, "Other items", "(3)", None),
    ]
    [table] = _tables(items)
    assert table.structure_status is TableStatus.STRUCTURED
    assert table.rows[-2] == TableRow("Hedging gains", (None, "17"))
    assert table.rows[-1] == TableRow("Other items", ("(3)", None))


def test_multi_line_headers_assigned_by_position() -> None:
    cols = (380.0, 450.0, 520.0)
    items: list[PlacedText] = [
        place_center(306, 72, "(In millions)"),
        place_right(cols[0], 100, "October"),
        place_right(cols[2], 100, "October"),
        place_right(cols[1], 106, "July 28,"),  # printed between the other header lines
        place_right(cols[0], 112, "27,"),
        place_right(cols[2], 112, "29,"),
        *(place_right(c, 124, y) for c, y in zip(cols, ("2024", "2024", "2023"), strict=True)),
    ]
    for n, y in enumerate((140, 154)):
        items.append((50.0, y, f"Line item {n}"))
        items.extend(place_right(c, y, f"{(n + 1) * (i + 1)}0") for i, c in enumerate(cols))
    [table] = _tables(items)
    assert table.column_headers == ("October\n27,\n2024", "July 28,\n2024", "October\n29,\n2023")
    assert table.structure_status is TableStatus.STRUCTURED


def test_wrapped_cell_is_flagged_not_merged() -> None:
    items = [
        *_basic_table(),
        place_right(COL_B, 165, "Down 0.5"),
        *_row(172, "Gross margin", "74.6%", "75.1%"),
        place_right(COL_B, 179, "pts"),
        *_row(193, "Net income", "10", "20"),
    ]
    [table] = _tables(items)
    assert table.structure_status is TableStatus.UNCERTAIN
    assert "rows_without_label" in {w.code for w in table.warnings}
    labels = [r.label for r in table.rows]
    assert labels[3:] == ["", "Gross margin", "", "Net income"]
    assert table.rows[3].cells == (None, "Down 0.5")  # kept in its column
    assert table.rows[5].cells == (None, "pts")


def test_label_printed_below_values_stays_on_the_row() -> None:
    items = [*_basic_table(), place_right(COL_A, 172, "16"), place_right(COL_B, 172, "51")]
    items.append((50.0, 176.5, "Sales, general and administrative"))  # 4.5 pt lower
    [table] = _tables(items)
    assert table.rows[-1] == TableRow("Sales, general and administrative", ("16", "51"))


def test_continuation_without_headers_or_units_is_uncertain() -> None:
    items = [
        *_row(60, "Operating income", "21,869", "10,417"),
        *_row(74, "Other, net", "36", "(66)"),
    ]
    [table] = _tables(items)
    assert table.structure_status is TableStatus.UNCERTAIN
    assert table.column_headers == (None, None)  # nothing invented
    assert table.unit_statements == ()
    assert {w.code for w in table.warnings} == {"no_header_rows", "no_unit_statement_on_page"}
    assert table.rows[1].cells == ("36", "(66)")


def test_ambiguous_columns_fall_back_without_structure() -> None:
    items = [
        *_basic_table(),
        (390.0, 172, "76.0% Down 1.6 pts and more"),
        *_row(186, "Tax", "1", "2"),
    ]
    items.insert(-3, (50.0, 172, "Gross margin"))
    [table] = _tables(items)
    assert table.structure_status is TableStatus.FALLBACK
    assert [w.code for w in table.warnings] == ["column_collision"]
    assert table.rows == ()
    assert table.column_headers == ()
    assert table.structural_problems() == []


def test_values_above_first_labelled_row_are_not_used_as_headers() -> None:
    items = [
        *_header(),
        place_right(COL_A, 124, "19,309"),
        place_right(COL_B, 124, "9,243"),
        *_row(140, "Depreciation", "1", "2"),
        *_row(154, "Other", "3", "4"),
    ]
    [table] = _tables(items)
    assert table.structure_status is TableStatus.UNCERTAIN
    assert "values_above_first_row" in {w.code for w in table.warnings}
    assert "19,309" not in " ".join(h or "" for h in table.column_headers)


def test_prose_page_has_no_tables() -> None:
    sentence = (
        "Revenue grew 15% in 2024 compared with the prior year across all segments of the business."
    )
    items: list[PlacedText] = [(50.0, 100.0 + 14 * i, sentence) for i in range(4)]
    assert _tables(items) == ()


def test_stable_ordered_table_ids() -> None:
    second = [(x, y + 300, text) for x, y, text in _basic_table()[2:]]
    items = [*_basic_table(), *second]
    first_run = _tables(items, page_id="doc-x:p0007")
    assert [t.table_id for t in first_run] == ["doc-x:p0007:t01", "doc-x:p0007:t02"]
    assert first_run[0].bounding_box[1] < first_run[1].bounding_box[1]
    assert _tables(items, page_id="doc-x:p0007") == first_run


def test_record_round_trip_and_consistency_checks() -> None:
    [table] = _tables(_basic_table())
    restored = TableRecord.from_dict(json.loads(json.dumps(table.to_dict())))
    assert restored == table
    bad_structured = TableRecord.from_dict(
        {**table.to_dict(), "warnings": [{"code": "x", "message": "y"}]}
    )
    assert any("structured tables cannot carry" in p for p in bad_structured.structural_problems())
    bad_cells = TableRecord.from_dict({**table.to_dict(), "rows": [{"label": "A", "cells": ["1"]}]})
    assert any("row cell count" in p for p in bad_cells.structural_problems())


@pytest.mark.parametrize(
    ("status", "warnings"),
    [(TableStatus.UNCERTAIN, ()), (TableStatus.FALLBACK, ())],
)
def test_non_structured_tables_must_explain_why(
    status: TableStatus, warnings: tuple[TableWarning, ...]
) -> None:
    [table] = _tables(_basic_table())
    data = table.to_dict()
    if status is TableStatus.FALLBACK:
        data.update(column_count=0, column_headers=[], spanning_headers=[], rows=[])
    record = TableRecord.from_dict({**data, "structure_status": status.value, "warnings": []})
    assert record.warnings == warnings
    assert any("must explain why" in p for p in record.structural_problems())
