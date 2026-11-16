"""Page record schema: identity, consistency checks and deterministic JSON."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from founderlens.ingestion.schema import (
    RECORD_FIELDS,
    ExtractionStatus,
    PageRecord,
    PageRecordError,
    make_page_id,
    parse_pages_jsonl,
    status_for_text,
)

SHA = "a" * 64


def _record(text: str = "Revenue $35,082", page_number: int = 1) -> PageRecord:
    return PageRecord.build(
        document_id="nvda-fy2025-q3",
        document_sha256=SHA,
        company="NVIDIA Corporation",
        fiscal_period="FY2025 Q3",
        source_url="https://example.com/results",
        page_number=page_number,
        text=text,
    )


# --- identity ----------------------------------------------------------------


def test_page_id_convention() -> None:
    assert make_page_id("nvda-fy2025-q3", 1) == "nvda-fy2025-q3:p0001"
    assert make_page_id("goog-2024-q4", 10) == "goog-2024-q4:p0010"
    assert make_page_id("goog-2024-q4", 1234) == "goog-2024-q4:p1234"


@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "1"])
def test_invalid_page_numbers_rejected(bad: Any) -> None:
    with pytest.raises(PageRecordError, match="page_number"):
        make_page_id("doc-a", bad)
    with pytest.raises(PageRecordError, match="page_number"):
        PageRecord.build(
            document_id="doc-a",
            document_sha256=SHA,
            company="C",
            fiscal_period="P",
            source_url="https://example.com",
            page_number=bad,
            text="x",
        )


def test_valid_record_derives_fields() -> None:
    record = _record("Revenue $35,082\n")
    assert record.page_id == "nvda-fy2025-q3:p0001"
    assert record.character_count == len("Revenue $35,082\n") == 16
    assert record.extraction_status is ExtractionStatus.OK


def test_page_id_must_match_document_and_page() -> None:
    record = _record(page_number=2)
    with pytest.raises(PageRecordError, match="page_id"):
        replace(record, page_id="nvda-fy2025-q3:p0001")
    with pytest.raises(PageRecordError, match="page_id"):
        replace(record, page_id="nvda-fy2025-q3:p2")


@pytest.mark.parametrize("bad_count", [0, 14, 16, True, "15"])
def test_inconsistent_character_count_rejected(bad_count: Any) -> None:
    record = _record("Revenue $35,082")  # 15 characters
    with pytest.raises(PageRecordError, match="character_count"):
        replace(record, character_count=bad_count)


def test_character_count_counts_code_points_not_bytes() -> None:
    record = _record("€ 日本")
    assert record.character_count == 4
    assert len(record.text.encode("utf-8")) > 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_id", "NVDA Q3"),
        ("document_sha256", "abc"),
        ("company", " "),
        ("source_url", ""),
    ],
)
def test_invalid_metadata_rejected(field: str, value: str) -> None:
    changes: dict[str, Any] = {field: value}
    record = _record()
    with pytest.raises(PageRecordError):
        replace(record, **changes)


# --- empty pages -------------------------------------------------------------


@pytest.mark.parametrize("text", ["", " ", "\n\n", " \t\n"])
def test_empty_page_is_explicit_and_keeps_raw_text(text: str) -> None:
    record = _record(text)
    assert record.extraction_status is ExtractionStatus.EMPTY
    assert record.text == text
    assert record.character_count == len(text)


def test_status_must_agree_with_text() -> None:
    assert status_for_text("x") is ExtractionStatus.OK
    with pytest.raises(PageRecordError, match="contradicts"):
        replace(_record("x"), extraction_status=ExtractionStatus.EMPTY)
    with pytest.raises(PageRecordError, match="contradicts"):
        replace(_record(""), extraction_status=ExtractionStatus.OK)


# --- serialization -----------------------------------------------------------


def test_json_line_is_deterministic_single_line_with_sorted_keys() -> None:
    record = _record("Revenue € (1,234.5)\n  12.5%\n")
    line = record.to_json_line()
    assert line == _record("Revenue € (1,234.5)\n  12.5%\n").to_json_line()
    assert line.endswith("\n")
    assert line.count("\n") == 1  # the text's own newlines are escaped
    assert "€" in line  # UTF-8 text, not \u escapes
    decoded = json.loads(line)
    assert list(decoded) == sorted(RECORD_FIELDS)
    assert decoded["text"] == "Revenue € (1,234.5)\n  12.5%\n"
    assert decoded["extraction_status"] == "ok"


def test_round_trip_through_json() -> None:
    record = _record("Line 1\nLine 2 — ™")
    assert PageRecord.from_dict(json.loads(record.to_json_line())) == record


def test_from_dict_rejects_missing_unknown_and_invalid_fields() -> None:
    data = _record().to_dict()
    with pytest.raises(PageRecordError, match="missing"):
        PageRecord.from_dict({k: v for k, v in data.items() if k != "page_id"})
    with pytest.raises(PageRecordError, match="unknown"):
        PageRecord.from_dict({**data, "path": "C:/x.pdf"})
    with pytest.raises(PageRecordError, match="extraction_status"):
        PageRecord.from_dict({**data, "extraction_status": "failed"})
    with pytest.raises(PageRecordError, match="character_count"):
        PageRecord.from_dict({**data, "character_count": 3})
    with pytest.raises(PageRecordError, match="JSON object"):
        PageRecord.from_dict([data])


def test_parse_jsonl_strict_format() -> None:
    body = _record(page_number=1).to_json_line() + _record(page_number=2).to_json_line()
    assert [r.page_number for r in parse_pages_jsonl(body)] == [1, 2]
    assert parse_pages_jsonl("") == []
    with pytest.raises(PageRecordError, match="LF"):
        parse_pages_jsonl(body.replace("\n", "\r\n"))
    with pytest.raises(PageRecordError, match="line feed"):
        parse_pages_jsonl(body.rstrip("\n"))
    with pytest.raises(PageRecordError, match="line 2: blank line"):
        parse_pages_jsonl(body.replace("\n", "\n\n", 1))
    with pytest.raises(PageRecordError, match="line 1"):
        parse_pages_jsonl("{not json}\n")
