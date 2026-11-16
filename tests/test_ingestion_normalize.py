"""Text normalization, normalized page records, artifacts and the normalize CLI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from founderlens.config import Settings
from founderlens.ingestion import normalize as nz
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import (
    NormalizationError,
    NormalizedPageRecord,
    normalize_page,
    normalize_text,
    parse_normalized_jsonl,
    run_normalization,
)
from founderlens.ingestion.schema import (
    ExtractionStatus,
    PageRecord,
    PageRecordError,
    parse_pages_jsonl,
)
from tests.conftest import (
    DOC_A,
    DOC_B,
    PlacedText,
    make_layout_pdf,
    make_text_pdf,
    place_right,
)

Factory = Callable[..., Path]

# --- text normalization ------------------------------------------------------


def test_line_endings_become_lf() -> None:
    assert normalize_text("a\r\nb\rc\n") == "a\nb\nc\n"


def test_unicode_nfc_only_no_compatibility_folding() -> None:
    assert normalize_text("Cafe\u0301\n") == "Caf\u00e9\n"  # combining accent composed
    kept = (
        "NVIDIA Isaac\u2122 \ufb01nance \u2460 \u00bd\n"  # trademark, fi ligature, circled 1, half
    )
    assert normalize_text(kept) == kept


def test_trailing_whitespace_and_blank_lines() -> None:
    raw = "\n\nTitle   \n\n\n\n   Body line\t \n\n\n"
    assert normalize_text(raw) == "Title\n\n   Body line\n"


def test_table_spacing_and_financial_notation_preserved() -> None:
    raw = (
        "Revenue                    $      35,082   $   18,120\n"
        "  Interest expense             (61)        (63)\n"
        "Margin(1)   74.6%   Down 0.5 pts   \u22125%   -   --   $0.78\n"
        "($ in millions, except earnings per share)  GAAP  Non-GAAP  Q3 FY25\n"
    )
    assert normalize_text(raw) == raw


@pytest.mark.parametrize("raw", ["", " ", "\n\n", " \t\r\n"])
def test_empty_input(raw: str) -> None:
    assert normalize_text(raw) == ""


@pytest.mark.parametrize(
    "raw",
    [
        "a\r\n\r\n\r\nb  \n",
        "  indented\n\n\n\ntext\u0301 \n",
        "x",
        "\n\n  $ (1,234.5)  \n\n",
        "",
    ],
)
def test_normalization_is_idempotent(raw: str) -> None:
    once = normalize_text(raw)
    assert normalize_text(once) == once


# --- normalized page record --------------------------------------------------


def _raw(text: str = "Revenue $ 1\n", page: int = 1) -> PageRecord:
    return PageRecord.build(
        document_id="doc-x",
        document_sha256="a" * 64,
        company="Example Corp",
        fiscal_period="FY2024 Q1",
        source_url="https://example.com",
        page_number=page,
        text=text,
    )


def test_normalized_record_preserves_identity_and_derives_counts() -> None:
    raw = _raw("Revenue   $ 1   \n\n\n")
    record = normalize_page(raw, ())
    assert (record.page_id, record.document_sha256, record.company, record.source_url) == (
        raw.page_id,
        raw.document_sha256,
        raw.company,
        raw.source_url,
    )
    assert record.raw_character_count == raw.character_count
    assert record.normalized_text == "Revenue   $ 1\n"
    assert record.normalized_character_count == len(record.normalized_text)
    assert record.table_text_overlap is False
    assert record.normalization_warnings == ()


def test_empty_page_record() -> None:
    record = normalize_page(_raw("  \n"), ())
    assert record.extraction_status is ExtractionStatus.EMPTY
    assert record.normalized_text == ""
    assert [w.code for w in record.normalization_warnings] == ["empty_page"]


def test_normalized_record_consistency_checks() -> None:
    record = normalize_page(_raw(), ())
    with pytest.raises(PageRecordError, match="page_id"):
        replace(record, page_id="doc-x:p0002")
    with pytest.raises(PageRecordError, match="normalized_character_count"):
        replace(record, normalized_character_count=1)
    with pytest.raises(PageRecordError, match="empty status"):
        replace(record, extraction_status=ExtractionStatus.EMPTY)
    with pytest.raises(PageRecordError, match="table_text_overlap"):
        replace(record, table_text_overlap=True)


def test_normalized_json_round_trip_and_strict_parsing() -> None:
    record = normalize_page(_raw("Line\u2014one \u2122\n"), ())
    line = record.to_json_line()
    obj = json.loads(line)
    assert list(obj) == sorted(obj)
    assert parse_normalized_jsonl(line) == [record]
    with pytest.raises(PageRecordError, match="LF"):
        parse_normalized_jsonl(line.replace("\n", "\r\n"))
    with pytest.raises(PageRecordError, match="unknown"):
        NormalizedPageRecord.from_dict({**obj, "local_path": "C:/x"})


# --- pipeline on synthetic PDFs ----------------------------------------------


def _table_page(prefix: str) -> list[PlacedText]:
    items: list[PlacedText] = [
        (50.0, 60.0, f"{prefix} results (in millions)"),
        place_right(420, 100, "2023"),
        place_right(500, 100, "2024"),
    ]
    for n, (label, a, b) in enumerate(
        [("Revenue", "$ 1,234", "$ 2,345"), ("Net loss", "(56)", "(78)"), ("Hedging", "", "17")]
    ):
        y = 120.0 + 14 * n
        items.append((50.0, y, label))
        items.extend(place_right(x, y, v) for x, v in ((420, a), (500, b)) if v)
    return items


def _project(make_project: Factory) -> Path:
    narrative = make_text_pdf([f"{DOC_A} narrative page", ""])
    tables = make_layout_pdf([_table_page("Example"), [(50.0, 80.0, "Closing notes")]])
    return make_project({DOC_A: (narrative, 2), DOC_B: (tables, 2)})


def _run(root: Path) -> nz.NormalizationOutcome:
    run_extraction(Settings(root), expected_document_count=2)
    return run_normalization(Settings(root), expected_document_count=2)


def _normalized(root: Path) -> Path:
    return root / "data" / "processed" / "pages.normalized.jsonl"


def _raw_artifact(root: Path) -> Path:
    return root / "data" / "processed" / "pages.raw.jsonl"


def test_pipeline_records_tables_and_metadata(make_project: Factory) -> None:
    root = _project(make_project)
    outcome = _run(root)
    raw = parse_pages_jsonl(_raw_artifact(root).read_text(encoding="utf-8"))
    records = parse_normalized_jsonl(_normalized(root).read_text(encoding="utf-8"))
    assert [r.page_id for r in records] == [r.page_id for r in raw]
    assert len(records) == 4
    for norm, src in zip(records, raw, strict=True):
        assert norm.document_sha256 == src.document_sha256
        assert norm.fiscal_period == src.fiscal_period
        assert norm.raw_character_count == src.character_count
    assert records[1].extraction_status is ExtractionStatus.EMPTY
    [table] = records[2].tables
    assert table.table_id == f"{DOC_B}:p0001:t01"
    assert table.column_headers == ("2023", "2024")
    assert [r.cells for r in table.rows] == [
        ("$ 1,234", "$ 2,345"),
        ("(56)", "(78)"),
        (None, "17"),
    ]
    assert table.unit_statements == ("Example results (in millions)",)
    assert records[2].table_text_overlap is True
    assert "Revenue" in records[2].normalized_text  # tables never replace page text
    assert outcome.report["tables"] == {
        "candidates": 1,
        "structured": 1,
        "uncertain": 0,
        "fallback": 0,
    }


def test_artifact_is_deterministic_and_raw_untouched(make_project: Factory) -> None:
    root = _project(make_project)
    first = _run(root)
    raw_bytes = _raw_artifact(root).read_bytes()
    first_bytes = _normalized(root).read_bytes()
    second = run_normalization(Settings(root), expected_document_count=2)
    assert _normalized(root).read_bytes() == first_bytes
    assert _raw_artifact(root).read_bytes() == raw_bytes
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.artifact_sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert b"\r" not in first_bytes
    assert str(root).encode() not in first_bytes
    report = json.loads((root / "reports" / "normalization_report.json").read_text("utf-8"))
    assert report["raw_artifact"]["sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert report["normalized_artifact"]["sha256"] == first.artifact_sha256
    assert set(report["operational"]) == {"note", "normalization_duration_seconds"}


def test_report_states_the_final_newline_rule(make_project: Factory) -> None:
    root = _project(make_project)
    _run(root)
    report = json.loads((root / "reports" / "normalization_report.json").read_text("utf-8"))
    rule = report["normalization_config"]["final_newline"]
    assert rule == "exactly one LF on non-empty text; empty text stays empty"
    # The stated rule matches the implemented behaviour.
    assert normalize_text("a") == normalize_text("a\n\n\n") == "a\n"
    assert normalize_text("") == normalize_text(" \n\t\n") == ""
    for record in parse_normalized_jsonl(_normalized(root).read_text("utf-8")):
        text = record.normalized_text
        assert text == "" or (text.endswith("\n") and not text.endswith("\n\n"))


def test_missing_raw_artifact(make_project: Factory) -> None:
    root = _project(make_project)
    with pytest.raises(NormalizationError, match=r"raw artifact not found.*ingestion\.extract"):
        run_normalization(Settings(root), expected_document_count=2)
    assert not (root / "data" / "processed").exists()


@pytest.mark.parametrize("damage", ["drop_last", "duplicate", "wrong_company", "not_json"])
def test_inconsistent_raw_artifact_rejected(make_project: Factory, damage: str) -> None:
    root = _project(make_project)
    run_extraction(Settings(root), expected_document_count=2)
    path = _raw_artifact(root)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if damage == "drop_last":
        lines = lines[:-1]
    elif damage == "duplicate":
        lines = [*lines[:-1], lines[0]]
    elif damage == "wrong_company":
        obj = json.loads(lines[0])
        obj["company"] = "Other Corp"
        lines[0] = json.dumps(obj, sort_keys=True) + "\n"
    else:
        lines[0] = "{broken\n"
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    with pytest.raises(NormalizationError, match="raw artifact"):
        run_normalization(Settings(root), expected_document_count=2)


def test_stale_raw_text_rejected(make_project: Factory) -> None:
    root = _project(make_project)
    run_extraction(Settings(root), expected_document_count=2)
    path = _raw_artifact(root)
    records = parse_pages_jsonl(path.read_text(encoding="utf-8"))
    edited = PageRecord.build(
        document_id=records[0].document_id,
        document_sha256=records[0].document_sha256,
        company=records[0].company,
        fiscal_period=records[0].fiscal_period,
        source_url=records[0].source_url,
        page_number=1,
        text="Edited text that the PDF does not contain\n",
    )
    body = "".join(r.to_json_line() for r in [edited, *records[1:]])
    path.write_text(body, encoding="utf-8", newline="\n")
    with pytest.raises(NormalizationError, match="differs from a fresh extraction"):
        run_normalization(Settings(root), expected_document_count=2)


def test_corpus_integrity_failure(make_project: Factory) -> None:
    root = _project(make_project)
    run_extraction(Settings(root), expected_document_count=2)
    pdf = root / "data" / "raw" / f"{DOC_B}.pdf"
    pdf.write_bytes(pdf.read_bytes() + b"\n%tampered")
    with pytest.raises(NormalizationError, match="corpus validation failed"):
        run_normalization(Settings(root), expected_document_count=2)


def test_failed_run_preserves_previous_normalized_artifact(make_project: Factory) -> None:
    root = _project(make_project)
    _run(root)
    before = _normalized(root).read_bytes()
    report_before = (root / "reports" / "normalization_report.json").read_bytes()
    _raw_artifact(root).unlink()
    with pytest.raises(NormalizationError):
        run_normalization(Settings(root), expected_document_count=2)
    assert _normalized(root).read_bytes() == before
    assert (root / "reports" / "normalization_report.json").read_bytes() == report_before
    assert not [p for p in (root / "data" / "processed").iterdir() if p.suffix == ".tmp"]


def test_written_artifact_is_verified_before_publishing(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(make_project)
    _run(root)
    before = _normalized(root).read_bytes()
    original = nz.serialize_normalized
    # Simulate a serializer bug that drops the last record.
    monkeypatch.setattr(nz, "serialize_normalized", lambda records: original(records[:-1]))
    with pytest.raises(NormalizationError, match="written artifact invalid"):
        run_normalization(Settings(root), expected_document_count=2)
    assert _normalized(root).read_bytes() == before


# --- CLI ---------------------------------------------------------------------


def _four_documents_project(make_project: Factory) -> Path:
    ids = ["aaa-2024-q1", "bbb-2024-q1", "ccc-2024-q1", "ddd-2024-q1"]
    docs = {doc_id: (make_layout_pdf([_table_page("SECRET-BODY")]), 1) for doc_id in ids}
    return make_project(docs)


def test_cli_success(make_project: Factory, capsys: pytest.CaptureFixture[str]) -> None:
    root = _four_documents_project(make_project)
    run_extraction(Settings(root))
    assert nz.main(["--project-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "Table candidates: 4 (structured 4, uncertain 0, fallback 0)" in out
    assert "Normalized artifact: data/processed/pages.normalized.jsonl" in out
    assert "SECRET-BODY" not in out


def test_cli_missing_raw_artifact(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _four_documents_project(make_project)
    assert nz.main(["--project-root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "Normalization FAILED" in err
    assert "python -m founderlens.ingestion.extract" in err
    assert "No artifact was published" in err


def test_cli_corpus_failure(make_project: Factory, capsys: pytest.CaptureFixture[str]) -> None:
    root = _four_documents_project(make_project)
    run_extraction(Settings(root))
    (root / "data" / "raw" / "bbb-2024-q1.pdf").unlink()
    assert nz.main(["--project-root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "corpus validation failed" in err
    assert "bbb-2024-q1: missing file" in err
