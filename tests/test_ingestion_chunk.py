"""Page-aware chunking: configuration, windows, records, collection checks, artifacts, CLI."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from founderlens.config import Settings
from founderlens.ingestion import chunk as ck
from founderlens.ingestion.chunk import (
    ChunkingConfig,
    ChunkingConfigError,
    ChunkingError,
    ChunkRecord,
    ChunkRecordError,
    chunk_page,
    count_words,
    page_overlaps,
    parse_chunks_jsonl,
    plan_windows,
    run_chunking,
    uncovered_characters,
    validate_chunks,
)
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import (
    NormalizedPageRecord,
    page_warnings,
    parse_normalized_jsonl,
    run_normalization,
)
from founderlens.ingestion.schema import make_page_id, status_for_text
from tests.conftest import DOC_A, DOC_B, PlacedText, make_layout_pdf, make_text_pdf, place_right

Factory = Callable[..., Path]


def _page(text: str, page_number: int = 1, document_id: str = DOC_A) -> NormalizedPageRecord:
    status = status_for_text(text)
    return NormalizedPageRecord(
        document_id=document_id,
        document_sha256="a" * 64,
        company="Example Corp",
        fiscal_period="Q1 2024",
        source_url="https://example.com/report.pdf",
        page_number=page_number,
        page_id=make_page_id(document_id, page_number),
        extraction_status=status,
        raw_character_count=len(text),
        normalized_text=text,
        normalized_character_count=len(text),
        tables=(),
        table_text_overlap=False,
        normalization_warnings=page_warnings(status, ()),
    )


def _words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def _chunk_texts(text: str, config: ChunkingConfig) -> list[str]:
    return [c.text for c in chunk_page(_page(text), config)]


# A page mixing paragraphs, short lines, a long line and column padding.
MIXED_TEXT = (
    "Fourth Quarter Results\n\n"
    f"{_words(25, 'a')}\n{_words(7, 'b')}\n\n"
    "                         Three Months Ended\n"
    "                       October 27,   October 29,\n"
    "Revenue             $     35,082   $     18,120\n"
    "Net loss                  (1,234)         (56)\n"
    "Gross margin                 74.6%        75.1%\n"
    "(In millions)\n\n"
    f"{_words(40, 'c')}\n"
)

# --- configuration ----------------------------------------------------------------


def test_default_configuration_and_identifier() -> None:
    config = ChunkingConfig()
    assert (config.chunk_size_words, config.chunk_overlap_words) == (120, 24)
    assert config.min_fill_words == 60
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    assert config.config_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    assert config.config_id == config.config_sha256[:12]
    assert ChunkingConfig(120, 24).config_id == config.config_id  # stable, not hash()
    assert config.to_dict()["table_candidates_indexed"] is False
    assert config.to_dict()["source_field"] == "normalized_text"


def test_identifier_changes_with_every_content_parameter() -> None:
    configs = [ChunkingConfig(120, 24), ChunkingConfig(160, 32), ChunkingConfig(120, 0)]
    configs.append(ChunkingConfig(121, 24))
    assert len({c.config_id for c in configs}) == len(configs)


@pytest.mark.parametrize(
    ("size", "overlap", "message"),
    [
        (0, 0, "chunk_size_words must be a positive integer"),
        (-5, 0, "chunk_size_words must be a positive integer"),
        (10, -1, "chunk_overlap_words must be a non-negative integer"),
        (10, 10, "must be smaller than chunk_size_words"),
        (10, 11, "must be smaller than chunk_size_words"),
        (True, 0, "chunk_size_words must be a positive integer"),
        (10.0, 2, "chunk_size_words must be a positive integer"),
        ("10", 2, "chunk_size_words must be a positive integer"),
        (10, None, "chunk_overlap_words must be a non-negative integer"),
    ],
)
def test_invalid_configuration_is_rejected(size: object, overlap: object, message: str) -> None:
    with pytest.raises(ChunkingConfigError, match=message):
        ChunkingConfig(size, overlap)  # type: ignore[arg-type]  # invalid types on purpose


# --- words and windows ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Revenue was $39,331 million.", 4),
        ("", 0),
        (" \n\t \n", 0),
        ("Net loss   (1,234)\n12.5%", 4),
    ],
)
def test_word_definition(text: str, expected: int) -> None:
    assert count_words(text) == expected


@pytest.mark.parametrize("text", ["", "   \n\n\t\n"])
def test_empty_or_whitespace_text_has_no_windows(text: str) -> None:
    assert plan_windows(text, ChunkingConfig()) == []


def test_empty_page_has_no_chunks() -> None:
    # Normalization turns whitespace-only pages into "" with status "empty".
    page = _page("")
    assert chunk_page(page, ChunkingConfig()) == ()
    assert validate_chunks([], [page], ChunkingConfig()) == []
    [chunk] = chunk_page(_page("x"), ChunkingConfig())
    stray = replace(chunk, document_id=page.document_id)
    assert validate_chunks([stray], [page], ChunkingConfig()) == [
        f"{page.page_id}: empty page must have no chunks"
    ]


@pytest.mark.parametrize("count", [3, 10])
def test_text_up_to_the_budget_is_one_chunk(count: int) -> None:
    text = f"  {_words(count)}\n"
    assert _chunk_texts(text, ChunkingConfig(10, 2)) == [_words(count)]


def test_paragraph_boundaries_are_preferred() -> None:
    paragraphs = [_words(30, p) for p in "xyz"]
    texts = _chunk_texts("\n\n".join(paragraphs) + "\n", ChunkingConfig(50, 0))
    assert texts == paragraphs


def test_line_boundaries_are_preferred_over_word_splits() -> None:
    lines = [_words(8, f"l{n}_") for n in range(6)]
    text = "\n".join(lines) + "\n"
    chunks = chunk_page(_page(text), ChunkingConfig(20, 0))
    assert [c.text for c in chunks] == [
        "\n".join(lines[0:2]),
        "\n".join(lines[2:4]),
        "\n".join(lines[4:6]),
    ]
    assert all(text[c.end_char] == "\n" for c in chunks)


def test_long_physical_line_is_split_at_word_boundaries() -> None:
    text = _words(25) + "\n"
    chunks = chunk_page(_page(text), ChunkingConfig(10, 0))
    assert [c.word_count for c in chunks] == [10, 10, 5]
    assert " ".join(c.text for c in chunks) == _words(25)


def test_overlap_starts_at_a_line_start_when_one_is_available() -> None:
    text = "\n".join(_words(5, f"r{n}_") for n in range(8)) + "\n"
    chunks = chunk_page(_page(text), ChunkingConfig(20, 7))
    for chunk in chunks[1:]:
        assert chunk.start_char == 0 or text[chunk.start_char - 1] == "\n"
    assert page_overlaps(text, chunks) == [5] * (len(chunks) - 1)  # one 5-word line fits in 7


@pytest.mark.parametrize(
    ("size", "overlap"), [(1, 0), (2, 1), (3, 2), (5, 0), (7, 3), (10, 9), (50, 0), (120, 24)]
)
@pytest.mark.parametrize("text", [MIXED_TEXT, _words(200) + "\n", "x\n\n\n\ny z\n"])
def test_invariants_hold_for_small_zero_and_high_overlap(
    size: int, overlap: int, text: str
) -> None:
    config = ChunkingConfig(size, overlap)
    page = _page(text)
    chunks = chunk_page(page, config)
    assert validate_chunks(chunks, [page], config) == []  # spans, budget, overlap, coverage
    assert uncovered_characters(text, chunks) == 0
    assert all(c.word_count <= size for c in chunks)
    assert all(a.end_char < b.end_char for a, b in pairwise(chunks))
    assert all(o <= overlap for o in page_overlaps(text, chunks))
    assert sum(c.word_count for c in chunks) == count_words(text) + sum(page_overlaps(text, chunks))


# --- financial content ------------------------------------------------------------------


def test_financial_values_and_layout_spacing_are_preserved() -> None:
    [whole] = _chunk_texts(MIXED_TEXT, ChunkingConfig())
    assert whole == MIXED_TEXT.strip()
    for snippet in (
        "Revenue             $     35,082   $     18,120",
        "Net loss                  (1,234)         (56)",
        "74.6%        75.1%",
        "October 27,   October 29,",
        "(In millions)",
    ):
        assert snippet in whole


def test_small_windows_never_split_inside_a_financial_expression() -> None:
    config = ChunkingConfig(4, 1)
    source = set(MIXED_TEXT.split())
    for chunk in chunk_page(_page(MIXED_TEXT), config):
        assert set(chunk.text.split()) <= source  # every word is a whole source word
    table_rows = [line for line in MIXED_TEXT.split("\n") if "$" in line or "%" in line]
    windows = _chunk_texts(MIXED_TEXT, ChunkingConfig(12, 0))
    assert all(any(row.strip() in w for w in windows) for row in table_rows)  # rows kept whole


# --- records and provenance ----------------------------------------------------------------


def test_record_fields_ids_and_exact_offsets() -> None:
    page = _page(MIXED_TEXT, page_number=3)
    config = ChunkingConfig(12, 3)
    chunks = chunk_page(page, config)
    assert chunk_page(page, config) == chunks  # deterministic
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"{DOC_A}:p0003:c{index:04d}:{config.config_id}"
        assert chunk.chunk_index == index
        assert (chunk.document_id, chunk.page_id, chunk.page_number) == (DOC_A, page.page_id, 3)
        assert (chunk.company, chunk.fiscal_period, chunk.source_url) == (
            page.company,
            page.fiscal_period,
            page.source_url,
        )
        assert chunk.text == MIXED_TEXT[chunk.start_char : chunk.end_char]
        assert chunk.character_count == len(chunk.text) == chunk.end_char - chunk.start_char
        assert chunk.page_table_status_summary.to_dict() == {
            "structured": 0,
            "uncertain": 0,
            "fallback": 0,
        }
        assert ChunkRecord.from_dict(json.loads(chunk.to_json_line())) == chunk
    other = chunk_page(page, ChunkingConfig(12, 2))
    assert not {c.chunk_id for c in chunks} & {c.chunk_id for c in other}


def test_offsets_are_unicode_code_points() -> None:
    text = "📈 Umsatz €1.234 café\n日本 売上 12,5 %\n"
    chunks = chunk_page(_page(text), ChunkingConfig(3, 0))
    assert [c.text for c in chunks] == ["📈 Umsatz €1.234", "café\n日本 売上", "12,5 %"]
    assert chunks[1].start_char == text.index("café") == 16
    assert all(text[c.start_char : c.end_char] == c.text for c in chunks)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"word_count": 99}, "word_count does not match"),
        ({"character_count": 1}, "character_count must equal"),
        ({"chunk_id": "x"}, "chunk_id"),
        ({"chunking_config_id": "ABC"}, "chunking_config_id"),
        ({"page_id": f"{DOC_A}:p0009"}, "page_id"),
        ({"text": "   "}, "non-whitespace"),
        ({"start_char": 5, "end_char": 5}, "offsets"),
        ({"chunk_index": -1}, "chunk_index"),
        ({"page_number": 0}, "page_number"),
    ],
)
def test_inconsistent_records_are_rejected(change: dict[str, object], message: str) -> None:
    [chunk] = chunk_page(_page("Revenue $ 1,234\n"), ChunkingConfig())
    with pytest.raises(ChunkRecordError, match=message):
        ChunkRecord.from_dict({**chunk.to_dict(), **change})


def test_unknown_or_missing_fields_are_rejected() -> None:
    [chunk] = chunk_page(_page("Revenue $ 1,234\n"), ChunkingConfig())
    data = chunk.to_dict()
    with pytest.raises(ChunkRecordError, match="unknown"):
        ChunkRecord.from_dict({**data, "path": "C:/x"})
    del data["word_count"]
    with pytest.raises(ChunkRecordError, match="missing"):
        ChunkRecord.from_dict(data)


# --- collection checks and overlap ----------------------------------------------------------


def test_overlap_counts_shared_positions_not_repeated_values() -> None:
    text = " ".join(["100", "200"] * 20) + "\n"
    chunks = chunk_page(_page(text), ChunkingConfig(10, 0))
    assert page_overlaps(text, chunks) == [0, 0, 0]
    assert chunks[0].text == chunks[1].text  # same values, different source positions
    overlapping = chunk_page(_page(text), ChunkingConfig(10, 4))
    assert set(page_overlaps(text, overlapping)) == {4}


def _relabel(chunk: ChunkRecord, **fields: object) -> ChunkRecord:
    """Rebuild a record with changed fields and a matching chunk_id."""
    data = {**chunk.to_dict(), **fields}
    data["chunk_id"] = ck.make_chunk_id(
        str(data["page_id"]), int(str(data["chunk_index"])), str(data["chunking_config_id"])
    )
    return ChunkRecord.from_dict(data)


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("drop_middle", "words between chunks are not covered"),
        ("duplicate", "duplicate chunk_id"),
        ("edited_text", "not the source span"),
        ("other_page", "not in the normalized artifact"),
    ],
)
def test_collection_validation_catches_tampering(kind: str, message: str) -> None:
    config = ChunkingConfig(10, 2)
    page = _page(_words(40) + "\n")
    chunks = list(chunk_page(page, config))
    assert validate_chunks(chunks, [page], config) == []
    if kind == "drop_middle":
        chunks = [chunks[0], *(_relabel(c, chunk_index=i) for i, c in enumerate(chunks[2:], 1))]
    elif kind == "duplicate":
        chunks.append(chunks[-1])
    elif kind == "edited_text":
        chunks[0] = replace(chunks[0], text=chunks[0].text.replace("w0", "W0"))
    else:
        chunks[0] = _relabel(chunks[0], page_number=7, page_id=make_page_id(DOC_A, 7))
    problems = validate_chunks(chunks, [page], config)
    assert any(message in p for p in problems), problems


def test_collection_validation_enforces_budget_and_overlap_limits() -> None:
    page = _page(_words(40) + "\n")
    chunks = chunk_page(page, ChunkingConfig(10, 6))
    for strict, message in (
        (ChunkingConfig(10, 2), "overlap exceeds 2"),
        (ChunkingConfig(5, 2), "exceed the word budget"),
    ):
        relabelled = [_relabel(c, chunking_config_id=strict.config_id) for c in chunks]
        problems = validate_chunks(relabelled, [page], strict)
        assert any(message in p for p in problems), problems
    assert validate_chunks([], [page], ChunkingConfig()) == [
        f"{page.page_id}: non-empty page has no chunks"
    ]


def test_chunks_never_cross_pages() -> None:
    pages = [_page(_words(15, "p"), 1), _page(_words(15, "q"), 2)]
    config = ChunkingConfig(10, 4)
    chunks = [c for p in pages for c in chunk_page(p, config)]
    assert validate_chunks(chunks, pages, config) == []
    assert all(("p" in c.text) != ("q" in c.text) for c in chunks)
    assert [c.chunk_index for c in chunks] == [0, 1, 0, 1]


# --- pipeline on synthetic PDFs --------------------------------------------------------------

NARRATIVE = (
    "Revenue was $39,331 million, up 12.5% from a year ago.\n"
    "Net loss was (1,234) in millions for the quarter.\n"
    "Operating margin(1) improved to 41.0% from 35.9%."
)


def _table_page() -> list[PlacedText]:
    items: list[PlacedText] = [
        (50.0, 60.0, "Results (in millions)"),
        place_right(420, 100, "2023"),
        place_right(500, 100, "2024"),
    ]
    for n, (label, a, b) in enumerate(
        [("Revenue", "$ 1,234", "$ 2,345"), ("Net loss", "(56)", "(78)")]
    ):
        y = 120.0 + 14 * n
        items.append((50.0, y, label))
        items.extend(place_right(x, y, v) for x, v in ((420, a), (500, b)))
    return items


def _project(make_project: Factory) -> Path:
    narrative = make_text_pdf([NARRATIVE, ""])
    tables = make_layout_pdf([_table_page(), [(50.0, 80.0, "Closing notes on the quarter")]])
    root = make_project({DOC_A: (narrative, 2), DOC_B: (tables, 2)})
    run_extraction(Settings(root), expected_document_count=2)
    run_normalization(Settings(root), expected_document_count=2)
    return root


SMALL = ChunkingConfig(8, 3)


def _chunks_path(root: Path) -> Path:
    return root / "data" / "processed" / "chunks.jsonl"


def _normalized_path(root: Path) -> Path:
    return root / "data" / "processed" / "pages.normalized.jsonl"


def _run(
    root: Path, output: str | None = None, report_output: str | None = None
) -> ck.ChunkingOutcome:
    return run_chunking(
        Settings(root),
        config=SMALL,
        output=output,
        report_output=report_output,
        expected_document_count=2,
    )


def test_pipeline_artifact_report_and_reproducibility(make_project: Factory) -> None:
    root = _project(make_project)
    normalized_before = _normalized_path(root).read_bytes()
    first = _run(root)
    first_bytes = _chunks_path(root).read_bytes()
    second = _run(root)
    assert _chunks_path(root).read_bytes() == first_bytes  # byte-identical rerun
    assert (
        second.artifact_sha256 == first.artifact_sha256 == hashlib.sha256(first_bytes).hexdigest()
    )
    assert _normalized_path(root).read_bytes() == normalized_before
    assert b"\r" not in first_bytes
    assert str(root).encode() not in first_bytes

    pages = parse_normalized_jsonl(normalized_before.decode("utf-8"))
    chunks = parse_chunks_jsonl(first_bytes.decode("utf-8"))
    assert chunks == list(first.result.chunks)
    assert validate_chunks(chunks, pages, SMALL) == []
    assert [c.document_id for c in chunks] == sorted(c.document_id for c in chunks)
    assert {c.page_id for c in chunks} == {f"{DOC_A}:p0001", f"{DOC_B}:p0001", f"{DOC_B}:p0002"}
    table_chunks = [c for c in chunks if c.page_id == f"{DOC_B}:p0001"]
    assert all(c.page_table_status_summary.structured == 1 for c in table_chunks)

    report = json.loads((root / "reports" / "chunking_report.json").read_text("utf-8"))
    assert report["chunk_artifact"] == {
        "path": "data/processed/chunks.jsonl",
        "sha256": first.artifact_sha256,
        "bytes": len(first_bytes),
        "records": len(chunks),
    }
    assert report["normalized_artifact"]["sha256"] == hashlib.sha256(normalized_before).hexdigest()
    assert report["chunking_config_id"] == SMALL.config_id
    assert report["empty_pages"] == [f"{DOC_A}:p0002"]
    assert report["chunks_per_page"][f"{DOC_A}:p0002"] == 0
    assert report["chunk_count"] == len(chunks) == sum(report["chunks_per_document"].values())
    assert report["source_words"] == sum(count_words(p.normalized_text) for p in pages)
    overlap = report["overlap"]
    assert overlap["max_words"] <= 3
    assert (
        report["emitted_chunk_words"]
        == report["source_words"] + overlap["repeated_word_occurrences"]
    )
    assert report["diagnostics"]["source_coverage"]["uncovered_non_whitespace_characters"] == 0
    assert report["chunk_word_count"]["max"] <= 8
    assert set(report["operational"]) == {"note", "chunking_duration_seconds"}


def test_forced_line_splits_are_reported(make_project: Factory) -> None:
    root = _project(make_project)
    outcome = run_chunking(Settings(root), config=ChunkingConfig(4, 1), expected_document_count=2)
    forced = outcome.report["diagnostics"]["forced_line_splits"]
    assert forced["count"] == len(forced["chunk_ids"]) > 0  # 10-word lines, 4-word budget
    assert outcome.report["diagnostics"]["physical_lines_over_word_budget"] > 0


def test_missing_normalized_artifact(make_project: Factory) -> None:
    root = make_project({DOC_A: (make_text_pdf(["x"]), 1), DOC_B: (make_text_pdf(["y"]), 1)})
    with pytest.raises(ChunkingError, match=r"normalized artifact not found.*ingestion\.normalize"):
        _run(root)
    assert not _chunks_path(root).exists()


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("not_json", "is invalid"),
        ("drop_last", "does not match the manifest"),
        ("duplicate", "does not match the manifest"),
        ("wrong_company", "company does not match the manifest"),
        ("inconsistent", "normalized_character_count mismatch"),
    ],
)
def test_invalid_normalized_artifact_keeps_previous_chunks(
    make_project: Factory, kind: str, message: str
) -> None:
    root = _project(make_project)
    _run(root)
    previous = _chunks_path(root).read_bytes()
    lines = _normalized_path(root).read_text("utf-8").splitlines(keepends=True)
    if kind == "not_json":
        lines[0] = "{not json\n"
    elif kind == "drop_last":
        lines = lines[:-1]
    elif kind == "duplicate":
        lines[1] = lines[0]
    else:
        record = json.loads(lines[0])
        if kind == "wrong_company":
            record["company"] = "Other Corp"
        else:
            record["normalized_text"] += "extra"
        lines[0] = json.dumps(record) + "\n"
    _normalized_path(root).write_text("".join(lines), encoding="utf-8", newline="\n")
    with pytest.raises(ChunkingError, match=message):
        _run(root)
    assert _chunks_path(root).read_bytes() == previous
    assert not list(_chunks_path(root).parent.glob(".chunks.jsonl.*.tmp"))


def test_failed_verification_publishes_nothing(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(make_project)
    _run(root)
    previous = _chunks_path(root).read_bytes()
    original = ck.serialize_chunks
    monkeypatch.setattr(ck, "serialize_chunks", lambda chunks: original(chunks[:-1]))
    with pytest.raises(ChunkingError, match="written artifact invalid"):
        _run(root)
    assert _chunks_path(root).read_bytes() == previous
    assert not list(_chunks_path(root).parent.glob(".chunks.jsonl.*.tmp"))


@pytest.mark.parametrize(
    ("output", "report_output", "message"),
    [
        ("data/processed/pages.normalized.jsonl", None, "would overwrite the normalized page"),
        ("data/processed/pages.raw.jsonl", None, "would overwrite the raw page artifact"),
        ("data/raw/chunks.jsonl", None, "inside the original PDF directory"),
        ("evaluation/chunks.jsonl", None, "inside the evaluation dataset directory"),
        ("../outside.jsonl", None, "must be inside the project root"),
        ("data/processed/chunks.txt", None, "must end with .jsonl"),
        (None, "data/manifest.json", "would overwrite the corpus manifest"),
        (None, "reports/normalization_report.json", "would overwrite the normalization report"),
        ("reports/same.jsonl", "reports/same.jsonl", "must end with .json"),
    ],
)
def test_unsafe_output_paths_are_rejected_before_any_work(
    make_project: Factory, output: str | None, report_output: str | None, message: str
) -> None:
    root = make_project({DOC_A: (make_text_pdf(["x"]), 1), DOC_B: (make_text_pdf(["y"]), 1)})
    manifest_before = (root / "data" / "manifest.json").read_bytes()
    with pytest.raises(ChunkingError, match=message):
        _run(root, output=output, report_output=report_output)
    assert (root / "data" / "manifest.json").read_bytes() == manifest_before
    assert not _chunks_path(root).exists()


def test_same_output_and_report_path_rejected(make_project: Factory) -> None:
    settings = Settings(make_project())
    path = settings.reports_dir / "x.json"
    assert "--output and --report-output must be different files" in ck.output_path_problems(
        settings, path, path
    )


def test_custom_output_paths_leave_the_default_baseline_alone(make_project: Factory) -> None:
    root = _project(make_project)
    outcome = _run(
        root, output="data/processed/chunks.8w3o.jsonl", report_output="reports/chunking.8w3o.json"
    )
    assert outcome.artifact_path == (root / "data/processed/chunks.8w3o.jsonl").resolve()
    assert (root / "reports" / "chunking.8w3o.json").is_file()
    assert not _chunks_path(root).exists()
    assert outcome.report["chunk_artifact"]["path"] == "data/processed/chunks.8w3o.jsonl"


# --- CLI ----------------------------------------------------------------------------------


def _four_documents_project(make_project: Factory) -> Path:
    ids = ["aaa-2024-q1", "bbb-2024-q1", "ccc-2024-q1", "ddd-2024-q1"]
    root = make_project(
        {i: (make_text_pdf([f"SECRET-BODY {i} revenue $1,234 million"]), 1) for i in ids}
    )
    run_extraction(Settings(root))
    run_normalization(Settings(root))
    return root


def test_cli_success_and_custom_outputs(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _four_documents_project(make_project)
    assert ck.main(["--project-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert f"config {ChunkingConfig().config_id}" in out
    assert "chunks: 4" in out
    assert "SECRET-BODY" not in out  # no document text is printed
    args = ["--chunk-size-words", "3", "--chunk-overlap-words", "1"]
    args += ["--output", "data/processed/chunks.3w1o.jsonl", "--report-output", "reports/c.json"]
    assert ck.main(["--project-root", str(root), *args]) == 0
    assert "3 words, overlap 1" in capsys.readouterr().out
    assert (root / "data/processed/chunks.3w1o.jsonl").is_file()
    assert (root / "reports/c.json").is_file()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--chunk-size-words", "10", "--chunk-overlap-words", "10"], "must be smaller"),
        (["--chunk-size-words", "0"], "positive integer"),
        (["--chunk-overlap-words", "-2"], "non-negative integer"),
        (["--chunk-size-words", "ten"], "invalid int value"),
    ],
)
def test_cli_invalid_configuration_exits_2(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        ck.main(args)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_missing_input_and_unsafe_output_exit_1(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = ["aaa-2024-q1", "bbb-2024-q1", "ccc-2024-q1", "ddd-2024-q1"]
    root = make_project({i: (make_text_pdf(["x"]), 1) for i in ids})
    assert ck.main(["--project-root", str(root)]) == 1
    err = capsys.readouterr().err
    assert "normalized artifact not found" in err
    assert "python -m founderlens.ingestion.normalize" in err
    assert "No chunk artifact was published" in err
    assert ck.main(["--project-root", str(root), "--report-output", "data/manifest.json"]) == 1
    assert "would overwrite the corpus manifest" in capsys.readouterr().err
