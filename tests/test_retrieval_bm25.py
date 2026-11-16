"""BM25 retriever on controlled synthetic chunks, collection loading and the CLI.

Scores come from the real rank-bm25 BM25Okapi implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]  # rank-bm25 0.2.2 has no type information

from founderlens.config import Settings
from founderlens.ingestion.chunk import ChunkingConfig, ChunkRecord, chunk_page, run_chunking
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import NormalizedPageRecord, page_warnings, run_normalization
from founderlens.ingestion.schema import make_page_id, status_for_text
from founderlens.retrieval import bm25
from founderlens.retrieval.bm25 import BM25Config, BM25Retriever
from founderlens.retrieval.schema import (
    ChunkCollection,
    RetrievalError,
    RetrievalResult,
    load_chunk_collection,
)
from founderlens.retrieval.tokenize import tokenize
from tests.conftest import DOC_A, DOC_B, make_text_pdf

Factory = Callable[..., Path]


def _chunks(*texts: str) -> list[ChunkRecord]:
    """One chunk per text, each on its own page of DOC_A, in the given order."""
    chunks: list[ChunkRecord] = []
    for number, text in enumerate(texts, start=1):
        page = NormalizedPageRecord(
            document_id=DOC_A,
            document_sha256="a" * 64,
            company="Example Corp",
            fiscal_period="Q1 2024",
            source_url="https://example.com/report.pdf",
            page_number=number,
            page_id=make_page_id(DOC_A, number),
            extraction_status=status_for_text(text),
            raw_character_count=len(text),
            normalized_text=text,
            normalized_character_count=len(text),
            tables=(),
            table_text_overlap=False,
            normalization_warnings=page_warnings(status_for_text(text), ()),
        )
        chunks.extend(chunk_page(page, ChunkingConfig(500, 0)))
    return chunks


def _ids(results: tuple[RetrievalResult, ...]) -> list[str]:
    return [r.chunk.page_id for r in results]


def _page(n: int) -> str:
    return make_page_id(DOC_A, n)


CORPUS = _chunks(
    "NVIDIA Blackwell demand was strong in the fourth quarter.",
    "Google Cloud revenue was $11,353 million, up 35%.",
    "Data center revenue was $35,580 million.",
    "Operating income and free cash flow for the quarter.",
)

# --- configuration -----------------------------------------------------------------


def test_default_configuration_and_identifier() -> None:
    config = BM25Config()
    assert (config.k1, config.b, config.epsilon, config.top_k) == (1.5, 0.75, 0.25, 5)
    scoring = config.scoring_dict()
    assert scoring["algorithm"] == "BM25Okapi"
    assert scoring["library_version"] == "0.2.2"
    assert scoring["tokenizer_version"] == 1
    assert len(config.config_id) == 12
    assert BM25Config(top_k=9).config_id == config.config_id  # top_k does not affect scores
    assert BM25Config(k1=1.2).config_id != config.config_id


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k1": 0}, "k1 must be"),
        ({"k1": math.inf}, "k1 must be"),
        ({"k1": True}, "k1 must be"),
        ({"b": -0.1}, "b must be"),
        ({"b": 1.5}, "b must be"),
        ({"b": math.nan}, "b must be"),
        ({"epsilon": -1.0}, "epsilon must be"),
        ({"top_k": 0}, "top_k must be"),
        ({"top_k": 2.0}, "top_k must be"),
        ({"top_k": True}, "top_k must be"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BM25Config(**kwargs)  # type: ignore[arg-type]  # invalid values on purpose


# --- scoring -------------------------------------------------------------------------


def test_scores_match_the_library_and_a_hand_calculation() -> None:
    chunks = _chunks("apple banana", "apple cherry cherry", "date")
    retriever = BM25Retriever(chunks)
    [hit] = retriever.search("cherry")
    # N = 3, n(cherry) = 1, avg length 2, tf = 2, length 3.
    idf = math.log((3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2 * (1.5 + 1) / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2))
    assert hit.chunk.page_id == _page(2)
    assert hit.score == pytest.approx(expected, rel=1e-12)
    library = BM25Okapi([list(tokenize(c.text)) for c in chunks])
    assert hit.score == pytest.approx(float(library.get_scores(["cherry"])[1]), rel=1e-12)
    # "apple" is in 2 of 3 chunks: negative idf is floored at epsilon * mean(idf).
    mean_idf = (math.log(1.5 / 2.5) + 3 * math.log(2.5 / 1.5)) / 4
    apple = retriever.search("apple")
    assert _ids(apple) == [_page(1), _page(2)]  # shorter chunk scores higher
    assert apple[0].score == pytest.approx(
        0.25 * mean_idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / 2)), rel=1e-12
    )


def test_known_lexical_matches_rank_first() -> None:
    retriever = BM25Retriever(CORPUS)
    assert _ids(retriever.search("NVIDIA Blackwell"))[0] == _page(1)
    assert _ids(retriever.search("google cloud"))[0] == _page(2)
    assert _ids(retriever.search("$35,580"))[0] == _page(3)
    assert _ids(retriever.search("free cash flow"))[0] == _page(4)
    assert all(math.isfinite(r.score) for r in retriever.search("revenue quarter"))


def test_non_matching_chunks_are_never_returned() -> None:
    results = BM25Retriever(CORPUS).search("Blackwell", top_k=4)
    assert _ids(results) == [_page(1)]  # one match, not padded to four


@pytest.mark.parametrize("query", ["", "   ", "$ % ( ) -", "zyzzyva quokka"])
def test_empty_punctuation_and_out_of_vocabulary_queries_return_nothing(query: str) -> None:
    retriever = BM25Retriever(CORPUS)
    assert retriever.search(query) == ()
    tokens, unknown = retriever.query_terms(query)
    assert unknown == tokens  # every token (if any) is out of vocabulary


def test_zero_and_negative_score_matches_are_returned_after_positive_ones() -> None:
    # "alpha" is in exactly half of four chunks, so idf = ln(2.5 / 2.5) = 0.
    zero = BM25Retriever(_chunks("alpha x", "alpha y", "z", "w")).search("alpha")
    assert _ids(zero) == [_page(1), _page(2)]
    assert [r.score for r in zero] == [0.0, 0.0]
    # Mean idf is negative here, so the epsilon floor is negative too.
    negative = BM25Retriever(_chunks("a b", "a b", "a c")).search("a c")
    assert _ids(negative)[0] == _page(3)  # the only chunk with "c"
    assert len(negative) == 3  # matching chunks kept
    assert negative[-1].score < 0


def test_ties_keep_collection_order_and_top_k_is_respected() -> None:
    chunks = _chunks(*["same revenue words"] * 4, "other text")
    retriever = BM25Retriever(chunks)
    results = retriever.search("revenue", top_k=3)
    assert _ids(results) == [_page(1), _page(2), _page(3)]
    assert len({r.score for r in results}) == 1
    assert [r.rank for r in results] == [1, 2, 3]
    assert len(retriever.search("revenue", top_k=10)) == 4  # fewer matches than k
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("revenue", top_k=0)


def test_results_keep_original_chunks_and_are_repeatable() -> None:
    retriever = BM25Retriever(CORPUS)
    first = retriever.search("revenue million")
    assert retriever.search("revenue million") == first
    assert BM25Retriever(CORPUS).search("revenue million") == first  # rebuilt index
    for result in first:
        assert result.method == "bm25"
        assert result.chunk in CORPUS  # the original record, unmodified
        data = result.to_dict()
        assert data["text"] == result.chunk.text
        assert (data["chunk_id"], data["page_number"], data["start_char"], data["end_char"]) == (
            result.chunk.chunk_id,
            result.chunk.page_number,
            result.chunk.start_char,
            result.chunk.end_char,
        )


def test_index_is_built_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []

    class CountingBM25(BM25Okapi):  # type: ignore[misc]  # untyped base class
        def __init__(self, *args: object, **kwargs: object) -> None:
            built.append(self)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(bm25, "BM25Okapi", CountingBM25)
    retriever = BM25Retriever(CORPUS)
    for query in ("revenue", "cloud", "nvidia"):
        retriever.search(query)
    assert len(built) == 1


def test_chunks_without_lexical_tokens_are_left_out_of_the_index() -> None:
    chunks = _chunks("$ % --", "revenue grew")
    retriever = BM25Retriever(chunks)
    assert retriever.unindexed_chunk_ids == (chunks[0].chunk_id,)
    assert (retriever.collection_size, retriever.indexed_chunk_count) == (2, 1)
    assert retriever.vocabulary_size == 2


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([], "empty chunk collection"),
        (_chunks("$ %", "( )"), "vocabulary is empty"),
        ([*CORPUS[:2], CORPUS[0]], "duplicate chunk_id"),
    ],
)
def test_invalid_corpus_is_rejected(chunks: list[ChunkRecord], message: str) -> None:
    with pytest.raises(RetrievalError, match=message):
        BM25Retriever(chunks)


def test_result_rejects_invalid_rank_and_score() -> None:
    with pytest.raises(ValueError, match="rank"):
        RetrievalResult("bm25", 0, 1.0, CORPUS[0])
    with pytest.raises(ValueError, match="finite"):
        RetrievalResult("bm25", 1, math.nan, CORPUS[0])


# --- collection loading on a synthetic project ------------------------------------------


def _project(make_project: Factory) -> Path:
    a = make_text_pdf(["NVIDIA Blackwell revenue was $35,580 million.", "Closing remarks"])
    b = make_text_pdf(["Google Cloud revenue grew 35% in Q4 FY2025."])
    root = make_project({DOC_A: (a, 2), DOC_B: (b, 1)})
    settings = Settings(root)
    run_extraction(settings, expected_document_count=2)
    run_normalization(settings, expected_document_count=2)
    run_chunking(settings, expected_document_count=2)
    return root


def _load(root: Path) -> ChunkCollection:
    s = Settings(root)
    return load_chunk_collection(
        s.processed_data_dir / "chunks.jsonl",
        s.corpus_manifest_path,
        s.processed_data_dir / "pages.normalized.jsonl",
    )


def test_valid_collection_loads_in_order_with_its_hash(make_project: Factory) -> None:
    root = _project(make_project)
    collection = _load(root)
    assert [c.page_id for c in collection.chunks] == [
        f"{DOC_A}:p0001",
        f"{DOC_A}:p0002",
        f"{DOC_B}:p0001",
    ]
    assert collection.chunking_config_id == ChunkingConfig().config_id
    data = (root / "data" / "processed" / "chunks.jsonl").read_bytes()
    assert collection.artifact_sha256 == hashlib.sha256(data).hexdigest()
    [hit] = BM25Retriever(collection.chunks).search("Google Cloud")
    assert hit.chunk.document_id == DOC_B


def _edit_chunks(root: Path, edit: Callable[[list[str]], list[str]]) -> None:
    path = root / "data" / "processed" / "chunks.jsonl"
    lines = path.read_text("utf-8").splitlines(keepends=True)
    path.write_text("".join(edit(lines)), encoding="utf-8", newline="\n")


def _set_field(line: str, **fields: object) -> str:
    return json.dumps({**json.loads(line), **fields}, ensure_ascii=False) + "\n"


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda lines: [*lines, lines[0]], "duplicate chunk_id"),
        (lambda lines: ["{oops\n", *lines[1:]], "is invalid"),
        (lambda lines: [lines[1], lines[0], *lines[2:]], "not in manifest document"),
        (lambda lines: [_set_field(lines[0], company="Other Corp"), *lines[1:]], "company"),
        (lambda lines: lines[1:], "do not cover the page text"),
        (
            lambda lines: [
                _set_field(
                    lines[0],
                    text=json.loads(lines[0])["text"].replace("NVIDIA", "NVIDIX"),
                ),
                *lines[1:],
            ],
            "not the recorded source span",
        ),
    ],
)
def test_invalid_collection_is_rejected(
    make_project: Factory, edit: Callable[[list[str]], list[str]], message: str
) -> None:
    root = _project(make_project)
    _edit_chunks(root, edit)
    with pytest.raises(RetrievalError, match=message):
        _load(root)


def test_missing_chunk_artifact(make_project: Factory) -> None:
    root = _project(make_project)
    (root / "data" / "processed" / "chunks.jsonl").unlink()
    with pytest.raises(RetrievalError, match=r"chunk artifact not found.*ingestion\.chunk"):
        _load(root)


# --- CLI ---------------------------------------------------------------------------------


def test_cli_query_text_and_json(make_project: Factory, capsys: pytest.CaptureFixture[str]) -> None:
    root = _project(make_project)
    assert bm25.main(["--project-root", str(root), "--query", "NVIDIA Blackwell"]) == 0
    out = capsys.readouterr().out
    assert "1. score" in out
    assert f"{DOC_A} p1" in out
    assert f"{DOC_A}:p0001:c0000" in out
    assert bm25.main(["--project-root", str(root), "--query", "revenue", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [r["rank"] for r in payload["results"]] == [1, 2]
    assert payload["results"][0]["text"]
    assert payload["bm25_config_id"] == BM25Config().config_id


def test_cli_zero_results_and_smoke_report(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(make_project)
    assert bm25.main(["--project-root", str(root), "--query", "zyzzyva"]) == 0
    assert "No results" in capsys.readouterr().out
    assert bm25.main(["--project-root", str(root), "--smoke"]) == 0
    report = json.loads((root / "reports" / "bm25_smoke_report.json").read_text("utf-8"))
    assert report["indexed_chunks"] == report["chunk_collection_size"] == 3
    assert "zyzzyva quokka" in report["zero_result_queries"]
    assert len(report["queries"]) == len(bm25.SMOKE_QUERIES)
    assert set(report["operational"]) == {"note", "index_build_seconds", "query_latency"}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--query", "x", "--top-k", "0"], "top_k must be"),
        (["--query", "x", "--b", "2"], "b must be"),
        (["--query", "x", "--smoke"], "not allowed with argument"),
        ([], "one of the arguments --query --smoke is required"),
    ],
)
def test_cli_invalid_arguments_exit_2(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        bm25.main(args)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_invalid_artifact_exits_1(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(make_project)
    args = ["--project-root", str(root), "--query", "x", "--chunks", "data/processed/none.jsonl"]
    assert bm25.main(args) == 1
    assert "chunk artifact not found" in capsys.readouterr().err
    _edit_chunks(root, lambda lines: [*lines, lines[0]])
    assert bm25.main(["--project-root", str(root), "--query", "x"]) == 1
    assert "duplicate chunk_id" in capsys.readouterr().err
