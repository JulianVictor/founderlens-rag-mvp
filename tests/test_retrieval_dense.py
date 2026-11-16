"""Exact dense retrieval: cosine math, ranking, query handling, validation, lifecycle, CLI.

Scoring uses real NumPy; only the query encoder is a deterministic fake. No model
is loaded and nothing touches the network.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from founderlens.config import Settings
from founderlens.ingestion.chunk import ChunkingConfig, ChunkRecord, chunk_page, run_chunking
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import NormalizedPageRecord, page_warnings, run_normalization
from founderlens.ingestion.schema import make_page_id, status_for_text
from founderlens.retrieval import dense as dn
from founderlens.retrieval import embeddings as em
from founderlens.retrieval.dense import (
    SCORE_BOUND_TOLERANCE,
    DenseConfig,
    DenseRetriever,
    QueryTooLongError,
    cosine_scores,
    load_dense_retriever,
    rank_indices,
)
from founderlens.retrieval.embeddings import EmbeddingArtifact, EmbeddingConfig
from founderlens.retrieval.model import MODEL_ID, MODEL_REVISION
from founderlens.retrieval.schema import ChunkCollection, RetrievalError, load_chunk_collection
from tests.conftest import DOC_A, DOC_B, make_text_pdf

Factory = Callable[..., Path]
DIM = 4
CONFIG = EmbeddingConfig(dimension=DIM, max_seq_length=16)


class FakeQueryEncoder:
    """Maps known queries to fixed vectors; others get a hash-derived vector.

    Tokens are whitespace words plus two special tokens. Counts every call.
    """

    def __init__(
        self,
        vectors: dict[str, Sequence[float]] | None = None,
        *,
        dimension: int = DIM,
        limit: int = 16,
        revision: str = MODEL_REVISION,
        output: Callable[[NDArray[np.float64]], NDArray[Any]] | None = None,
    ) -> None:
        self.vectors = vectors or {}
        self._dimension = dimension
        self._limit = limit
        self.revision = revision
        self.output = output
        self.calls: dict[str, int] = {"describe": 0, "token_lengths": 0, "encode": 0}
        self.encoded: list[list[str]] = []

    @property
    def max_seq_length(self) -> int:
        return self._limit

    @property
    def dimension(self) -> int:
        return self._dimension

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        self.calls["token_lengths"] += 1
        return [len(t.split()) + 2 for t in texts]

    def encode(self, texts: Sequence[str]) -> NDArray[Any]:
        self.calls["encode"] += 1
        self.encoded.append(list(texts))
        rows = []
        for text in texts:
            if text in self.vectors:
                rows.append(list(self.vectors[text]))
            else:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                rows.append(
                    [(digest[i % 32] - 127.5) * (1 + i // 32) for i in range(self._dimension)]
                )
        array = np.asarray(rows, dtype=np.float64)
        return self.output(array) if self.output else array

    def describe(self) -> dict[str, Any]:
        self.calls["describe"] += 1
        return {"model_id": MODEL_ID, "model_revision": self.revision}


def _unit(*values: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def _chunks(texts: Sequence[str]) -> tuple[ChunkRecord, ...]:
    chunks: list[ChunkRecord] = []
    for number, text in enumerate(texts, start=1):
        status = status_for_text(text)
        page = NormalizedPageRecord(
            document_id=DOC_A,
            document_sha256="a" * 64,
            company="Example Corp",
            fiscal_period="Q1 2024",
            source_url="https://example.com/report.pdf",
            page_number=number,
            page_id=make_page_id(DOC_A, number),
            extraction_status=status,
            raw_character_count=len(text),
            normalized_text=text,
            normalized_character_count=len(text),
            tables=(),
            table_text_overlap=False,
            normalization_warnings=page_warnings(status, ()),
        )
        chunks.extend(chunk_page(page, ChunkingConfig(500, 0)))
    return tuple(chunks)


TEXTS = ["alpha revenue", "beta margin", "gamma cash", "delta growth"]
ROWS = [_unit(1, 0, 0, 0), _unit(0, 1, 0, 0), _unit(1, 1, 0, 0), _unit(-1, 0, 0, 0)]


def _pair(
    rows: Sequence[Sequence[float]] = ROWS,
    texts: Sequence[str] = TEXTS,
    *,
    config: EmbeddingConfig = CONFIG,
    **manifest_changes: Any,
) -> tuple[ChunkCollection, EmbeddingArtifact]:
    chunks = _chunks(texts)
    collection = ChunkCollection(
        chunks=chunks,
        artifact_sha256="c" * 64,
        chunking_config_id=chunks[0].chunking_config_id,
        normalized_sha256="d" * 64,
    )
    matrix = np.asarray(rows, dtype=np.float32)
    manifest: dict[str, Any] = {
        "embedding_config": config.semantic_dict(),
        "embedding_config_id": config.config_id,
        "source_chunks": {"sha256": collection.artifact_sha256},
        "matrix": {"sha256": hashlib.sha256(em.matrix_bytes(matrix)).hexdigest()},
    }
    manifest.update(manifest_changes)
    artifact = EmbeddingArtifact(matrix, tuple(c.chunk_id for c in chunks), manifest)
    return collection, artifact


def _retriever(
    encoder: FakeQueryEncoder | None = None, **kwargs: Any
) -> tuple[DenseRetriever, FakeQueryEncoder]:
    collection, artifact = _pair(**kwargs)
    encoder = encoder or FakeQueryEncoder()
    return DenseRetriever(collection, artifact, encoder, embedding_config=CONFIG), encoder


# --- cosine mathematics ------------------------------------------------------------


def test_identical_orthogonal_and_opposite_vectors() -> None:
    matrix = np.asarray([_unit(1, 0, 0, 0), _unit(0, 1, 0, 0), _unit(-1, 0, 0, 0)])
    scores = cosine_scores(matrix, np.asarray(_unit(1, 0, 0, 0)))
    assert scores.tolist() == [1.0, 0.0, -1.0]


def test_scores_match_a_manual_cosine_calculation() -> None:
    a, b = [3.0, 4.0, 0.0, 0.0], [1.0, 2.0, 2.0, 0.0]
    manual = sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )
    scores = cosine_scores(np.asarray([_unit(*a), _unit(0, 0, 0, 1)]), np.asarray(_unit(*b)))
    assert scores[0] == pytest.approx(manual, abs=1e-12)  # 11/15
    assert scores[1] == 0.0


def test_scores_near_the_bounds_are_kept_raw_and_beyond_them_rejected() -> None:
    inside = 1.0 + SCORE_BOUND_TOLERANCE / 2
    scores = cosine_scores(np.asarray([[inside, 0.0, 0.0, 0.0]]), np.asarray([1.0, 0, 0, 0]))
    assert scores[0] == inside  # not clipped
    with pytest.raises(RetrievalError, match="outside"):
        cosine_scores(np.asarray([[1.001, 0.0, 0, 0]]), np.asarray([1.0, 0, 0, 0]))
    with pytest.raises(RetrievalError, match="NaN or infinite"):
        cosine_scores(np.asarray([[np.nan, 0.0, 0, 0]]), np.asarray([1.0, 0, 0, 0]))
    with pytest.raises(RetrievalError, match="does not match"):
        cosine_scores(np.asarray([[1.0, 0, 0, 0]]), np.asarray([1.0, 0, 0]))


# --- ranking -------------------------------------------------------------------------


def test_descending_order_with_stable_ties() -> None:
    scores = np.asarray([0.2, 0.9, 0.5, 0.9, -0.3, 0.0])
    assert rank_indices(scores, 6) == [1, 3, 2, 0, 5, 4]  # tie 1/3 kept in collection order
    assert rank_indices(scores, 1) == [1]
    assert rank_indices(scores, 100) == [1, 3, 2, 0, 5, 4]


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "3"])
def test_invalid_top_k_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        rank_indices(np.asarray([0.1]), bad)  # type: ignore[arg-type]  # invalid on purpose
    with pytest.raises(ValueError, match="top_k"):
        DenseConfig(top_k=bad)  # type: ignore[arg-type]  # invalid on purpose


def test_results_ranks_scores_and_provenance() -> None:
    query = "which chunk is about alpha"
    retriever, _ = _retriever(FakeQueryEncoder({query: _unit(1, 0.2, 0, 0)}))
    results = retriever.search(query, top_k=4)
    expected = sorted(range(4), key=lambda i: -float(np.dot(ROWS[i], _unit(1, 0.2, 0, 0))))
    assert [r.chunk.text for r in results] == [TEXTS[i] for i in expected]  # row i -> chunk i
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert [r.method for r in results] == ["dense"] * 4
    assert results[-1].score < 0  # negative similarity retained
    assert len({r.chunk.chunk_id for r in results}) == 4
    for r in results:
        assert r.chunk.page_id == make_page_id(DOC_A, r.chunk.page_number)
        assert r.score == pytest.approx(
            float(np.dot(ROWS[TEXTS.index(r.chunk.text)], _unit(1, 0.2, 0, 0))), abs=1e-6
        )
    assert retriever.search(query, top_k=4) == results  # stable across calls
    assert len(retriever.search(query, top_k=10)) == 4  # k > collection size
    assert len(retriever.search(query)) == 4  # default k = 5 > 4 chunks


def test_exact_ties_keep_collection_order() -> None:
    rows = [_unit(1, 1, 0, 0)] * 3 + [_unit(0, 0, 1, 0)]
    retriever, _ = _retriever(FakeQueryEncoder({"tie": _unit(1, 1, 0, 0)}), rows=rows)
    results = retriever.search("tie", top_k=3)
    assert [r.chunk.page_number for r in results] == [1, 2, 3]
    assert len({r.score for r in results}) == 1


# --- query handling --------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   \n\t", "?!... $ % ( ) -", "— …"])
def test_queries_without_letters_or_digits_return_nothing_and_skip_the_model(query: str) -> None:
    retriever, encoder = _retriever()
    assert retriever.search(query) == ()
    assert encoder.calls == {"describe": 0, "token_lengths": 0, "encode": 0}


@pytest.mark.parametrize("query", ["39,331 37.5%", "Umsatz €1.234 日本", "zyzzyva quokka", "(146)"])
def test_numeric_unicode_and_unknown_words_are_encoded_as_given(query: str) -> None:
    retriever, encoder = _retriever()
    results = retriever.search(query, top_k=2)
    assert len(results) == 2  # no vocabulary: unknown words still get neighbours
    assert encoder.encoded[-1] == [query]  # exact text, no prefix or normalization


def test_query_at_the_limit_fits_and_over_the_limit_is_rejected_before_encoding() -> None:
    retriever, encoder = _retriever()
    at_limit = " ".join(["word"] * 14)  # 14 + 2 special tokens = 16
    assert len(retriever.search(at_limit, top_k=1)) == 1
    before = encoder.calls["encode"]
    with pytest.raises(QueryTooLongError, match=r"query has 17 tokens .* at most 16"):
        retriever.search(at_limit + " more")
    assert encoder.calls["encode"] == before  # never encoded, never truncated


def test_non_string_query_is_rejected() -> None:
    retriever, _ = _retriever()
    with pytest.raises(TypeError, match="must be a string"):
        retriever.search(None)  # type: ignore[arg-type]  # invalid on purpose


# --- query embedding validation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (lambda a: a[:, :3], "returned shape \\(1, 3\\)"),
        (lambda a: a.astype(np.int64), "dtype int64"),
        (lambda a: a * 0, "zero vectors"),
        (lambda a: a * np.nan, "NaN or infinite"),
        (lambda a: a * np.inf, "NaN or infinite"),
    ],
)
def test_invalid_query_embeddings_are_rejected(
    output: Callable[[NDArray[np.float64]], NDArray[Any]], message: str
) -> None:
    retriever, _ = _retriever(FakeQueryEncoder(output=output))
    with pytest.raises(RetrievalError, match=message):
        retriever.search("valid query")


def test_query_vectors_are_normalized_and_unnormalized_vectors_rejected() -> None:
    retriever, _ = _retriever(FakeQueryEncoder({"q": [3.0, 4.0, 0.0, 0.0]}))
    vector = retriever.encode_query("q")
    assert vector is not None
    assert vector.dtype == np.float32
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(RetrievalError, match="not unit L2 norm"):
        retriever.rank(np.asarray([3.0, 4.0, 0.0, 0.0], dtype=np.float32))
    with pytest.raises(RetrievalError, match="1-D"):
        retriever.rank(np.zeros((2, DIM), dtype=np.float32))


def test_query_encoder_with_another_model_revision_is_rejected() -> None:
    retriever, _ = _retriever(FakeQueryEncoder(revision="f" * 40))
    with pytest.raises(RetrievalError, match="documents were embedded with"):
        retriever.search("valid query")


# --- artifact validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rows": [ROWS[0], ROWS[1], ROWS[2], [2.0, 0, 0, 0]]}, "not unit L2 norm"),
        ({"rows": ROWS[:3]}, "do not match the collection's chunk order"),
        ({"source_chunks": {"sha256": "e" * 64}}, "different chunk collection"),
        ({"embedding_config_id": "000000000000"}, "embedding config 000000000000"),
    ],
)
def test_invalid_document_artifacts_are_rejected(change: dict[str, Any], message: str) -> None:
    rows = change.pop("rows", ROWS)
    collection, artifact = _pair(rows, **change)
    if len(rows) != len(collection.chunks):
        artifact = EmbeddingArtifact(
            artifact.matrix, artifact.chunk_ids[: len(rows)], artifact.manifest
        )
    with pytest.raises(RetrievalError, match=message):
        DenseRetriever(collection, artifact, FakeQueryEncoder(), embedding_config=CONFIG)


def test_reordered_chunks_model_revision_and_config_mismatches_are_rejected() -> None:
    collection, artifact = _pair()
    reordered = ChunkCollection(
        tuple(reversed(collection.chunks)),
        collection.artifact_sha256,
        collection.chunking_config_id,
        collection.normalized_sha256,
    )
    with pytest.raises(RetrievalError, match="chunk order"):
        DenseRetriever(reordered, artifact, FakeQueryEncoder(), embedding_config=CONFIG)
    other_revision = {**CONFIG.semantic_dict(), "model_revision": "f" * 40}
    _, wrong_model = _pair(embedding_config=other_revision)
    with pytest.raises(RetrievalError, match="different model or revision"):
        DenseRetriever(collection, wrong_model, FakeQueryEncoder(), embedding_config=CONFIG)
    other = EmbeddingConfig(dimension=DIM, max_seq_length=32)
    with pytest.raises(RetrievalError, match="embedding config"):
        DenseRetriever(collection, artifact, FakeQueryEncoder(), embedding_config=other)


# --- lifecycle ------------------------------------------------------------------------------


def test_lazy_model_use_and_reuse_across_queries() -> None:
    retriever, encoder = _retriever()
    assert encoder.calls == {"describe": 0, "token_lengths": 0, "encode": 0}  # nothing at init
    for query in ("first query", "second query", "third query"):
        retriever.search(query)
    assert encoder.calls["describe"] == 1  # identity checked once
    assert encoder.calls["encode"] == 3  # one query vector per search, no document re-embedding
    assert all(len(batch) == 1 for batch in encoder.encoded)


def test_identity_keeps_every_identifier_separate() -> None:
    retriever, _ = _retriever()
    ident = retriever.identity()
    assert ident["method"] == "dense"
    assert ident["dense_config_id"] == DenseConfig().config_id
    assert ident["dense_config_id"] == DenseConfig(top_k=9).config_id  # top_k is not identity
    assert ident["embedding_config_id"] == CONFIG.config_id
    assert ident["collection_sha256"] == "c" * 64
    assert ident["model_revision"] == MODEL_REVISION
    assert len(ident["matrix_sha256"]) == 64


# --- loading from a project and the CLI -----------------------------------------------------


class ProjectEncoder(FakeQueryEncoder):
    """Full-size fake (384 dimensions, 256 tokens) with the pinned model identity."""

    def __init__(self, config: EmbeddingConfig | None = None, **_: object) -> None:
        super().__init__(dimension=384, limit=256)


def _project(make_project: Factory, *, embed: bool = True) -> Path:
    a = make_text_pdf(["NVIDIA Blackwell revenue was $35,580 million.", "Closing remarks"])
    b = make_text_pdf(["Google Cloud revenue grew 35% in Q4 FY2025."])
    root = make_project({DOC_A: (a, 2), DOC_B: (b, 1)})
    settings = Settings(root)
    run_extraction(settings, expected_document_count=2)
    run_normalization(settings, expected_document_count=2)
    run_chunking(settings, expected_document_count=2)
    if embed:
        collection = load_chunk_collection(
            settings.processed_data_dir / "chunks.jsonl",
            settings.corpus_manifest_path,
            settings.processed_data_dir / "pages.normalized.jsonl",
        )
        em.run_build(
            settings,
            ProjectEncoder(),
            EmbeddingConfig(),
            collection,
            settings.processed_data_dir / "chunks.jsonl",
        )
    return root


def test_artifacts_are_loaded_once_and_never_rebuilt(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(make_project)
    encoder = ProjectEncoder()
    retriever = load_dense_retriever(Settings(root), "v1", encoder=encoder)

    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("artifact reloaded or documents re-embedded during search")

    monkeypatch.setattr(dn, "load_embedding_artifact", forbidden)
    monkeypatch.setattr(em, "write_embedding_artifact", forbidden)
    for _ in range(3):
        assert len(retriever.search("cloud revenue", top_k=2)) == 2
    assert all(len(batch) == 1 for batch in encoder.encoded)


def test_missing_or_corrupted_artifact_fails_with_a_remedy(make_project: Factory) -> None:
    root = _project(make_project, embed=False)
    with pytest.raises(RetrievalError, match="embedding artifact not found"):
        load_dense_retriever(Settings(root), "v1", encoder=ProjectEncoder())
    root = _project(make_project)
    matrix = next((root / "artifacts" / "embeddings").glob("*/embeddings.npy"))
    data = bytearray(matrix.read_bytes())
    data[-1] ^= 0xFF
    matrix.write_bytes(bytes(data))
    with pytest.raises(RetrievalError, match="SHA-256 does not match") as info:
        load_dense_retriever(Settings(root), "v1", encoder=ProjectEncoder())
    assert any("cannot be embedded" in p for p in info.value.problems)  # v1 remedy


def test_cli_query_json_empty_oversized_and_errors(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(make_project)
    monkeypatch.setattr(dn, "SentenceTransformerEncoder", ProjectEncoder)
    base = ["--project-root", str(root), "--collection", "v1"]
    assert dn.main([*base, "--query", "cloud revenue", "--top-k", "2"]) == 0
    out = capsys.readouterr().out
    assert "Dense (exact cosine" in out
    assert "1. cosine" in out
    assert "not relevance probabilities" in out
    assert dn.main([*base, "--query", "cloud revenue", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "dense"
    assert payload["collection_id"] == ChunkingConfig().config_id
    assert [r["rank"] for r in payload["results"]] == [1, 2, 3]
    assert payload["results"][0]["text"]
    assert dn.main([*base, "--query", "?!"]) == 0
    assert "no letters or digits" in capsys.readouterr().out
    assert dn.main([*base, "--query", "word " * 300]) == 1
    assert "never truncated" in capsys.readouterr().err
    assert dn.main(["--project-root", str(root), "--query", "x"]) == 1  # default v2: not built
    err = capsys.readouterr().err
    assert "python -m founderlens.retrieval.embeddings build-collection" in err
    assert dn.main([*base, "--smoke"]) == 0
    report = json.loads(
        next((root / "reports").glob("dense_smoke_report.*.json")).read_text("utf-8")
    )
    assert report["dense"]["collection_id"] == ChunkingConfig().config_id
    assert {"dense", "bm25"} <= set(report["queries"][0])
    assert set(report["operational"]) >= {"cold_start_model_load_seconds", "per_query"}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--query", "x", "--top-k", "0"], "top_k must be"),
        (["--query", "x", "--collection", "v3"], "invalid choice"),
        ([], "one of the arguments --query --smoke is required"),
    ],
)
def test_cli_invalid_arguments_exit_2(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        dn.main(args)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
