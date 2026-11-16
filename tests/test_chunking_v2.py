"""Chunking V2: the version-1 window rule plus a hard model-token budget.

A fake tokenizer stands in for the model: every run of letters, every run of
digits and every punctuation mark is one token, plus two special tokens. So
``$35,082`` costs 5 tokens while ``revenue`` costs 1, like WordPiece on
financial tables. No model is loaded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from founderlens.config import Settings
from founderlens.ingestion.chunk import (
    ChunkingConfig,
    ChunkingConfigError,
    ChunkingError,
    TokenBudget,
    chunk_page,
    count_words,
    page_overlaps,
    plan_windows,
    run_chunking,
    token_budget_problems,
    uncovered_characters,
    validate_chunks,
)
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import NormalizedPageRecord, page_warnings, run_normalization
from founderlens.ingestion.schema import make_page_id, status_for_text
from founderlens.retrieval import embeddings as em
from founderlens.retrieval.collections import (
    CHUNKING_V2_CONFIG,
    collection_path,
    load_named_collection,
)
from founderlens.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    load_embedding_artifact,
)
from founderlens.retrieval.model import MODEL_ID, MODEL_REVISION, TOKENIZER_CLASS
from founderlens.retrieval.schema import RetrievalError
from tests.conftest import DOC_A, make_text_pdf

Factory = Callable[..., Path]
_PIECE = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
REVISION = "b" * 40


class FakeTokenizer:
    """Letters, digit runs and punctuation are one token each; plus [CLS] and [SEP]."""

    def __init__(self, limit: int, *, penalty: Callable[[str], int] | None = None) -> None:
        self.limit = limit
        self.penalty = penalty
        self.calls = 0

    @property
    def max_seq_length(self) -> int:
        return self.limit

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        self.calls += 1
        extra = self.penalty or (lambda _: 0)
        return [len(_PIECE.findall(t)) + 2 + extra(t) for t in texts]


def _config(limit: int, size: int = 20, overlap: int = 4) -> ChunkingConfig:
    budget = TokenBudget(
        max_tokens=limit, model_id="fake/model", model_revision=REVISION, tokenizer="Fake"
    )
    return ChunkingConfig(size, overlap, token_budget=budget)


def _page(text: str, number: int = 1, document_id: str = DOC_A) -> NormalizedPageRecord:
    status = status_for_text(text)
    return NormalizedPageRecord(
        document_id=document_id,
        document_sha256="a" * 64,
        company="Example Corp",
        fiscal_period="Q1 2024",
        source_url="https://example.com/report.pdf",
        page_number=number,
        page_id=make_page_id(document_id, number),
        extraction_status=status,
        raw_character_count=len(text),
        normalized_text=text,
        normalized_character_count=len(text),
        tables=(),
        table_text_overlap=False,
        normalization_warnings=page_warnings(status, ()),
    )


PROSE = "\n".join(" ".join(f"word{i}x" for i in range(j, j + 6)) for j in range(0, 60, 6)) + "\n"
TABLE = (
    "\n".join(f"Line item {n}   $ {n},082.5   $ ({n}8,120)   {n}.4%   {n}.1%" for n in range(1, 13))
    + "\n"
)


def _check_all(page: NormalizedPageRecord, config: ChunkingConfig, counter: FakeTokenizer) -> None:
    chunks = chunk_page(page, config, counter)
    text = page.normalized_text
    assert validate_chunks(chunks, [page], config, counter) == []  # spans, words, overlap, tokens
    assert uncovered_characters(text, chunks) == 0
    assert all(n <= counter.limit for n in counter.token_lengths([c.text for c in chunks]))
    assert all(c.word_count <= config.chunk_size_words for c in chunks)
    assert all(a.end_char < b.end_char for a, b in pairwise(chunks))  # forward progress
    assert all(o <= config.chunk_overlap_words for o in page_overlaps(text, chunks))


# --- configuration -------------------------------------------------------------------


def test_v1_identity_is_unchanged_and_v2_is_distinct() -> None:
    v1 = ChunkingConfig()
    assert v1.config_id == "a8a0288e0348"  # the frozen 162-chunk baseline
    assert "token_budget" not in v1.to_dict()
    assert v1.algorithm_version == 1
    v2 = CHUNKING_V2_CONFIG.to_dict()
    assert v2["algorithm_version"] == 2
    assert v2["chunk_size_words"] == 120
    assert v2["chunk_overlap_words"] == 24
    assert v2["token_budget"] == {
        "max_tokens": 256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer": TOKENIZER_CLASS,
        "special_tokens": "included",
        "truncation": "disabled",
        "padding": "disabled",
        "counted_text": "exact chunk text",
    }
    assert CHUNKING_V2_CONFIG.config_id not in {v1.config_id, _config(256, 120, 24).config_id}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_tokens": 2}, "max_tokens must be"),
        ({"model_revision": "main"}, "40-character commit SHA"),
        ({"tokenizer": " "}, "tokenizer must be"),
    ],
)
def test_invalid_token_budget(kwargs: dict[str, Any], message: str) -> None:
    base: dict[str, Any] = {
        "max_tokens": 16,
        "model_id": "fake/model",
        "model_revision": REVISION,
        "tokenizer": "Fake",
    }
    with pytest.raises(ChunkingConfigError, match=message):
        TokenBudget(**{**base, **kwargs})


def test_budget_requires_a_matching_counter() -> None:
    with pytest.raises(ChunkingConfigError, match="requires a token counter"):
        plan_windows(PROSE, _config(32))
    with pytest.raises(ChunkingConfigError, match="limit 64 != token budget 32"):
        plan_windows(PROSE, _config(32), FakeTokenizer(64))
    assert plan_windows(PROSE, ChunkingConfig(20, 4))  # version 1 needs no counter


# --- token budget ---------------------------------------------------------------------


def test_unaffected_text_is_chunked_exactly_as_in_version_1() -> None:
    v1 = plan_windows(PROSE, ChunkingConfig(20, 4))
    v2 = plan_windows(PROSE, _config(64), FakeTokenizer(64))
    assert [(w.start_char, w.end_char) for w in v2] == [(w.start_char, w.end_char) for w in v1]
    assert not any(w.token_limited for w in v2)


def test_numeric_dense_table_is_cut_to_fit_and_fully_covered() -> None:
    counter = FakeTokenizer(40)
    v1 = plan_windows(TABLE, ChunkingConfig(20, 4))
    assert max(counter.token_lengths([TABLE[w.start_char : w.end_char] for w in v1])) > 40
    windows = plan_windows(TABLE, _config(40), counter)
    assert any(w.token_limited for w in windows)
    _check_all(_page(TABLE), _config(40), counter)


def test_special_tokens_count_and_the_exact_limit_fits() -> None:
    text = "one two three four five six seven eight nine ten eleven"  # 11 words + 2 specials
    [window] = plan_windows(text, _config(13), FakeTokenizer(13))  # exactly 13 tokens: fits
    assert (window.start_char, window.end_char) == (0, len(text))
    assert not window.token_limited
    split = plan_windows(text, _config(12), FakeTokenizer(12))  # one token over: split
    assert len(split) == 2
    assert all(w.token_limited for w in split[:1])
    _check_all(_page(text), _config(12), FakeTokenizer(12))


def test_line_boundaries_are_preferred_inside_the_token_budget() -> None:
    text = "\n".join(f"a{n} b{n} c{n} 1,2{n}" for n in range(10)) + "\n"  # 6 tokens + 4 words/line
    config = _config(26, size=20, overlap=0)  # 20 words would be 5 lines = 32 tokens
    chunks = chunk_page(_page(text), config, FakeTokenizer(26))
    assert all(text[c.end_char] == "\n" for c in chunks)  # every cut at a line end
    _check_all(_page(text), config, FakeTokenizer(26))


@pytest.mark.parametrize(
    ("size", "overlap", "limit"), [(20, 19, 30), (12, 6, 20), (5, 0, 9), (2, 1, 9)]
)
def test_overlap_shrinks_but_progress_and_coverage_hold(
    size: int, overlap: int, limit: int
) -> None:
    _check_all(_page(TABLE), _config(limit, size, overlap), FakeTokenizer(limit))


def test_non_monotonic_token_counts_are_handled() -> None:
    # Odd-length candidates get a large penalty, so counts jump up and down as the end moves.
    counter = FakeTokenizer(40, penalty=lambda t: 30 if len(t) % 2 else 0)
    config = _config(40, size=16, overlap=3)
    _check_all(_page(TABLE), config, counter)


def test_an_unsplittable_word_fails_with_provenance() -> None:
    text = "fine words then 1,2,3,4,5,6,7,8,9,0,1,2,3,4,5 more"
    with pytest.raises(ChunkingError, match=r"doc-a.*\[16, 45\) does not fit the 12-token budget"):
        chunk_page(_page(text, document_id="doc-a"), _config(12), FakeTokenizer(12))


def test_budget_violations_are_detected_independently() -> None:
    chunks = chunk_page(_page(TABLE), ChunkingConfig(20, 4))  # version 1, too long for 48
    problems = token_budget_problems(chunks, _config(40), FakeTokenizer(40))
    assert problems
    assert all("exceed the 40-token budget" in p for p in problems)


def test_v2_is_deterministic() -> None:
    first = chunk_page(_page(TABLE), _config(40), FakeTokenizer(40))
    assert chunk_page(_page(TABLE), _config(40), FakeTokenizer(40)) == first
    assert all(c.chunking_config_id == _config(40).config_id for c in first)
    assert sum(c.word_count for c in first) >= count_words(TABLE)


# --- collections on a synthetic project -----------------------------------------------


class V2Encoder:
    """Fake encoder with the V2 identity (pinned model, BertTokenizer, 256 tokens)."""

    def __init__(self, config: EmbeddingConfig | None = None, **_: object) -> None:
        self.tokenizer = FakeTokenizer(256)

    @property
    def max_seq_length(self) -> int:
        return 256

    @property
    def dimension(self) -> int:
        return 384

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        return self.tokenizer.token_lengths(texts)

    def encode(self, texts: Sequence[str]) -> NDArray[Any]:
        rows = [[float((hash_ + i) % 7 + 1) for i in range(384)] for hash_ in map(len, texts)]
        return np.asarray(rows, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_class": TOKENIZER_CLASS,
            "effective_max_seq_length": 256,
            "output_dimension": 384,
        }


DENSE = " ".join(f"$ {n},{n}{n}{n}.{n} ({n}{n},{n}{n}{n})" for n in range(1, 10)) * 6


def _project(make_project: Factory) -> Path:
    a = make_text_pdf(["Revenue grew strongly this quarter.", DENSE[:900]])
    b = make_text_pdf(["Google Cloud revenue grew 35% in Q4 FY2025."])
    root = make_project(
        {"aaa-2024-q1": (a, 2), "bbb-2024-q1": (b, 1), "ccc-2024-q1": (b, 1), "ddd-2024-q1": (b, 1)}
    )
    settings = Settings(root)
    run_extraction(settings)
    run_normalization(settings)
    run_chunking(settings)  # the V1 baseline
    return root


def test_v2_collection_is_separate_and_v1_is_untouched(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(make_project)
    settings = Settings(root)
    v1_path = collection_path(settings, "v1")
    v1_bytes = v1_path.read_bytes()
    monkeypatch.setattr(em, "SentenceTransformerEncoder", V2Encoder)
    base = ["--project-root", str(root)]
    assert em.main([*base, "build-collection"]) == 0
    v2_path = collection_path(settings, "v2")
    first = v2_path.read_bytes()
    assert em.main([*base, "build-collection"]) == 0
    assert v2_path.read_bytes() == first  # byte-identical rebuild
    assert v1_path.read_bytes() == v1_bytes  # V1 never overwritten
    assert v2_path.name == f"chunks.{CHUNKING_V2_CONFIG.config_id}.jsonl"
    v2 = load_named_collection(settings, "v2")
    assert {c.chunking_config_id for c in v2.chunks} == {CHUNKING_V2_CONFIG.config_id}
    report = json.loads(
        (root / "reports" / f"chunking_report.{CHUNKING_V2_CONFIG.config_id}.json").read_text(
            "utf-8"
        )
    )
    assert report["chunking_config"]["token_budget"]["max_tokens"] == 256
    assert report["diagnostics"]["token_budget"]["over_budget"] == 0
    assert em.main([*base, "validate-collection", "--collection", "v2"]) == 0
    assert em.main([*base, "validate-collection", "--collection", "v1"]) == 0
    capsys.readouterr()

    # Embed V2, then show that its artifact cannot be used with V1.
    assert em.main([*base, "build", "--collection", "v2"]) == 0
    assert em.main([*base, "validate", "--collection", "v2"]) == 0
    v2_dir = root / "artifacts" / "embeddings" / CHUNKING_V2_CONFIG.config_id
    v1 = load_named_collection(settings, "v1")
    with pytest.raises(EmbeddingError, match="different chunk collection"):
        load_embedding_artifact(v2_dir, v1, EmbeddingConfig())
    reordered = type(v2)(
        chunks=tuple(reversed(v2.chunks)),
        artifact_sha256=v2.artifact_sha256,
        chunking_config_id=v2.chunking_config_id,
        normalized_sha256=v2.normalized_sha256,
    )
    with pytest.raises(EmbeddingError, match="collection order"):
        load_embedding_artifact(v2_dir, reordered, EmbeddingConfig())
    capsys.readouterr()
    assert em.main([*base, "validate", "--collection", "v1"]) == 1  # no V1 artifact exists
    assert "embedding artifact not found" in capsys.readouterr().err


def test_v2_cannot_overwrite_the_v1_files(make_project: Factory) -> None:
    root = _project(make_project)
    with pytest.raises(ChunkingError, match="version-1 baseline"):
        run_chunking(Settings(root), config=CHUNKING_V2_CONFIG, counter=V2Encoder())


def test_named_collection_rejects_the_wrong_configuration(make_project: Factory) -> None:
    root = _project(make_project)
    settings = Settings(root)
    collection_path(settings, "v2").write_bytes(collection_path(settings, "v1").read_bytes())
    with pytest.raises(RetrievalError, match="must have chunking_config_id"):
        load_named_collection(settings, "v2")


def test_build_collection_rejects_a_different_tokenizer(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(make_project)

    class OtherTokenizer(V2Encoder):
        def describe(self) -> dict[str, Any]:
            return {**super().describe(), "tokenizer_class": "BertTokenizerFast"}

    monkeypatch.setattr(em, "SentenceTransformerEncoder", OtherTokenizer)
    assert em.main(["--project-root", str(root), "build-collection"]) == 1
    assert "does not match the chunking V2 token budget" in capsys.readouterr().err
    assert not collection_path(Settings(root), "v2").exists()
