"""Embedding configuration, token audit, embedding invariants, artifacts and CLI.

All tests use a deterministic fake encoder; no model is downloaded or loaded.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from founderlens.config import Settings
from founderlens.ingestion.chunk import ChunkingConfig, run_chunking
from founderlens.ingestion.extract import run_extraction
from founderlens.ingestion.normalize import run_normalization
from founderlens.retrieval import embeddings as em
from founderlens.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    audit_token_lengths,
    embed_texts,
    load_embedding_artifact,
    run_build,
)
from founderlens.retrieval.schema import ChunkCollection, load_chunk_collection
from tests.conftest import DOC_A, DOC_B, make_text_pdf

Factory = Callable[..., Path]
SPECIAL_TOKENS = 2  # [CLS] and [SEP]


class FakeEncoder:
    """Deterministic stand-in: one token per whitespace word plus two special tokens."""

    def __init__(
        self,
        *,
        dimension: int = 8,
        limit: int = 16,
        corrupt: Callable[[NDArray[np.float64], Sequence[str]], NDArray[Any]] | None = None,
        fail: bool = False,
    ) -> None:
        self._dimension = dimension
        self._limit = limit
        self.corrupt = corrupt
        self.fail = fail
        self.encode_calls: list[list[str]] = []

    @property
    def max_seq_length(self) -> int:
        return self._limit

    @property
    def dimension(self) -> int:
        return self._dimension

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) + SPECIAL_TOKENS for t in texts]

    def encode(self, texts: Sequence[str]) -> NDArray[Any]:
        self.encode_calls.append(list(texts))
        if self.fail:
            raise RuntimeError("simulated model failure")
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append([(digest[i % 32] - 127.5) * 3.0 for i in range(self._dimension)])
        output = np.asarray(rows, dtype=np.float64)
        return self.corrupt(output, texts) if self.corrupt else output

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": "fake/encoder",
            "model_revision": "0" * 40,
            "tokenizer_class": "WhitespaceFake",
            "effective_max_seq_length": self._limit,
            "output_dimension": self._dimension,
        }


SMALL = EmbeddingConfig(dimension=8, max_seq_length=16, batch_size=2)

# --- configuration -------------------------------------------------------------------


def test_default_configuration_and_identifier() -> None:
    config = EmbeddingConfig()
    assert config.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.model_revision == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert (config.dimension, config.max_seq_length, config.batch_size) == (384, 256, 32)
    assert (config.dtype, config.normalization, config.device) == ("float32", "l2", "cpu")
    canonical = json.dumps(config.semantic_dict(), sort_keys=True, separators=(",", ":"))
    assert config.config_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    assert config.config_id == config.config_sha256[:12] == EmbeddingConfig().config_id
    # Execution settings are not part of the semantic identity.
    assert EmbeddingConfig(batch_size=7).config_id == config.config_id
    assert EmbeddingConfig(max_seq_length=128).config_id != config.config_id
    assert EmbeddingConfig(model_revision="a" * 40).config_id != config.config_id


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_revision": "main"}, "full 40-character"),
        ({"model_revision": "A" * 40}, "full 40-character"),
        ({"model_revision": "abc123"}, "full 40-character"),
        ({"model_id": "no-namespace"}, "organization/name"),
        ({"dimension": 0}, "dimension must be"),
        ({"dimension": True}, "dimension must be"),
        ({"max_seq_length": -1}, "max_seq_length must be"),
        ({"batch_size": 0}, "batch_size must be"),
        ({"batch_size": 5000}, "batch_size must be"),
        ({"batch_size": 2.0}, "batch_size must be"),
        ({"dtype": "float16"}, "dtype must be"),
        ({"normalization": "none"}, "normalization must be"),
        ({"device": "cuda"}, "device must be"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingConfig(**kwargs)  # type: ignore[arg-type]  # invalid values on purpose


# --- token-length audit --------------------------------------------------------------


def _collection_chunks(make_project: Factory) -> tuple[Path, ChunkCollection]:
    root = _project(make_project)
    return root, _load_collection(root)


def test_audit_counts_special_tokens_and_reports_overflows(make_project: Factory) -> None:
    _, collection = _collection_chunks(make_project)
    words = [c.word_count for c in collection.chunks]
    limit = max(words) + SPECIAL_TOKENS - 1  # the longest chunk is one token over
    audit = audit_token_lengths(collection.chunks, FakeEncoder(limit=limit))
    assert audit.lengths == tuple(w + SPECIAL_TOKENS for w in words)
    assert audit.chunk_ids == tuple(c.chunk_id for c in collection.chunks)
    longest = [c for c in collection.chunks if c.word_count == max(words)]
    assert [o["chunk_id"] for o in audit.overflows] == [c.chunk_id for c in longest]
    assert audit.overflows[0]["page_id"] == longest[0].page_id
    assert audit.overflows[0]["excess_tokens"] == 1
    report = audit.to_dict()
    assert report["measurement"] == {
        "add_special_tokens": True,
        "truncation": False,
        "padding": False,
        "text": "exact chunk text intended for embedding",
        "percentiles": "nearest-rank",
    }
    assert report["over_limit_count"] == len(longest)
    assert report["all_within_limit"] is False
    assert report["token_length"]["max"] == max(words) + SPECIAL_TOKENS


def test_audit_boundary_length_is_within_limit(make_project: Factory) -> None:
    _, collection = _collection_chunks(make_project)
    limit = max(c.word_count for c in collection.chunks) + SPECIAL_TOKENS  # exactly at the limit
    audit = audit_token_lengths(collection.chunks, FakeEncoder(limit=limit))
    assert audit.ok
    assert audit.to_dict()["headroom_tokens"] == 0


def test_audit_of_no_chunks_is_empty() -> None:
    report = audit_token_lengths([], FakeEncoder()).to_dict()
    assert report["chunks_inspected"] == 0
    assert report["token_length"]["max"] is None


# --- embedding -----------------------------------------------------------------------

TEXTS = ["alpha beta", "gamma", "delta epsilon zeta", "eta", "theta iota"]


def test_embeddings_are_batched_ordered_float32_and_unit_norm() -> None:
    encoder = FakeEncoder()
    matrix = embed_texts(TEXTS, encoder, SMALL)
    assert [len(call) for call in encoder.encode_calls] == [2, 2, 1]
    assert [t for call in encoder.encode_calls for t in call] == TEXTS  # order preserved
    assert matrix.shape == (5, 8)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=em.NORM_TOLERANCE)
    single = embed_texts([TEXTS[2]], FakeEncoder(), SMALL)
    assert np.array_equal(single[0], matrix[2])  # row i belongs to text i, batch-independent
    raw = FakeEncoder().encode([TEXTS[0]])[0]
    assert np.allclose(matrix[0], raw / np.linalg.norm(raw), atol=1e-6)


def test_oversized_input_is_rejected_before_encoding() -> None:
    encoder = FakeEncoder(limit=4)
    with pytest.raises(EmbeddingError, match=r"exceed the 4-token limit and would be truncated"):
        embed_texts(
            ["one two", "one two three"], encoder, EmbeddingConfig(dimension=8, max_seq_length=4)
        )
    assert encoder.encode_calls == []  # nothing reached the model


@pytest.mark.parametrize(
    ("texts", "message"),
    [([], "input is empty"), (["ok", "   "], "blank or non-string"), (["ok", ""], "blank")],
)
def test_empty_or_blank_input_is_rejected(texts: list[str], message: str) -> None:
    with pytest.raises(EmbeddingError, match=message):
        embed_texts(texts, FakeEncoder(), SMALL)


def _set_row(value: float) -> Callable[[NDArray[np.float64], Sequence[str]], NDArray[Any]]:
    def corrupt(output: NDArray[np.float64], texts: Sequence[str]) -> NDArray[Any]:
        if "gamma" in texts:
            output[list(texts).index("gamma")] = value
        return output

    return corrupt


@pytest.mark.parametrize(
    ("encoder", "message"),
    [
        (FakeEncoder(corrupt=_set_row(0.0)), "zero vectors for \\['text 1'\\]"),
        (FakeEncoder(corrupt=_set_row(float("nan"))), "NaN or infinite values for \\['text 1'\\]"),
        (FakeEncoder(corrupt=_set_row(float("inf"))), "NaN or infinite"),
        (FakeEncoder(corrupt=lambda out, _: out[:, :4]), "returned shape \\(2, 4\\)"),
        (FakeEncoder(corrupt=lambda out, _: out[:1]), "returned shape \\(1, 8\\)"),
        (FakeEncoder(fail=True), "encoder failed on rows 0..1: RuntimeError"),
        (FakeEncoder(dimension=4), "encoder dimension 4 != 8"),
        (FakeEncoder(limit=32), "encoder sequence limit 32 != 16"),
    ],
)
def test_invalid_encoder_output_fails_explicitly(encoder: FakeEncoder, message: str) -> None:
    with pytest.raises(EmbeddingError, match=message):
        embed_texts(TEXTS, encoder, SMALL)


def test_matrix_problems_checks_every_invariant() -> None:
    good = embed_texts(TEXTS, FakeEncoder(), SMALL)
    assert em.matrix_problems(good, 5, 8) == []
    assert "shape" in em.matrix_problems(good, 4, 8)[0]
    assert "dtype" in em.matrix_problems(good.astype(np.float64), 5, 8)[0]
    assert "unit L2 norm" in em.matrix_problems(good * 2, 5, 8)[0]
    bad = good.copy()
    bad[0, 0] = np.nan
    assert "NaN" in em.matrix_problems(bad, 5, 8)[0]


# --- artifacts on a synthetic project -------------------------------------------------


def _project(make_project: Factory) -> Path:
    a = make_text_pdf(["NVIDIA Blackwell revenue was $35,580 million.", "Closing remarks"])
    b = make_text_pdf(["Google Cloud revenue grew 35% in Q4 FY2025."])
    root = make_project({DOC_A: (a, 2), DOC_B: (b, 1)})
    settings = Settings(root)
    run_extraction(settings, expected_document_count=2)
    run_normalization(settings, expected_document_count=2)
    run_chunking(settings, expected_document_count=2)
    return root


def _load_collection(root: Path) -> ChunkCollection:
    s = Settings(root)
    return load_chunk_collection(
        s.processed_data_dir / "chunks.jsonl",
        s.corpus_manifest_path,
        s.processed_data_dir / "pages.normalized.jsonl",
    )


V1_ID = ChunkingConfig().config_id


def _artifact_dir(root: Path) -> Path:
    return root / "artifacts" / "embeddings" / V1_ID


def _build(
    root: Path,
    encoder: FakeEncoder | None = None,
    config: EmbeddingConfig = SMALL,
    output_dir: str | None = None,
) -> em.BuildOutcome:
    collection = _load_collection(root)
    path = root / "data" / "processed" / "chunks.jsonl"
    encoder = encoder or FakeEncoder()
    return run_build(Settings(root), encoder, config, collection, path, output_dir=output_dir)


def test_build_round_trip_and_reproducibility(make_project: Factory) -> None:
    root = _project(make_project)
    collection = _load_collection(root)
    outcome = _build(root)
    first = (_artifact_dir(root) / "embeddings.npy").read_bytes()
    manifest_text = (_artifact_dir(root) / "manifest.json").read_text("utf-8")
    artifact = load_embedding_artifact(_artifact_dir(root), collection, SMALL)
    assert artifact.chunk_ids == tuple(c.chunk_id for c in collection.chunks)
    assert np.array_equal(artifact.matrix, outcome.matrix)
    manifest = artifact.manifest
    assert manifest["source_chunks"]["sha256"] == collection.artifact_sha256
    assert manifest["matrix"]["sha256"] == hashlib.sha256(first).hexdigest()
    assert manifest["matrix"]["shape"] == [3, 8]
    assert manifest["embedding_config_id"] == SMALL.config_id
    assert "seconds" not in manifest_text
    assert "timestamp" not in manifest_text
    assert np.load(_artifact_dir(root) / "embeddings.npy", allow_pickle=False).dtype == np.float32
    _build(root)  # rebuild replaces the artifact with identical content
    assert (_artifact_dir(root) / "embeddings.npy").read_bytes() == first
    assert (_artifact_dir(root) / "manifest.json").read_text("utf-8") == manifest_text
    report = json.loads(
        (root / "reports" / f"embedding_build_report.{V1_ID}.json").read_text("utf-8")
    )
    assert report["matrix_sha256"] == manifest["matrix"]["sha256"]
    assert set(report["operational"]) == {"note", "embedding_seconds"}
    audit = json.loads(
        (root / "reports" / f"embedding_token_audit.{V1_ID}.json").read_text("utf-8")
    )
    assert audit["chunks_inspected"] == 3
    assert audit["all_within_limit"] is True


def test_build_stops_before_encoding_when_a_chunk_overflows(make_project: Factory) -> None:
    root = _project(make_project)
    _build(root)
    previous = (_artifact_dir(root) / "embeddings.npy").read_bytes()
    encoder = FakeEncoder(limit=4)
    with pytest.raises(EmbeddingError, match="exceed the 4-token limit; no embeddings"):
        _build(root, encoder, EmbeddingConfig(dimension=8, max_seq_length=4))
    assert encoder.encode_calls == []
    assert (_artifact_dir(root) / "embeddings.npy").read_bytes() == previous  # untouched
    audit = json.loads(
        (root / "reports" / f"embedding_token_audit.{V1_ID}.json").read_text("utf-8")
    )
    assert audit["over_limit_count"] > 0
    assert audit["over_limit_chunks"][0]["chunk_id"].startswith(DOC_A)


def test_failed_build_leaves_no_partial_artifact(make_project: Factory) -> None:
    root = _project(make_project)
    with pytest.raises(EmbeddingError, match="encoder failed"):
        _build(root, FakeEncoder(fail=True))
    artifacts = root / "artifacts" / "embeddings"
    assert not _artifact_dir(root).exists()
    leftovers = list(artifacts.iterdir()) if artifacts.exists() else []
    assert leftovers == []  # staging directory removed


def _tamper(directory: Path, kind: str) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    matrix_path = directory / "embeddings.npy"
    if kind == "missing_manifest":
        manifest_path.unlink()
        return
    if kind == "malformed_manifest":
        manifest_path.write_text("{not json", encoding="utf-8")
        return
    if kind == "corrupted_matrix":
        data = bytearray(matrix_path.read_bytes())
        data[-1] ^= 0xFF
        matrix_path.write_bytes(bytes(data))
        return
    if kind == "truncated_matrix":
        matrix_path.write_bytes(matrix_path.read_bytes()[:40])
        return
    if kind == "wrong_dtype":
        array = np.load(matrix_path, allow_pickle=False).astype(np.float64)
        matrix_path.write_bytes(em.matrix_bytes(array))  # wrong dtype on purpose
        manifest["matrix"]["sha256"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    elif kind == "unnormalized":
        array = np.load(matrix_path, allow_pickle=False) * np.float32(2)
        matrix_path.write_bytes(em.matrix_bytes(array))
        manifest["matrix"]["sha256"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    elif kind == "wrong_matrix_hash":
        manifest["matrix"]["sha256"] = "0" * 64
    elif kind == "wrong_source_hash":
        manifest["source_chunks"]["sha256"] = "1" * 64
    elif kind == "reordered_rows":
        manifest["rows"] = list(reversed(manifest["rows"]))
    elif kind == "duplicate_rows":
        manifest["rows"][1] = manifest["rows"][0]
    elif kind == "missing_key":
        del manifest["rows"]
    elif kind == "wrong_schema":
        manifest["artifact_schema_version"] = 99
    elif kind == "wrong_config_id":
        manifest["embedding_config_id"] = "deadbeef0000"
    elif kind == "wrong_dimension":
        manifest["embedding_config"]["dimension"] = 16
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing_manifest", "incomplete artifact"),
        ("malformed_manifest", "not valid JSON"),
        ("corrupted_matrix", "SHA-256 does not match"),
        ("truncated_matrix", "SHA-256 does not match"),
        ("wrong_dtype", "dtype float64 != float32"),
        ("unnormalized", "not unit L2 norm"),
        ("wrong_matrix_hash", "SHA-256 does not match"),
        ("wrong_source_hash", "different chunk collection \\(stale\\)"),
        ("reordered_rows", "does not match the collection order"),
        ("duplicate_rows", "duplicate chunk ids"),
        ("missing_key", "unexpected structure"),
        ("wrong_schema", "unsupported artifact_schema_version"),
        ("wrong_config_id", "embedding_config_id does not match"),
        ("wrong_dimension", "shape"),
    ],
)
def test_invalid_artifacts_are_rejected(make_project: Factory, kind: str, message: str) -> None:
    root = _project(make_project)
    _build(root)
    _tamper(_artifact_dir(root), kind)
    with pytest.raises(EmbeddingError, match=message):
        load_embedding_artifact(_artifact_dir(root), _load_collection(root), SMALL)


def test_artifact_for_another_configuration_is_rejected(make_project: Factory) -> None:
    root = _project(make_project)
    _build(root)
    with pytest.raises(EmbeddingError, match="requested config"):
        load_embedding_artifact(
            _artifact_dir(root),
            _load_collection(root),
            EmbeddingConfig(dimension=8, max_seq_length=32),
        )


def test_output_must_be_inside_artifacts_and_not_clobber_other_files(make_project: Factory) -> None:
    root = _project(make_project)
    with pytest.raises(EmbeddingError, match="subdirectory of artifacts"):
        _build(root, output_dir="data/processed")
    other = root / "artifacts" / "other"
    other.mkdir(parents=True)
    (other / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(EmbeddingError, match="contains other files"):
        _build(root, output_dir="artifacts/other")
    assert (other / "notes.txt").read_text("utf-8") == "keep"


# --- import side effects and CLI ------------------------------------------------------


def test_import_and_entry_point_do_not_load_the_model() -> None:
    code = (
        "import sys, founderlens.retrieval.embeddings, founderlens.retrieval.dense;"
        "import founderlens.__main__;"
        "print(any(m in sys.modules for m in ('torch', 'sentence_transformers', 'transformers')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "False"
    encoder = em.SentenceTransformerEncoder(EmbeddingConfig())
    assert encoder._model is None  # constructing the encoder loads nothing


def _fake_factory(**kwargs: Any) -> Callable[..., FakeEncoder]:
    def factory(config: EmbeddingConfig, *, allow_download: bool = False) -> FakeEncoder:
        return FakeEncoder(dimension=config.dimension, limit=config.max_seq_length, **kwargs)

    return factory


def test_cli_audit_build_and_validate(
    make_project: Factory, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(make_project)
    monkeypatch.setattr(em, "SentenceTransformerEncoder", _fake_factory())
    base = ["--project-root", str(root)]
    assert em.main([*base, "audit", "--collection", "v1"]) == 0
    assert f"Token audit ({V1_ID}): 3 chunks, limit 256 tokens" in capsys.readouterr().out
    assert em.main([*base, "build", "--collection", "v1", "--batch-size", "2"]) == 0
    out = capsys.readouterr().out
    assert "Embedded 3 chunks -> 384-d float32" in out
    assert em.main([*base, "validate", "--collection", "v1"]) == 0
    assert "Embedding artifact valid" in capsys.readouterr().out
    _tamper(_artifact_dir(root), "corrupted_matrix")
    assert em.main([*base, "validate", "--collection", "v1"]) == 1
    assert "SHA-256 does not match" in capsys.readouterr().err


def test_cli_overflow_and_failures_exit_1(
    make_project: Factory, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(make_project)
    base = ["--project-root", str(root)]

    class Tiny(FakeEncoder):
        @property
        def max_seq_length(self) -> int:
            return 3

    monkeypatch.setattr(em, "SentenceTransformerEncoder", lambda config, **_: Tiny(dimension=384))
    assert em.main([*base, "audit", "--collection", "v1"]) == 1
    assert "OVER " in capsys.readouterr().out
    assert em.main([*base, "build", "--collection", "v1"]) == 1
    err = capsys.readouterr().err
    assert "Nothing was published" in err
    assert not _artifact_dir(root).exists()
    assert em.main([*base, "validate", "--collection", "v1"]) == 1
    assert "embedding artifact not found" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["build", "--collection", "v1", "--batch-size", "0"], "batch_size must be"),
        (["build", "--collection", "v1", "--batch-size", "x"], "invalid int value"),
        (["build"], "one of the arguments --collection --chunks is required"),
        (["audit", "--collection", "v3"], "invalid choice"),
        ([], "the following arguments are required: command"),
    ],
)
def test_cli_invalid_arguments_exit_2(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        em.main(args)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
