"""Reproducible local chunk embeddings for dense retrieval (Commit 3.2).

Usage::

    python -m founderlens.retrieval.embeddings audit [--allow-download]
    python -m founderlens.retrieval.embeddings build [--allow-download] [--batch-size 32]
    python -m founderlens.retrieval.embeddings validate

The model is ``sentence-transformers/all-MiniLM-L6-v2`` pinned to an
immutable Hugging Face commit. It runs locally on CPU with
``trust_remote_code=False`` and safetensors weights. Nothing is sent to a
hosted embedding service. By default the model is loaded from the local
Hugging Face cache only; ``--allow-download`` permits fetching the pinned
revision once.

Token-length audit. Each chunk's exact text is tokenized with the model's own
tokenizer, special tokens included, truncation and padding disabled. A chunk
longer than the effective sequence limit (the smallest of the model's
``max_seq_length``, the tokenizer's ``model_max_length`` and the position
embedding count) is never embedded: Sentence Transformers would silently
truncate it. ``build`` stops before encoding if any chunk overflows, and
:func:`embed_texts` repeats the check itself.

Embeddings. Chunk text is encoded exactly as stored (no prefix, prompt or
metadata), in collection order, in batches. Each output row is checked for
shape and finiteness, rejected if its norm is zero, L2-normalized and stored
as float32. Rows must have unit norm within :data:`NORM_TOLERANCE`.

Artifact (``artifacts/embeddings/`` by default): ``embeddings.npy`` (no
pickle) plus ``manifest.json`` with the configuration, source chunk artifact
SHA-256, matrix SHA-256, the ordered row-to-chunk mapping and the execution
environment. Both files are staged in a temporary directory, validated, then
moved into place, so a failed build never leaves a partial artifact.

Identity. ``embedding_config_id`` hashes the semantic configuration only
(model, revision, dimension, dtype, normalization, sequence limit, input
policy). Corpus identity is the source chunk SHA-256; artifact identity is the
matrix SHA-256. Device, batch size, threads and library versions are recorded
as execution details, not hashed into the configuration identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import re
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.corpus.manifest import EXPECTED_DOCUMENT_COUNT, ManifestError, load_manifest
from founderlens.ingestion.chunk import (
    CHUNKS_FILENAME,
    ChunkingConfigError,
    ChunkingError,
    ChunkingOutcome,
    ChunkRecord,
    load_normalized_artifact,
    run_chunking,
    validate_chunks,
)
from founderlens.ingestion.extract import publish_atomic
from founderlens.ingestion.normalize import PAGES_NORMALIZED_FILENAME
from founderlens.retrieval.collections import (
    CHUNKING_V2_CONFIG,
    COLLECTION_NAMES,
    CollectionName,
    chunking_report_path,
    collection_config,
    collection_path,
    load_named_collection,
)
from founderlens.retrieval.model import (
    EMBEDDING_DIMENSION,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    TOKENIZER_CLASS,
)
from founderlens.retrieval.schema import ChunkCollection, RetrievalError, load_chunk_collection

EMBEDDING_CONFIG_SCHEMA_VERSION = 1
EMBEDDING_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_MODEL_ID = MODEL_ID
DEFAULT_MODEL_REVISION = MODEL_REVISION
DEFAULT_DIMENSION = EMBEDDING_DIMENSION
DEFAULT_MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
NEAR_LIMIT_FRACTION = 0.9  # audit reports chunks above 90% of the limit as "near the limit"
DEFAULT_BATCH_SIZE = 32
MAX_BATCH_SIZE = 1024
NORM_TOLERANCE = 1e-5
EMBEDDINGS_DIRNAME = "embeddings"
MATRIX_FILENAME = "embeddings.npy"
MANIFEST_FILENAME = "manifest.json"
AUDIT_REPORT_STEM = "embedding_token_audit"
BUILD_REPORT_STEM = "embedding_build_report"
INPUT_POLICY = "exact chunk text; no prefix, prompt or metadata"
DEPENDENCIES = (
    "sentence-transformers",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "torch",
    "numpy",
    "safetensors",
)

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_FILES = frozenset({MATRIX_FILENAME, MANIFEST_FILENAME})

Float32Matrix = NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """Embedding run or artifact is invalid. ``problems`` lists every actionable reason."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Semantic embedding settings plus execution settings (device, batch size)."""

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    dimension: int = DEFAULT_DIMENSION
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    dtype: Literal["float32"] = "float32"
    normalization: Literal["l2"] = "l2"
    device: Literal["cpu"] = "cpu"
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        model_id: object = self.model_id
        revision: object = self.model_revision
        if not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id):
            raise ValueError(f"model_id must look like 'organization/name', got {model_id!r}")
        if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
            raise ValueError(
                "model_revision must be a full 40-character lowercase commit SHA "
                f"(not a branch or tag), got {revision!r}"
            )
        for name in ("dimension", "max_seq_length"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        batch: object = self.batch_size
        if (
            isinstance(batch, bool)
            or not isinstance(batch, int)
            or not 1 <= batch <= MAX_BATCH_SIZE
        ):
            raise ValueError(f"batch_size must be an integer in 1..{MAX_BATCH_SIZE}, got {batch!r}")
        for name, allowed in (("dtype", "float32"), ("normalization", "l2"), ("device", "cpu")):
            if getattr(self, name) != allowed:
                raise ValueError(f"{name} must be {allowed!r}, got {getattr(self, name)!r}")

    def semantic_dict(self) -> dict[str, Any]:
        """Settings that define what the vectors mean. Hashed into :attr:`config_id`."""
        return {
            "schema_version": EMBEDDING_CONFIG_SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "dtype": self.dtype,
            "normalization": self.normalization,
            "input_policy": INPUT_POLICY,
            "overflow_policy": "reject (no truncation)",
        }

    @property
    def config_sha256(self) -> str:
        canonical = json.dumps(self.semantic_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def config_id(self) -> str:
        return self.config_sha256[:12]


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #


class TextEncoder(Protocol):
    """The model-specific part: tokenization lengths and raw sentence embeddings."""

    @property
    def max_seq_length(self) -> int:
        """Effective input limit in tokens, special tokens included."""
        ...

    @property
    def dimension(self) -> int: ...

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Token counts with special tokens, no truncation and no padding."""
        ...

    def encode(self, texts: Sequence[str]) -> NDArray[Any]:
        """One raw embedding row per text, in order."""
        ...

    def describe(self) -> dict[str, Any]:
        """Model, tokenizer and backend facts for the artifact manifest."""
        ...


class SentenceTransformerEncoder:
    """Pinned Sentence Transformers model, loaded lazily on first use (CPU, eval mode)."""

    def __init__(self, config: EmbeddingConfig, *, allow_download: bool = False) -> None:
        self.config = config
        self.allow_download = allow_download
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(
                self.config.model_id,
                revision=self.config.model_revision,
                device=self.config.device,
                local_files_only=not self.allow_download,
                trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
            )
        except OSError as exc:
            hint = (
                ""
                if self.allow_download
                else "; re-run with --allow-download to fetch the "
                "pinned revision into the local Hugging Face cache"
            )
            raise EmbeddingError(
                [f"cannot load {self.config.model_id}@{self.config.model_revision}: {exc}{hint}"]
            ) from exc
        model.eval()
        limits = self._limits(model)
        effective = min(limits.values())
        problems = []
        if effective != self.config.max_seq_length:
            problems.append(
                f"model effective sequence limit is {effective} ({limits}), configuration says "
                f"{self.config.max_seq_length}"
            )
        dimension = model.get_embedding_dimension()
        if dimension != self.config.dimension:
            problems.append(
                f"model dimension is {dimension}, configuration says {self.config.dimension}"
            )
        if model.default_prompt_name is not None:
            problems.append(f"model has a default prompt {model.default_prompt_name!r}")
        if type(model.tokenizer).__name__ != TOKENIZER_CLASS:
            problems.append(
                f"tokenizer is {type(model.tokenizer).__name__}, expected {TOKENIZER_CLASS}"
            )
        if problems:
            raise EmbeddingError(problems)
        self._model = model
        return model

    @staticmethod
    def _limits(model: Any) -> dict[str, int]:
        return {
            "model_max_seq_length": int(model.max_seq_length),
            "tokenizer_model_max_length": int(model.tokenizer.model_max_length),
            "max_position_embeddings": int(model[0].auto_model.config.max_position_embeddings),
        }

    @property
    def max_seq_length(self) -> int:
        return min(self._limits(self._load()).values())

    @property
    def dimension(self) -> int:
        return int(self._load().get_embedding_dimension())

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        encoded = self._load().tokenizer(
            list(texts), add_special_tokens=True, truncation=False, padding=False
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def encode(self, texts: Sequence[str]) -> NDArray[Any]:
        import torch

        model = self._load()
        with torch.inference_mode():
            output = model.encode(
                list(texts),
                batch_size=len(texts),
                prompt=None,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        return np.asarray(output)

    def describe(self) -> dict[str, Any]:
        import torch

        model = self._load()
        return {
            "model_id": self.config.model_id,
            "model_revision": self.config.model_revision,
            "model_license": MODEL_LICENSE,
            "backend": "sentence-transformers (PyTorch)",
            "modules": [type(module).__name__ for module in model],
            "tokenizer_class": type(model.tokenizer).__name__,
            "tokenizer_vocab_size": int(model.tokenizer.vocab_size),
            "sequence_limits": self._limits(model),
            "effective_max_seq_length": self.max_seq_length,
            "output_dimension": self.dimension,
            "trust_remote_code": False,
            "weights_format": "safetensors",
            "torch_num_threads": int(torch.get_num_threads()),
        }


# --------------------------------------------------------------------------- #
# Token-length audit
# --------------------------------------------------------------------------- #


def _nearest_rank(sorted_values: Sequence[int], percent: float) -> int:
    """Nearest-rank percentile: the smallest value with at least ``percent``% at or below it."""
    rank = max(1, math.ceil(percent / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


@dataclass(frozen=True, slots=True)
class TokenAudit:
    chunk_ids: tuple[str, ...]
    lengths: tuple[int, ...]
    limit: int
    overflows: tuple[dict[str, Any], ...]

    @property
    def ok(self) -> bool:
        return not self.overflows

    @property
    def near_threshold(self) -> int:
        return math.ceil(NEAR_LIMIT_FRACTION * self.limit)

    def to_dict(self) -> dict[str, Any]:
        values = sorted(self.lengths)
        stats: dict[str, Any] = {"min": None, "max": None, "mean": None, "median": None}
        if values:
            stats = {
                "min": values[0],
                "max": values[-1],
                "mean": round(statistics.fmean(values), 3),
                "median": statistics.median(values),
                "p90": _nearest_rank(values, 90),
                "p95": _nearest_rank(values, 95),
                "p99": _nearest_rank(values, 99),
            }
        return {
            "measurement": {
                "add_special_tokens": True,
                "truncation": False,
                "padding": False,
                "text": "exact chunk text intended for embedding",
                "percentiles": "nearest-rank",
            },
            "chunks_inspected": len(self.lengths),
            "effective_max_seq_length": self.limit,
            "token_length": stats,
            "headroom_tokens": None if not values else self.limit - values[-1],
            "near_limit": {
                "definition": f"within the limit and at least {self.near_threshold} tokens "
                f"({NEAR_LIMIT_FRACTION:.0%} of {self.limit})",
                "count": sum(1 for n in values if self.near_threshold <= n <= self.limit),
            },
            "over_limit_count": len(self.overflows),
            "over_limit_chunks": list(self.overflows),
            "all_within_limit": self.ok,
        }


def audit_token_lengths(chunks: Sequence[ChunkRecord], encoder: TextEncoder) -> TokenAudit:
    """Measure every chunk with the encoder's own tokenizer (no truncation)."""
    limit = encoder.max_seq_length
    lengths = encoder.token_lengths([c.text for c in chunks])
    if len(lengths) != len(chunks):
        raise EmbeddingError(["tokenizer returned a different number of lengths than texts"])
    overflows = tuple(
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "page_id": c.page_id,
            "page_number": c.page_number,
            "word_count": c.word_count,
            "token_length": n,
            "excess_tokens": n - limit,
        }
        for c, n in zip(chunks, lengths, strict=True)
        if n > limit
    )
    return TokenAudit(tuple(c.chunk_id for c in chunks), tuple(lengths), limit, overflows)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #


def matrix_problems(matrix: NDArray[Any], rows: int, dimension: int) -> list[str]:
    """Invariants of a stored embedding matrix."""
    if matrix.ndim != 2 or matrix.shape != (rows, dimension):
        return [f"matrix shape {matrix.shape} != expected ({rows}, {dimension})"]
    problems = []
    if matrix.dtype != np.float32:
        problems.append(f"matrix dtype {matrix.dtype} != float32")
    if not np.all(np.isfinite(matrix)):
        problems.append("matrix contains NaN or infinite values")
        return problems
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    bad = np.flatnonzero(np.abs(norms - 1.0) > NORM_TOLERANCE)
    if bad.size:
        problems.append(
            f"{bad.size} row(s) are not unit L2 norm within {NORM_TOLERANCE} (first row {bad[0]})"
        )
    return problems


def embed_texts(
    texts: Sequence[str],
    encoder: TextEncoder,
    config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
    *,
    labels: Sequence[str] | None = None,
) -> Float32Matrix:
    """Encode ``texts`` in order and return validated, L2-normalized float32 rows.

    Raises :class:`EmbeddingError` for empty input, blank texts, any text over
    the encoder's sequence limit (checked before encoding anything), encoder
    failures, wrong output shapes, non-finite values or zero vectors.
    """
    names = list(labels) if labels is not None else [f"text {i}" for i in range(len(texts))]
    if not texts:
        raise EmbeddingError(["nothing to embed: the input is empty"])
    if len(names) != len(texts):
        raise EmbeddingError(["labels and texts differ in length"])
    items: Sequence[object] = texts  # callers may pass untyped values
    blank = [
        n for n, t in zip(names, items, strict=True) if not isinstance(t, str) or not t.strip()
    ]
    if blank:
        raise EmbeddingError([f"blank or non-string input: {blank[:5]}"])
    if encoder.dimension != config.dimension:
        raise EmbeddingError([f"encoder dimension {encoder.dimension} != {config.dimension}"])
    limit = encoder.max_seq_length
    if limit != config.max_seq_length:
        raise EmbeddingError([f"encoder sequence limit {limit} != {config.max_seq_length}"])
    lengths = encoder.token_lengths(texts)
    over = [f"{n} ({k} tokens)" for n, k in zip(names, lengths, strict=True) if k > limit]
    if over:
        raise EmbeddingError(
            [
                f"{len(over)} input(s) exceed the {limit}-token limit and would be truncated: "
                f"{over[:5]}"
            ]
        )
    batches: list[NDArray[np.float64]] = []
    for start in range(0, len(texts), config.batch_size):
        batch = texts[start : start + config.batch_size]
        try:
            output = np.asarray(encoder.encode(batch))
        except EmbeddingError:
            raise
        except Exception as exc:  # any model/runtime failure becomes an actionable error
            raise EmbeddingError(
                [f"encoder failed on rows {start}..{start + len(batch) - 1}: {exc!r}"]
            ) from exc
        if output.shape != (len(batch), config.dimension):
            raise EmbeddingError(
                [
                    f"encoder returned shape {output.shape} for rows "
                    f"{start}..{start + len(batch) - 1}, "
                    f"expected ({len(batch)}, {config.dimension})"
                ]
            )
        batches.append(output.astype(np.float64))
    raw = np.vstack(batches)
    if not np.all(np.isfinite(raw)):
        rows = sorted({int(r) for r in np.argwhere(~np.isfinite(raw))[:, 0]})
        raise EmbeddingError(
            [f"encoder produced NaN or infinite values for {[names[r] for r in rows][:5]}"]
        )
    norms = np.linalg.norm(raw, axis=1)
    zero = np.flatnonzero(norms <= np.finfo(np.float32).tiny)
    if zero.size:
        raise EmbeddingError(
            [f"encoder produced zero vectors for {[names[int(r)] for r in zero][:5]}"]
        )
    matrix = (raw / norms[:, None]).astype(np.float32)
    problems = matrix_problems(matrix, len(texts), config.dimension)
    if problems:
        raise EmbeddingError(problems)
    return matrix


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def matrix_bytes(matrix: Float32Matrix) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    return buffer.getvalue()


def build_manifest(
    matrix: Float32Matrix,
    chunk_ids: Sequence[str],
    collection: ChunkCollection,
    config: EmbeddingConfig,
    model: dict[str, Any],
    *,
    chunks_path: str,
) -> dict[str, Any]:
    """Deterministic artifact metadata (no timestamps or durations)."""
    return {
        "artifact_schema_version": EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        "embedding_config": config.semantic_dict(),
        "embedding_config_id": config.config_id,
        "embedding_config_sha256": config.config_sha256,
        "source_chunks": {
            "path": chunks_path,
            "sha256": collection.artifact_sha256,
            "chunk_count": len(collection.chunks),
            "chunking_config_id": collection.chunking_config_id,
        },
        "matrix": {
            "file": MATRIX_FILENAME,
            "sha256": hashlib.sha256(matrix_bytes(matrix)).hexdigest(),
            "shape": list(matrix.shape),
            "dtype": "float32",
            "normalization": "l2",
            "norm_tolerance": NORM_TOLERANCE,
        },
        "rows": list(chunk_ids),
        "model": model,
        "execution": {
            "device": config.device,
            "batch_size": config.batch_size,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": dependency_versions(),
        },
    }


@dataclass(frozen=True, slots=True)
class EmbeddingArtifact:
    matrix: Float32Matrix
    chunk_ids: tuple[str, ...]
    manifest: dict[str, Any] = field(repr=False)

    @property
    def config_id(self) -> str:
        return str(self.manifest["embedding_config_id"])

    @property
    def matrix_sha256(self) -> str:
        return str(self.manifest["matrix"]["sha256"])


_CONFIG_FIELDS = frozenset(
    {"model_id", "model_revision", "dimension", "max_seq_length", "dtype", "normalization"}
)
_MANIFEST_KEYS = frozenset(
    {
        "artifact_schema_version",
        "embedding_config",
        "embedding_config_id",
        "embedding_config_sha256",
        "source_chunks",
        "matrix",
        "rows",
        "model",
        "execution",
    }
)


def load_embedding_artifact(
    directory: Path,
    collection: ChunkCollection | None = None,
    config: EmbeddingConfig | None = None,
) -> EmbeddingArtifact:
    """Load and fully validate an artifact; optionally require a collection and config.

    Rejects missing or extra files, malformed or incompatible manifests, hash
    mismatches, wrong shape or dtype, non-finite or non-unit rows, duplicate
    or reordered row mappings, and artifacts built from another chunk
    collection (stale) or configuration.
    """
    if not directory.is_dir():
        raise EmbeddingError([f"embedding artifact not found: {directory.name}/"])
    present = {p.name for p in directory.iterdir()}
    if not present >= _ARTIFACT_FILES:
        raise EmbeddingError([f"incomplete artifact: missing {sorted(_ARTIFACT_FILES - present)}"])
    try:
        manifest = json.loads((directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingError([f"{MANIFEST_FILENAME} is not valid JSON: {exc}"]) from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        keys = sorted(manifest) if isinstance(manifest, dict) else type(manifest).__name__
        raise EmbeddingError([f"{MANIFEST_FILENAME} has unexpected structure: {keys}"])
    if manifest["artifact_schema_version"] != EMBEDDING_ARTIFACT_SCHEMA_VERSION:
        raise EmbeddingError(
            [f"unsupported artifact_schema_version {manifest['artifact_schema_version']!r}"]
        )
    problems: list[str] = []
    if not isinstance(manifest["embedding_config"], dict):
        raise EmbeddingError(["embedding_config must be an object"])
    try:
        stored = EmbeddingConfig(
            **{k: v for k, v in manifest["embedding_config"].items() if k in _CONFIG_FIELDS}
        )
    except (TypeError, ValueError) as exc:
        raise EmbeddingError([f"invalid embedding_config: {exc}"]) from exc
    if stored.semantic_dict() != manifest["embedding_config"]:
        problems.append("embedding_config does not match the current configuration schema")
    if manifest["embedding_config_id"] != stored.config_id:
        problems.append("embedding_config_id does not match embedding_config")
    if config is not None and stored.config_id != config.config_id:
        problems.append(
            f"artifact config {stored.config_id} != requested config {config.config_id}"
        )
    info = manifest["matrix"]
    data = (directory / MATRIX_FILENAME).read_bytes()
    if not isinstance(info, dict) or info.get("file") != MATRIX_FILENAME:
        raise EmbeddingError(["manifest matrix entry is malformed"])
    if not isinstance(info.get("sha256"), str) or not _SHA256_RE.fullmatch(info["sha256"]):
        problems.append("manifest matrix sha256 is malformed")
    elif hashlib.sha256(data).hexdigest() != info["sha256"]:
        problems.append(
            f"{MATRIX_FILENAME} SHA-256 does not match the manifest (corrupted or replaced)"
        )
    try:
        matrix = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise EmbeddingError(
            [*problems, f"{MATRIX_FILENAME} cannot be read as .npy: {exc}"]
        ) from exc
    rows = manifest["rows"]
    if not isinstance(rows, list) or not all(isinstance(r, str) for r in rows):
        raise EmbeddingError([*problems, "rows must be a list of chunk ids"])
    if len(set(rows)) != len(rows):
        problems.append("rows contain duplicate chunk ids")
    if info.get("shape") != [len(rows), stored.dimension] or info.get("dtype") != "float32":
        problems.append("manifest shape or dtype does not match rows and configuration")
    problems.extend(matrix_problems(matrix, len(rows), stored.dimension))
    source = manifest["source_chunks"]
    if not isinstance(source, dict) or source.get("chunk_count") != len(rows):
        problems.append("source chunk_count does not match the row mapping")
    if collection is not None:
        if not isinstance(source, dict) or source.get("sha256") != collection.artifact_sha256:
            problems.append(
                "artifact was built from a different chunk collection (stale); rebuild embeddings"
            )
        if rows != [c.chunk_id for c in collection.chunks]:
            problems.append("row-to-chunk mapping does not match the collection order")
    if problems:
        raise EmbeddingError(problems)
    return EmbeddingArtifact(matrix=matrix, chunk_ids=tuple(rows), manifest=manifest)


def write_embedding_artifact(
    directory: Path,
    matrix: Float32Matrix,
    manifest: dict[str, Any],
    collection: ChunkCollection,
    config: EmbeddingConfig,
) -> None:
    """Stage both files in a temporary directory, validate, then move into place."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        unexpected = {p.name for p in directory.iterdir()} - _ARTIFACT_FILES
        if not directory.is_dir() or unexpected:
            raise EmbeddingError(
                [
                    f"refusing to replace {directory.name}: it contains other files "
                    f"{sorted(unexpected)}"
                ]
            )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.", suffix=".tmp", dir=directory.parent)
    )
    backup: Path | None = None
    try:
        (staging / MATRIX_FILENAME).write_bytes(matrix_bytes(matrix))
        text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        (staging / MANIFEST_FILENAME).write_text(text, encoding="utf-8", newline="\n")
        load_embedding_artifact(staging, collection, config)  # validate before publishing
        if directory.exists():
            backup = directory.with_name(f".{directory.name}.old-{uuid.uuid4().hex[:8]}")
            directory.replace(backup)
        try:
            staging.replace(directory)
        except BaseException:
            if backup is not None:
                backup.replace(directory)
                backup = None
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Orchestration and CLI
# --------------------------------------------------------------------------- #


def _resolve(settings: Settings, value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else settings.project_root / path


def _relative(path: Path, root: Path) -> str:
    resolved, base = path.resolve(), root.resolve()
    return resolved.relative_to(base).as_posix() if resolved.is_relative_to(base) else path.name


def output_dir_problems(settings: Settings, directory: Path) -> list[str]:
    artifacts = settings.artifacts_dir.resolve()
    resolved = directory.resolve()
    if resolved == artifacts or not resolved.is_relative_to(artifacts):
        return ["--output-dir must be a subdirectory of artifacts/"]
    return []


def default_artifact_dir(settings: Settings, collection: ChunkCollection) -> Path:
    """``artifacts/embeddings/<chunking_config_id>``: one artifact per chunk collection."""
    return settings.artifacts_dir / EMBEDDINGS_DIRNAME / collection.chunking_config_id


def _report_path(settings: Settings, stem: str, collection: ChunkCollection) -> Path:
    return settings.reports_dir / f"{stem}.{collection.chunking_config_id}.json"


def load_collection(
    settings: Settings,
    *,
    collection: CollectionName | None = None,
    chunks: str | None = None,
) -> tuple[ChunkCollection, Path]:
    """Load an explicitly selected collection (by name, or by path) and verify it."""
    try:
        if chunks is not None:
            path = _resolve(settings, chunks, settings.processed_data_dir / CHUNKS_FILENAME)
            loaded = load_chunk_collection(
                path,
                settings.corpus_manifest_path,
                settings.processed_data_dir / PAGES_NORMALIZED_FILENAME,
            )
        elif collection is not None:
            path = collection_path(settings, collection)
            loaded = load_named_collection(settings, collection)
        else:
            raise EmbeddingError(["select a collection with --collection v1|v2 or --chunks PATH"])
    except RetrievalError as exc:
        raise EmbeddingError(exc.problems) from exc
    return loaded, path


@dataclass(frozen=True, slots=True)
class BuildOutcome:
    directory: Path
    matrix: Float32Matrix
    manifest: dict[str, Any] = field(repr=False)
    audit: TokenAudit = field(repr=False)
    report: dict[str, Any] = field(repr=False)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    publish_atomic(path, data.encode("utf-8"))


def run_audit(
    settings: Settings,
    encoder: TextEncoder,
    collection: ChunkCollection,
    chunks_path: Path,
) -> tuple[TokenAudit, dict[str, Any]]:
    """Audit every chunk; write ``reports/embedding_token_audit.<collection id>.json``."""
    started = time.perf_counter()
    audit = audit_token_lengths(collection.chunks, encoder)
    seconds = time.perf_counter() - started
    described = encoder.describe()
    report = {
        "report_type": "founderlens-embedding-token-audit",
        "chunk_artifact": {
            "path": _relative(chunks_path, settings.project_root),
            "sha256": collection.artifact_sha256,
            "chunking_config_id": collection.chunking_config_id,
        },
        "model": {
            key: described.get(key) for key in ("model_id", "model_revision", "tokenizer_class")
        },
        **audit.to_dict(),
        "operational": {
            "note": "Varies between runs; not part of the deterministic results.",
            "audit_seconds": round(seconds, 4),
        },
    }
    _write_report(_report_path(settings, AUDIT_REPORT_STEM, collection), report)
    return audit, report


def run_build(
    settings: Settings,
    encoder: TextEncoder,
    config: EmbeddingConfig,
    collection: ChunkCollection,
    chunks_path: Path,
    *,
    output_dir: str | None = None,
) -> BuildOutcome:
    """Audit, embed, validate and publish the artifact; write the build report."""
    directory = _resolve(settings, output_dir, default_artifact_dir(settings, collection))
    problems = output_dir_problems(settings, directory)
    if problems:
        raise EmbeddingError(problems)
    audit, audit_report = run_audit(settings, encoder, collection, chunks_path)
    if not audit.ok:
        report_name = _report_path(settings, AUDIT_REPORT_STEM, collection).name
        raise EmbeddingError(
            [
                f"{len(audit.overflows)} chunk(s) exceed the {audit.limit}-token limit; no "
                f"embeddings were generated (see reports/{report_name})"
            ]
            + [f"{o['chunk_id']}: {o['token_length']} tokens" for o in audit.overflows]
        )
    started = time.perf_counter()
    matrix = embed_texts(
        [c.text for c in collection.chunks],
        encoder,
        config,
        labels=[c.chunk_id for c in collection.chunks],
    )
    seconds = time.perf_counter() - started
    manifest = build_manifest(
        matrix,
        [c.chunk_id for c in collection.chunks],
        collection,
        config,
        encoder.describe(),
        chunks_path=_relative(chunks_path, settings.project_root),
    )
    write_embedding_artifact(directory, matrix, manifest, collection, config)
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    report = {
        "report_type": "founderlens-embedding-build-report",
        "artifact_directory": _relative(directory, settings.project_root),
        "embedding_config_id": manifest["embedding_config_id"],
        "chunking_config_id": collection.chunking_config_id,
        "source_chunk_sha256": collection.artifact_sha256,
        "matrix_sha256": manifest["matrix"]["sha256"],
        "matrix_bytes": (directory / MATRIX_FILENAME).stat().st_size,
        "manifest_bytes": (directory / MANIFEST_FILENAME).stat().st_size,
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "l2_norm": {
            "min": float(norms.min()),
            "max": float(norms.max()),
            "max_abs_deviation_from_1": float(np.abs(norms - 1).max()),
        },
        "token_audit": {
            k: audit_report[k]
            for k in ("token_length", "over_limit_count", "near_limit", "effective_max_seq_length")
        },
        "operational": {
            "note": "Varies between runs; not part of the deterministic artifact.",
            "embedding_seconds": round(seconds, 3),
        },
    }
    _write_report(_report_path(settings, BUILD_REPORT_STEM, collection), report)
    return BuildOutcome(directory, matrix, manifest, audit, report)


def _check_budget_identity(encoder: TextEncoder) -> None:
    """The tokenizer that enforces the V2 budget must be the one the configuration names."""
    budget = CHUNKING_V2_CONFIG.token_budget
    if budget is None:
        raise EmbeddingError(["chunking V2 has no token budget"])
    described = encoder.describe()
    actual = (
        described.get("model_id"),
        described.get("model_revision"),
        described.get("tokenizer_class"),
    )
    expected = (budget.model_id, budget.model_revision, budget.tokenizer)
    if actual != expected or encoder.max_seq_length != budget.max_tokens:
        raise EmbeddingError(
            [
                f"encoder {actual} with limit {encoder.max_seq_length} does not match the "
                f"chunking V2 token budget {expected} with limit {budget.max_tokens}"
            ]
        )


def run_build_collection(
    settings: Settings,
    encoder: TextEncoder,
    *,
    expected_document_count: int = EXPECTED_DOCUMENT_COUNT,
) -> ChunkingOutcome:
    """Build chunking V2 (token-budgeted) into its own file; V1 is never touched."""
    _check_budget_identity(encoder)
    try:
        return run_chunking(
            settings,
            config=CHUNKING_V2_CONFIG,
            output=collection_path(settings, "v2"),
            report_output=chunking_report_path(settings, "v2"),
            expected_document_count=expected_document_count,
            counter=encoder,
        )
    except ChunkingError as exc:
        raise EmbeddingError(exc.problems) from exc
    except ChunkingConfigError as exc:
        raise EmbeddingError([str(exc)]) from exc


def run_validate_collection(
    settings: Settings, name: CollectionName, encoder: TextEncoder | None
) -> ChunkCollection:
    """Full collection checks: spans, coverage, order, word budget, overlap, token budget."""
    collection, _ = load_collection(settings, collection=name)
    config = collection_config(name)
    counter = encoder if config.token_budget is not None else None
    try:
        manifest = load_manifest(settings.corpus_manifest_path)
        pages, _ = load_normalized_artifact(
            settings.processed_data_dir / PAGES_NORMALIZED_FILENAME, manifest
        )
        problems = validate_chunks(collection.chunks, pages, config, counter)
    except ChunkingError as exc:
        raise EmbeddingError(exc.problems) from exc
    except (ManifestError, ChunkingConfigError) as exc:
        raise EmbeddingError([str(exc)]) from exc
    if problems:
        raise EmbeddingError(problems)
    return collection


def _add_selection(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collection", choices=COLLECTION_NAMES, help="Named chunk collection.")
    group.add_argument("--chunks", default=None, help="Chunk artifact path (advanced).")


def _add_download(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow fetching the pinned model revision into the Hugging Face cache.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.retrieval.embeddings",
        description="Chunking V2, token-length audit, embedding generation and validation.",
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    commands = parser.add_subparsers(dest="command", required=True)
    build_collection = commands.add_parser(
        "build-collection", help="Build the token-budgeted chunk collection V2."
    )
    _add_download(build_collection)
    validate_collection = commands.add_parser(
        "validate-collection", help="Fully validate a named chunk collection."
    )
    validate_collection.add_argument("--collection", choices=COLLECTION_NAMES, required=True)
    _add_download(validate_collection)
    audit = commands.add_parser("audit", help="Measure every chunk's token length.")
    _add_selection(audit)
    _add_download(audit)
    build = commands.add_parser("build", help="Audit, embed all chunks, publish the artifact.")
    _add_selection(build)
    _add_download(build)
    build.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    build.add_argument("--output-dir", default=None, help="Default artifacts/embeddings/<id>.")
    validate = commands.add_parser("validate", help="Validate a stored embedding artifact.")
    _add_selection(validate)
    validate.add_argument("--output-dir", default=None, help="Default artifacts/embeddings/<id>.")
    return parser


def _run_command(args: argparse.Namespace, settings: Settings, config: EmbeddingConfig) -> int:
    root = settings.project_root
    if args.command == "validate":
        collection, _ = load_collection(settings, collection=args.collection, chunks=args.chunks)
        directory = _resolve(settings, args.output_dir, default_artifact_dir(settings, collection))
        artifact = load_embedding_artifact(directory, collection, config)
        print(
            f"Embedding artifact valid: {_relative(directory, root)} shape "
            f"{tuple(artifact.matrix.shape)} float32, config {artifact.config_id}, "
            f"matrix sha256 {artifact.matrix_sha256}, collection "
            f"{collection.chunking_config_id} sha256 {collection.artifact_sha256}"
        )
        return 0
    encoder = SentenceTransformerEncoder(config, allow_download=args.allow_download)
    if args.command == "build-collection":
        outcome = run_build_collection(settings, encoder)
        r = outcome.report
        tokens = r["diagnostics"]["token_budget"]
        print(
            f"Chunking V2 (config {r['chunking_config_id']}): {r['chunk_count']} chunks; tokens "
            f"max {tokens['chunk_token_count']['max']} / {tokens['max_tokens']}; "
            f"token-limited chunks {tokens['token_limited_chunks']['count']}"
        )
        print(f"Collection: {r['chunk_artifact']['path']}  sha256 {r['chunk_artifact']['sha256']}")
        print(f"Report: {_relative(outcome.report_path, root)}")
        return 0
    if args.command == "validate-collection":
        validated = run_validate_collection(settings, args.collection, encoder)
        print(
            f"Collection {args.collection} valid: {len(validated.chunks)} chunks, config "
            f"{validated.chunking_config_id}, sha256 {validated.artifact_sha256}"
        )
        return 0
    collection, chunks_path = load_collection(
        settings, collection=args.collection, chunks=args.chunks
    )
    if args.command == "audit":
        audit, report = run_audit(settings, encoder, collection, chunks_path)
        stats = report["token_length"]
        print(
            f"Token audit ({collection.chunking_config_id}): {report['chunks_inspected']} chunks, "
            f"limit {report['effective_max_seq_length']} tokens; min {stats['min']}, median "
            f"{stats['median']}, mean {stats['mean']}, p95 {stats.get('p95')}, max "
            f"{stats['max']}; near limit {report['near_limit']['count']}; over limit: "
            f"{report['over_limit_count']}"
        )
        for overflow in audit.overflows:
            print(f"  OVER {overflow['chunk_id']}: {overflow['token_length']} tokens")
        print(f"Report: {_relative(_report_path(settings, AUDIT_REPORT_STEM, collection), root)}")
        return 0 if audit.ok else 1
    built = run_build(
        settings, encoder, config, collection, chunks_path, output_dir=args.output_dir
    )
    r = built.report
    print(
        f"Embedded {r['shape'][0]} chunks -> {r['shape'][1]}-d float32 (config "
        f"{r['embedding_config_id']}); max |norm - 1| "
        f"{r['l2_norm']['max_abs_deviation_from_1']:.2e}"
    )
    print(f"Artifact: {r['artifact_directory']}  matrix sha256 {r['matrix_sha256']}")
    print(f"Source collection {r['chunking_config_id']} sha256 {r['source_chunk_sha256']}")
    print(f"Embedding time: {r['operational']['embedding_seconds']} s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = EmbeddingConfig(batch_size=getattr(args, "batch_size", DEFAULT_BATCH_SIZE))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        settings = load_settings(args.project_root)
        return _run_command(args, settings, config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except EmbeddingError as exc:
        print(f"Embedding {args.command} FAILED ({len(exc.problems)} problem(s)):", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        if args.command in {"build", "build-collection"}:
            print("Nothing was published; any previous artifact is unchanged.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Embedding {args.command} FAILED: cannot read or write: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
