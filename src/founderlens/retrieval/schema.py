"""Retrieval result type and validated loading of a chunk collection.

Shared by the BM25 baseline and (later) dense retrieval, so both rank the
same chunk records and return the same result shape.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from founderlens.corpus.manifest import CorpusManifest, ManifestError, load_manifest
from founderlens.ingestion.chunk import (
    ChunkingError,
    ChunkRecord,
    ChunkRecordError,
    load_normalized_artifact,
    parse_chunks_jsonl,
    uncovered_characters,
)
from founderlens.ingestion.normalize import NormalizedPageRecord


class RetrievalError(RuntimeError):
    """Retrieval input or run is invalid. ``problems`` lists every actionable reason."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked chunk. ``score`` is the retriever's raw ranking score, not a probability."""

    method: str
    rank: int
    score: float
    chunk: ChunkRecord

    def __post_init__(self) -> None:
        # Values may come from untyped sources (e.g. numpy), so check types at runtime.
        rank: object = self.rank
        score: object = self.score
        if not self.method:
            raise ValueError("method must be non-empty")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("rank must be a positive integer (one-based)")
        if not isinstance(score, float) or not math.isfinite(score):
            raise ValueError("score must be a finite float")

    def to_dict(self) -> dict[str, Any]:
        c = self.chunk
        return {
            "method": self.method,
            "rank": self.rank,
            "score": self.score,
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "document_sha256": c.document_sha256,
            "page_id": c.page_id,
            "page_number": c.page_number,
            "company": c.company,
            "fiscal_period": c.fiscal_period,
            "source_url": c.source_url,
            "chunk_index": c.chunk_index,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "chunking_config_id": c.chunking_config_id,
            "text": c.text,
        }


@dataclass(frozen=True, slots=True)
class ChunkCollection:
    """A validated chunk artifact and the hash that identifies it."""

    chunks: tuple[ChunkRecord, ...]
    artifact_sha256: str
    chunking_config_id: str
    normalized_sha256: str


def collection_problems(
    chunks: Sequence[ChunkRecord],
    manifest: CorpusManifest,
    pages: Sequence[NormalizedPageRecord],
) -> list[str]:
    """Check identity, order and provenance of chunks against the manifest and source pages.

    Word budget and overlap are checked when the collection is built
    (``founderlens.ingestion.chunk.validate_chunks``); they need the chunking
    configuration, which the records carry only as an identifier.
    """
    if not chunks:
        return ["chunk collection is empty"]
    problems: list[str] = []
    duplicates = sorted(i for i, n in Counter(c.chunk_id for c in chunks).items() if n > 1)
    if duplicates:
        problems.append(f"duplicate chunk_id: {duplicates[:5]}")
    config_ids = sorted({c.chunking_config_id for c in chunks})
    if len(config_ids) != 1:
        problems.append(f"chunks come from several chunking configurations: {config_ids}")
    order = {d.document_id: i for i, d in enumerate(manifest.documents)}
    page_by_id = {p.page_id: p for p in pages}
    keys = []
    for chunk in chunks:
        entry = manifest.get(chunk.document_id)
        if entry is None:
            problems.append(f"{chunk.chunk_id}: document not in the corpus manifest")
            continue
        for name, expected in (
            ("document_sha256", entry.sha256),
            ("company", entry.company),
            ("fiscal_period", entry.fiscal_period),
            ("source_url", entry.source_url),
        ):
            if getattr(chunk, name) != expected:
                problems.append(f"{chunk.chunk_id}: {name} does not match the manifest")
        page = page_by_id.get(chunk.page_id)
        if page is None:
            problems.append(f"{chunk.chunk_id}: page not in the normalized artifact")
        elif page.normalized_text[chunk.start_char : chunk.end_char] != chunk.text:
            problems.append(f"{chunk.chunk_id}: text is not the recorded source span")
        keys.append((order[chunk.document_id], chunk.page_number, chunk.chunk_index))
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        problems.append("chunks are not in manifest document, page and chunk order")
    by_page: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_page.setdefault(chunk.page_id, []).append(chunk)
    for page in pages:
        page_chunks = by_page.get(page.page_id, [])
        if [c.chunk_index for c in page_chunks] != list(range(len(page_chunks))):
            problems.append(f"{page.page_id}: chunk_index values are not 0..n-1")
        if uncovered_characters(page.normalized_text, page_chunks):
            problems.append(f"{page.page_id}: chunks do not cover the page text")
    return problems


def load_chunk_collection(
    chunks_path: Path, manifest_path: Path, normalized_path: Path
) -> ChunkCollection:
    """Read ``chunks.jsonl`` and verify it against the manifest and the normalized pages."""
    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ManifestError) as exc:
        raise RetrievalError([f"cannot load corpus manifest: {exc}"]) from exc
    if not chunks_path.is_file():
        raise RetrievalError(
            [
                f"chunk artifact not found: {chunks_path.name}; "
                "run python -m founderlens.ingestion.chunk"
            ]
        )
    try:
        pages, normalized_sha256 = load_normalized_artifact(normalized_path, manifest)
    except ChunkingError as exc:
        raise RetrievalError(exc.problems) from exc
    data = chunks_path.read_bytes()
    try:
        chunks = parse_chunks_jsonl(data.decode("utf-8"))
    except (UnicodeDecodeError, ChunkRecordError) as exc:
        raise RetrievalError([f"chunk artifact {chunks_path.name} is invalid: {exc}"]) from exc
    problems = collection_problems(chunks, manifest, pages)
    if problems:
        raise RetrievalError(
            [f"chunk artifact {chunks_path.name}: {p}" for p in problems]
            + ["re-run python -m founderlens.ingestion.chunk"]
        )
    return ChunkCollection(
        chunks=tuple(chunks),
        artifact_sha256=hashlib.sha256(data).hexdigest(),
        chunking_config_id=chunks[0].chunking_config_id,
        normalized_sha256=normalized_sha256,
    )
