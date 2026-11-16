"""Named chunk collections, selected explicitly (never a mutable "latest").

- ``v1``: the frozen 120/24-word baseline (algorithm version 1),
  ``data/processed/chunks.jsonl``. Nine chunks exceed the embedding model's
  256-token limit, so it cannot be embedded; it stays the historical BM25
  baseline.
- ``v2``: the same words and overlap plus a hard 256-token budget in the
  pinned MiniLM tokenizer (algorithm version 2),
  ``data/processed/chunks.<config_id>.jsonl``. BM25 and dense retrieval are
  compared on this collection.

A collection is identified by its chunking configuration id (in every chunk
record and file name) and its artifact SHA-256 (recorded downstream). Loading
a named collection checks that its records carry the expected configuration id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from founderlens.config import Settings
from founderlens.ingestion.chunk import (
    CHUNKS_FILENAME,
    DEFAULT_CHUNKING_CONFIG,
    ChunkingConfig,
    TokenBudget,
)
from founderlens.ingestion.normalize import PAGES_NORMALIZED_FILENAME
from founderlens.retrieval.model import MAX_SEQ_LENGTH, MODEL_ID, MODEL_REVISION, TOKENIZER_CLASS
from founderlens.retrieval.schema import ChunkCollection, RetrievalError, load_chunk_collection

CollectionName = Literal["v1", "v2"]
COLLECTION_NAMES: tuple[CollectionName, ...] = ("v1", "v2")

CHUNKING_V2_CONFIG = ChunkingConfig(
    token_budget=TokenBudget(
        max_tokens=MAX_SEQ_LENGTH,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer=TOKENIZER_CLASS,
    )
)
COLLECTION_CONFIGS: dict[CollectionName, ChunkingConfig] = {
    "v1": DEFAULT_CHUNKING_CONFIG,
    "v2": CHUNKING_V2_CONFIG,
}


def collection_config(name: CollectionName) -> ChunkingConfig:
    return COLLECTION_CONFIGS[name]


def collection_path(settings: Settings, name: CollectionName) -> Path:
    if name == "v1":
        return settings.processed_data_dir / CHUNKS_FILENAME
    return settings.processed_data_dir / f"chunks.{collection_config(name).config_id}.jsonl"


def chunking_report_path(settings: Settings, name: CollectionName) -> Path:
    config_id = collection_config(name).config_id
    return settings.reports_dir / f"chunking_report.{config_id}.json"


def load_named_collection(settings: Settings, name: CollectionName) -> ChunkCollection:
    """Load and verify a collection, and require its records' chunking configuration id."""
    collection = load_chunk_collection(
        collection_path(settings, name),
        settings.corpus_manifest_path,
        settings.processed_data_dir / PAGES_NORMALIZED_FILENAME,
    )
    expected = collection_config(name).config_id
    if collection.chunking_config_id != expected:
        raise RetrievalError(
            [
                f"collection {name} must have chunking_config_id {expected}, "
                f"found {collection.chunking_config_id}"
            ]
        )
    return collection
