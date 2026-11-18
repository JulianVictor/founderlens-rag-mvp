"""Frozen retrieval policy for the grounded generation pipeline (Commit 4.3).

Usage::

    python -m founderlens.retrieval.policy          # verify V2 and print the policy
    python -m founderlens.retrieval.policy --json

The decision and its evidence are recorded in ``docs/retrieval_strategy.md``:
BM25 over collection V2, ten ranked candidates per query, no score threshold,
no page deduplication and no fallback to dense or hybrid retrieval. Dense and
hybrid retrieval stay available as evaluated alternatives; they are not used
here.

:func:`load_policy_retriever` is the entry point the generation stage (roadmap
Stage 4) will call. It loads and validates V2 with the existing loader, checks
the collection identity,
the pinned artifact SHA-256 and the BM25 scoring identity, builds the existing
:class:`~founderlens.retrieval.bm25.BM25Retriever` once, and reuses it for
every query. It imports no embedding model, PyTorch or Sentence Transformers
code, reads no embedding matrix and no benchmark report, and never rebuilds an
artifact. Any mismatch raises :class:`RetrievalError`; nothing falls back.

Candidate top-k (10) is the number of ranked chunks handed to generation. It is
not the benchmark's primary k (5) and not the final LLM context, which generation
selects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.retrieval.bm25 import BM25Config, BM25Retriever
from founderlens.retrieval.collections import (
    COLLECTION_NAMES,
    CollectionName,
    collection_config,
    load_named_collection,
)
from founderlens.retrieval.schema import ChunkCollection, RetrievalError, RetrievalResult

POLICY_SCHEMA_VERSION = 1
# The collection V2 artifact the decision was evaluated on (chunks.0c2f87411627.jsonl).
# Verified on load; recorded separately from the policy identity.
V2_ARTIFACT_SHA256 = "f9a8f12a8d1c61aec3580f7f95dd7fd44a26860ff751d85d5a778693f59c55a8"
TIE_BREAKING = "descending raw score, then collection order (document, page, chunk index)"
MATCHING = "a chunk is returned only if it contains a query token; empty queries return nothing"


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """The selected retrieval path. Hashed into :attr:`policy_id`."""

    schema_version: int = POLICY_SCHEMA_VERSION
    method: Literal["bm25"] = "bm25"
    collection: CollectionName = "v2"
    collection_config_id: str = "0c2f87411627"
    bm25_config_id: str = "c21bdd0691f8"
    candidate_top_k: int = 10
    score_threshold: None = None
    fallback: None = None
    page_deduplication: bool = False

    def __post_init__(self) -> None:
        # Fields may come from untyped sources, so check them at runtime.
        problems = []
        method: object = self.method
        top_k: object = self.candidate_top_k
        threshold: object = self.score_threshold
        fallback: object = self.fallback
        if self.schema_version != POLICY_SCHEMA_VERSION:
            problems.append(f"unsupported policy schema_version {self.schema_version!r}")
        if method != "bm25":
            problems.append(f"method must be 'bm25' (the only selected path), got {method!r}")
        if self.collection not in COLLECTION_NAMES:
            problems.append(f"unknown collection {self.collection!r}")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            problems.append(f"candidate_top_k must be a positive integer, got {top_k!r}")
        if threshold is not None:
            problems.append("score_threshold must be None (BM25 scores are not thresholded)")
        if fallback is not None:
            problems.append("fallback must be None (no dense or hybrid fallback)")
        if self.page_deduplication is not False:
            problems.append("page_deduplication must be False at the retrieval layer")
        if problems:
            raise ValueError("; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        """The complete policy, including the fixed ranking invariants."""
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "collection": self.collection,
            "collection_config_id": self.collection_config_id,
            "bm25_config_id": self.bm25_config_id,
            "candidate_top_k": self.candidate_top_k,
            "score_threshold": self.score_threshold,
            "fallback": self.fallback,
            "page_deduplication": self.page_deduplication,
            "matching": MATCHING,
            "tie_breaking": TIE_BREAKING,
        }

    @property
    def policy_id(self) -> str:
        """First 12 hex characters of the SHA-256 of the canonical policy JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


SELECTED_POLICY = RetrievalPolicy()


class PolicyRetriever:
    """The selected BM25 retriever over a verified collection, built once and reused."""

    __slots__ = ("_collection", "_policy", "_retriever")

    def __init__(
        self, policy: RetrievalPolicy, collection: ChunkCollection, retriever: BM25Retriever
    ) -> None:
        self._policy = policy
        self._collection = collection
        self._retriever = retriever

    @property
    def policy(self) -> RetrievalPolicy:
        """The frozen policy this retriever was verified against (read-only)."""
        return self._policy

    @property
    def collection(self) -> ChunkCollection:
        """The verified collection (frozen; read-only)."""
        return self._collection

    def search(self, query: str) -> tuple[RetrievalResult, ...]:
        """Up to ``candidate_top_k`` ranked chunks with their original text and provenance."""
        return self._retriever.search(query, top_k=self.policy.candidate_top_k)

    def identities(self) -> dict[str, Any]:
        """The policy identity and, separately, the actual artifact and scoring identities."""
        config = self._retriever.config
        return {
            "policy_id": self.policy.policy_id,
            "policy": self.policy.to_dict(),
            "collection_config_id": self.collection.chunking_config_id,
            "collection_artifact_sha256": self.collection.artifact_sha256,
            "normalized_artifact_sha256": self.collection.normalized_sha256,
            "chunks": len(self.collection.chunks),
            "indexed_chunks": self._retriever.indexed_chunk_count,
            "bm25_config_id": config.config_id,
            "bm25_config": config.scoring_dict(),
        }


def load_policy_retriever(
    settings: Settings,
    policy: RetrievalPolicy = SELECTED_POLICY,
    *,
    expected_artifact_sha256: str = V2_ARTIFACT_SHA256,
) -> PolicyRetriever:
    """Load and verify the policy's collection and build its BM25 index once.

    Raises :class:`RetrievalError` with actionable reasons when the collection is
    missing or invalid, or when any identity differs from the policy. There is no
    fallback to another collection or retriever.
    """
    expected_config = collection_config(policy.collection).config_id
    if policy.collection_config_id != expected_config:
        raise RetrievalError(
            [
                f"policy expects collection {policy.collection} with chunking_config_id "
                f"{policy.collection_config_id}, but {policy.collection} is {expected_config}"
            ]
        )
    bm25_config = BM25Config(top_k=policy.candidate_top_k)
    if bm25_config.config_id != policy.bm25_config_id:
        raise RetrievalError(
            [
                f"BM25 scoring configuration is {bm25_config.config_id}, policy expects "
                f"{policy.bm25_config_id}; check the installed rank-bm25 version and tokenizer"
            ]
        )
    collection = load_named_collection(settings, policy.collection)
    if collection.artifact_sha256 != expected_artifact_sha256:
        raise RetrievalError(
            [
                f"collection {policy.collection} artifact sha256 {collection.artifact_sha256} "
                f"differs from the evaluated artifact {expected_artifact_sha256}; restore it "
                "(python -m founderlens.retrieval.embeddings build-collection rebuilds it "
                "byte-identically) instead of changing the policy"
            ]
        )
    return PolicyRetriever(policy, collection, BM25Retriever(collection.chunks, bm25_config))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.retrieval.policy",
        description="Verify the selected retrieval policy against the local artifacts.",
    )
    parser.add_argument("--json", action="store_true", help="Print the identities as JSON.")
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)
    try:
        retriever = load_policy_retriever(load_settings(args.project_root))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RetrievalError as exc:
        print(f"Retrieval policy check FAILED ({len(exc.problems)} problem(s)):", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Retrieval policy check FAILED: cannot read: {exc}", file=sys.stderr)
        return 1
    ids = retriever.identities()
    if args.json:
        print(json.dumps(ids, indent=2, sort_keys=True))
        return 0
    p = retriever.policy
    print(
        f"Retrieval policy {ids['policy_id']} OK: {p.method} on collection {p.collection} "
        f"({ids['collection_config_id']}, {ids['chunks']} chunks, sha256 "
        f"{ids['collection_artifact_sha256']}), BM25 config {ids['bm25_config_id']}, "
        f"top {p.candidate_top_k}, no threshold, no fallback, no page deduplication"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
