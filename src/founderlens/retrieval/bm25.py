"""BM25 lexical retrieval baseline over the chunk collection (Commit 3.1).

Usage::

    python -m founderlens.retrieval.bm25 --query "NVIDIA Blackwell" --top-k 5
    python -m founderlens.retrieval.bm25 --query "Google Cloud" --json
    python -m founderlens.retrieval.bm25 --smoke   # writes reports/bm25_smoke_report.json

Scoring is ``rank_bm25.BM25Okapi`` (rank-bm25 0.2.2), unmodified. For a term in
``n`` of ``N`` indexed chunks it uses ``idf = ln((N - n + 0.5) / (n + 0.5))``;
terms in more than half of the chunks get a negative value, which the library
replaces by ``epsilon * mean(idf)``. A term in exactly half of the chunks has
``idf = 0``. A chunk's score is the sum over query tokens (duplicates count
again) of ``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avglen))``, with
lengths in lexical tokens (``founderlens.retrieval.tokenize``).

Ranking policy:

- A chunk **matches** when it contains at least one query token. Only matching
  chunks are returned; a chunk is never returned just because it is in the
  collection. Matching is decided by token presence, not by score sign, so a
  matching chunk with a zero or negative score is still returned (ranked last).
- Matches are sorted by descending score; equal scores keep collection order
  (manifest document order, then page, then chunk index).
- At most ``top_k`` results; fewer when fewer chunks match.
- An empty, punctuation-only or all-out-of-vocabulary query returns no
  results. That is not evidence that the corpus cannot answer the question.

Scores are ranking scores, not probabilities, and are not thresholded.
The index is built in memory once per retriever and reused by every query.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]  # rank-bm25 0.2.2 has no type information

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.ingestion.chunk import CHUNKS_FILENAME, DEFAULT_CHUNKING_CONFIG, ChunkRecord
from founderlens.ingestion.extract import publish_atomic
from founderlens.ingestion.normalize import PAGES_NORMALIZED_FILENAME
from founderlens.retrieval.collections import (
    COLLECTION_NAMES,
    CollectionName,
    collection_path,
    load_named_collection,
)
from founderlens.retrieval.schema import (
    ChunkCollection,
    RetrievalError,
    RetrievalResult,
    load_chunk_collection,
)
from founderlens.retrieval.tokenize import TOKENIZER_NAME, TOKENIZER_VERSION, tokenize

METHOD = "bm25"
ALGORITHM = "BM25Okapi"
LIBRARY = "rank-bm25"
SMOKE_REPORT_FILENAME = "bm25_smoke_report.json"
PREVIEW_CHARACTERS = 160

# Ordinary exploratory queries written for this smoke test; not taken from the
# evaluation dataset. They show behaviour, not measured retrieval quality.
SMOKE_QUERIES: tuple[tuple[str, str], ...] = (
    ("company and product terminology", "NVIDIA Blackwell"),
    ("company and product terminology", "Google Cloud"),
    ("financial metric terminology", "operating income"),
    ("financial metric terminology", "free cash flow"),
    ("reporting period terminology", "fourth quarter fiscal 2025"),
    ("reporting period terminology", "three months ended September 30, 2024"),
    ("multi-term question", "How did data center revenue change compared with a year ago?"),
    ("out of vocabulary", "zyzzyva quokka"),
    ("punctuation only", "$ % ( ) -"),
)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_top_k(top_k: object) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    return top_k


def library_version() -> str:
    return importlib.metadata.version(LIBRARY)


@dataclass(frozen=True, slots=True)
class BM25Config:
    """BM25Okapi parameters plus the default result count."""

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    top_k: int = 5

    def __post_init__(self) -> None:
        k1, b, epsilon = self.k1, self.b, self.epsilon
        if not _is_number(k1) or not math.isfinite(k1) or k1 <= 0:
            raise ValueError(f"k1 must be a finite positive number, got {k1!r}")
        if not _is_number(b) or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError(f"b must be a finite number between 0 and 1, got {b!r}")
        if not _is_number(epsilon) or not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError(f"epsilon must be a finite non-negative number, got {epsilon!r}")
        _check_top_k(self.top_k)

    def scoring_dict(self) -> dict[str, Any]:
        """Everything that affects scores. Hashed into :attr:`config_id`; excludes top_k."""
        return {
            "algorithm": ALGORITHM,
            "library": LIBRARY,
            "library_version": library_version(),
            "tokenizer": TOKENIZER_NAME,
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": float(self.k1),
            "b": float(self.b),
            "epsilon": float(self.epsilon),
        }

    @property
    def config_id(self) -> str:
        """First 12 hex characters of the SHA-256 of the canonical scoring settings.

        Identifies the scoring configuration only; index identity also needs
        the chunk artifact SHA-256.
        """
        canonical = json.dumps(self.scoring_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


DEFAULT_BM25_CONFIG = BM25Config()


class BM25Retriever:
    """In-memory BM25Okapi index over chunk text, built once and reused for every query."""

    def __init__(self, chunks: Sequence[ChunkRecord], config: BM25Config = DEFAULT_BM25_CONFIG):
        if not chunks:
            raise RetrievalError(["cannot build a BM25 index from an empty chunk collection"])
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise RetrievalError(["duplicate chunk_id in the chunk collection"])
        tokenized = [(chunk, tokenize(chunk.text)) for chunk in chunks]
        self.config = config
        self.collection_size = len(chunks)
        # A chunk with no lexical token can never match; it is left out of the
        # index (and of N and the mean length) and listed here instead.
        self.unindexed_chunk_ids: tuple[str, ...] = tuple(c.chunk_id for c, t in tokenized if not t)
        self._chunks: tuple[ChunkRecord, ...] = tuple(c for c, t in tokenized if t)
        documents = [list(t) for _, t in tokenized if t]
        if not documents:
            raise RetrievalError(["no chunk contains a lexical token; the vocabulary is empty"])
        self.token_counts: tuple[int, ...] = tuple(len(d) for d in documents)
        postings: dict[str, list[int]] = {}
        for index, document in enumerate(documents):
            for term in dict.fromkeys(document):
                postings.setdefault(term, []).append(index)
        self._postings = postings
        self._bm25 = BM25Okapi(documents, k1=config.k1, b=config.b, epsilon=config.epsilon)

    @property
    def indexed_chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def indexed_chunks(self) -> tuple[ChunkRecord, ...]:
        """The indexed chunk records, in collection order (read-only)."""
        return self._chunks

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def query_terms(self, query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """(query tokens, tokens absent from the index vocabulary)."""
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        terms = tokenize(query)
        return terms, tuple(t for t in dict.fromkeys(terms) if t not in self._postings)

    def search(self, query: str, top_k: int | None = None) -> tuple[RetrievalResult, ...]:
        """Matching chunks ranked by BM25 score; see the module docstring for the policy."""
        k = self.config.top_k if top_k is None else _check_top_k(top_k)
        terms, _ = self.query_terms(query)
        matched = sorted({i for t in set(terms) for i in self._postings.get(t, ())})
        if not matched:
            return ()
        scores = np.asarray(self._bm25.get_scores(list(terms)), dtype=np.float64)
        if scores.shape != (len(self._chunks),) or not np.all(np.isfinite(scores)):
            raise RetrievalError(["BM25 produced non-finite or misshaped scores"])
        ranked = sorted(matched, key=lambda i: (-float(scores[i]), i))[:k]
        return tuple(
            RetrievalResult(METHOD, rank, float(scores[i]), self._chunks[i])
            for rank, i in enumerate(ranked, start=1)
        )


# --------------------------------------------------------------------------- #
# Smoke report and CLI
# --------------------------------------------------------------------------- #


def _load(
    settings: Settings, chunks: str | None, name: CollectionName | None = None
) -> tuple[ChunkCollection, Path]:
    """A named collection (v1 or v2), an explicit path, or by default the v1 baseline."""
    if name is not None:
        return load_named_collection(settings, name), collection_path(settings, name)
    root = settings.project_root
    default = settings.processed_data_dir / CHUNKS_FILENAME
    path = default if chunks is None else Path(chunks)
    path = path if path.is_absolute() else root / path
    collection = load_chunk_collection(
        path, settings.corpus_manifest_path, settings.processed_data_dir / PAGES_NORMALIZED_FILENAME
    )
    return collection, path


def smoke_report_path(settings: Settings, collection: ChunkCollection) -> Path:
    """V1 keeps ``bm25_smoke_report.json``; other collections add their configuration id."""
    if collection.chunking_config_id == DEFAULT_CHUNKING_CONFIG.config_id:
        return settings.reports_dir / SMOKE_REPORT_FILENAME
    return settings.reports_dir / f"bm25_smoke_report.{collection.chunking_config_id}.json"


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(root.resolve()).as_posix()
        if resolved.is_relative_to(root.resolve())
        else path.name
    )


def index_summary(collection: ChunkCollection, retriever: BM25Retriever) -> dict[str, Any]:
    counts = retriever.token_counts
    return {
        "chunk_collection_size": retriever.collection_size,
        "indexed_chunks": retriever.indexed_chunk_count,
        "unindexed_chunk_ids": list(retriever.unindexed_chunk_ids),
        "vocabulary_size": retriever.vocabulary_size,
        "lexical_tokens_per_chunk": {
            "min": min(counts),
            "max": max(counts),
            "mean": round(statistics.fmean(counts), 3),
            "median": statistics.median(counts),
            "total": sum(counts),
        },
        "chunk_artifact_sha256": collection.artifact_sha256,
        "normalized_artifact_sha256": collection.normalized_sha256,
        "chunking_config_id": collection.chunking_config_id,
    }


def build_smoke_report(
    collection: ChunkCollection,
    retriever: BM25Retriever,
    *,
    chunks_path: str,
    build_seconds: float,
    queries: Sequence[tuple[str, str]] = SMOKE_QUERIES,
) -> dict[str, Any]:
    """Run the smoke queries. Everything except ``operational`` is deterministic."""
    config = retriever.config
    results: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    for category, query in queries:
        started = time.perf_counter()
        hits = retriever.search(query)
        elapsed_ms = (time.perf_counter() - started) * 1000
        terms, unknown = retriever.query_terms(query)
        results.append(
            {
                "category": category,
                "query": query,
                "query_tokens": list(terms),
                "out_of_vocabulary_tokens": list(unknown),
                "result_count": len(hits),
                "results": [
                    {
                        "rank": h.rank,
                        "score": h.score,
                        "chunk_id": h.chunk.chunk_id,
                        "document_id": h.chunk.document_id,
                        "page_number": h.chunk.page_number,
                    }
                    for h in hits
                ],
            }
        )
        latencies.append({"query": query, "search_ms": round(elapsed_ms, 3)})
    return {
        "report_type": "founderlens-bm25-smoke-report",
        "note": "Functional smoke test with exploratory queries written for this report, not "
        "evaluation questions. Results are not relevance judgements and do not measure "
        "retrieval accuracy.",
        "chunk_artifact_path": chunks_path,
        **index_summary(collection, retriever),
        "bm25_config": config.scoring_dict(),
        "bm25_config_id": config.config_id,
        "top_k": config.top_k,
        "ranking_policy": "matching = contains a query token; descending raw score; ties by "
        "collection order; no score threshold; no padding with non-matching chunks",
        "queries": results,
        "zero_result_queries": [r["query"] for r in results if r["result_count"] == 0],
        "operational": {
            "note": "Varies between runs; not part of the deterministic results.",
            "index_build_seconds": round(build_seconds, 4),
            "query_latency": latencies,
        },
    }


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARACTERS else flat[: PREVIEW_CHARACTERS - 3] + "..."


def _print_results(
    query: str,
    retriever: BM25Retriever,
    collection: ChunkCollection,
    hits: Sequence[RetrievalResult],
) -> None:
    c = retriever.config
    print(
        f"BM25 ({LIBRARY} {library_version()}, k1 {c.k1}, b {c.b}, epsilon {c.epsilon}; "
        f"config {c.config_id}) over {retriever.indexed_chunk_count} chunks "
        f"(chunks sha256 {collection.artifact_sha256[:12]})"
    )
    terms, unknown = retriever.query_terms(query)
    print(
        f"Query tokens: {', '.join(terms) or '(none)'}; "
        f"out of vocabulary: {', '.join(unknown) or 'none'}"
    )
    if not hits:
        print(
            "No results: no chunk contains any query token. This does not show that the "
            "corpus cannot answer the question."
        )
        return
    for h in hits:
        ch = h.chunk
        print(
            f"{h.rank:>3}. score {h.score:.4f}  {ch.document_id} p{ch.page_number}  {ch.chunk_id}"
        )
        print(f"     {_preview(ch.text)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.retrieval.bm25",
        description="BM25 lexical retrieval over the chunk collection (in-memory index).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="Query text.")
    mode.add_argument(
        "--smoke",
        action="store_true",
        help=f"Run the smoke queries; write reports/{SMOKE_REPORT_FILENAME}.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Results to return (default 5).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--collection",
        choices=COLLECTION_NAMES,
        default=None,
        help="Named chunk collection (default: the v1 baseline).",
    )
    source.add_argument(
        "--chunks", default=None, help=f"Chunk artifact (default data/processed/{CHUNKS_FILENAME})."
    )
    parser.add_argument("--k1", type=float, default=DEFAULT_BM25_CONFIG.k1)
    parser.add_argument("--b", type=float, default=DEFAULT_BM25_CONFIG.b)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_BM25_CONFIG.epsilon)
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)
    try:
        top_k = DEFAULT_BM25_CONFIG.top_k if args.top_k is None else args.top_k
        config = BM25Config(k1=args.k1, b=args.b, epsilon=args.epsilon, top_k=top_k)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        settings = load_settings(args.project_root)
        collection, chunks_path = _load(settings, args.chunks, args.collection)
        started = time.perf_counter()
        retriever = BM25Retriever(collection.chunks, config)
        build_seconds = time.perf_counter() - started
        if args.smoke:
            report = build_smoke_report(
                collection,
                retriever,
                chunks_path=_relative(chunks_path, settings.project_root),
                build_seconds=build_seconds,
            )
            report_path = smoke_report_path(settings, collection)
            data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            publish_atomic(report_path, data.encode("utf-8"))
            print(
                f"BM25 smoke: {len(report['queries'])} queries over "
                f"{report['indexed_chunks']} chunks; zero-result queries: "
                f"{len(report['zero_result_queries'])}; report "
                f"{_relative(report_path, settings.project_root)}"
            )
            return 0
        hits = retriever.search(args.query)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RetrievalError as exc:
        print(f"BM25 retrieval FAILED ({len(exc.problems)} problem(s)):", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"BM25 retrieval FAILED: cannot read or write: {exc}", file=sys.stderr)
        return 1
    if args.json:
        payload = {
            "query": args.query,
            "bm25_config": config.scoring_dict(),
            "bm25_config_id": config.config_id,
            "chunk_artifact_sha256": collection.artifact_sha256,
            "results": [h.to_dict() for h in hits],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_results(args.query, retriever, collection, hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
