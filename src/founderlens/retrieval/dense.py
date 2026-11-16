"""Exact dense semantic retrieval over a validated embedding artifact (Commit 3.3).

Usage::

    python -m founderlens.retrieval.dense --query "How fast is the data center business growing?"
    python -m founderlens.retrieval.dense --query "Google Cloud" --top-k 10 --json
    python -m founderlens.retrieval.dense --smoke   # reports/dense_smoke_report.<collection>.json

Lifecycle. :func:`load_dense_retriever` loads the named chunk collection
(default V2) and its embedding artifact once, with every check of
:func:`~founderlens.retrieval.embeddings.load_embedding_artifact` (matrix and
source collection hashes, row order, configuration, model revision, shape,
dtype, finiteness, unit norms). Missing or invalid artifacts fail with the
command that rebuilds them; nothing is rebuilt or downloaded implicitly. The
document matrix is kept in memory (one float64 copy for scoring) and reused
for every query. The model loads lazily on the first non-empty query and is
reused afterwards.

Queries. A query with no Unicode letter or digit (empty, whitespace or
punctuation only) returns no results without touching the model. Any other
query is encoded exactly as given (no prefix, prompt, stemming or
punctuation removal) with the same pinned model, tokenizer, sequence limit,
normalization and dtype as the documents. Its token count (special tokens
included, no truncation or padding) must fit the 256-token limit; a longer
query is rejected with its count, never truncated. Dense retrieval has no
vocabulary, so words unknown to BM25 are fine.

Scoring. Rows and the query are unit vectors, so cosine similarity is the dot
product: ``scores = matrix @ query``, computed exhaustively in float64 (one
score per chunk, O(N x d); 164 x 384 here). Scores are raw cosine values in
[-1, 1]; unit norms are only exact to ``NORM_TOLERANCE`` (1e-5), so a score may
exceed +-1 by at most :data:`SCORE_BOUND_TOLERANCE`, and is reported unchanged.
Anything outside that bound, or non-finite, is an error. Scores are not
probabilities, are not normalized, and have no relevance threshold: the
nearest neighbours are returned even when none is relevant.

Ranking. Descending score; exact ties keep collection order (stable sort);
top-k defaults to 5 and returns every chunk when k exceeds the collection
size. No page or document deduplication (evaluation protocol v1.1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.ingestion.chunk import ChunkRecord
from founderlens.ingestion.extract import publish_atomic
from founderlens.retrieval.bm25 import DEFAULT_BM25_CONFIG as BM25_DEFAULTS
from founderlens.retrieval.bm25 import SMOKE_QUERIES, BM25Retriever
from founderlens.retrieval.collections import (
    COLLECTION_NAMES,
    CollectionName,
    load_named_collection,
)
from founderlens.retrieval.embeddings import (
    DEFAULT_EMBEDDING_CONFIG,
    NORM_TOLERANCE,
    EmbeddingArtifact,
    EmbeddingConfig,
    EmbeddingError,
    SentenceTransformerEncoder,
    TextEncoder,
    default_artifact_dir,
    embed_texts,
    load_embedding_artifact,
    matrix_problems,
)
from founderlens.retrieval.schema import ChunkCollection, RetrievalError, RetrievalResult

METHOD = "dense"
DENSE_CONFIG_SCHEMA_VERSION = 1
DEFAULT_TOP_K = 5
# |cos| <= |q| |d| <= (1 + 1e-5)^2 for vectors that are unit-norm within NORM_TOLERANCE.
SCORE_BOUND_TOLERANCE = 2 * NORM_TOLERANCE + NORM_TOLERANCE**2
PREVIEW_CHARACTERS = 160

# Paraphrased and unrelated queries written for this smoke test (not evaluation questions),
# added to the BM25 smoke queries so both retrievers are compared on the same inputs.
DENSE_SMOKE_QUERIES: tuple[tuple[str, str], ...] = (
    *SMOKE_QUERIES,
    ("natural-language paraphrase", "How quickly is the AI chip maker's main business growing?"),
    ("natural-language paraphrase", "money the search company spent on servers and data centers"),
    ("numeric or financial term", "diluted earnings per share"),
    ("clearly unrelated question", "What is a good recipe for banana bread?"),
    ("empty input", ""),
    ("oversized input", "revenue " * 300),
)


def _check_top_k(top_k: object) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
    return top_k


def has_content(query: str) -> bool:
    """A query needs at least one Unicode letter or digit to be encoded."""
    return any(ch.isalnum() for ch in query)


@dataclass(frozen=True, slots=True)
class DenseConfig:
    """Exact cosine retrieval settings. ``top_k`` is a default, not part of the identity."""

    top_k: int = DEFAULT_TOP_K
    similarity: Literal["cosine"] = "cosine"
    search: Literal["exact"] = "exact"

    def __post_init__(self) -> None:
        _check_top_k(self.top_k)
        similarity: object = self.similarity  # may come from untyped input
        search: object = self.search
        if similarity != "cosine" or search != "exact":
            raise ValueError("only exact cosine search is supported")

    def scoring_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DENSE_CONFIG_SCHEMA_VERSION,
            "method": METHOD,
            "similarity": "cosine = dot product of L2-normalized vectors (float64 accumulation)",
            "search": "exact exhaustive",
            "tie_break": "collection order (stable sort)",
            "query_policy": "exact text; no prefix or prompt; reject over the token limit",
            "empty_query": "no Unicode letter or digit -> no results",
            "score_threshold": None,
        }

    @property
    def config_id(self) -> str:
        """Retriever identity only; embedding, collection and matrix identities are separate."""
        canonical = json.dumps(self.scoring_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


DEFAULT_DENSE_CONFIG = DenseConfig()


class QueryTooLongError(RetrievalError):
    """The query exceeds the model's token limit and would be truncated."""


# --------------------------------------------------------------------------- #
# Pure scoring and ranking
# --------------------------------------------------------------------------- #


def cosine_scores(matrix: NDArray[np.float64], query: NDArray[Any]) -> NDArray[np.float64]:
    """One raw cosine score per row; both sides must already be unit vectors."""
    if matrix.ndim != 2 or query.shape != (matrix.shape[1],):
        raise RetrievalError(
            [f"query shape {query.shape} does not match matrix shape {matrix.shape}"]
        )
    scores: NDArray[np.float64] = matrix @ query.astype(np.float64)
    if not np.all(np.isfinite(scores)):
        raise RetrievalError(["cosine scores contain NaN or infinite values"])
    worst = float(np.max(np.abs(scores))) if scores.size else 0.0
    if worst > 1.0 + SCORE_BOUND_TOLERANCE:
        raise RetrievalError(
            [f"cosine score {worst} is outside [-1, 1] beyond tolerance {SCORE_BOUND_TOLERANCE}"]
        )
    return scores


def rank_indices(scores: NDArray[np.float64], top_k: int) -> list[int]:
    """Indices by descending score; exact ties keep the original (collection) order."""
    order = np.argsort(-scores, kind="stable")
    return [int(i) for i in order[: _check_top_k(top_k)]]


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #


def _artifact_problems(
    collection: ChunkCollection, artifact: EmbeddingArtifact, embedding: EmbeddingConfig
) -> list[str]:
    """Re-check the pairing even for artifacts not produced by the loader."""
    problems = []
    manifest = artifact.manifest
    if artifact.chunk_ids != tuple(c.chunk_id for c in collection.chunks):
        problems.append("embedding rows do not match the collection's chunk order")
    if manifest.get("source_chunks", {}).get("sha256") != collection.artifact_sha256:
        problems.append("embedding artifact was built from a different chunk collection")
    if artifact.config_id != embedding.config_id:
        problems.append(f"embedding config {artifact.config_id} != expected {embedding.config_id}")
    stored = manifest.get("embedding_config", {})
    if (stored.get("model_id"), stored.get("model_revision")) != (
        embedding.model_id,
        embedding.model_revision,
    ):
        problems.append("embedding artifact was built with a different model or revision")
    problems.extend(matrix_problems(artifact.matrix, len(collection.chunks), embedding.dimension))
    return problems


class DenseRetriever:
    """Exact cosine search over one validated (collection, embedding artifact) pair."""

    def __init__(
        self,
        collection: ChunkCollection,
        artifact: EmbeddingArtifact,
        encoder: TextEncoder,
        *,
        embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
        config: DenseConfig = DEFAULT_DENSE_CONFIG,
    ) -> None:
        problems = _artifact_problems(collection, artifact, embedding_config)
        if problems:
            raise RetrievalError(problems)
        self.collection = collection
        self.artifact = artifact
        self.encoder = encoder
        self.embedding_config = embedding_config
        self.config = config
        self._chunks: tuple[ChunkRecord, ...] = collection.chunks
        self._matrix: NDArray[np.float64] = artifact.matrix.astype(np.float64)
        self._encoder_checked = False

    def identity(self) -> dict[str, Any]:
        """Every identity a result depends on, kept separate."""
        return {
            "method": METHOD,
            "dense_config_id": self.config.config_id,
            "embedding_config_id": self.embedding_config.config_id,
            "model_id": self.embedding_config.model_id,
            "model_revision": self.embedding_config.model_revision,
            "collection_id": self.collection.chunking_config_id,
            "collection_sha256": self.collection.artifact_sha256,
            "matrix_sha256": self.artifact.matrix_sha256,
            "chunks": len(self._chunks),
        }

    def _check_encoder(self) -> None:
        """On first use: the query encoder must be the model the documents were embedded with."""
        if self._encoder_checked:
            return
        facts = self.encoder.describe()
        expected = (self.embedding_config.model_id, self.embedding_config.model_revision)
        actual = (facts.get("model_id"), facts.get("model_revision"))
        if actual != expected:
            raise RetrievalError(
                [f"query encoder is {actual}, but the documents were embedded with {expected}"]
            )
        self._encoder_checked = True

    def query_token_count(self, query: str) -> int:
        return self.encoder.token_lengths([query])[0]

    def encode_query(self, query: str) -> NDArray[np.float32] | None:
        """Validated unit query vector, or ``None`` for a query without letters or digits."""
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        if not has_content(query):
            return None
        self._check_encoder()
        limit = self.encoder.max_seq_length
        tokens = self.query_token_count(query)
        if tokens > limit:
            raise QueryTooLongError(
                [
                    f"query has {tokens} tokens (special tokens included); the model accepts "
                    f"at most {limit}. Shorten the query; it is never truncated."
                ]
            )
        try:
            matrix = embed_texts([query], self.encoder, self.embedding_config, labels=["query"])
        except EmbeddingError as exc:
            raise RetrievalError(exc.problems) from exc
        vector: NDArray[np.float32] = matrix[0]
        return vector

    def rank(
        self, query_vector: NDArray[Any], top_k: int | None = None
    ) -> tuple[RetrievalResult, ...]:
        """Score every chunk exactly and return the top k, ties in collection order."""
        k = self.config.top_k if top_k is None else _check_top_k(top_k)
        vector = np.asarray(query_vector)
        if vector.ndim != 1:
            raise RetrievalError([f"query vector must be 1-D, got shape {vector.shape}"])
        problems = matrix_problems(vector.reshape(1, -1), 1, self.embedding_config.dimension)
        if problems:
            raise RetrievalError([f"invalid query vector: {p}" for p in problems])
        scores = cosine_scores(self._matrix, vector)
        return tuple(
            RetrievalResult(METHOD, rank, float(scores[i]), self._chunks[i])
            for rank, i in enumerate(rank_indices(scores, k), start=1)
        )

    def search(self, query: str, top_k: int | None = None) -> tuple[RetrievalResult, ...]:
        k = self.config.top_k if top_k is None else _check_top_k(top_k)
        vector = self.encode_query(query)
        return () if vector is None else self.rank(vector, k)


def load_dense_retriever(
    settings: Settings,
    collection_name: CollectionName = "v2",
    *,
    encoder: TextEncoder | None = None,
    allow_download: bool = False,
    embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
    config: DenseConfig = DEFAULT_DENSE_CONFIG,
) -> DenseRetriever:
    """Load and validate the collection and its artifact once; never rebuild either."""
    collection = load_named_collection(settings, collection_name)
    directory = default_artifact_dir(settings, collection)
    try:
        artifact = load_embedding_artifact(directory, collection, embedding_config)
    except EmbeddingError as exc:
        remedy = (
            "collection v1 cannot be embedded (9 chunks exceed the 256-token limit); "
            "use --collection v2"
            if collection_name == "v1"
            else f"rebuild with: python -m founderlens.retrieval.embeddings build "
            f"--collection {collection_name}"
        )
        raise RetrievalError([*exc.problems, remedy]) from exc
    model = encoder or SentenceTransformerEncoder(embedding_config, allow_download=allow_download)
    return DenseRetriever(
        collection, artifact, model, embedding_config=embedding_config, config=config
    )


# --------------------------------------------------------------------------- #
# Smoke report and CLI
# --------------------------------------------------------------------------- #


def _hits(results: Sequence[RetrievalResult]) -> list[dict[str, Any]]:
    return [
        {
            "rank": r.rank,
            "score": r.score,
            "chunk_id": r.chunk.chunk_id,
            "document_id": r.chunk.document_id,
            "page_number": r.chunk.page_number,
        }
        for r in results
    ]


def build_smoke_report(
    retriever: DenseRetriever,
    *,
    top_k: int = DEFAULT_TOP_K,
    queries: Sequence[tuple[str, str]] = DENSE_SMOKE_QUERIES,
) -> dict[str, Any]:
    """Dense and BM25 on the same collection and queries. ``operational`` holds the timings."""
    started = time.perf_counter()
    retriever.encoder.describe()  # loads the model: cold start
    cold_start = time.perf_counter() - started
    bm25 = BM25Retriever(retriever.collection.chunks)
    entries: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for category, query in queries:
        entry: dict[str, Any] = {"category": category, "query": query, "top_k": top_k}
        timing: dict[str, Any] = {"query": query[:60]}
        t0 = time.perf_counter()
        try:
            vector = retriever.encode_query(query)
        except QueryTooLongError as exc:
            entry["dense"] = {"rejected": exc.problems[0], "results": []}
            entry["bm25"] = {"results": _hits(bm25.search(query, top_k))}
            entries.append(entry)
            timings.append(timing)
            continue
        t1 = time.perf_counter()
        dense = () if vector is None else retriever.rank(vector, top_k)
        t2 = time.perf_counter()
        timing.update(
            encode_ms=round((t1 - t0) * 1000, 3),
            score_and_rank_ms=round((t2 - t1) * 1000, 3),
            end_to_end_ms=round((t2 - t0) * 1000, 3),
        )
        lexical = bm25.search(query, top_k)
        dense_pages = {(r.chunk.document_id, r.chunk.page_number) for r in dense}
        lexical_pages = {(r.chunk.document_id, r.chunk.page_number) for r in lexical}
        entry["query_tokens"] = None if vector is None else retriever.query_token_count(query)
        entry["dense"] = {"results": _hits(dense)}
        entry["bm25"] = {"results": _hits(lexical)}
        entry["shared_pages_in_top_k"] = len(dense_pages & lexical_pages)
        entries.append(entry)
        timings.append(timing)
    warm = [t["encode_ms"] for t in timings if "encode_ms" in t and t["encode_ms"] > 0]
    return {
        "report_type": "founderlens-dense-smoke-report",
        "note": "Functional smoke test with queries written for this report (not evaluation "
        "questions). Both retrievers use the same collection. Results are not relevance "
        "judgements and do not measure retrieval quality.",
        "dense": retriever.identity(),
        "dense_config": retriever.config.scoring_dict(),
        "bm25_config_id": BM25_DEFAULTS.config_id,
        "queries": entries,
        "operational": {
            "note": "Varies between runs and machines; not part of the deterministic results.",
            "cold_start_model_load_seconds": round(cold_start, 3),
            "warm_query_encode_ms_median": round(statistics.median(warm), 3) if warm else None,
            "per_query": timings,
        },
    }


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARACTERS else flat[: PREVIEW_CHARACTERS - 3] + "..."


def _print_results(
    query: str, retriever: DenseRetriever, results: Sequence[RetrievalResult], tokens: int | None
) -> None:
    ident = retriever.identity()
    print(
        f"Dense (exact cosine, config {ident['dense_config_id']}; embedding "
        f"{ident['embedding_config_id']}, {ident['model_id']}@{ident['model_revision'][:12]}) "
        f"over collection {ident['collection_id']} ({ident['chunks']} chunks, sha256 "
        f"{ident['collection_sha256'][:12]})"
    )
    if tokens is None:
        print("No results: the query has no letters or digits.")
        return
    print(f"Query tokens: {tokens} (limit {retriever.encoder.max_seq_length})")
    print("Scores are raw cosine similarities, not relevance probabilities.")
    for r in results:
        c = r.chunk
        print(f"{r.rank:>3}. cosine {r.score:.4f}  {c.document_id} p{c.page_number}  {c.chunk_id}")
        print(f"     {_preview(c.text)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.retrieval.dense",
        description="Exact dense (cosine) retrieval over a validated embedding artifact.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="Query text (encoded exactly as given).")
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run the smoke queries with dense and BM25 on the same collection.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Results (default 5).")
    parser.add_argument(
        "--collection",
        choices=COLLECTION_NAMES,
        default="v2",
        help="Chunk collection (default v2; v1 has no embedding artifact).",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow fetching the pinned model revision into the Hugging Face cache.",
    )
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)
    try:
        config = DenseConfig(top_k=args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        settings = load_settings(args.project_root)
        retriever = load_dense_retriever(
            settings, args.collection, allow_download=args.allow_download, config=config
        )
        if args.smoke:
            report = build_smoke_report(retriever, top_k=config.top_k)
            path = (
                settings.reports_dir
                / f"dense_smoke_report.{retriever.collection.chunking_config_id}.json"
            )
            data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            publish_atomic(path, data.encode("utf-8"))
            print(
                f"Dense smoke: {len(report['queries'])} queries over collection "
                f"{report['dense']['collection_id']}; cold start "
                f"{report['operational']['cold_start_model_load_seconds']} s; report "
                f"{path.relative_to(settings.project_root).as_posix()}"
            )
            return 0
        vector = retriever.encode_query(args.query)
        tokens = None if vector is None else retriever.query_token_count(args.query)
        results = () if vector is None else retriever.rank(vector, config.top_k)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RetrievalError as exc:
        print(f"Dense retrieval FAILED ({len(exc.problems)} problem(s)):", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Dense retrieval FAILED: cannot read or write: {exc}", file=sys.stderr)
        return 1
    if args.json:
        payload = {
            "query": args.query,
            "query_tokens": tokens,
            **retriever.identity(),
            "dense_config": config.scoring_dict(),
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_results(args.query, retriever, results, tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
