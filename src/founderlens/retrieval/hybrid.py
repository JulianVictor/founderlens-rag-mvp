"""Hybrid retrieval: Reciprocal Rank Fusion of BM25 and dense rankings (Commit 4.1).

Usage::

    python -m founderlens.retrieval.hybrid --query "NVIDIA Blackwell"
    python -m founderlens.retrieval.hybrid --query "Google Cloud" --top-k 10 --json
    python -m founderlens.retrieval.hybrid --smoke   # reports/hybrid_smoke_report.<collection>.json

Why ranks, not scores. A BM25 score (unbounded, corpus-dependent) and a cosine
similarity (in [-1, 1]) are on different scales, so adding or comparing them is
meaningless without a calibration this project does not have. RRF uses only
each retriever's ranks.

Fixed configuration (fusion algorithm version 1). Both retrievers run on the
same validated V2 collection, each asked once per query for its top 10 with the
original query text. For each distinct chunk ID in the union::

    rrf(c) = sum over retrievers r containing c of 1 / (60 + rank_r(c))

Ranks are one-based; a retriever that did not return the chunk contributes 0;
both retrievers have weight 1. The union is sorted by descending RRF score and
exact ties keep collection order (manifest document, page, chunk index). The
output top-k (1-10, default 5) is cut afterwards, so top 5 is always the prefix
of top 10; candidate depth never changes with k. No threshold, no page
deduplication: distinct chunks on one page stay distinct.

With constant 60 and depth 10, a chunk in both lists scores at least
2 / 70 = 0.0286, more than any single-list chunk can (at most 1 / 61 = 0.0164),
so every shared candidate outranks every single-list candidate. Fused scores
are ranking scores, not probabilities.

Failures. A query without letters or digits returns no results and loads no
model. A dense query over the token limit is rejected, never truncated. An
empty BM25 ranking (no lexical match) is legitimate and the dense ranking is
fused alone; any component exception or invalid component ranking is raised,
never replaced by the other retriever's result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.ingestion.chunk import ChunkRecord
from founderlens.ingestion.extract import publish_atomic
from founderlens.retrieval.bm25 import DEFAULT_BM25_CONFIG, BM25Retriever
from founderlens.retrieval.collections import (
    CHUNKING_V2_CONFIG,
    COLLECTION_NAMES,
    load_named_collection,
)
from founderlens.retrieval.dense import (
    DenseRetriever,
    QueryTooLongError,
    has_content,
    load_dense_retriever,
)
from founderlens.retrieval.schema import ChunkCollection, RetrievalError, RetrievalResult

METHOD = "hybrid_rrf"
HYBRID_CONFIG_SCHEMA_VERSION = 1
FUSION_ALGORITHM_VERSION = 1
RRF_CONSTANT = 60
CANDIDATE_DEPTH = 10
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
COMPONENTS = ("bm25", "dense")  # fixed summation and report order
PREVIEW_CHARACTERS = 160

# The established smoke queries (written for smoke tests, not evaluation questions).
HYBRID_SMOKE_QUERIES: tuple[tuple[str, str], ...] = (
    ("company and product terminology", "NVIDIA Blackwell"),
    ("company and product terminology", "Google Cloud"),
    ("reporting period terminology", "fourth quarter fiscal 2025"),
    ("numeric or financial term", "diluted earnings per share"),
    ("natural-language paraphrase", "How quickly is the AI chip maker's main business growing?"),
    ("clearly unrelated question", "What is a good recipe for banana bread?"),
    ("empty input", ""),
    ("punctuation only", "$ % ( ) -"),
    ("oversized input", "revenue " * 300),
)


def check_top_k(top_k: object) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be an integer from 1 to {MAX_TOP_K}, got {top_k!r}")
    return top_k


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """The frozen fusion settings. ``top_k`` is an execution default, not identity."""

    rrf_constant: int = RRF_CONSTANT
    bm25_depth: int = CANDIDATE_DEPTH
    dense_depth: int = CANDIDATE_DEPTH
    top_k: int = DEFAULT_TOP_K

    def __post_init__(self) -> None:
        fixed = (self.rrf_constant, self.bm25_depth, self.dense_depth)
        if fixed != (RRF_CONSTANT, CANDIDATE_DEPTH, CANDIDATE_DEPTH):
            raise ValueError(
                f"the initial hybrid configuration is fixed: constant {RRF_CONSTANT}, depths "
                f"{CANDIDATE_DEPTH}/{CANDIDATE_DEPTH}; got {fixed}"
            )
        check_top_k(self.top_k)

    def semantic_dict(self, bm25_config_id: str, dense_config_id: str) -> dict[str, Any]:
        return {
            "schema_version": HYBRID_CONFIG_SCHEMA_VERSION,
            "method": METHOD,
            "fusion": "reciprocal rank fusion",
            "fusion_algorithm_version": FUSION_ALGORITHM_VERSION,
            "rrf_constant": self.rrf_constant,
            "candidate_depths": {"bm25": self.bm25_depth, "dense": self.dense_depth},
            "contributions": "equal (weight 1 per retriever)",
            "candidate_identity": "chunk_id",
            "tie_break": "collection order",
            "score_threshold": None,
            "page_deduplication": False,
            "components": {"bm25": bm25_config_id, "dense": dense_config_id},
        }

    def config_id(self, bm25_config_id: str, dense_config_id: str) -> str:
        canonical = json.dumps(
            self.semantic_dict(bm25_config_id, dense_config_id),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


DEFAULT_HYBRID_CONFIG = HybridConfig()


# --------------------------------------------------------------------------- #
# Pure fusion
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ComponentEvidence:
    """One retriever's view of a fused chunk: rank and raw score, or ``None`` if absent."""

    method: str
    rank: int | None
    score: float | None
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "contribution": self.contribution,
        }


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    chunk: ChunkRecord
    rrf_score: float
    components: tuple[ComponentEvidence, ...]


def ranking_problems(
    method: str,
    results: Sequence[RetrievalResult],
    depth: int,
    collection: Mapping[str, ChunkRecord],
) -> list[str]:
    """A component ranking must be original chunks, ranks 1..n, unique, finite, within depth."""
    problems = []
    if len(results) > depth:
        problems.append(f"{method}: {len(results)} results exceed the candidate depth {depth}")
    if [r.rank for r in results] != list(range(1, len(results) + 1)):
        problems.append(f"{method}: ranks are not 1..n in order")
    ids = [r.chunk.chunk_id for r in results]
    if len(set(ids)) != len(ids):
        problems.append(f"{method}: duplicate chunk ids in the ranking")
    for r in results:
        if r.method != method:
            problems.append(f"{method}: result labelled {r.method!r}")
        known = collection.get(r.chunk.chunk_id)
        if known is None:
            problems.append(f"{method}: unknown chunk {r.chunk.chunk_id}")
        elif known != r.chunk:
            problems.append(f"{method}: {r.chunk.chunk_id} differs from the collection record")
        if not math.isfinite(r.score):
            problems.append(f"{method}: {r.chunk.chunk_id} has a non-finite score")
    return problems


def rrf_fuse(
    rankings: Mapping[str, Sequence[RetrievalResult]],
    order: Mapping[str, int],
    *,
    constant: int = RRF_CONSTANT,
    methods: Sequence[str] = COMPONENTS,
) -> list[FusedCandidate]:
    """Fuse already-validated rankings; returns every candidate, best first.

    Contributions are summed in the fixed ``methods`` order with ``math.fsum``,
    so the score of a chunk does not depend on which retriever ran first.
    Raw component scores are recorded but never used.
    """
    positions: dict[str, dict[str, RetrievalResult]] = {
        method: {r.chunk.chunk_id: r for r in rankings.get(method, ())} for method in methods
    }
    chunks: dict[str, ChunkRecord] = {}
    for method in methods:
        for result in rankings.get(method, ()):
            chunks.setdefault(result.chunk.chunk_id, result.chunk)
    fused = []
    for chunk_id, chunk in chunks.items():
        evidence = []
        for method in methods:
            hit = positions[method].get(chunk_id)
            contribution = 0.0 if hit is None else 1.0 / (constant + hit.rank)
            evidence.append(
                ComponentEvidence(
                    method,
                    None if hit is None else hit.rank,
                    None if hit is None else hit.score,
                    contribution,
                )
            )
        score = math.fsum(e.contribution for e in evidence)
        if not math.isfinite(score) or score <= 0:
            raise RetrievalError([f"invalid RRF score {score!r} for {chunk_id}"])
        fused.append(FusedCandidate(chunk, score, tuple(evidence)))
    fused.sort(key=lambda c: (-c.rrf_score, order[c.chunk.chunk_id]))
    return fused


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HybridResult:
    """A fused result: the shared result type plus each retriever's rank and score."""

    result: RetrievalResult
    components: tuple[ComponentEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.result.to_dict(),
            "components": {e.method: e.to_dict() for e in self.components},
        }


@dataclass(frozen=True, slots=True)
class HybridRun:
    """One query: component rankings, fused results and timings (ms)."""

    query: str
    bm25: tuple[RetrievalResult, ...]
    dense: tuple[RetrievalResult, ...]
    fused: tuple[HybridResult, ...]
    timings_ms: dict[str, float]


def collection_problems(
    collection: ChunkCollection, bm25: BM25Retriever, dense: DenseRetriever
) -> list[str]:
    """Both retrievers must index exactly this V2 collection, in the same order."""
    problems = []
    if collection.chunking_config_id != CHUNKING_V2_CONFIG.config_id:
        problems.append(
            f"hybrid retrieval fuses collection v2 ({CHUNKING_V2_CONFIG.config_id}) only, "
            f"got {collection.chunking_config_id}"
        )
    if bm25.indexed_chunks != collection.chunks:
        problems.append("BM25 does not index exactly this collection in collection order")
    if (dense.collection.chunking_config_id, dense.collection.artifact_sha256) != (
        collection.chunking_config_id,
        collection.artifact_sha256,
    ):
        problems.append(
            f"dense runs on collection {dense.collection.chunking_config_id} "
            f"({dense.collection.artifact_sha256[:12]}), not {collection.chunking_config_id} "
            f"({collection.artifact_sha256[:12]})"
        )
    elif dense.collection.chunks != collection.chunks:
        problems.append("dense chunks differ from this collection or its order")
    return problems


class HybridRetriever:
    """RRF over a BM25 and a dense retriever that share one validated V2 collection."""

    def __init__(
        self,
        collection: ChunkCollection,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        config: HybridConfig = DEFAULT_HYBRID_CONFIG,
    ) -> None:
        problems = collection_problems(collection, bm25, dense)
        if problems:
            raise RetrievalError(problems)
        self.collection = collection
        self.bm25 = bm25
        self.dense = dense
        self.config = config
        self._chunks = {c.chunk_id: c for c in collection.chunks}
        self._order = {c.chunk_id: i for i, c in enumerate(collection.chunks)}

    @property
    def config_id(self) -> str:
        return self.config.config_id(self.bm25.config.config_id, self.dense.config.config_id)

    def identity(self) -> dict[str, Any]:
        dense = self.dense.identity()
        return {
            "method": METHOD,
            "hybrid_config_id": self.config_id,
            "bm25_config_id": self.bm25.config.config_id,
            "dense_config_id": dense["dense_config_id"],
            "embedding_config_id": dense["embedding_config_id"],
            "matrix_sha256": dense["matrix_sha256"],
            "model_id": dense["model_id"],
            "model_revision": dense["model_revision"],
            "collection_id": self.collection.chunking_config_id,
            "collection_sha256": self.collection.artifact_sha256,
            "rrf_constant": self.config.rrf_constant,
            "candidate_depths": {"bm25": self.config.bm25_depth, "dense": self.config.dense_depth},
            "chunks": len(self.collection.chunks),
        }

    def run(self, query: str, top_k: int | None = None) -> HybridRun:
        """Both component rankings (once each, depth 10), fused, cut to top k."""
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        k = self.config.top_k if top_k is None else check_top_k(top_k)
        if not has_content(query):
            return HybridRun(query, (), (), (), {})
        t0 = time.perf_counter()
        bm25 = self.bm25.search(query, top_k=self.config.bm25_depth)
        t1 = time.perf_counter()
        dense = self.dense.search(query, top_k=self.config.dense_depth)  # raises on failure
        t2 = time.perf_counter()
        problems = ranking_problems("bm25", bm25, self.config.bm25_depth, self._chunks)
        problems += ranking_problems("dense", dense, self.config.dense_depth, self._chunks)
        if problems:
            raise RetrievalError(problems)
        fused = rrf_fuse(
            {"bm25": bm25, "dense": dense}, self._order, constant=self.config.rrf_constant
        )
        results = tuple(
            HybridResult(RetrievalResult(METHOD, rank, c.rrf_score, c.chunk), c.components)
            for rank, c in enumerate(fused[:k], start=1)
        )
        t3 = time.perf_counter()
        timings = {
            "bm25_ms": (t1 - t0) * 1000,
            "dense_ms": (t2 - t1) * 1000,
            "fusion_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }
        return HybridRun(query, bm25, dense, results, timings)

    def search_detailed(self, query: str, top_k: int | None = None) -> tuple[HybridResult, ...]:
        return self.run(query, top_k).fused

    def search(self, query: str, top_k: int | None = None) -> tuple[RetrievalResult, ...]:
        """The shared result type only (for callers that treat retrievers alike)."""
        return tuple(h.result for h in self.run(query, top_k).fused)


def load_hybrid_retriever(
    settings: Settings,
    *,
    allow_download: bool = False,
    config: HybridConfig = DEFAULT_HYBRID_CONFIG,
) -> HybridRetriever:
    """Load V2 once for BM25 and validate the dense artifact independently against it."""
    collection = load_named_collection(settings, "v2")
    bm25 = BM25Retriever(collection.chunks, DEFAULT_BM25_CONFIG)
    dense = load_dense_retriever(settings, "v2", allow_download=allow_download)
    return HybridRetriever(collection, bm25, dense, config)


# --------------------------------------------------------------------------- #
# Smoke report and CLI
# --------------------------------------------------------------------------- #


def _ranking(results: Sequence[RetrievalResult]) -> list[dict[str, Any]]:
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


def _fused(results: Sequence[HybridResult]) -> list[dict[str, Any]]:
    return [
        {
            "rank": h.result.rank,
            "rrf_score": h.result.score,
            "chunk_id": h.result.chunk.chunk_id,
            "document_id": h.result.chunk.document_id,
            "page_number": h.result.chunk.page_number,
            "components": {e.method: e.to_dict() for e in h.components},
        }
        for h in results
    ]


def build_smoke_report(
    retriever: HybridRetriever,
    *,
    top_k: int = MAX_TOP_K,
    queries: Sequence[tuple[str, str]] = HYBRID_SMOKE_QUERIES,
) -> dict[str, Any]:
    """Every smoke query through the hybrid retriever. Timings go to ``operational``."""
    started = time.perf_counter()
    retriever.dense.encoder.describe()  # model load: cold start
    cold = time.perf_counter() - started
    entries: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for category, query in queries:
        entry: dict[str, Any] = {"category": category, "query": query[:80], "top_k": top_k}
        try:
            run = retriever.run(query, top_k)
        except QueryTooLongError as exc:
            entry.update(rejected=exc.problems[0], fused=[])
            entries.append(entry)
            continue
        bm25_ids = {r.chunk.chunk_id for r in run.bm25}
        dense_ids = {r.chunk.chunk_id for r in run.dense}
        entry.update(
            bm25=_ranking(run.bm25),
            dense=_ranking(run.dense),
            candidate_overlap=len(bm25_ids & dense_ids),
            candidate_union=len(bm25_ids | dense_ids),
            fused=_fused(run.fused),
        )
        entries.append(entry)
        if run.timings_ms:
            timings.append(
                {"query": query[:60], **{k: round(v, 3) for k, v in run.timings_ms.items()}}
            )
    warm = [t["total_ms"] for t in timings]
    return {
        "report_type": "founderlens-hybrid-smoke-report",
        "note": "Functional smoke test with queries written for smoke tests (not evaluation "
        "questions). Not a relevance judgement; retrieval quality is not measured here.",
        "hybrid": retriever.identity(),
        "hybrid_config": retriever.config.semantic_dict(
            retriever.bm25.config.config_id, retriever.dense.config.config_id
        ),
        "queries": entries,
        "operational": {
            "note": "Varies between runs and machines; not part of the deterministic results.",
            "cold_start_model_load_seconds": round(cold, 3),
            "warm_total_ms_median": round(statistics.median(warm), 3) if warm else None,
            "per_query": timings,
        },
    }


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARACTERS else flat[: PREVIEW_CHARACTERS - 3] + "..."


def _print_run(retriever: HybridRetriever, run: HybridRun) -> None:
    ident = retriever.identity()
    print(
        f"Hybrid RRF (config {ident['hybrid_config_id']}; constant {ident['rrf_constant']}; "
        f"depths bm25 {ident['candidate_depths']['bm25']} / dense "
        f"{ident['candidate_depths']['dense']}; components bm25 {ident['bm25_config_id']}, "
        f"dense {ident['dense_config_id']}, embedding {ident['embedding_config_id']}) over "
        f"collection {ident['collection_id']} (sha256 {ident['collection_sha256'][:12]})"
    )
    if not run.fused:
        print("No results: the query has no letters or digits, or no candidates were found.")
        return
    print("RRF scores are rank-fusion scores, not relevance probabilities.")
    for h in run.fused:
        c = h.result.chunk
        parts = " ".join(
            f"{e.method}#{e.rank if e.rank is not None else '-'}" for e in h.components
        )
        print(
            f"{h.result.rank:>3}. rrf {h.result.score:.5f}  [{parts}]  {c.document_id} "
            f"p{c.page_number}  {c.chunk_id}"
        )
        print(f"     {_preview(c.text)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.retrieval.hybrid",
        description="Hybrid retrieval: reciprocal rank fusion of BM25 and dense (both on V2).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="Query text (passed unchanged to both retrievers).")
    mode.add_argument("--smoke", action="store_true", help="Run the hybrid smoke queries.")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help=f"Results, 1-{MAX_TOP_K} (default 5)."
    )
    parser.add_argument(
        "--collection",
        choices=COLLECTION_NAMES,
        default="v2",
        help="Chunk collection (only v2 is supported).",
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
        config = HybridConfig(top_k=args.top_k)
    except ValueError as exc:
        parser.error(str(exc))
    if args.collection != "v2":
        print(
            "Hybrid retrieval FAILED: it fuses collection v2 only. Collection v1 has no "
            "embedding artifact (9 chunks exceed the model's 256-token limit), so BM25 v1 "
            "cannot be fused with dense v2. Use --collection v2.",
            file=sys.stderr,
        )
        return 1
    try:
        settings = load_settings(args.project_root)
        retriever = load_hybrid_retriever(
            settings, allow_download=args.allow_download, config=config
        )
        if args.smoke:
            report = build_smoke_report(retriever)
            name = f"hybrid_smoke_report.{retriever.collection.chunking_config_id}.json"
            path = settings.reports_dir / name
            data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            publish_atomic(path, data.encode("utf-8"))
            print(
                f"Hybrid smoke: {len(report['queries'])} queries; config "
                f"{report['hybrid']['hybrid_config_id']}; cold start "
                f"{report['operational']['cold_start_model_load_seconds']} s; report "
                f"{path.relative_to(settings.project_root).as_posix()}"
            )
            return 0
        run = retriever.run(args.query, config.top_k)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RetrievalError as exc:
        print(f"Hybrid retrieval FAILED ({len(exc.problems)} problem(s)):", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Hybrid retrieval FAILED: cannot read or write: {exc}", file=sys.stderr)
        return 1
    if args.json:
        payload = {
            "query": args.query,
            "top_k": config.top_k,
            **retriever.identity(),
            "hybrid_config": config.semantic_dict(
                retriever.bm25.config.config_id, retriever.dense.config.config_id
            ),
            "components": {"bm25": _ranking(run.bm25), "dense": _ranking(run.dense)},
            "results": [h.to_dict() for h in run.fused],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_run(retriever, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
