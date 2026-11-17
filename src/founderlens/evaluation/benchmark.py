"""Comparative retrieval benchmark on the development split (Commit 3.4).

Usage::

    python -m founderlens.evaluation.benchmark            # BM25 and dense, both on V2, dev split

Implements protocol v1.1 §3 as recorded in ``docs/retrieval_evaluation.md``.
Only ``evaluation/questions.dev.jsonl`` is read (its SHA-256 is checked
against ``evaluation/dataset.json``); the held-out file is never opened and
``--split test`` is refused. Both systems run on the explicitly selected V2
collection with the same queries, depth and metrics. Each system is asked once
per question for its top ``depth`` chunks; every k is a prefix of that list.

Writes ``reports/retrieval_benchmark.dev.json`` and ``.md``. Everything except
the ``operational`` section (timings, timestamps, machine, git state) is
deterministic for a fixed environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeGuard, runtime_checkable

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.corpus.manifest import ManifestError, load_manifest
from founderlens.evaluation import metrics as m
from founderlens.evaluation.schema import Category, EvalQuestion, parse_jsonl
from founderlens.evaluation.validate import DATASET_METADATA_FILENAME, SPLIT_FILES, check_question
from founderlens.ingestion.extract import publish_atomic
from founderlens.retrieval import hybrid as hy
from founderlens.retrieval.bm25 import DEFAULT_BM25_CONFIG, LIBRARY, BM25Retriever, library_version
from founderlens.retrieval.collections import CollectionName, load_named_collection
from founderlens.retrieval.dense import DenseRetriever, load_dense_retriever
from founderlens.retrieval.embeddings import dependency_versions
from founderlens.retrieval.hybrid import HybridRetriever, load_hybrid_retriever
from founderlens.retrieval.schema import ChunkCollection, RetrievalError, RetrievalResult
from founderlens.retrieval.tokenize import TOKENIZER_NAME, TOKENIZER_VERSION

EVALUATION_SCHEMA_VERSION = 1
DEV_SPLIT = "dev"
DEFAULT_KS = (1, 3, 5, 10)
PRIMARY_K = 5
MRR_CUTOFF = 10
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261116
COLLECTION: CollectionName = "v2"
WARMUP_QUERY = "quarterly revenue growth"  # written for warm-up; not an evaluation question
NOT_APPLICABLE = "not_applicable"
REPORT_STEM = "retrieval_benchmark"
RELEVANCE_METRICS = ("recall", "fact_coverage", "complete_coverage", "mrr")
HYBRID_METHOD = "hybrid"
THREE_WAY_REPORT_STEM = "retrieval_benchmark_three_way"
# Absolute tolerance (zero relative tolerance) for raw retrieval scores: hybrid components vs
# standalone retrievers, and the rerun baseline vs the historical two-method report.
SCORE_TOLERANCE = 1e-6


class EvaluationError(RuntimeError):
    """The benchmark cannot run or its inputs are invalid. ``problems`` says why."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Everything that defines the measurement. Hashed into :attr:`config_id`."""

    split: str = DEV_SPLIT
    ks: tuple[int, ...] = DEFAULT_KS
    primary_k: int = PRIMARY_K
    mrr_cutoff: int = MRR_CUTOFF
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = BOOTSTRAP_SEED
    collection: str = COLLECTION
    include_hybrid: bool = False

    def __post_init__(self) -> None:
        if self.split != DEV_SPLIT:
            raise ValueError(
                f"only the development split may be evaluated in this commit, got {self.split!r}; "
                "held-out evaluation is not authorized"
            )
        if not self.ks or tuple(sorted(set(self.ks))) != self.ks:
            raise ValueError("ks must be a non-empty, strictly increasing tuple")
        for k in (*self.ks, self.primary_k, self.mrr_cutoff):
            m.check_k(k)
        if self.primary_k not in self.ks:
            raise ValueError("primary_k must be one of ks")
        if self.mrr_cutoff > self.depth:
            raise ValueError("mrr_cutoff cannot exceed the retrieval depth")
        if self.collection != COLLECTION:
            raise ValueError("the controlled comparison uses collection v2 for every system")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")
        if self.include_hybrid and self.depth > hy.MAX_TOP_K:
            raise ValueError(f"hybrid returns at most {hy.MAX_TOP_K} results")

    @property
    def depth(self) -> int:
        """Chunks requested once per question; every k is a prefix of this list."""
        return max(self.ks)

    def to_dict(self) -> dict[str, Any]:
        """Without ``include_hybrid`` this is the historical two-method dictionary."""
        return {
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "metrics_version": m.METRICS_VERSION,
            "split": self.split,
            "collection": self.collection,
            "ks": list(self.ks),
            "primary_k": self.primary_k,
            "depth": self.depth,
            "mrr_cutoff": self.mrr_cutoff,
            "relevance_unit": "page (document_id, page_number)",
            "recall_definition": "hit rate: 1 if any accepted page is in the top k",
            "bootstrap": {
                "resamples": self.bootstrap_resamples,
                "seed": self.bootstrap_seed,
                "interval": "95% percentile, over questions",
            },
            "unanswerable": "excluded from relevance metrics; reported descriptively",
        } | (
            {
                "methods": ["bm25", "dense", HYBRID_METHOD],
                "hybrid_diagnostics": (
                    "candidate-pool coverage, fact outcomes,"
                    " exclusive-candidate displacement, evidence diversity"
                ),
            }
            if self.include_hybrid
            else {}
        )

    @property
    def config_id(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


DEFAULT_EVALUATION_CONFIG = EvaluationConfig()


# --------------------------------------------------------------------------- #
# Development split
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DevSplit:
    questions: tuple[EvalQuestion, ...]
    dataset_version: str
    dataset_sha256: str
    split_sha256: str


def load_dev_split(evaluation_dir: Path, manifest_path: Path) -> DevSplit:
    """Read and validate the development split only. The held-out file is never opened."""
    metadata_path = evaluation_dir / DATASET_METADATA_FILENAME
    dev_path = evaluation_dir / SPLIT_FILES[DEV_SPLIT]
    try:
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        dev_bytes = dev_path.read_bytes()
        manifest = load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        raise EvaluationError([f"cannot load the development split: {exc}"]) from exc
    problems: list[str] = []
    expected = metadata.get("files", {}).get(SPLIT_FILES[DEV_SPLIT], {})
    actual_sha = hashlib.sha256(dev_bytes).hexdigest()
    if expected.get("sha256") != actual_sha:
        problems.append(f"{dev_path.name} SHA-256 does not match {metadata_path.name}")
    questions, parse_problems = parse_jsonl(dev_bytes.decode("utf-8"), source=dev_path.name)
    problems.extend(parse_problems)
    if expected.get("question_count") != len(questions):
        problems.append("development question count does not match dataset.json")
    page_counts = {d.document_id: d.page_count for d in manifest.documents}
    for question in questions:
        if question.split != DEV_SPLIT:
            problems.append(f"{question.question_id} is not a development question")
        problems.extend(check_question(question, page_counts))
    if problems:
        raise EvaluationError(problems)
    return DevSplit(
        questions=tuple(questions),
        dataset_version=str(metadata.get("dataset_version")),
        dataset_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        split_sha256=actual_sha,
    )


def gold_evidence(question: EvalQuestion) -> m.GoldEvidence | None:
    """Accepted pages per required fact; ``None`` for unanswerable questions."""
    if not question.answerable:
        return None
    pages: dict[str, list[m.PageId]] = {f.fact_id: [] for f in question.required_facts}
    for ref in question.accepted_evidence:
        pages[ref.fact_id].append((ref.document_id, ref.page_number))
    return m.GoldEvidence.from_mapping(pages)


# --------------------------------------------------------------------------- #
# Systems under test
# --------------------------------------------------------------------------- #


class RetrievalSystem(Protocol):
    """What the benchmark needs from a retriever; a later hybrid adds one adapter."""

    @property
    def method(self) -> str: ...

    def identity(self) -> dict[str, Any]: ...

    def run(self, query: str, depth: int) -> tuple[tuple[RetrievalResult, ...], dict[str, float]]:
        """Ranked results and timings in milliseconds."""
        ...


class BM25System:
    method = "bm25"

    def __init__(self, retriever: BM25Retriever, collection: ChunkCollection) -> None:
        self.retriever = retriever
        self.collection = collection

    def identity(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "retriever_config_id": self.retriever.config.config_id,
            "retriever_config": {
                **self.retriever.config.scoring_dict(),
                "library": LIBRARY,
                "library_version": library_version(),
                "tokenizer": f"{TOKENIZER_NAME} v{TOKENIZER_VERSION}",
            },
            "collection_id": self.collection.chunking_config_id,
            "collection_sha256": self.collection.artifact_sha256,
            "embedding_config_id": NOT_APPLICABLE,
            "matrix_sha256": NOT_APPLICABLE,
            "model_id": NOT_APPLICABLE,
            "model_revision": NOT_APPLICABLE,
            "indexed_chunks": self.retriever.indexed_chunk_count,
        }

    def run(self, query: str, depth: int) -> tuple[tuple[RetrievalResult, ...], dict[str, float]]:
        started = time.perf_counter()
        results = self.retriever.search(query, top_k=depth)
        return results, {"total_ms": (time.perf_counter() - started) * 1000}


class DenseSystem:
    method = "dense"

    def __init__(self, retriever: DenseRetriever) -> None:
        self.retriever = retriever

    def identity(self) -> dict[str, Any]:
        ident = self.retriever.identity()
        return {
            "method": self.method,
            "retriever_config_id": ident["dense_config_id"],
            "retriever_config": self.retriever.config.scoring_dict(),
            "collection_id": ident["collection_id"],
            "collection_sha256": ident["collection_sha256"],
            "embedding_config_id": ident["embedding_config_id"],
            "matrix_sha256": ident["matrix_sha256"],
            "model_id": ident["model_id"],
            "model_revision": ident["model_revision"],
            "indexed_chunks": ident["chunks"],
        }

    def run(self, query: str, depth: int) -> tuple[tuple[RetrievalResult, ...], dict[str, float]]:
        started = time.perf_counter()
        vector = self.retriever.encode_query(query)
        encoded = time.perf_counter()
        results = () if vector is None else self.retriever.rank(vector, depth)
        done = time.perf_counter()
        return results, {
            "encode_ms": (encoded - started) * 1000,
            "score_and_rank_ms": (done - encoded) * 1000,
            "total_ms": (done - started) * 1000,
        }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def check_systems(systems: Sequence[RetrievalSystem], collection: ChunkCollection) -> None:
    """Every system must run on exactly the given collection (controlled comparison)."""
    problems = []
    methods = [s.method for s in systems]
    if len(set(methods)) != len(methods):
        problems.append(f"duplicate system methods: {methods}")
    for system in systems:
        ident = system.identity()
        if (ident["collection_id"], ident["collection_sha256"]) != (
            collection.chunking_config_id,
            collection.artifact_sha256,
        ):
            problems.append(
                f"{system.method} runs on collection {ident['collection_id']} "
                f"({str(ident['collection_sha256'])[:12]}), not {collection.chunking_config_id} "
                f"({collection.artifact_sha256[:12]})"
            )
    if problems:
        raise EvaluationError(problems)


def check_gold_reachable(split: DevSplit, collection: ChunkCollection) -> None:
    """Every accepted evidence page must have at least one chunk in the collection."""
    pages = {(c.document_id, c.page_number) for c in collection.chunks}
    missing = sorted(
        f"{q.question_id}:{ref.document_id}:p{ref.page_number}"
        for q in split.questions
        for ref in q.accepted_evidence
        if (ref.document_id, ref.page_number) not in pages
    )
    if missing:
        raise EvaluationError([f"evidence pages with no chunk in the collection: {missing}"])


def result_problems(
    results: Sequence[RetrievalResult], collection: Mapping[str, Any], depth: int
) -> list[str]:
    """Retrieved results must be original collection chunks with ranks 1..n."""
    problems = []
    if len(results) > depth:
        problems.append(f"{len(results)} results exceed the requested depth {depth}")
    if [r.rank for r in results] != list(range(1, len(results) + 1)):
        problems.append("ranks are not 1..n in order")
    ids = [r.chunk.chunk_id for r in results]
    if len(set(ids)) != len(ids):
        problems.append("duplicate chunk ids in results")
    for r in results:
        if collection.get(r.chunk.chunk_id) != r.chunk:
            problems.append(f"{r.chunk.chunk_id} is not an original chunk of the collection")
        if not math.isfinite(r.score):
            problems.append(f"{r.chunk.chunk_id} has a non-finite score")
    return problems


@dataclass(slots=True)
class BenchmarkRun:
    report: dict[str, Any]
    complete: bool
    timings: dict[str, Any] = field(default_factory=dict)


def _evaluate_question(
    question: EvalQuestion,
    systems: Sequence[RetrievalSystem],
    chunks: Mapping[str, Any],
    config: EvaluationConfig,
    timings: dict[str, list[dict[str, Any]]],
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold = gold_evidence(question)
    record: dict[str, Any] = {
        "question_id": question.question_id,
        "category": question.category.value,
        "answerable": question.answerable,
        "gold": None
        if gold is None
        else {fact: sorted(f"{d}:p{p}" for d, p in pages) for fact, pages in gold.facts},
        "systems": {},
    }
    for system in systems:
        entry: dict[str, Any] = {"status": "ok", "error": None}
        try:
            results, timing = system.run(question.question, config.depth)
        except Exception as exc:  # execution failure, recorded and never scored as a miss
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
            record["systems"][system.method] = entry
            continue
        problems = result_problems(results, chunks, config.depth)
        if problems:
            entry.update(status="error", error="invalid results: " + "; ".join(problems))
            record["systems"][system.method] = entry
            continue
        timings[system.method].append({"question_id": question.question_id, **timing})
        if raw is not None:
            raw[system.method] = results
            if isinstance(system, ReportsComponents):
                raw[f"{system.method}_components"] = dict(system.last_components)
        ranking: list[m.PageId] = [(r.chunk.document_id, r.chunk.page_number) for r in results]
        entry["returned"] = len(results)
        entry["results"] = [
            {
                "rank": r.rank,
                "chunk_id": r.chunk.chunk_id,
                "document_id": r.chunk.document_id,
                "page_number": r.chunk.page_number,
                "score": r.score,
            }
            for r in results
        ]
        if gold is None:
            entry["metrics"] = None
            entry["observation"] = {
                "top_score": results[0].score if results else None,
                "score_at_depth": results[-1].score if len(results) == config.depth else None,
            }
        else:
            primary = config.primary_k
            entry["metrics"] = m.question_metrics(
                ranking, gold, config.ks, mrr_cutoff=config.mrr_cutoff
            )
            entry["first_relevant_rank"] = m.first_relevant_rank(ranking, gold)
            entry["matched_pages_at_depth"] = sorted(
                f"{d}:p{p}" for d, p in m.pages_at_k(ranking, config.depth) & gold.pages
            )
            covered = set(m.covered_facts(ranking, gold, primary))
            entry[f"missing_facts_at_{primary}"] = [f for f, _ in gold.facts if f not in covered]
        record["systems"][system.method] = entry
    return record


def _metric_values(
    records: Sequence[dict[str, Any]], method: str, metric: str, key: str
) -> list[float | int | None]:
    return [r["systems"][method]["metrics"][metric][key] for r in records]


def _keys(metric: str, config: EvaluationConfig) -> list[str]:
    return [str(config.mrr_cutoff)] if metric == "mrr" else [str(k) for k in config.ks]


def _add_pair(
    paired: dict[str, Any],
    answerable: Sequence[dict[str, Any]],
    baseline: str,
    candidate: str,
    config: EvaluationConfig,
) -> None:
    """Candidate minus baseline on the same answerable questions (paired bootstrap)."""
    paired[f"{candidate}_minus_{baseline}"] = {
        name: {
            key: m.paired_comparison(
                _metric_values(answerable, baseline, name, key),
                _metric_values(answerable, candidate, name, key),
                resamples=config.bootstrap_resamples,
                seed=config.bootstrap_seed,
            )
            for key in _keys(name, config)
        }
        for name in RELEVANCE_METRICS
    }


def aggregate(
    records: Sequence[dict[str, Any]], methods: Sequence[str], config: EvaluationConfig
) -> dict[str, Any]:
    """Overall (with bootstrap intervals), per-category and paired results."""
    answerable = [r for r in records if r["answerable"]]
    metric_names = (*RELEVANCE_METRICS, "unique_pages", "duplicate_page_rate")
    overall: dict[str, Any] = {}
    by_category: dict[str, Any] = {}
    for method in methods:
        overall[method] = {
            name: {
                key: m.summarize(
                    _metric_values(answerable, method, name, key),
                    resamples=config.bootstrap_resamples,
                    seed=config.bootstrap_seed,
                )
                for key in _keys(name, config)
            }
            for name in metric_names
        }
        by_category[method] = {}
        for category in Category:
            subset = [r for r in answerable if r["category"] == category.value]
            if not subset:
                continue
            by_category[method][category.value] = {
                "n": len(subset),
                **{
                    name: {
                        key: m.summarize(_metric_values(subset, method, name, key))
                        for key in _keys(name, config)
                    }
                    for name in RELEVANCE_METRICS
                },
            }
    paired: dict[str, Any] = {}
    for index, baseline in enumerate(methods):
        for candidate in methods[index + 1 :]:
            _add_pair(paired, answerable, baseline, candidate, config)
    primary = str(config.primary_k)
    outcomes: dict[str, list[str]] = {"both_hit": [], "neither": []}
    if len(methods) >= 2:
        a, b = methods[0], methods[1]
        outcomes.update({f"{a}_only": [], f"{b}_only": []})
        for r in answerable:
            hit_a = r["systems"][a]["metrics"]["recall"][primary] == 1
            hit_b = r["systems"][b]["metrics"]["recall"][primary] == 1
            label = (
                "both_hit"
                if hit_a and hit_b
                else f"{a}_only"
                if hit_a
                else f"{b}_only"
                if hit_b
                else "neither"
            )
            outcomes[label].append(r["question_id"])
    unanswerable = {
        method: {
            "n": sum(1 for r in records if not r["answerable"]),
            "returned": [r["systems"][method]["returned"] for r in records if not r["answerable"]],
            "top_scores": [
                r["systems"][method]["observation"]["top_score"]
                for r in records
                if not r["answerable"]
            ],
        }
        for method in methods
    }
    return {
        "answerable_n": len(answerable),
        "overall": overall,
        "by_category": by_category,
        "paired": paired,
        f"recall_at_{primary}_outcomes": outcomes,
        "unanswerable": unanswerable,
    }


def run_benchmark(
    split: DevSplit,
    systems: Sequence[RetrievalSystem],
    collection: ChunkCollection,
    config: EvaluationConfig = DEFAULT_EVALUATION_CONFIG,
    *,
    load_timings_ms: Mapping[str, float] | None = None,
    historical: Mapping[str, Any] | None = None,
) -> BenchmarkRun:
    """Warm up, run every development question through every system, aggregate."""
    check_systems(systems, collection)
    check_gold_reachable(split, collection)
    chunks = {c.chunk_id: c for c in collection.chunks}
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    total = time.perf_counter()
    warmup: dict[str, float] = {}
    for system in systems:
        t0 = time.perf_counter()
        system.run(WARMUP_QUERY, config.depth)  # cold first query (model load for dense)
        warmup[system.method] = (time.perf_counter() - t0) * 1000
    timings: dict[str, list[dict[str, Any]]] = {s.method: [] for s in systems}
    raws: dict[str, dict[str, Any]] = {q.question_id: {} for q in split.questions}
    records = [
        _evaluate_question(q, systems, chunks, config, timings, raws[q.question_id])
        for q in split.questions
    ]
    failures = [
        f"{r['question_id']}/{method}: {entry['error']}"
        for r in records
        for method, entry in r["systems"].items()
        if entry["status"] != "ok"
    ]
    complete = not failures
    methods = [s.method for s in systems]
    identities = {s.method: s.identity() for s in systems}
    if config.include_hybrid and methods != ["bm25", "dense", HYBRID_METHOD]:
        raise EvaluationError(
            [f"the three-method experiment needs bm25, dense, hybrid; got {methods}"]
        )
    report: dict[str, Any] = {
        "report_type": "founderlens-retrieval-benchmark",
        "evaluation_config": DEFAULT_EVALUATION_CONFIG.to_dict()
        if config == DEFAULT_EVALUATION_CONFIG
        else config.to_dict(),
        "evaluation_config_id": experiment_id(config, identities),
        "dataset": {
            "dataset_version": split.dataset_version,
            "dataset_json_sha256": split.dataset_sha256,
            "split": DEV_SPLIT,
            "split_file": SPLIT_FILES[DEV_SPLIT],
            "split_sha256": split.split_sha256,
            "questions": len(split.questions),
            "answerable": sum(1 for q in split.questions if q.answerable),
            "unanswerable": sum(1 for q in split.questions if not q.answerable),
            "categories": {
                c.value: sum(1 for q in split.questions if q.category is c) for c in Category
            },
        },
        "collection": {
            "collection_id": collection.chunking_config_id,
            "collection_sha256": collection.artifact_sha256,
            "chunks": len(collection.chunks),
        },
        "systems": identities,
        "status": {"complete": complete, "failures": failures},
        "per_question": records,
        "aggregates": aggregate(records, methods, config) if complete else None,
        "operational": _operational(
            systems, timings, warmup, load_timings_ms or {}, started_at, total
        ),
    }
    if config.include_hybrid:
        _add_three_way(report, split, records, raws, collection, methods, config, historical)
    return BenchmarkRun(report=report, complete=complete, timings=report["operational"])


def _add_three_way(
    report: dict[str, Any],
    split: DevSplit,
    records: Sequence[dict[str, Any]],
    raws: Mapping[str, dict[str, Any]],
    collection: ChunkCollection,
    methods: Sequence[str],
    config: EvaluationConfig,
    historical: Mapping[str, Any] | None,
) -> None:
    """Hybrid diagnostics, baseline reproducibility and identities (three-method only)."""
    order = {c.chunk_id: i for i, c in enumerate(collection.chunks)}
    meta = _doc_meta(collection)
    # The two-method experiment with the same measurement settings (3fd7b205dc81 by default).
    report["historical_baseline_config_id"] = replace(config, include_hybrid=False).config_id
    if not report["status"]["complete"]:
        report["hybrid_diagnostics"] = None
        return
    diagnostics = {
        q.question_id: hybrid_question_diagnostics(
            gold_evidence(q),
            {"bm25": raws[q.question_id]["bm25"], "dense": raws[q.question_id]["dense"]},
            raws[q.question_id][f"{HYBRID_METHOD}_components"],
            raws[q.question_id][HYBRID_METHOD],
            order,
            meta,
            config,
        )
        for q in split.questions
    }
    report["hybrid_diagnostics"] = {
        "per_question": diagnostics,
        "aggregates": hybrid_aggregates(records, diagnostics, methods, config),
    }
    agreement = report["hybrid_diagnostics"]["aggregates"]["component_agreement"]["all_agree"]
    report["status"]["hybrid_components_agree"] = agreement
    report["baseline_reproducibility"] = (
        None if historical is None else compare_with_baseline(report, historical)
    )


def _latency(values: Sequence[float]) -> dict[str, float | int | None]:
    p50, p95 = m.percentile(values, 50), m.percentile(values, 95)
    return {
        "n": len(values),
        "p50_ms": None if p50 is None else round(p50, 3),
        "p95_ms": None if p95 is None else round(p95, 3),
        "max_ms": round(max(values), 3) if values else None,
    }


def _git_state(root: Path | None = None) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=root
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=root
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_clean": None}
    return {"commit": commit, "working_tree_clean": dirty == ""}


def _operational(
    systems: Sequence[RetrievalSystem],
    timings: Mapping[str, list[dict[str, Any]]],
    warmup: Mapping[str, float],
    load_timings_ms: Mapping[str, float],
    started_at: str,
    total_started: float,
) -> dict[str, Any]:
    torch = sys.modules.get("torch")
    latency: dict[str, Any] = {}
    for system in systems:
        rows = timings[system.method]
        latency[system.method] = {
            "load_ms": round(load_timings_ms.get(system.method, float("nan")), 3)
            if system.method in load_timings_ms
            else None,
            "cold_first_query_ms": round(warmup[system.method], 3),
            "warm_query": _latency([r["total_ms"] for r in rows]),
            **{
                part: _latency([r[part] for r in rows])
                for part in (rows[0] if rows else {})
                if part not in {"question_id", "total_ms"}
            },
        }
    return {
        "note": "Varies between runs and machines; excluded from deterministic comparisons.",
        "started_at_utc": started_at,
        "total_seconds": round(time.perf_counter() - total_started, 3),
        "latency": latency,
        "per_query_ms": {
            method: [
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                for r in rows
            ]
            for method, rows in timings.items()
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "device": "cpu",
            "torch_num_threads": int(torch.get_num_threads()) if torch is not None else None,
            "dependencies": {**dependency_versions(), LIBRARY: library_version()},
        },
        "git": _git_state(),
    }


def deterministic_view(report: Mapping[str, Any]) -> dict[str, Any]:
    """The report without volatile fields, for reproducibility comparisons."""
    return {k: v for k, v in report.items() if k != "operational"}


# --------------------------------------------------------------------------- #
# Three-method experiment: hybrid adapter and diagnostics (Commit 4.2)
# --------------------------------------------------------------------------- #


@runtime_checkable
class ReportsComponents(Protocol):
    """A system that exposes the component rankings behind its last result."""

    @property
    def last_components(self) -> Mapping[str, tuple[RetrievalResult, ...]]: ...


class HybridSystem:
    """The implemented RRF retriever, run for real; its component rankings are kept."""

    method = HYBRID_METHOD

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever
        self._components: dict[str, tuple[RetrievalResult, ...]] = {}

    @property
    def last_components(self) -> Mapping[str, tuple[RetrievalResult, ...]]:
        return self._components

    def identity(self) -> dict[str, Any]:
        ident = self.retriever.identity()
        return {
            "method": self.method,
            "retriever_config_id": ident["hybrid_config_id"],
            "retriever_config": self.retriever.config.semantic_dict(
                ident["bm25_config_id"], ident["dense_config_id"]
            ),
            "component_config_ids": {
                "bm25": ident["bm25_config_id"],
                "dense": ident["dense_config_id"],
            },
            "rrf_constant": ident["rrf_constant"],
            "candidate_depths": ident["candidate_depths"],
            "collection_id": ident["collection_id"],
            "collection_sha256": ident["collection_sha256"],
            "embedding_config_id": ident["embedding_config_id"],
            "matrix_sha256": ident["matrix_sha256"],
            "model_id": ident["model_id"],
            "model_revision": ident["model_revision"],
            "indexed_chunks": ident["chunks"],
        }

    def run(self, query: str, depth: int) -> tuple[tuple[RetrievalResult, ...], dict[str, float]]:
        started = time.perf_counter()
        run = self.retriever.run(query, depth)
        total = (time.perf_counter() - started) * 1000
        self._components = {"bm25": run.bm25, "dense": run.dense}
        timings = {k: v for k, v in run.timings_ms.items() if k != "total_ms"}
        return tuple(h.result for h in run.fused), {**timings, "total_ms": total}


def experiment_id(config: EvaluationConfig, identities: Mapping[str, Mapping[str, Any]]) -> str:
    """The two-method experiment keeps its historical id; the three-method one adds systems."""
    if not config.include_hybrid:
        return config.config_id
    payload = {
        **config.to_dict(),
        "retriever_config_ids": {m: i["retriever_config_id"] for m, i in identities.items()},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _pages(results: Sequence[RetrievalResult]) -> list[m.PageId]:
    return [(r.chunk.document_id, r.chunk.page_number) for r in results]


def _label(page: m.PageId) -> str:
    return f"{page[0]}:p{page[1]}"


def component_agreement(
    standalone: Mapping[str, Sequence[RetrievalResult]],
    components: Mapping[str, Sequence[RetrievalResult]],
    *,
    tolerance: float = SCORE_TOLERANCE,
) -> dict[str, Any]:
    """Do the hybrid's component rankings equal the standalone retrievers' rankings?"""
    out: dict[str, Any] = {}
    for method in ("bm25", "dense"):
        a, b = list(standalone.get(method, ())), list(components.get(method, ()))
        same_ids = [(r.chunk.chunk_id, r.rank) for r in a] == [
            (r.chunk.chunk_id, r.rank) for r in b
        ]
        diff = max((abs(x.score - y.score) for x, y in zip(a, b, strict=False)), default=0.0)
        out[method] = {
            "same_chunks_and_ranks": same_ids,
            "max_abs_score_difference": diff,
            "agrees": same_ids and diff <= tolerance,
        }
    return out


def _doc_meta(collection: ChunkCollection) -> dict[str, tuple[str, str]]:
    return {c.document_id: (c.company, c.fiscal_period) for c in collection.chunks}


def hybrid_question_diagnostics(
    gold: m.GoldEvidence | None,
    standalone: Mapping[str, tuple[RetrievalResult, ...]],
    components: Mapping[str, tuple[RetrievalResult, ...]],
    fused: Sequence[RetrievalResult],
    order: Mapping[str, int],
    meta: Mapping[str, tuple[str, str]],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Candidate pool, fact outcomes, exclusive-candidate suppression and diversity."""
    bm25, dense = components["bm25"], components["dense"]
    bm25_ids = {r.chunk.chunk_id for r in bm25}
    dense_ids = {r.chunk.chunk_id for r in dense}
    union = {r.chunk.chunk_id: r.chunk for r in (*bm25, *dense)}
    pool_pages = {(c.document_id, c.page_number) for c in union.values()}
    fused_pages = _pages(fused)
    record: dict[str, Any] = {
        "component_agreement": component_agreement(standalone, components),
        "pool": {
            "bm25_candidates": len(bm25_ids),
            "dense_candidates": len(dense_ids),
            "shared": len(bm25_ids & dense_ids),
            "bm25_exclusive": len(bm25_ids - dense_ids),
            "dense_exclusive": len(dense_ids - bm25_ids),
            "union": len(union),
            "union_pages": len(pool_pages),
        },
    }
    if gold is None:
        return record
    primary = config.primary_k
    covered_by_pool = set(m.facts_covered_by_pages(pool_pages, gold))
    record["pool"].update(
        relevant_pages_in_union=sorted(_label(p) for p in pool_pages & gold.pages),
        facts_covered_by_union=[f for f, _ in gold.facts if f in covered_by_pool],
        facts_missing_from_union=[f for f, _ in gold.facts if f not in covered_by_pool],
        union_fact_coverage=len(covered_by_pool) / len(gold.facts),
    )
    record["fact_outcomes"] = {
        str(k): m.classify_facts(fused_pages, pool_pages, gold, k) for k in config.ks
    }
    # Full fused order of the whole pool (the implemented RRF function), for positions > 10.
    full = hy.rrf_fuse({"bm25": bm25, "dense": dense}, order)
    position = {c.chunk.chunk_id: i for i, c in enumerate(full, start=1)}
    hybrid_top = set(m.covered_facts(fused_pages, gold, primary))
    hybrid_top_chunks = {r.chunk.chunk_id for r in fused[:primary]}
    suppression = []
    for comparator in ("bm25", "dense"):
        ranking = standalone[comparator][:primary]
        comparator_facts = set(m.covered_facts(_pages(ranking), gold, primary))
        for fact_id, accepted in gold.facts:
            if fact_id not in comparator_facts:
                continue
            for r in ranking:
                page = (r.chunk.document_id, r.chunk.page_number)
                if page not in accepted or r.chunk.chunk_id in hybrid_top_chunks:
                    continue
                cid = r.chunk.chunk_id
                shared = cid in bm25_ids and cid in dense_ids
                suppression.append(
                    {
                        "comparator": comparator,
                        "fact_id": fact_id,
                        "chunk_id": cid,
                        "page": _label(page),
                        "comparator_rank": r.rank,
                        "candidate_type": "shared" if shared else f"{comparator}_exclusive",
                        "fused_position": position.get(cid),
                        "shared_candidates_in_pool": record["pool"]["shared"],
                        "evidence_lost": fact_id not in hybrid_top,
                    }
                )
    record[f"displaced_relevant_chunks_at_{primary}"] = suppression
    diversity = {}
    required_companies = sorted({meta[d][0] for d, _ in gold.pages})
    required_periods = sorted({f"{meta[d][0]} {meta[d][1]}" for d, _ in gold.pages})
    for method, results in (*standalone.items(), (HYBRID_METHOD, tuple(fused))):
        top = results[:primary]
        hits = {(r.chunk.document_id, r.chunk.page_number) for r in top} & gold.pages
        evidence_companies = sorted({meta[d][0] for d, _ in hits})
        diversity[method] = {
            "documents": len({r.chunk.document_id for r in top}),
            "companies": len({r.chunk.company for r in top}),
            "companies_with_evidence": evidence_companies,
            "periods_with_evidence": sorted({f"{meta[d][0]} {meta[d][1]}" for d, _ in hits}),
        }
    record[f"diversity_at_{primary}"] = {
        "required_companies": required_companies,
        "required_company_periods": required_periods,
        "systems": diversity,
    }
    return record


def _pair_outcomes(
    answerable: Sequence[dict[str, Any]], a: str, b: str, key: str
) -> dict[str, list[str]]:
    outcome: dict[str, list[str]] = {
        "both_hit": [],
        f"{a}_only": [],
        f"{b}_only": [],
        "neither": [],
    }
    for r in answerable:
        hit_a = r["systems"][a]["metrics"]["recall"][key] == 1
        hit_b = r["systems"][b]["metrics"]["recall"][key] == 1
        label = (
            "both_hit"
            if hit_a and hit_b
            else f"{a}_only"
            if hit_a
            else f"{b}_only"
            if hit_b
            else "neither"
        )
        outcome[label].append(r["question_id"])
    return outcome


def hybrid_aggregates(
    records: Sequence[dict[str, Any]],
    diagnostics: Mapping[str, dict[str, Any]],
    methods: Sequence[str],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Three-way extras: pair outcomes, fact outcome counts, pool coverage, diversity."""
    answerable = [r for r in records if r["answerable"]]
    primary = str(config.primary_k)
    pairs = {
        f"{b}_vs_{a}": _pair_outcomes(answerable, b, a, primary)
        for i, a in enumerate(methods)
        for b in methods[i + 1 :]
    }
    fact_total = sum(len(r["gold"]) for r in answerable)
    outcome_counts = {
        str(k): {
            "facts": fact_total,
            **{
                o: sum(
                    1
                    for r in answerable
                    for v in diagnostics[r["question_id"]]["fact_outcomes"][str(k)].values()
                    if v == o
                )
                for o in m.FACT_OUTCOMES
            },
        }
        for k in config.ks
    }
    union_cov = [diagnostics[r["question_id"]]["pool"]["union_fact_coverage"] for r in answerable]
    union_micro = sum(
        len(diagnostics[r["question_id"]]["pool"]["facts_covered_by_union"]) for r in answerable
    )
    pools = [d["pool"] for d in diagnostics.values()]
    displaced = [
        {"question_id": qid, **item}
        for qid, d in diagnostics.items()
        for item in d.get(f"displaced_relevant_chunks_at_{primary}", [])
    ]
    single_evidence = {
        method: [
            r["question_id"]
            for r in answerable
            if len(diagnostics[r["question_id"]][f"diversity_at_{primary}"]["required_companies"])
            >= 2
            and len(
                diagnostics[r["question_id"]][f"diversity_at_{primary}"]["systems"][method][
                    "companies_with_evidence"
                ]
            )
            <= 1
        ]
        for method in methods
    }
    agreement = all(
        side["agrees"] for d in diagnostics.values() for side in d["component_agreement"].values()
    )
    max_diff = max(
        (
            side["max_abs_score_difference"]
            for d in diagnostics.values()
            for side in d["component_agreement"].values()
        ),
        default=0.0,
    )
    score_ranges = {}
    for method in methods:
        ans = [
            r["systems"][method]["results"][0]["score"]
            for r in answerable
            if r["systems"][method]["results"]
        ]
        una = [
            r["systems"][method]["results"][0]["score"]
            for r in records
            if not r["answerable"] and r["systems"][method]["results"]
        ]
        score_ranges[method] = {
            "answerable_top_score_range": [min(ans), max(ans)] if ans else None,
            "unanswerable_top_score_range": [min(una), max(una)] if una else None,
            "ranges_overlap": bool(ans and una and min(una) <= max(ans) and min(ans) <= max(una)),
            "unanswerable_unique_pages": [
                len({(x["document_id"], x["page_number"]) for x in r["systems"][method]["results"]})
                for r in records
                if not r["answerable"]
            ],
        }
    return {
        f"recall_at_{primary}_pair_outcomes": pairs,
        "fact_outcome_counts": {
            "definition": "fact-level micro counts over all required facts of answerable "
            "questions; mutually exclusive per fact at each k",
            "by_k": outcome_counts,
        },
        "candidate_pool": {
            "definition": "unordered union of the BM25 and dense top-10 candidates; a "
            "diagnostic evidence pool, not a ranked or deployable retriever",
            "union_fact_coverage_macro": m.summarize(union_cov),
            "union_facts_covered_micro": union_micro,
            "facts_total": fact_total,
            "mean_shared": statistics.fmean(p["shared"] for p in pools),
            "mean_union": statistics.fmean(p["union"] for p in pools),
            "min_union": min(p["union"] for p in pools),
            "max_union": max(p["union"] for p in pools),
        },
        f"displaced_relevant_chunks_at_{primary}": displaced,
        f"single_company_evidence_at_{primary}": single_evidence,
        "component_agreement": {"all_agree": agreement, "max_abs_score_difference": max_diff},
        "score_ranges": score_ranges,
    }


def _without_scores(entry: Mapping[str, Any]) -> dict[str, Any]:
    results = [{k: v for k, v in r.items() if k != "score"} for r in entry.get("results", [])]
    return {**entry, "results": results}


def _finite_score(value: object) -> TypeGuard[float]:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def compare_with_baseline(
    report: Mapping[str, Any],
    historical: Mapping[str, Any],
    *,
    tolerance: float = SCORE_TOLERANCE,
) -> dict[str, Any]:
    """Do the rerun BM25 and dense results reproduce the historical two-method report?

    Exact equality is required for the experiment, dataset, collection and retriever
    identities; for every per-question entry apart from raw scores (chunk IDs, ranks,
    document/page provenance, returned counts, metric values, applicability and
    denominators); and for the BM25/dense aggregates and their paired comparison. A ranking
    change therefore fails even when the scores are numerically close.

    Raw retrieval scores are compared at full precision (never rounded) with absolute
    tolerance ``tolerance`` (default :data:`SCORE_TOLERANCE`, the same 1e-6 used for
    component agreement) and zero relative tolerance. A missing or non-finite score fails.
    ``scores_identical`` records separately whether every raw score matched exactly.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance must be finite and >= 0, got {tolerance!r}")
    methods = ("bm25", "dense")
    differences: list[str] = []
    max_diff = 0.0
    identical = True
    expected_id = report.get("historical_baseline_config_id")
    if historical.get("evaluation_config_id") != expected_id:
        differences.append(
            f"historical report is experiment {historical.get('evaluation_config_id')!r},"
            f" expected {expected_id!r}"
        )
    for key in ("dataset", "collection"):
        if report[key] != historical.get(key):
            differences.append(f"{key} identity differs")
    for method in methods:
        if report["systems"][method] != historical.get("systems", {}).get(method):
            differences.append(f"{method} identity differs")
    old = {r["question_id"]: r for r in historical.get("per_question", [])}
    for record in report["per_question"]:
        qid = record["question_id"]
        before = old.get(qid)
        if before is None:
            differences.append(f"{qid} missing from the historical report")
            continue
        for method in methods:
            a, b = record["systems"][method], before["systems"][method]
            ids_a = [(x["chunk_id"], x["rank"]) for x in a.get("results", [])]
            ids_b = [(x["chunk_id"], x["rank"]) for x in b.get("results", [])]
            if ids_a != ids_b:
                differences.append(f"{qid}/{method}: chunks or ranks differ")
            if a.get("metrics") != b.get("metrics"):
                differences.append(f"{qid}/{method}: metrics differ")
            if _without_scores(a) != _without_scores(b):
                differences.append(f"{qid}/{method}: result entries differ")
            for x, y in zip(a.get("results", []), b.get("results", []), strict=False):
                sx, sy = x.get("score"), y.get("score")
                if _finite_score(sx) and _finite_score(sy):
                    if sx != sy:
                        identical = False
                        max_diff = max(max_diff, abs(sx - sy))
                else:
                    differences.append(f"{qid}/{method}: non-finite or missing score")
                    identical = False
    if max_diff > tolerance:
        differences.append(
            f"retrieval scores differ beyond tolerance {tolerance!r}"
            f" (max abs difference {max_diff!r})"
        )
    new_agg, old_agg = report.get("aggregates") or {}, historical.get("aggregates") or {}
    for method in methods:
        for section in ("overall", "by_category"):
            if new_agg.get(section, {}).get(method) != old_agg.get(section, {}).get(method):
                differences.append(f"{method} {section} aggregates differ")
    pair = "dense_minus_bm25"
    if new_agg.get("paired", {}).get(pair) != old_agg.get("paired", {}).get(pair):
        differences.append(f"{pair} paired comparison differs")
    return {
        "historical_evaluation_config_id": historical.get("evaluation_config_id"),
        "reproduced": not differences,
        "score_tolerance": {"absolute": tolerance, "relative": 0.0},
        "scores_identical": identical,
        "max_abs_score_difference": max_diff,
        "differences": differences,
    }


# --------------------------------------------------------------------------- #
# Markdown summary
# --------------------------------------------------------------------------- #


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(summary: Mapping[str, Any]) -> str:
    ci = summary.get("ci95")
    return "n/a" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


METRIC_LABELS = {
    "recall": "Recall@k (hit)",
    "fact_coverage": "Fact coverage@k",
    "complete_coverage": "Complete coverage@k (≥2 facts)",
    "mrr": "MRR@10",
}


def render_markdown(report: Mapping[str, Any]) -> str:
    cfg, data, agg = report["evaluation_config"], report["dataset"], report["aggregates"]
    systems = report["systems"]
    methods = list(systems)
    primary = str(cfg["primary_k"])
    lines = [
        "# Retrieval benchmark: development split",
        "",
        "Generated by `python -m founderlens.evaluation.benchmark`. Retrieval only; no answers",
        "are generated or judged. Page-level relevance is a proxy, not passage-level proof.",
        "",
        "## 1. Experimental design",
        "",
        f"- Evaluation config `{report['evaluation_config_id']}`; split `{cfg['split']}`;"
        f" k ∈ {cfg['ks']}, **primary k = {cfg['primary_k']}** (fixed before running);"
        f" depth {cfg['depth']}; MRR cut-off {cfg['mrr_cutoff']}.",
        f"- Collection `{report['collection']['collection_id']}`"
        f" ({report['collection']['chunks']} chunks) for every system.",
        f"- Bootstrap: {cfg['bootstrap']['resamples']} resamples, seed {cfg['bootstrap']['seed']}.",
        "",
        "## 2. Dataset",
        "",
        f"- Dataset {data['dataset_version']}, split file `{data['split_file']}`"
        f" (sha256 `{data['split_sha256'][:12]}…`): {data['questions']} questions,"
        f" {data['answerable']} answerable, {data['unanswerable']} unanswerable.",
        "- Categories: " + ", ".join(f"{k} {v}" for k, v in data["categories"].items() if v),
        "",
        "## 3. Systems",
        "",
        "| System | Retriever config | Collection | Embedding config | Matrix |",
        "|--------|------------------|------------|------------------|--------|",
    ]
    for method, ident in systems.items():
        lines.append(
            f"| {method} | `{ident['retriever_config_id']}` | `{ident['collection_id']}` |"
            f" `{ident['embedding_config_id']}` | `{str(ident['matrix_sha256'])[:12]}` |"
        )
    if agg is None:
        lines += ["", "**Benchmark incomplete:** " + "; ".join(report["status"]["failures"])]
        return "\n".join(lines) + "\n"
    n = agg["answerable_n"]
    lines += [
        "",
        f"## 4. Primary results (k = {primary}, {n} answerable questions)",
        "",
        "| Metric | "
        + " | ".join(f"{x} (95% CI)" for x in methods)
        + " | Δ "
        + " - ".join(reversed(methods[:2]))
        + " (95% CI) |",
        "|--------|" + "---|" * (len(methods) + 1),
    ]
    pair_key = next(iter(agg["paired"]), None)
    for metric in RELEVANCE_METRICS:
        key = str(cfg["mrr_cutoff"]) if metric == "mrr" else primary
        cells = []
        for method in methods:
            s = agg["overall"][method][metric][key]
            cells.append(f"{_fmt(s['mean'])} {_ci(s)} (n={s['n']})")
        delta = agg["paired"][pair_key][metric][key] if pair_key else None
        dcell = "n/a" if delta is None else f"{_fmt(delta['mean_difference'])} {_ci(delta)}"
        lines.append(f"| {METRIC_LABELS[metric]} | " + " | ".join(cells) + f" | {dcell} |")
    lines += [
        "",
        "## 5. All k",
        "",
        "| Metric | k | " + " | ".join(methods) + " |",
        "|--------|---|" + "---|" * len(methods),
    ]
    for metric in ("recall", "fact_coverage", "complete_coverage"):
        for k in cfg["ks"]:
            lines.append(
                f"| {METRIC_LABELS[metric]} | {k} | "
                + " | ".join(_fmt(agg["overall"][x][metric][str(k)]["mean"]) for x in methods)
                + " |"
            )
    lines += [
        "",
        "## 6. By category (point estimates; n is small)",
        "",
        "| Category | n | "
        + " | ".join(f"{x} R@{primary} / Cov@{primary} / MRR" for x in methods)
        + " |",
        "|----------|---|" + "---|" * len(methods),
    ]
    for category, stats in agg["by_category"][methods[0]].items():
        cells = []
        for method in methods:
            s = agg["by_category"][method][category]
            cells.append(
                f"{_fmt(s['recall'][primary]['mean'], 2)} / "
                f"{_fmt(s['fact_coverage'][primary]['mean'], 2)} / "
                f"{_fmt(s['mrr'][str(cfg['mrr_cutoff'])]['mean'], 2)}"
            )
        lines.append(f"| {category} | {stats['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## 7. Paired comparison", ""]
    for pair, metrics_ in agg["paired"].items():
        lines.append(f"`{pair}`:")
        lines.append("")
        lines.append(
            "| Metric | k | n | Δ mean | 95% CI | candidate better | baseline better | ties |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for metric, by_k in metrics_.items():
            for key, p in by_k.items():
                lines.append(
                    f"| {METRIC_LABELS[metric]} | {key} | {p['n']} | {_fmt(p['mean_difference'])}"
                    f" | {_ci(p)} | {p['candidate_better']} | {p['baseline_better']}"
                    f" | {p['ties']} |"
                )
        lines.append("")
    outcomes = agg[f"recall_at_{primary}_outcomes"]
    lines += [f"## 8. Recall@{primary} outcomes", ""]
    for label, ids in outcomes.items():
        lines.append(f"- {label}: {len(ids)} ({', '.join(ids) or '—'})")
    lines += ["", "## 9. Unanswerable questions (descriptive only)", ""]
    for method, u in agg["unanswerable"].items():
        scores = ", ".join(_fmt(s) for s in u["top_scores"])
        lines.append(f"- {method}: n={u['n']}, returned {u['returned']}, top scores [{scores}]")
    lat = report["operational"]["latency"]
    lines += [
        "",
        "## 10. Latency (this run; not deterministic)",
        "",
        "| System | load ms | cold first query ms | warm p50 ms | warm p95 ms |",
        "|---|---|---|---|---|",
    ]
    for method, latency in lat.items():
        lines.append(
            f"| {method} | {_fmt(latency['load_ms'], 1)}"
            f" | {_fmt(latency['cold_first_query_ms'], 1)}"
            f" | {_fmt(latency['warm_query']['p50_ms'], 2)}"
            f" | {_fmt(latency['warm_query']['p95_ms'], 2)} |"
        )
    lines += [
        "",
        "## 11. Limitations",
        "",
        f"- {n} answerable development questions; intervals are wide and category subsets tiny.",
        "- Page-level relevance: a hit means a retrieved chunk is on an accepted page, not that",
        "  it contains the fact. No answer quality is measured.",
        "- Development results only; nothing here was tuned, and held-out data was not used.",
        "",
        "Error analysis and recommendations: `docs/retrieval_evaluation.md`.",
    ]
    return "\n".join(lines) + "\n"


def render_three_way_markdown(report: Mapping[str, Any]) -> str:
    """Markdown summary of the three-method experiment (sections 1-16)."""
    cfg, data, agg = report["evaluation_config"], report["dataset"], report["aggregates"]
    methods = list(report["systems"])
    primary, cutoff = str(cfg["primary_k"]), str(cfg["mrr_cutoff"])
    lines = [
        "# Retrieval benchmark: BM25 vs dense vs hybrid RRF (development split)",
        "",
        "Generated by `python -m founderlens.evaluation.benchmark --include-hybrid`. Retrieval",
        "only; page-level relevance is a proxy, not passage-level proof. Not a strategy choice.",
        "",
        "## 1. Experimental design",
        "",
        f"- Experiment `{report['evaluation_config_id']}` (historical two-method baseline"
        f" `{report['historical_baseline_config_id']}`); split `{cfg['split']}`; k ∈ {cfg['ks']},"
        f" **primary k = {cfg['primary_k']}** (unchanged); depth {cfg['depth']}; MRR cut-off"
        f" {cfg['mrr_cutoff']}; bootstrap {cfg['bootstrap']['resamples']} resamples, seed"
        f" {cfg['bootstrap']['seed']}.",
        f"- All systems on collection `{report['collection']['collection_id']}`"
        f" ({report['collection']['chunks']} chunks); identical query text; one top-10 request"
        " per question and system.",
        "",
        "## 2. Dataset",
        "",
        f"- {data['questions']} development questions: {data['answerable']} answerable,"
        f" {data['unanswerable']} unanswerable; split sha256 `{data['split_sha256'][:12]}…`.",
        "",
        "## 3. Configuration and identities",
        "",
        "| System | Retriever config | Embedding config | Matrix |",
        "|---|---|---|---|",
    ]
    for method, ident in report["systems"].items():
        lines.append(
            f"| {method} | `{ident['retriever_config_id']}` | `{ident['embedding_config_id']}` |"
            f" `{str(ident['matrix_sha256'])[:12]}` |"
        )
    hybrid = report["systems"].get(HYBRID_METHOD, {})
    lines.append("")
    lines.append(
        f"Hybrid: RRF constant {hybrid.get('rrf_constant')}, depths"
        f" {hybrid.get('candidate_depths')}, equal weights, ties in collection order."
    )
    if agg is None:
        lines += ["", "**Benchmark incomplete:** " + "; ".join(report["status"]["failures"])]
        return "\n".join(lines) + "\n"
    base = report.get("baseline_reproducibility")
    lines += ["", "## 4. Baseline reproducibility", ""]
    if base is None:
        lines.append("- Historical report not available; not verified.")
    else:
        lines.append(
            f"- Against `{base['historical_evaluation_config_id']}`: reproduced ="
            f" **{base['reproduced']}**; raw scores identical {base['scores_identical']};"
            f" max abs score difference {base['max_abs_score_difference']} (absolute"
            f" tolerance {base['score_tolerance']['absolute']}, relative"
            f" {base['score_tolerance']['relative']}); differences:"
            f" {base['differences'] or 'none'}."
        )
        lines.append(
            "- Identities, chunk IDs, ranks, provenance and metrics must match exactly. The"
            " reference is the two-method report in the reports directory; on a fresh checkout"
            " it is regenerated locally and is not the original Commit 3.4 run."
        )
    n = agg["answerable_n"]
    lines += [
        "",
        f"## 5. Quality ({n} answerable questions)",
        "",
        "| Metric | k | " + " | ".join(methods) + " |",
        "|---|---|" + "---|" * len(methods),
    ]
    for metric in ("recall", "fact_coverage", "complete_coverage"):
        for k in cfg["ks"]:
            cells = []
            for method in methods:
                s = agg["overall"][method][metric][str(k)]
                cells.append(f"{_fmt(s['mean'])} {_ci(s)} (n={s['n']})")
            lines.append(f"| {METRIC_LABELS[metric]} | {k} | " + " | ".join(cells) + " |")
    cells = [
        f"{_fmt(agg['overall'][x]['mrr'][cutoff]['mean'])} {_ci(agg['overall'][x]['mrr'][cutoff])}"
        for x in methods
    ]
    lines.append("| MRR@10 | - | " + " | ".join(cells) + " |")
    for metric in ("unique_pages", "duplicate_page_rate"):
        for k in cfg["ks"]:
            lines.append(
                f"| {metric} | {k} | "
                + " | ".join(_fmt(agg["overall"][x][metric][str(k)]["mean"]) for x in methods)
                + " |"
            )
    lines += [
        "",
        "## 6. Paired differences (exploratory; three comparisons, n = 19)",
        "",
        "| Pair | Metric | k | Δ mean | 95% CI | first better | second better | ties | n |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pair, metrics_ in agg["paired"].items():
        for metric, by_k in metrics_.items():
            for key, p in by_k.items():
                lines.append(
                    f"| {pair} | {METRIC_LABELS[metric]} | {key} | {_fmt(p['mean_difference'])}"
                    f" | {_ci(p)} | {p['candidate_better']} | {p['baseline_better']}"
                    f" | {p['ties']} | {p['n']} |"
                )
    extra = report["hybrid_diagnostics"]["aggregates"]
    lines += ["", f"Recall@{primary} outcomes per pair:", ""]
    for pair, outcome in extra[f"recall_at_{primary}_pair_outcomes"].items():
        parts = ", ".join(
            f"{label} {len(ids)} ({', '.join(ids) or '—'})" for label, ids in outcome.items()
        )
        lines.append(f"- {pair}: {parts}")
    lines += [
        "",
        f"## 7. By category (k = {primary}; R / Cov / Complete / MRR; small n)",
        "",
        "| Category | n | " + " | ".join(methods) + " |",
        "|---|---|" + "---|" * len(methods),
    ]
    for category, stats in agg["by_category"][methods[0]].items():
        cells = []
        for method in methods:
            s = agg["by_category"][method][category]
            cells.append(
                f"{_fmt(s['recall'][primary]['mean'], 2)}"
                f" / {_fmt(s['fact_coverage'][primary]['mean'], 2)}"
                f" / {_fmt(s['complete_coverage'][primary]['mean'], 2)}"
                f" (n={s['complete_coverage'][primary]['n']})"
                f" / {_fmt(s['mrr'][cutoff]['mean'], 2)}"
            )
        lines.append(f"| {category} | {stats['n']} | " + " | ".join(cells) + " |")
    lines += ["", f"## 8. Per question (first relevant rank; facts missing at k = {primary})", ""]
    lines.append("| Question | Category | " + " | ".join(methods) + " |")
    lines.append("|---|---|" + "---|" * len(methods))
    for r in report["per_question"]:
        if not r["answerable"]:
            continue
        cells = []
        for method in methods:
            e = r["systems"][method]
            missing = len(e[f"missing_facts_at_{primary}"])
            cells.append(f"rank {e['first_relevant_rank']}, missing {missing}/{len(r['gold'])}")
        lines.append(f"| {r['question_id']} | {r['category']} | " + " | ".join(cells) + " |")
    pool = extra["candidate_pool"]
    lines += [
        "",
        "## 9. Candidate-pool coverage (diagnostic, not a retriever)",
        "",
        f"- Union of BM25 and dense top-10: mean {pool['mean_union']:.2f} chunks (min"
        f" {pool['min_union']}, max {pool['max_union']}); mean shared {pool['mean_shared']:.2f}.",
        f"- Union fact coverage (macro) {_fmt(pool['union_fact_coverage_macro']['mean'])}"
        f" (n={pool['union_fact_coverage_macro']['n']}); facts covered (micro)"
        f" {pool['union_facts_covered_micro']}/{pool['facts_total']}.",
        "",
        "## 10. Fact outcomes for hybrid (micro counts over all required facts)",
        "",
        "| k | facts | retained | candidate retrieval failure | fusion/cut-off failure |",
        "|---|---|---|---|---|",
    ]
    for k, c in extra["fact_outcome_counts"]["by_k"].items():
        lines.append(
            f"| {k} | {c['facts']} | {c[m.RETAINED]} | {c[m.CANDIDATE_RETRIEVAL_FAILURE]}"
            f" | {c[m.FUSION_CUTOFF_FAILURE]} |"
        )
    displaced = extra[f"displaced_relevant_chunks_at_{primary}"]
    lost = [d for d in displaced if d["evidence_lost"]]
    lines += [
        "",
        f"## 11. Relevant chunks in a baseline top {primary} but not in the hybrid top {primary}",
        "",
        f"- {len(displaced)} displaced relevant chunks; {len(lost)} with page-level evidence lost"
        " (the fact is not covered by any hybrid top-k chunk).",
    ]
    for d in lost:
        lines.append(
            f"  - {d['question_id']} {d['fact_id']} via {d['comparator']} rank"
            f" {d['comparator_rank']} ({d['candidate_type']}), fused position {d['fused_position']}"
        )
    lines += ["", f"## 12. Evidence from one company only at k = {primary} (≥ 2 required)", ""]
    for method, ids in extra[f"single_company_evidence_at_{primary}"].items():
        lines.append(f"- {method}: {len(ids)} ({', '.join(ids) or '—'})")
    lines += ["", "## 13. Unanswerable questions (descriptive; scales differ per method)", ""]
    for method, s in extra["score_ranges"].items():
        lines.append(
            f"- {method}: answerable top-score range {s['answerable_top_score_range']},"
            f" unanswerable {s['unanswerable_top_score_range']}, overlap {s['ranges_overlap']},"
            f" unique pages {s['unanswerable_unique_pages']}"
        )
    lat = report["operational"]["latency"]
    lines += [
        "",
        "## 14. Latency (same run; not deterministic)",
        "",
        "| System | load ms | cold first query ms | warm p50 ms | warm p95 ms | parts (p50 ms) |",
        "|---|---|---|---|---|---|",
    ]
    for method, latency in lat.items():
        parts = ", ".join(
            f"{name} {_fmt(v['p50_ms'], 2)}"
            for name, v in latency.items()
            if name.endswith("_ms") and isinstance(v, dict)
        )
        lines.append(
            f"| {method} | {_fmt(latency['load_ms'], 1)}"
            f" | {_fmt(latency['cold_first_query_ms'], 1)}"
            f" | {_fmt(latency['warm_query']['p50_ms'], 2)}"
            f" | {_fmt(latency['warm_query']['p95_ms'], 2)}"
            f" | {parts} |"
        )
    lines += [
        "",
        "## 15. Limitations",
        "",
        f"- {n} answerable development questions; several comparisons; category subsets tiny.",
        "- An interval containing zero is not evidence of equivalence; one excluding zero is not",
        "  proof of general superiority.",
        "- Page-level relevance; no answer quality measured; development results only.",
        "",
        "## 16. Recommendations",
        "",
        "See `docs/retrieval_evaluation.md` (observations, hypotheses and recommendations are",
        "kept separate there). No retrieval strategy is selected here.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_systems(
    settings: Settings, *, allow_download: bool = False, include_hybrid: bool = False
) -> tuple[ChunkCollection, list[RetrievalSystem], dict[str, float]]:
    """BM25 and dense, both on V2, with their load timings in milliseconds."""
    collection = load_named_collection(settings, COLLECTION)
    t0 = time.perf_counter()
    bm25 = BM25System(BM25Retriever(collection.chunks, DEFAULT_BM25_CONFIG), collection)
    t1 = time.perf_counter()
    dense = DenseSystem(load_dense_retriever(settings, COLLECTION, allow_download=allow_download))
    t2 = time.perf_counter()
    systems: list[RetrievalSystem] = [bm25, dense]
    load = {"bm25": (t1 - t0) * 1000, "dense": (t2 - t1) * 1000}
    if include_hybrid:
        # Its own BM25 index, dense artifact and model: nothing is shared with the baselines.
        t3 = time.perf_counter()
        systems.append(HybridSystem(load_hybrid_retriever(settings, allow_download=allow_download)))
        load[HYBRID_METHOD] = (time.perf_counter() - t3) * 1000
    return collection, systems, load


def write_reports(report: Mapping[str, Any], directory: Path) -> tuple[Path, Path]:
    three_way = "hybrid_diagnostics" in report
    stem = THREE_WAY_REPORT_STEM if three_way else REPORT_STEM
    json_path = directory / f"{stem}.{DEV_SPLIT}.json"
    md_path = directory / f"{stem}.{DEV_SPLIT}.md"
    data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    publish_atomic(json_path, data.encode("utf-8"))
    markdown = render_three_way_markdown(report) if three_way else render_markdown(report)
    publish_atomic(md_path, markdown.encode("utf-8"))
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.benchmark",
        description="Compare BM25 and dense retrieval on the development split (both on V2).",
    )
    parser.add_argument("--split", default=DEV_SPLIT, help="Only 'dev' is allowed.")
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Three-method experiment: also evaluate RRF hybrid (separate reports).",
    )
    parser.add_argument("--output-dir", default=None, help="Report directory (default reports/).")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow fetching the pinned model revision into the Hugging Face cache.",
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)
    try:
        config = EvaluationConfig(split=args.split, include_hybrid=args.include_hybrid)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        settings = load_settings(args.project_root)
        split = load_dev_split(settings.evaluation_dir, settings.corpus_manifest_path)
        collection, systems, load_ms = build_systems(
            settings, allow_download=args.allow_download, include_hybrid=args.include_hybrid
        )
        historical = _historical_report(settings) if args.include_hybrid else None
        run = run_benchmark(
            split, systems, collection, config, load_timings_ms=load_ms, historical=historical
        )
        out = settings.reports_dir if args.output_dir is None else Path(args.output_dir)
        out = out if out.is_absolute() else settings.project_root / out
        json_path, md_path = write_reports(run.report, out)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (EvaluationError, RetrievalError) as exc:
        print(f"Benchmark FAILED ({len(exc.problems)} problem(s)):", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Benchmark FAILED: cannot read or write: {exc}", file=sys.stderr)
        return 1
    _print_summary(run.report, json_path, md_path)
    if not run.complete:
        return 1
    if args.include_hybrid:
        problems = _three_way_problems(run.report)
        for problem in problems:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        return 1 if problems else 0
    return 0


def _historical_report(settings: Settings) -> dict[str, Any] | None:
    """The committed-experiment baseline report, read only (never rewritten)."""
    path = settings.reports_dir / f"{REPORT_STEM}.{DEV_SPLIT}.json"
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _three_way_problems(report: Mapping[str, Any]) -> list[str]:
    problems = []
    if not report["status"].get("hybrid_components_agree", False):
        problems.append("hybrid component rankings disagree with the standalone retrievers")
    baseline = report.get("baseline_reproducibility")
    if baseline is None:
        problems.append(
            f"historical two-method report {REPORT_STEM}.{DEV_SPLIT}.json not found in the"
            " reports directory; baseline not verified. Run `python -m"
            " founderlens.evaluation.benchmark` first (it writes that report), then rerun"
            " with --include-hybrid"
        )
    elif not baseline["reproduced"]:
        problems.append(
            "baseline differs from the historical report: " + "; ".join(baseline["differences"][:5])
        )
    return problems


def _print_summary(report: Mapping[str, Any], json_path: Path, md_path: Path) -> None:
    data = report["dataset"]
    print(
        f"Retrieval benchmark (config {report['evaluation_config_id']}): {data['questions']} dev "
        f"questions ({data['answerable']} answerable) on collection "
        f"{report['collection']['collection_id']}"
    )
    if report["aggregates"] is None:
        print("INCOMPLETE: " + "; ".join(report["status"]["failures"]))
    else:
        cfg, agg = report["evaluation_config"], report["aggregates"]
        primary, cutoff = str(cfg["primary_k"]), str(cfg["mrr_cutoff"])
        for method in report["systems"]:
            o = agg["overall"][method]
            print(
                f"  {method:<6} Recall@{primary} {_fmt(o['recall'][primary]['mean'])} "
                f"{_ci(o['recall'][primary])}  FactCov@{primary} "
                f"{_fmt(o['fact_coverage'][primary]['mean'])}  MRR@{cutoff} "
                f"{_fmt(o['mrr'][cutoff]['mean'])}"
            )
    print(f"Reports: {json_path.name}, {md_path.name} in {json_path.parent}")


if __name__ == "__main__":
    raise SystemExit(main())
