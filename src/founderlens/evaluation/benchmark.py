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
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.corpus.manifest import ManifestError, load_manifest
from founderlens.evaluation import metrics as m
from founderlens.evaluation.schema import Category, EvalQuestion, parse_jsonl
from founderlens.evaluation.validate import DATASET_METADATA_FILENAME, SPLIT_FILES, check_question
from founderlens.ingestion.extract import publish_atomic
from founderlens.retrieval.bm25 import DEFAULT_BM25_CONFIG, LIBRARY, BM25Retriever, library_version
from founderlens.retrieval.collections import CollectionName, load_named_collection
from founderlens.retrieval.dense import DenseRetriever, load_dense_retriever
from founderlens.retrieval.embeddings import dependency_versions
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

    @property
    def depth(self) -> int:
        """Chunks requested once per question; every k is a prefix of this list."""
        return max(self.ks)

    def to_dict(self) -> dict[str, Any]:
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
        }

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
    baseline = methods[0]
    for candidate in methods[1:]:
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
    records = [_evaluate_question(q, systems, chunks, config, timings) for q in split.questions]
    failures = [
        f"{r['question_id']}/{method}: {entry['error']}"
        for r in records
        for method, entry in r["systems"].items()
        if entry["status"] != "ok"
    ]
    complete = not failures
    methods = [s.method for s in systems]
    report: dict[str, Any] = {
        "report_type": "founderlens-retrieval-benchmark",
        "evaluation_config": DEFAULT_EVALUATION_CONFIG.to_dict()
        if config == DEFAULT_EVALUATION_CONFIG
        else config.to_dict(),
        "evaluation_config_id": config.config_id,
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
        "systems": {s.method: s.identity() for s in systems},
        "status": {"complete": complete, "failures": failures},
        "per_question": records,
        "aggregates": aggregate(records, methods, config) if complete else None,
        "operational": _operational(
            systems, timings, warmup, load_timings_ms or {}, started_at, total
        ),
    }
    return BenchmarkRun(report=report, complete=complete, timings=report["operational"])


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
                for part in ("encode_ms", "score_and_rank_ms")
                if rows and part in rows[0]
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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_systems(
    settings: Settings, *, allow_download: bool = False
) -> tuple[ChunkCollection, list[RetrievalSystem], dict[str, float]]:
    """BM25 and dense, both on V2, with their load timings in milliseconds."""
    collection = load_named_collection(settings, COLLECTION)
    t0 = time.perf_counter()
    bm25 = BM25System(BM25Retriever(collection.chunks, DEFAULT_BM25_CONFIG), collection)
    t1 = time.perf_counter()
    dense = DenseSystem(load_dense_retriever(settings, COLLECTION, allow_download=allow_download))
    t2 = time.perf_counter()
    return collection, [bm25, dense], {"bm25": (t1 - t0) * 1000, "dense": (t2 - t1) * 1000}


def write_reports(report: Mapping[str, Any], directory: Path) -> tuple[Path, Path]:
    json_path = directory / f"{REPORT_STEM}.{DEV_SPLIT}.json"
    md_path = directory / f"{REPORT_STEM}.{DEV_SPLIT}.md"
    data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    publish_atomic(json_path, data.encode("utf-8"))
    publish_atomic(md_path, render_markdown(report).encode("utf-8"))
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.benchmark",
        description="Compare BM25 and dense retrieval on the development split (both on V2).",
    )
    parser.add_argument("--split", default=DEV_SPLIT, help="Only 'dev' is allowed.")
    parser.add_argument("--output-dir", default=None, help="Report directory (default reports/).")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow fetching the pinned model revision into the Hugging Face cache.",
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)
    try:
        config = EvaluationConfig(split=args.split)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        settings = load_settings(args.project_root)
        split = load_dev_split(settings.evaluation_dir, settings.corpus_manifest_path)
        collection, systems, load_ms = build_systems(settings, allow_download=args.allow_download)
        run = run_benchmark(split, systems, collection, config, load_timings_ms=load_ms)
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
    return 0 if run.complete else 1


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
