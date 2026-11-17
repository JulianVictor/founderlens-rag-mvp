"""Retrieval benchmark runner, reports and CLI with fake retrieval systems.

No model, no network, no real corpus. The held-out file in the synthetic
project is deliberately unreadable to prove it is never opened.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from founderlens.config import Settings
from founderlens.evaluation import benchmark as bm
from founderlens.evaluation.benchmark import (
    DevSplit,
    EvaluationConfig,
    EvaluationError,
    deterministic_view,
    load_dev_split,
    render_markdown,
    run_benchmark,
)
from founderlens.evaluation.schema import parse_record
from founderlens.ingestion.chunk import ChunkingConfig, ChunkRecord, chunk_page
from founderlens.ingestion.normalize import NormalizedPageRecord, page_warnings
from founderlens.ingestion.schema import make_page_id, status_for_text
from founderlens.retrieval.schema import ChunkCollection, RetrievalResult
from tests.conftest import DOC_A, DOC_B, make_pdf_bytes

Factory = Callable[..., Path]
HELD_OUT_SENTINEL = "HELD-OUT-SENTINEL"


def _collection() -> ChunkCollection:
    chunks: list[ChunkRecord] = []
    for document_id in (DOC_A, DOC_B):
        for number in (1, 2, 3):
            text = f"{document_id} page {number} text more words"  # two 3-word chunks
            page = NormalizedPageRecord(
                document_id=document_id,
                document_sha256="a" * 64,
                company="Example",
                fiscal_period="Q1 2024",
                source_url="https://example.com/x.pdf",
                page_number=number,
                page_id=make_page_id(document_id, number),
                extraction_status=status_for_text(text),
                raw_character_count=len(text),
                normalized_text=text,
                normalized_character_count=len(text),
                tables=(),
                table_text_overlap=False,
                normalization_warnings=page_warnings(status_for_text(text), ()),
            )
            chunks.extend(chunk_page(page, ChunkingConfig(3, 0)))
    return ChunkCollection(tuple(chunks), "c" * 64, chunks[0].chunking_config_id, "d" * 64)


COLLECTION = _collection()


def _question(
    qid: str, category: str, evidence: dict[str, list[tuple[str, int]]]
) -> dict[str, Any]:
    answerable = category != "unanswerable"
    return {
        "question_id": qid,
        "question": f"What does question {qid} ask?",
        "category": category,
        "answerable": answerable,
        "reference_answer": "Reference." if answerable else None,
        "required_facts": [{"fact_id": f"{qid}-{f}", "statement": "A fact."} for f in evidence],
        "accepted_evidence": [
            {"fact_id": f"{qid}-{f}", "document_id": d, "page_number": p}
            for f, pages in evidence.items()
            for d, p in pages
        ],
        "expected_abstention_reason": None if answerable else "Not in the corpus.",
    }


RECORDS = [
    _question("dev-001", "single_doc_factual", {"f1": [(DOC_A, 1)]}),
    _question("dev-002", "single_doc_factual", {"f1": [(DOC_A, 2), (DOC_A, 3)]}),
    _question("dev-003", "cross_doc_comparison", {"f1": [(DOC_A, 1)], "f2": [(DOC_B, 1)]}),
    _question("dev-004", "unanswerable", {}),
]
QUESTIONS = tuple(parse_record(r) for r in RECORDS)
SPLIT = DevSplit(QUESTIONS, "1.0.0", "e" * 64, "f" * 64)


def _chunk(document_id: str, page: int, index: int = 0) -> ChunkRecord:
    return next(
        c
        for c in COLLECTION.chunks
        if (c.document_id, c.page_number, c.chunk_index) == (document_id, page, index)
    )


class FakeSystem:
    """Returns fixed rankings per query and records every query it receives."""

    def __init__(
        self,
        method: str,
        rankings: dict[str, list[ChunkRecord]],
        *,
        collection: ChunkCollection = COLLECTION,
        fail_on: str | None = None,
    ) -> None:
        self.method = method
        self.rankings = rankings
        self.collection = collection
        self.fail_on = fail_on
        self.queries: list[tuple[str, int]] = []

    def identity(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "retriever_config_id": f"{self.method}-cfg",
            "collection_id": self.collection.chunking_config_id,
            "collection_sha256": self.collection.artifact_sha256,
            "embedding_config_id": bm.NOT_APPLICABLE,
            "matrix_sha256": bm.NOT_APPLICABLE,
        }

    def run(self, query: str, depth: int) -> tuple[tuple[RetrievalResult, ...], dict[str, float]]:
        self.queries.append((query, depth))
        if query == self.fail_on:
            raise RuntimeError("simulated model failure")
        chunks = self.rankings.get(query, [])[:depth]
        results = tuple(
            RetrievalResult(self.method, rank, 1.0 / rank, chunk)
            for rank, chunk in enumerate(chunks, start=1)
        )
        return results, {"total_ms": 0.5}


def _q(qid: str) -> str:
    return f"What does question {qid} ask?"


def _systems(**dense_kwargs: Any) -> list[FakeSystem]:
    bm25 = FakeSystem(
        "bm25",
        {
            _q("dev-001"): [_chunk(DOC_A, 1)],  # hit at rank 1
            _q("dev-002"): [_chunk(DOC_B, 2), _chunk(DOC_A, 3)],  # hit at rank 2
            _q("dev-003"): [_chunk(DOC_A, 1), _chunk(DOC_A, 1, 1)],  # same page twice
            _q("dev-004"): [_chunk(DOC_B, 3)],
        },
    )
    dense = FakeSystem(
        "dense",
        {
            _q("dev-001"): [_chunk(DOC_B, 1)],  # miss
            _q("dev-002"): [_chunk(DOC_A, 2)],  # hit at rank 1
            _q("dev-003"): [_chunk(DOC_B, 1), _chunk(DOC_A, 1)],  # both facts
            _q("dev-004"): [],
        },
        **dense_kwargs,
    )
    return [bm25, dense]


CONFIG = EvaluationConfig(bootstrap_resamples=200)


# --- configuration ---------------------------------------------------------------------


def test_configuration_identity_and_validation() -> None:
    config = EvaluationConfig()
    assert (config.ks, config.primary_k, config.depth, config.mrr_cutoff) == (
        (1, 3, 5, 10),
        5,
        10,
        10,
    )
    assert config.config_id == EvaluationConfig().config_id
    assert config.config_id != CONFIG.config_id  # resamples are part of the measurement
    assert config.to_dict()["recall_definition"].startswith("hit rate")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"split": "test"}, "held-out evaluation is not authorized"),
        ({"ks": (5, 1)}, "strictly increasing"),
        ({"ks": (1, 3), "primary_k": 5}, "primary_k must be one of ks"),
        ({"ks": (0, 5)}, "positive integer"),
        ({"collection": "v1"}, "collection v2 for every system"),
        ({"mrr_cutoff": 20}, "cannot exceed the retrieval depth"),
    ],
)
def test_invalid_configuration(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EvaluationConfig(**kwargs)


# --- runner --------------------------------------------------------------------------------


def test_both_systems_get_identical_queries_once_each_at_full_depth() -> None:
    systems = _systems()
    run_benchmark(SPLIT, systems, COLLECTION, CONFIG)
    expected = [(bm.WARMUP_QUERY, 10)] + [(q.question, 10) for q in QUESTIONS]
    assert systems[0].queries == expected  # warm-up, then each question exactly once
    assert systems[1].queries == expected


def test_per_question_metrics_and_prefix_evaluation() -> None:
    report = run_benchmark(SPLIT, _systems(), COLLECTION, CONFIG).report
    by_id = {r["question_id"]: r for r in report["per_question"]}
    bm25 = by_id["dev-002"]["systems"]["bm25"]
    assert bm25["metrics"]["recall"] == {"1": 0, "3": 1, "5": 1, "10": 1}  # from one ranking
    assert bm25["metrics"]["mrr"] == {"10": 0.5}
    assert bm25["first_relevant_rank"] == 2
    cross = by_id["dev-003"]["systems"]
    assert cross["bm25"]["metrics"]["fact_coverage"]["5"] == 0.5
    assert cross["bm25"]["metrics"]["complete_coverage"]["5"] == 0
    assert cross["bm25"]["metrics"]["duplicate_page_rate"]["5"] == 0.5
    assert cross["bm25"]["missing_facts_at_5"] == ["dev-003-f2"]
    assert cross["dense"]["metrics"]["complete_coverage"]["5"] == 1
    assert by_id["dev-001"]["systems"]["dense"]["metrics"]["mrr"] == {"10": 0.0}
    unanswerable = by_id["dev-004"]["systems"]
    assert unanswerable["bm25"]["metrics"] is None  # no gold evidence, no relevance metric
    assert unanswerable["dense"]["returned"] == 0
    assert unanswerable["dense"]["observation"]["top_score"] is None


def test_aggregates_denominators_categories_and_pairs() -> None:
    agg = run_benchmark(SPLIT, _systems(), COLLECTION, CONFIG).report["aggregates"]
    assert agg["answerable_n"] == 3  # unanswerable excluded
    recall = agg["overall"]["bm25"]["recall"]["5"]
    assert (recall["mean"], recall["n"], recall["undefined"]) == (1.0, 3, 0)
    complete = agg["overall"]["dense"]["complete_coverage"]["5"]
    assert (complete["n"], complete["undefined"]) == (1, 2)  # only dev-003 has two facts
    assert agg["overall"]["dense"]["mrr"]["10"]["mean"] == pytest.approx((0 + 1 + 1) / 3)
    assert agg["by_category"]["bm25"]["single_doc_factual"]["n"] == 2
    assert agg["by_category"]["bm25"]["cross_doc_comparison"]["n"] == 1
    assert "unanswerable" not in agg["by_category"]["bm25"]
    pair = agg["paired"]["dense_minus_bm25"]["recall"]["5"]
    assert (pair["candidate_better"], pair["baseline_better"], pair["ties"]) == (0, 1, 2)
    assert agg["recall_at_5_outcomes"] == {
        "both_hit": ["dev-002", "dev-003"],
        "neither": [],
        "bm25_only": ["dev-001"],
        "dense_only": [],
    }
    assert agg["unanswerable"]["bm25"] == {"n": 1, "returned": [1], "top_scores": [1.0]}


def test_execution_failures_are_reported_never_scored() -> None:
    run = run_benchmark(SPLIT, _systems(fail_on=_q("dev-002")), COLLECTION, CONFIG)
    assert run.complete is False
    assert run.report["aggregates"] is None  # no partial means presented as results
    assert run.report["status"]["failures"] == [
        "dev-002/dense: RuntimeError: simulated model failure"
    ]
    entry = next(r for r in run.report["per_question"] if r["question_id"] == "dev-002")
    assert "metrics" not in entry["systems"]["dense"]
    assert "Benchmark incomplete" in render_markdown(run.report)


def test_invalid_results_are_execution_failures() -> None:
    foreign = replace(_chunk(DOC_A, 1), company="Other Corp")  # valid record, not ours
    systems = _systems()
    systems[1].rankings[_q("dev-001")] = [foreign]
    run = run_benchmark(SPLIT, systems, COLLECTION, CONFIG)
    assert run.complete is False
    assert "not an original chunk" in run.report["status"]["failures"][0]


def test_incompatible_collections_and_unreachable_gold_are_rejected() -> None:
    other = ChunkCollection(COLLECTION.chunks, "0" * 64, COLLECTION.chunking_config_id, "d" * 64)
    with pytest.raises(EvaluationError, match="runs on collection"):
        run_benchmark(
            SPLIT, [_systems()[0], FakeSystem("dense", {}, collection=other)], COLLECTION, CONFIG
        )
    missing_page = DevSplit(
        (parse_record(_question("dev-009", "single_doc_factual", {"f1": [(DOC_A, 7)]})),),
        "1.0.0",
        "e" * 64,
        "f" * 64,
    )
    with pytest.raises(EvaluationError, match="no chunk in the collection"):
        run_benchmark(missing_page, _systems(), COLLECTION, CONFIG)


def test_deterministic_content_is_stable_and_volatile_fields_are_separate() -> None:
    first = run_benchmark(SPLIT, _systems(), COLLECTION, CONFIG).report
    second = run_benchmark(SPLIT, _systems(), COLLECTION, CONFIG).report
    assert deterministic_view(first) == deterministic_view(second)
    assert "latency" in first["operational"]
    assert "operational" not in deterministic_view(first)
    assert json.dumps(deterministic_view(first)).count("_ms") == 0  # no timings outside
    text = json.dumps(first)
    assert "What does question" not in text  # question text is not stored
    systems = first["systems"]
    assert systems["bm25"]["embedding_config_id"] == "not_applicable"
    md = render_markdown(first)
    for heading in ("## 1. Experimental design", "## 4. Primary results (k = 5", "## 7. Paired"):
        assert heading in md


# --- development split loading ------------------------------------------------------------


def _write_evaluation(
    root: Path, records: Sequence[dict[str, Any]], *, sha: str | None = None
) -> None:
    evaluation = root / "evaluation"
    evaluation.mkdir(exist_ok=True)
    dev = "".join(json.dumps(r) + "\n" for r in records).encode("utf-8")
    (evaluation / "questions.dev.jsonl").write_bytes(dev)
    (evaluation / "questions.test.jsonl").write_text(HELD_OUT_SENTINEL + " {not json", "utf-8")
    metadata = {
        "dataset_version": "1.0.0",
        "files": {
            "questions.dev.jsonl": {
                "split": "dev",
                "question_count": len(records),
                "sha256": sha or hashlib.sha256(dev).hexdigest(),
            },
            "questions.test.jsonl": {"split": "test", "question_count": 1, "sha256": "0" * 64},
        },
    }
    (evaluation / "dataset.json").write_text(json.dumps(metadata), "utf-8")


def _project(make_project: Factory) -> Path:
    return make_project({DOC_A: (make_pdf_bytes(3), 3), DOC_B: (make_pdf_bytes(3), 3)})


def test_dev_split_loads_without_opening_the_held_out_file(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(make_project)
    _write_evaluation(root, RECORDS)
    opened: list[str] = []
    original = Path.read_bytes

    def guarded(self: Path) -> bytes:
        opened.append(self.name)
        if self.name == "questions.test.jsonl":
            raise AssertionError("held-out file opened")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    split = load_dev_split(root / "evaluation", root / "data" / "manifest.json")
    assert [q.question_id for q in split.questions] == ["dev-001", "dev-002", "dev-003", "dev-004"]
    assert "questions.test.jsonl" not in opened


@pytest.mark.parametrize(
    ("records", "sha", "message"),
    [
        (RECORDS, "0" * 64, "SHA-256 does not match"),
        (
            [*RECORDS[:3], {**RECORDS[3], "question_id": "test-004"}],
            None,
            "not a development question",
        ),
        ([{**RECORDS[0], "accepted_evidence": []}], None, "has no accepted evidence"),
        ([{**RECORDS[0], "category": "nonsense"}], None, "invalid category"),
    ],
)
def test_invalid_development_split_is_rejected(
    make_project: Factory, records: list[dict[str, Any]], sha: str | None, message: str
) -> None:
    root = _project(make_project)
    _write_evaluation(root, records, sha=sha)
    with pytest.raises(EvaluationError, match=message):
        load_dev_split(root / "evaluation", root / "data" / "manifest.json")


# --- CLI -------------------------------------------------------------------------------------


def test_cli_rejects_held_out_split(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        bm.main(["--split", "test"])
    assert exc_info.value.code == 2
    assert "held-out evaluation is not authorized" in capsys.readouterr().err


def test_cli_runs_and_writes_reports(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(make_project)
    _write_evaluation(root, RECORDS)
    monkeypatch.setattr(
        bm,
        "build_systems",
        lambda settings, **_: (COLLECTION, _systems(), {"bm25": 1.0, "dense": 2.0}),
    )
    monkeypatch.setattr(bm, "DEFAULT_EVALUATION_CONFIG", CONFIG)
    assert bm.main(["--project-root", str(root), "--output-dir", "out"]) == 0
    out = capsys.readouterr().out
    assert "Recall@5" in out
    assert "4 dev questions (3 answerable)" in out
    report = json.loads((root / "out" / "retrieval_benchmark.dev.json").read_text("utf-8"))
    assert report["status"]["complete"] is True
    assert report["operational"]["latency"]["bm25"]["load_ms"] == 1.0
    assert (root / "out" / "retrieval_benchmark.dev.md").is_file()

    monkeypatch.setattr(
        bm,
        "build_systems",
        lambda settings, **_: (COLLECTION, _systems(fail_on=_q("dev-001")), {}),
    )
    assert bm.main(["--project-root", str(root), "--output-dir", "out"]) == 1
    assert "INCOMPLETE" in capsys.readouterr().out


def test_cli_reports_missing_inputs(
    make_project: Factory, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(make_project)
    assert bm.main(["--project-root", str(root)]) == 1
    assert "cannot load the development split" in capsys.readouterr().err


def test_settings_paths_are_used(make_project: Factory) -> None:
    root = _project(make_project)
    assert Settings(root).evaluation_dir == root / "evaluation"
