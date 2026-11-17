"""Three-method benchmark (BM25, dense, hybrid RRF) with fake systems.

No model, no network, no real corpus. The fake hybrid fuses the fake component
rankings with the implemented ``rrf_fuse`` so fused positions are the real ones.
Hand-calculated scenario at k = 5 (chunks are ``<doc><page>_<index>``)::

    dev-001  bm25  A2_0 A2_1 B2_0 B2_1 A1_0        (hit at 5, bm25-exclusive)
             dense A2_0 A2_1 B2_0 B3_0 B2_1 B3_1   (miss)
             fused A2_0 A2_1 B2_0 B2_1 B3_0 A1_0 B3_1 -> evidence at position 6
    dev-002  bm25  A1_0 A2_0 / dense B1_0 A1_0 -> fused A1_0 B1_0 A2_0 (both facts)
    dev-003  bm25  B3_1 / dense B2_1 -> evidence page A3 never retrieved
    dev-004  unanswerable
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from founderlens.evaluation import benchmark as bm
from founderlens.evaluation import metrics as m
from founderlens.evaluation.benchmark import (
    DevSplit,
    EvaluationConfig,
    EvaluationError,
    compare_with_baseline,
    deterministic_view,
    experiment_id,
    render_three_way_markdown,
    run_benchmark,
    write_reports,
)
from founderlens.evaluation.schema import parse_record
from founderlens.ingestion.chunk import ChunkingConfig, ChunkRecord, chunk_page
from founderlens.ingestion.normalize import NormalizedPageRecord, page_warnings
from founderlens.ingestion.schema import make_page_id, status_for_text
from founderlens.retrieval import hybrid as hy
from founderlens.retrieval.schema import ChunkCollection, RetrievalResult
from tests.conftest import DOC_A, DOC_B
from tests.test_evaluation_benchmark import _project, _question, _write_evaluation

Factory = Callable[..., Path]
COMPANIES = {DOC_A: "Alpha", DOC_B: "Beta"}


def _collection() -> ChunkCollection:
    chunks: list[ChunkRecord] = []
    for document_id in (DOC_A, DOC_B):
        for number in (1, 2, 3):
            text = f"{document_id} page {number} text more words"  # two 3-word chunks
            page = NormalizedPageRecord(
                document_id=document_id,
                document_sha256="a" * 64,
                company=COMPANIES[document_id],
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
ORDER = {c.chunk_id: i for i, c in enumerate(COLLECTION.chunks)}
RECORDS = [
    _question("dev-001", "single_doc_factual", {"f1": [(DOC_A, 1)]}),
    _question("dev-002", "cross_doc_comparison", {"f1": [(DOC_A, 1)], "f2": [(DOC_B, 1)]}),
    _question("dev-003", "single_doc_factual", {"f1": [(DOC_A, 3)]}),
    _question("dev-004", "unanswerable", {}),
]
QUESTIONS = tuple(parse_record(r) for r in RECORDS)
SPLIT = DevSplit(QUESTIONS, "1.0.0", "e" * 64, "f" * 64)
CONFIG = EvaluationConfig(bootstrap_resamples=200)
THREE_WAY = EvaluationConfig(bootstrap_resamples=200, include_hybrid=True)


def _c(name: str) -> ChunkRecord:
    """``"A2_1"`` -> chunk 1 of page 2 of document A."""
    document_id = DOC_A if name[0] == "A" else DOC_B
    page, index = int(name[1]), int(name[3])
    return next(
        c
        for c in COLLECTION.chunks
        if (c.document_id, c.page_number, c.chunk_index) == (document_id, page, index)
    )


def _q(qid: str) -> str:
    return f"What does question {qid} ask?"


RANKINGS = {
    "bm25": {
        _q("dev-001"): ["A2_0", "A2_1", "B2_0", "B2_1", "A1_0"],
        _q("dev-002"): ["A1_0", "A2_0"],
        _q("dev-003"): ["B3_1"],
        _q("dev-004"): ["B3_0"],
    },
    "dense": {
        _q("dev-001"): ["A2_0", "A2_1", "B2_0", "B3_0", "B2_1", "B3_1"],
        _q("dev-002"): ["B1_0", "A1_0"],
        _q("dev-003"): ["B2_1"],
        _q("dev-004"): [],
    },
}


def _results(method: str, query: str, depth: int) -> tuple[RetrievalResult, ...]:
    names = RANKINGS[method].get(query, [])[:depth]
    return tuple(
        RetrievalResult(method, rank, 1.0 / rank, _c(name))
        for rank, name in enumerate(names, start=1)
    )


class FakeSystem:
    def __init__(self, method: str) -> None:
        self.method = method
        self.queries: list[tuple[str, int]] = []

    def identity(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "retriever_config_id": f"{self.method}-cfg",
            "collection_id": COLLECTION.chunking_config_id,
            "collection_sha256": COLLECTION.artifact_sha256,
            "embedding_config_id": bm.NOT_APPLICABLE,
            "matrix_sha256": bm.NOT_APPLICABLE,
        }

    def run(self, query: str, depth: int) -> tuple[tuple[RetrievalResult, ...], dict[str, float]]:
        self.queries.append((query, depth))
        return _results(self.method, query, depth), {"total_ms": 0.5}


class FakeHybrid:
    """Its own component rankings (optionally perturbed), fused by the real RRF function."""

    method = bm.HYBRID_METHOD

    def __init__(self, *, dense_score_shift: float = 0.0, config_id: str = "hybrid-cfg") -> None:
        self.shift = dense_score_shift
        self.config_id = config_id
        self.queries: list[tuple[str, int]] = []
        self._components: dict[str, tuple[RetrievalResult, ...]] = {}

    @property
    def last_components(self) -> Mapping[str, tuple[RetrievalResult, ...]]:
        return self._components

    def identity(self) -> dict[str, Any]:
        return {
            **FakeSystem("hybrid").identity(),
            "retriever_config_id": self.config_id,
            "component_config_ids": {"bm25": "bm25-cfg", "dense": "dense-cfg"},
            "rrf_constant": hy.RRF_CONSTANT,
            "candidate_depths": {"bm25": 10, "dense": 10},
        }

    def run(self, query: str, depth: int) -> tuple[tuple[RetrievalResult, ...], dict[str, float]]:
        self.queries.append((query, depth))
        dense = tuple(
            RetrievalResult(r.method, r.rank, r.score + self.shift, r.chunk)
            for r in _results("dense", query, 10)
        )
        self._components = {"bm25": _results("bm25", query, 10), "dense": dense}
        fused = hy.rrf_fuse(self._components, ORDER)[:depth]
        results = tuple(
            RetrievalResult(self.method, rank, c.rrf_score, c.chunk)
            for rank, c in enumerate(fused, start=1)
        )
        return results, {"bm25_ms": 0.1, "dense_ms": 0.2, "fusion_ms": 0.05, "total_ms": 0.4}


def _two() -> list[Any]:
    return [FakeSystem("bm25"), FakeSystem("dense")]


def _three(**hybrid_kwargs: Any) -> list[Any]:
    return [*_two(), FakeHybrid(**hybrid_kwargs)]


def _historical() -> dict[str, Any]:
    return run_benchmark(SPLIT, _two(), COLLECTION, CONFIG).report


def _report(**hybrid_kwargs: Any) -> dict[str, Any]:
    return run_benchmark(
        SPLIT, _three(**hybrid_kwargs), COLLECTION, THREE_WAY, historical=_historical()
    ).report


# --- identity and execution ----------------------------------------------------------------


def test_two_method_identity_is_unchanged_and_the_experiment_has_its_own() -> None:
    assert EvaluationConfig().config_id == "3fd7b205dc81"  # historical Commit 3.4 experiment
    assert "methods" not in EvaluationConfig().to_dict()
    report = _report()
    assert report["evaluation_config_id"] not in {"3fd7b205dc81", CONFIG.config_id}
    # The baseline is the two-method experiment with the same measurement settings.
    assert report["historical_baseline_config_id"] == CONFIG.config_id
    default_three_way = EvaluationConfig(include_hybrid=True)
    assert replace(default_three_way, include_hybrid=False).config_id == "3fd7b205dc81"
    other = {**report["systems"], "hybrid": {"retriever_config_id": "different"}}
    assert experiment_id(THREE_WAY, other) != report["evaluation_config_id"]
    assert experiment_id(CONFIG, report["systems"]) == CONFIG.config_id
    with pytest.raises(ValueError, match="at most 10"):
        EvaluationConfig(ks=(1, 5, 20), include_hybrid=True)


def test_all_three_systems_get_identical_queries_once_each() -> None:
    systems = _three()
    run_benchmark(SPLIT, systems, COLLECTION, THREE_WAY, historical=_historical())
    expected = [(bm.WARMUP_QUERY, 10)] + [(q.question, 10) for q in QUESTIONS]
    assert [s.queries for s in systems] == [expected] * 3


def test_the_three_method_experiment_needs_exactly_bm25_dense_hybrid() -> None:
    with pytest.raises(EvaluationError, match="needs bm25, dense, hybrid"):
        run_benchmark(SPLIT, _two(), COLLECTION, THREE_WAY)


# --- metrics, pairs and diagnostics -----------------------------------------------------------


def test_three_way_metrics_and_paired_comparisons() -> None:
    report = _report()
    agg = report["aggregates"]
    hybrid = agg["overall"]["hybrid"]
    assert hybrid["recall"]["5"]["mean"] == pytest.approx(1 / 3)
    assert hybrid["recall"]["10"]["mean"] == pytest.approx(2 / 3)
    assert hybrid["complete_coverage"]["5"] == {"mean": 1.0, "n": 1, "undefined": 2, "ci95": None}
    by_id = {r["question_id"]: r for r in report["per_question"]}
    assert by_id["dev-001"]["systems"]["hybrid"]["first_relevant_rank"] == 6
    assert list(agg["paired"]) == ["dense_minus_bm25", "hybrid_minus_bm25", "hybrid_minus_dense"]
    pair = agg["paired"]["hybrid_minus_bm25"]["recall"]["5"]
    assert (pair["candidate_better"], pair["baseline_better"], pair["ties"]) == (0, 1, 2)
    outcomes = report["hybrid_diagnostics"]["aggregates"]["recall_at_5_pair_outcomes"]
    assert outcomes["hybrid_vs_bm25"] == {
        "both_hit": ["dev-002"],
        "hybrid_only": [],
        "bm25_only": ["dev-001"],
        "neither": ["dev-003"],
    }
    assert outcomes["hybrid_vs_dense"]["neither"] == ["dev-001", "dev-003"]
    assert outcomes["dense_vs_bm25"]["bm25_only"] == ["dev-001"]


def test_candidate_pool_and_mutually_exclusive_fact_outcomes() -> None:
    diagnostics = _report()["hybrid_diagnostics"]
    pool = diagnostics["per_question"]["dev-001"]["pool"]
    assert {k: pool[k] for k in ("shared", "bm25_exclusive", "dense_exclusive", "union")} == {
        "shared": 4,
        "bm25_exclusive": 1,
        "dense_exclusive": 2,
        "union": 7,
    }
    assert diagnostics["per_question"]["dev-003"]["pool"]["facts_missing_from_union"] == [
        "dev-003-f1"
    ]
    extra = diagnostics["aggregates"]
    union = extra["candidate_pool"]
    assert union["union_fact_coverage_macro"]["mean"] == pytest.approx(2 / 3)
    assert (union["union_facts_covered_micro"], union["facts_total"]) == (3, 4)
    counts = extra["fact_outcome_counts"]["by_k"]
    expected = {"1": (1, 1, 2), "3": (2, 1, 1), "5": (2, 1, 1), "10": (3, 1, 0)}
    for k, (retained, candidate, cutoff) in expected.items():
        c = counts[k]
        assert (c[m.RETAINED], c[m.CANDIDATE_RETRIEVAL_FAILURE], c[m.FUSION_CUTOFF_FAILURE]) == (
            retained,
            candidate,
            cutoff,
        )
        assert retained + candidate + cutoff == c["facts"] == 4  # each fact exactly once


def test_displaced_exclusive_evidence_and_diversity() -> None:
    extra = _report()["hybrid_diagnostics"]["aggregates"]
    assert extra["displaced_relevant_chunks_at_5"] == [
        {
            "question_id": "dev-001",
            "comparator": "bm25",
            "fact_id": "dev-001-f1",
            "chunk_id": _c("A1_0").chunk_id,
            "page": f"{DOC_A}:p1",
            "comparator_rank": 5,
            "candidate_type": "bm25_exclusive",
            "fused_position": 6,
            "shared_candidates_in_pool": 4,
            "evidence_lost": True,
        }
    ]
    assert extra["single_company_evidence_at_5"] == {
        "bm25": ["dev-002"],  # cross-company question, evidence from Alpha only
        "dense": [],
        "hybrid": [],
    }
    assert extra["component_agreement"] == {"all_agree": True, "max_abs_score_difference": 0.0}
    assert extra["score_ranges"]["bm25"]["ranges_overlap"] is True
    assert extra["score_ranges"]["dense"]["unanswerable_top_score_range"] is None


def test_component_disagreement_is_detected_and_fails_the_run() -> None:
    report = _report(dense_score_shift=0.01)
    assert report["status"]["hybrid_components_agree"] is False
    assert any("disagree" in p for p in bm._three_way_problems(report))
    assert bm._three_way_problems(_report()) == []


# --- baseline reproducibility -----------------------------------------------------------------


def _score(historical: dict[str, Any], qid_index: int, method: str, rank_index: int) -> Any:
    return historical["per_question"][qid_index]["systems"][method]["results"][rank_index]


def test_baseline_with_exactly_equal_scores_is_reproduced() -> None:
    base = _report()["baseline_reproducibility"]
    assert base == {
        "historical_evaluation_config_id": CONFIG.config_id,
        "reproduced": True,
        "score_tolerance": {"absolute": bm.SCORE_TOLERANCE, "relative": 0.0},
        "scores_identical": True,
        "max_abs_score_difference": 0.0,
        "differences": [],
    }
    assert bm.SCORE_TOLERANCE == 1e-6


def test_score_differences_within_tolerance_reproduce_but_are_not_identical() -> None:
    report, historical = _report(), _historical()
    _score(historical, 0, "dense", 0)["score"] += 4e-7
    _score(historical, 1, "bm25", 1)["score"] -= 2.5e-7
    result = compare_with_baseline(report, historical)
    assert result["reproduced"] is True
    assert result["scores_identical"] is False
    assert result["differences"] == []
    assert result["max_abs_score_difference"] == pytest.approx(4e-7, abs=1e-12)


def test_score_differences_beyond_tolerance_fail_and_report_the_maximum() -> None:
    report, historical = _report(), _historical()
    _score(historical, 0, "dense", 0)["score"] += 3e-6
    _score(historical, 0, "bm25", 1)["score"] -= 2e-6
    result = compare_with_baseline(report, historical)
    assert result["reproduced"] is False
    assert result["scores_identical"] is False
    assert result["max_abs_score_difference"] == pytest.approx(3e-6, abs=1e-12)
    assert len(result["differences"]) == 1
    assert result["differences"][0].startswith("retrieval scores differ beyond tolerance 1e-06")
    # The tolerance is absolute only: the same shift passes under a looser explicit bound.
    assert compare_with_baseline(report, historical, tolerance=1e-5)["reproduced"] is True


def test_ranking_change_fails_even_with_numerically_close_scores() -> None:
    report, historical = _report(), _historical()
    results = historical["per_question"][1]["systems"]["bm25"]["results"]
    first, second = results[0], results[1]
    first["chunk_id"], second["chunk_id"] = second["chunk_id"], first["chunk_id"]
    first["page_number"], second["page_number"] = second["page_number"], first["page_number"]
    first["document_id"], second["document_id"] = second["document_id"], first["document_id"]
    result = compare_with_baseline(report, historical, tolerance=1.0)  # scores ignored
    assert result["reproduced"] is False
    assert "dev-002/bm25: chunks or ranks differ" in result["differences"]
    assert "dev-002/bm25: result entries differ" in result["differences"]
    historical = _historical()
    _score(historical, 1, "dense", 0)["page_number"] = 3  # provenance changed, same chunk ID
    differences = compare_with_baseline(report, historical)["differences"]
    assert differences == ["dev-002/dense: result entries differ"]
    historical = _historical()
    historical["aggregates"]["overall"]["bm25"]["recall"]["5"]["mean"] = 0.0
    assert compare_with_baseline(report, historical)["differences"] == [
        "bm25 overall aggregates differ"
    ]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "0.5"])
def test_non_finite_or_missing_scores_fail(bad: object) -> None:
    report, historical = _report(), _historical()
    _score(historical, 0, "dense", 2)["score"] = bad
    result = compare_with_baseline(report, historical)
    assert result["reproduced"] is False
    assert result["scores_identical"] is False
    assert "dev-001/dense: non-finite or missing score" in result["differences"]
    assert result["max_abs_score_difference"] == 0.0  # never poisoned by NaN
    for tolerance in (float("nan"), float("inf"), -1e-9):
        with pytest.raises(ValueError, match="tolerance"):
            compare_with_baseline(report, _historical(), tolerance=tolerance)


def test_baseline_must_be_the_matching_two_method_experiment() -> None:
    report, historical = _report(), _historical()
    historical["evaluation_config_id"] = "000000000000"
    differences = compare_with_baseline(report, historical)["differences"]
    assert differences == [
        f"historical report is experiment '000000000000', expected {CONFIG.config_id!r}"
    ]
    (problem,) = bm._three_way_problems({**report, "baseline_reproducibility": None})
    assert "retrieval_benchmark.dev.json not found" in problem
    assert "python -m founderlens.evaluation.benchmark` first" in problem


# --- determinism and reports ------------------------------------------------------------------


def test_deterministic_content_and_hybrid_latency_parts() -> None:
    first, second = _report(), _report()
    assert deterministic_view(first) == deterministic_view(second)
    assert json.dumps(deterministic_view(first)).count("_ms") == 0
    latency = first["operational"]["latency"]["hybrid"]
    assert {"bm25_ms", "dense_ms", "fusion_ms", "warm_query"} <= set(latency)
    md = render_three_way_markdown(first)
    for heading in (
        "## 4. Baseline reproducibility",
        "## 9. Candidate-pool coverage",
        "## 10. Fact outcomes",
        "## 14. Latency",
    ):
        assert heading in md
    assert "reproduced = **True**" in md


def test_three_way_reports_never_overwrite_the_two_method_reports(tmp_path: Path) -> None:
    two_json, two_md = write_reports(_historical(), tmp_path)
    before = (two_json.read_bytes(), two_md.read_bytes())
    three_json, three_md = write_reports(_report(), tmp_path)
    assert (three_json.name, three_md.name) == (
        "retrieval_benchmark_three_way.dev.json",
        "retrieval_benchmark_three_way.dev.md",
    )
    assert (two_json.read_bytes(), two_md.read_bytes()) == before


def test_incomplete_three_way_run_has_no_diagnostics() -> None:
    class Failing(FakeHybrid):
        def run(
            self, query: str, depth: int
        ) -> tuple[tuple[RetrievalResult, ...], dict[str, float]]:
            if query == _q("dev-002"):
                raise RuntimeError("simulated failure")
            return super().run(query, depth)

    run = run_benchmark(SPLIT, [*_two(), Failing()], COLLECTION, THREE_WAY)
    assert run.complete is False
    assert run.report["hybrid_diagnostics"] is None
    assert "Benchmark incomplete" in render_three_way_markdown(run.report)


# --- CLI ------------------------------------------------------------------------------------


def test_cli_rejects_held_out_split_with_hybrid(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        bm.main(["--include-hybrid", "--split", "test"])
    assert exc_info.value.code == 2
    assert "held-out evaluation is not authorized" in capsys.readouterr().err


def test_cli_three_way_run_reads_but_never_rewrites_the_historical_report(
    make_project: Factory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(make_project)
    _write_evaluation(root, RECORDS)

    def build(
        settings: Any, *, allow_download: bool = False, include_hybrid: bool = False
    ) -> tuple[ChunkCollection, list[Any], dict[str, float]]:
        systems = _three() if include_hybrid else _two()
        return COLLECTION, systems, {s.method: 1.0 for s in systems}

    monkeypatch.setattr(bm, "build_systems", build)
    reports = root / "reports"
    assert bm.main(["--project-root", str(root), "--include-hybrid"]) == 1  # no baseline yet
    assert "baseline not verified" in capsys.readouterr().err
    assert bm.main(["--project-root", str(root)]) == 0  # the two-method default
    historical = (reports / "retrieval_benchmark.dev.json").read_bytes()
    assert bm.main(["--project-root", str(root), "--include-hybrid"]) == 0
    out = capsys.readouterr().out
    assert "hybrid" in out
    assert (reports / "retrieval_benchmark.dev.json").read_bytes() == historical
    report = json.loads((reports / "retrieval_benchmark_three_way.dev.json").read_text("utf-8"))
    assert report["baseline_reproducibility"]["reproduced"] is True
    assert report["status"]["hybrid_components_agree"] is True
