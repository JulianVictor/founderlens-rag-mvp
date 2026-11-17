"""Page-level retrieval metrics against hand-calculated values (protocol v1.1 §3)."""

from __future__ import annotations

import pytest

from founderlens.evaluation import metrics as m
from founderlens.evaluation.metrics import GoldEvidence

A, B = "aaa-2024-q1", "bbb-2024-q1"
# f1 has two interchangeable pages; f2 is on another document's page 1.
GOLD = GoldEvidence.from_mapping({"f1": [(A, 1), (A, 3)], "f2": [(B, 1)]})
# Rank 1 irrelevant, rank 2 covers f2, rank 3 repeats a page, rank 4 covers f1.
RANKING = [(A, 2), (B, 1), (A, 2), (A, 1)]


@pytest.mark.parametrize(
    ("k", "recall", "coverage", "complete", "unique", "duplicate"),
    [
        (1, 0, 0.0, 0, 1, 0.0),
        (2, 1, 0.5, 0, 2, 0.0),
        (3, 1, 0.5, 0, 2, 1 / 3),
        (4, 1, 1.0, 1, 3, 0.25),
        (10, 1, 1.0, 1, 3, 0.25),  # fewer than k results
    ],
)
def test_hand_calculated_metrics_at_each_k(
    k: int, recall: int, coverage: float, complete: int, unique: int, duplicate: float
) -> None:
    assert m.recall_at_k(RANKING, GOLD, k) == recall
    assert m.fact_coverage_at_k(RANKING, GOLD, k) == coverage
    assert m.complete_coverage_at_k(RANKING, GOLD, k) == complete
    assert m.unique_pages_at_k(RANKING, k) == unique
    assert m.duplicate_page_rate_at_k(RANKING, k) == pytest.approx(duplicate)


def test_reciprocal_rank_uses_the_original_rank_and_the_cutoff() -> None:
    assert m.first_relevant_rank(RANKING, GOLD) == 2
    assert m.reciprocal_rank(RANKING, GOLD) == 0.5
    late = [(A, 9)] * 10 + [(A, 1)]
    assert m.first_relevant_rank(late, GOLD) == 11
    assert m.reciprocal_rank(late, GOLD, cutoff=10) == 0.0  # outside the cut-off
    assert m.reciprocal_rank(late, GOLD, cutoff=11) == pytest.approx(1 / 11)


def test_duplicates_do_not_inflate_recall_or_coverage() -> None:
    repeated = [(B, 1)] * 5
    assert m.fact_coverage_at_k(repeated, GOLD, 5) == 0.5
    assert m.unique_pages_at_k(repeated, 5) == 1
    assert m.duplicate_page_rate_at_k(repeated, 5) == pytest.approx(0.8)


def test_same_page_number_in_another_document_is_not_relevant() -> None:
    assert m.recall_at_k([(B, 3)], GOLD, 1) == 0  # (A, 3) is gold, (B, 3) is not
    assert m.recall_at_k([(A, 3)], GOLD, 1) == 1  # either alternative page for f1 counts


def test_empty_results_and_single_fact_questions() -> None:
    assert m.recall_at_k([], GOLD, 5) == 0
    assert m.fact_coverage_at_k([], GOLD, 5) == 0.0
    assert m.reciprocal_rank([], GOLD) == 0.0
    assert m.duplicate_page_rate_at_k([], 5) is None  # undefined, not zero
    single = GoldEvidence.from_mapping({"f1": [(A, 1)]})
    assert m.complete_coverage_at_k([(A, 1)], single, 1) is None  # protocol: |F| >= 2 only


def test_question_metrics_bundle() -> None:
    result = m.question_metrics(RANKING, GOLD, (1, 4), mrr_cutoff=10)
    assert result["recall"] == {"1": 0, "4": 1}
    assert result["fact_coverage"] == {"1": 0.0, "4": 1.0}
    assert result["mrr"] == {"10": 0.5}


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_invalid_k_is_rejected(bad: object) -> None:
    with pytest.raises(ValueError, match="k must be a positive integer"):
        m.recall_at_k(RANKING, GOLD, bad)  # type: ignore[arg-type]  # invalid on purpose


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        ({}, "at least one required fact"),
        ({"f1": []}, "no accepted evidence page"),
    ],
)
def test_invalid_gold_evidence_is_rejected(facts: dict[str, list[m.PageId]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GoldEvidence.from_mapping(facts)
    with pytest.raises(ValueError, match="duplicate fact ids"):
        GoldEvidence((("f1", frozenset({(A, 1)})), ("f1", frozenset({(A, 2)}))))


# --- aggregation -------------------------------------------------------------------


def test_summarize_reports_denominators_and_undefined_values() -> None:
    summary = m.summarize([1, 0, None, 1])
    assert summary == {"mean": pytest.approx(2 / 3), "n": 3, "undefined": 1}
    assert m.summarize([]) == {"mean": None, "n": 0, "undefined": 0}
    assert m.summarize([None, None]) == {"mean": None, "n": 0, "undefined": 2}


def test_bootstrap_interval_is_seeded_and_bounded() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0]
    first = m.bootstrap_interval(values, resamples=2000, seed=7)
    assert first == m.bootstrap_interval(values, resamples=2000, seed=7)
    assert first is not None
    assert 0.0 <= first[0] <= 0.6 <= first[1] <= 1.0
    assert m.bootstrap_interval([1.0, 1.0, 1.0], resamples=100, seed=1) == (1.0, 1.0)
    assert m.bootstrap_interval([1.0], resamples=100, seed=1) is None  # too few to resample
    with_ci = m.summarize([1, 0, 1], resamples=500, seed=3)
    assert with_ci["ci95"] is not None


def test_paired_comparison_counts_and_skips_undefined_pairs() -> None:
    result = m.paired_comparison([1, 0, 1, None], [1, 1, 0, 1], resamples=500, seed=3)
    assert result["n"] == 3
    assert result["mean_difference"] == 0.0
    assert (result["candidate_better"], result["baseline_better"], result["ties"]) == (1, 1, 1)
    with pytest.raises(ValueError, match="same length"):
        m.paired_comparison([1], [1, 0], resamples=10, seed=1)
    empty = m.paired_comparison([None], [1], resamples=10, seed=1)
    assert empty["mean_difference"] is None
    assert empty["n"] == 0


def test_percentile() -> None:
    assert m.percentile([], 50) is None
    assert m.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_facts_covered_by_an_unordered_page_set() -> None:
    assert m.facts_covered_by_pages({(A, 3)}, GOLD) == ("f1",)  # alternative page counts
    assert m.facts_covered_by_pages([(B, 1), (A, 1), (A, 1)], GOLD) == ("f1", "f2")
    assert m.facts_covered_by_pages(set(), GOLD) == ()


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, {"f1": m.FUSION_CUTOFF_FAILURE, "f2": m.CANDIDATE_RETRIEVAL_FAILURE}),
        (2, {"f1": m.RETAINED, "f2": m.CANDIDATE_RETRIEVAL_FAILURE}),
    ],
)
def test_fact_outcomes_are_exclusive_per_fact(k: int, expected: dict[str, str]) -> None:
    fused = [(A, 2), (A, 3)]  # f1 via its alternative page at rank 2; f2 never retrieved
    pool = {(A, 2), (A, 3), (B, 2)}
    outcomes = m.classify_facts(fused, pool, GOLD, k)
    assert outcomes == expected
    assert set(outcomes.values()) <= set(m.FACT_OUTCOMES)


def test_fact_outcomes_reject_results_outside_the_pool() -> None:
    with pytest.raises(ValueError, match="outside the candidate pool"):
        m.classify_facts([(B, 1)], {(A, 1)}, GOLD, 1)
