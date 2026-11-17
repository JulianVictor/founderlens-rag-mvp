"""Page-level retrieval metrics (evaluation protocol v1.1 §3) and their aggregation.

Pure functions over typed inputs; no I/O. A ranking is the list of page
identities ``(document_id, page_number)`` of the retrieved chunks, in rank
order (rank 1 first), duplicates kept. See ``docs/retrieval_evaluation.md``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

METRICS_VERSION = 1
PageId = tuple[str, int]


def check_k(k: object) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


@dataclass(frozen=True, slots=True)
class GoldEvidence:
    """Required facts, each with its accepted (alternative) evidence pages."""

    facts: tuple[tuple[str, frozenset[PageId]], ...]

    def __post_init__(self) -> None:
        if not self.facts:
            raise ValueError("gold evidence needs at least one required fact")
        ids = [fact_id for fact_id, _ in self.facts]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate fact ids in gold evidence")
        for fact_id, pages in self.facts:
            if not pages:
                raise ValueError(f"fact {fact_id} has no accepted evidence page")

    @classmethod
    def from_mapping(cls, facts: Mapping[str, Sequence[PageId]]) -> GoldEvidence:
        return cls(tuple((fact_id, frozenset(pages)) for fact_id, pages in facts.items()))

    @property
    def pages(self) -> frozenset[PageId]:
        """``G(q)``: every accepted page of every fact."""
        return frozenset().union(*(pages for _, pages in self.facts))


def pages_at_k(ranking: Sequence[PageId], k: int) -> frozenset[PageId]:
    """``P_k``: distinct pages among the first k chunks (no deduplication before the cut)."""
    return frozenset(ranking[: check_k(k)])


def recall_at_k(ranking: Sequence[PageId], gold: GoldEvidence, k: int) -> int:
    """Evidence-page Recall@k: 1 if any accepted page is in the top k (a hit rate)."""
    return int(bool(pages_at_k(ranking, k) & gold.pages))


def covered_facts(ranking: Sequence[PageId], gold: GoldEvidence, k: int) -> tuple[str, ...]:
    found = pages_at_k(ranking, k)
    return tuple(fact_id for fact_id, pages in gold.facts if found & pages)


def facts_covered_by_pages(pages: Collection[PageId], gold: GoldEvidence) -> tuple[str, ...]:
    """Facts with an accepted page in an unordered page set (no rank, no cut-off)."""
    found = frozenset(pages)
    return tuple(fact_id for fact_id, accepted in gold.facts if found & accepted)


RETAINED = "retained"
CANDIDATE_RETRIEVAL_FAILURE = "candidate_retrieval_failure"
FUSION_CUTOFF_FAILURE = "fusion_cutoff_failure"
FACT_OUTCOMES = (RETAINED, CANDIDATE_RETRIEVAL_FAILURE, FUSION_CUTOFF_FAILURE)


def classify_facts(
    fused: Sequence[PageId], pool: Collection[PageId], gold: GoldEvidence, k: int
) -> dict[str, str]:
    """Per required fact, exactly one outcome at k for a fused ranking drawn from ``pool``.

    ``retained``: an accepted page is in the fused top k. ``candidate_retrieval_failure``:
    no accepted page anywhere in the candidate pool. ``fusion_cutoff_failure``: an
    accepted page is in the pool but not in the fused top k. Alternative pages count.
    """
    top = pages_at_k(fused, k)
    pool_pages = frozenset(pool)
    if not top <= pool_pages:
        raise ValueError("fused results contain pages outside the candidate pool")
    outcomes = {}
    for fact_id, accepted in gold.facts:
        if top & accepted:
            outcomes[fact_id] = RETAINED
        elif pool_pages & accepted:
            outcomes[fact_id] = FUSION_CUTOFF_FAILURE
        else:
            outcomes[fact_id] = CANDIDATE_RETRIEVAL_FAILURE
    return outcomes


def fact_coverage_at_k(ranking: Sequence[PageId], gold: GoldEvidence, k: int) -> float:
    """Fraction of required facts with at least one accepted page in the top k."""
    return len(covered_facts(ranking, gold, k)) / len(gold.facts)


def complete_coverage_at_k(ranking: Sequence[PageId], gold: GoldEvidence, k: int) -> int | None:
    """1 if every fact is covered; ``None`` for single-fact questions (protocol: |F| >= 2)."""
    if len(gold.facts) < 2:
        return None
    return int(len(covered_facts(ranking, gold, k)) == len(gold.facts))


def first_relevant_rank(ranking: Sequence[PageId], gold: GoldEvidence) -> int | None:
    """One-based rank of the first chunk on an accepted page, before any deduplication."""
    return next((i for i, page in enumerate(ranking, start=1) if page in gold.pages), None)


def reciprocal_rank(ranking: Sequence[PageId], gold: GoldEvidence, cutoff: int = 10) -> float:
    """``1 / rank_first`` if the first relevant chunk is within ``cutoff``, else 0."""
    rank = first_relevant_rank(ranking[: check_k(cutoff)], gold)
    return 0.0 if rank is None else 1.0 / rank


def unique_pages_at_k(ranking: Sequence[PageId], k: int) -> int:
    return len(pages_at_k(ranking, k))


def duplicate_page_rate_at_k(ranking: Sequence[PageId], k: int) -> float | None:
    """Share of the top-k chunks that repeat a page already seen; ``None`` if none returned."""
    returned = len(ranking[: check_k(k)])
    return None if returned == 0 else 1 - unique_pages_at_k(ranking, k) / returned


def question_metrics(
    ranking: Sequence[PageId], gold: GoldEvidence, ks: Sequence[int], *, mrr_cutoff: int
) -> dict[str, dict[str, float | int | None]]:
    """All metrics for one answerable question, keyed by metric then ``str(k)``."""
    return {
        "recall": {str(k): recall_at_k(ranking, gold, k) for k in ks},
        "fact_coverage": {str(k): fact_coverage_at_k(ranking, gold, k) for k in ks},
        "complete_coverage": {str(k): complete_coverage_at_k(ranking, gold, k) for k in ks},
        "unique_pages": {str(k): unique_pages_at_k(ranking, k) for k in ks},
        "duplicate_page_rate": {str(k): duplicate_page_rate_at_k(ranking, k) for k in ks},
        "mrr": {str(mrr_cutoff): reciprocal_rank(ranking, gold, mrr_cutoff)},
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _defined(values: Sequence[float | int | None]) -> list[float]:
    return [float(v) for v in values if v is not None]


def bootstrap_interval(
    values: Sequence[float], *, resamples: int, seed: int, level: float = 0.95
) -> tuple[float, float] | None:
    """Percentile bootstrap interval of the mean over questions; ``None`` if n < 2."""
    if len(values) < 2:
        return None
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(resamples, len(data)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def summarize(
    values: Sequence[float | int | None],
    *,
    resamples: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Mean of the defined values with its denominator; ``None`` mean for an empty set."""
    defined = _defined(values)
    summary: dict[str, object] = {
        "mean": math.fsum(defined) / len(defined) if defined else None,
        "n": len(defined),
        "undefined": len(values) - len(defined),
    }
    if resamples is not None:
        interval = bootstrap_interval(defined, resamples=resamples, seed=seed)
        summary["ci95"] = None if interval is None else list(interval)
    return summary


def paired_comparison(
    baseline: Sequence[float | int | None],
    candidate: Sequence[float | int | None],
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    """Candidate minus baseline over questions where both are defined."""
    if len(baseline) != len(candidate):
        raise ValueError("paired values must have the same length")
    pairs = [
        (float(a), float(b))
        for a, b in zip(baseline, candidate, strict=True)
        if a is not None and b is not None
    ]
    diffs = [b - a for a, b in pairs]
    interval = bootstrap_interval(diffs, resamples=resamples, seed=seed)
    return {
        "n": len(pairs),
        "mean_difference": math.fsum(diffs) / len(diffs) if diffs else None,
        "ci95": None if interval is None else list(interval),
        "candidate_better": sum(1 for d in diffs if d > 0),
        "baseline_better": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (NumPy default); ``None`` for no values."""
    return None if not values else float(np.percentile(np.asarray(values), q))
