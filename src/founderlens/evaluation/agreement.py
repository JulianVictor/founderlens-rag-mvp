"""Agreement statistics for judge calibration (Stage 6, Commit 7.4; protocol §18).

Usage (offline; prints public aggregates only)::

    python -m founderlens.evaluation.agreement --run <run_id>
    python -m founderlens.evaluation.agreement --run <run_id> --judge <private.jsonl> --write

Two comparisons are kept apart:

* **human-human**: two reviewers' own, independently submitted current decisions
  on the same unit. Adjudications are never substituted for either decision.
  Distinct pseudonyms are taken as distinct reviewers; that is an assertion the
  workflow cannot verify.
* **human-judge**: the human reference label of a unit (single-reviewed,
  independently agreed or adjudicated; never disputed) against a judge label for
  the identical unit, rubric, dimension, run and record. Available with a single
  human reviewer. No judge exists yet: only synthetic fixtures exercise it.

Synthetic stress-test variants form their own population and are never pooled
into natural-output figures. Every population is split by the
calibration/evaluation partition of the original question.

Statistics (:data:`CONFIG`) are computed per dimension, on pairs where both labels
are substantive: raw agreement, Cohen's kappa, Gwet's AC1 and, for ordinal
dimensions only, linearly weighted kappa. ``not_assessable``, ``not_applicable``
and ``unverified`` pairs are excluded from those figures but counted, and the
confusion matrix keeps them as distinct classes. Confidence intervals come from
a seeded bootstrap over original-question clusters and are withheld when too few
clusters exist. Agreement measures consistency between raters, not factual truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from founderlens.config import ConfigError, load_settings
from founderlens.evaluation import annotation as an
from founderlens.evaluation import calibration as cal
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation import judging_input as ji
from founderlens.evaluation.benchmark import EvaluationError
from founderlens.evaluation.judgments import (
    RUBRIC,
    Dimension,
    Judgment,
    Role,
    Source,
    labels_for,
    rubric_id,
)
from founderlens.retrieval.schema import RetrievalError

AGREEMENT_VERSION: Final = 1
NON_SUBSTANTIVE: Final = ("not_assessable", "not_applicable", "unverified")

CONFIG: Final[dict[str, Any]] = {
    "agreement_version": AGREEMENT_VERSION,
    "unit": "one judgment unit and dimension; statistics are per dimension, never pooled "
    "across dimensions with different label sets",
    "substantive_labels": "rubric labels other than not_assessable, not_applicable and unverified",
    "primary_population": "pairs where both labels are substantive",
    "non_substantive_policy": "pairs with a non-substantive label are excluded from the "
    "primary statistics, counted as assessability_mismatch (one side substantive) or "
    "both_non_substantive, and kept as distinct classes in the confusion matrix",
    "cohen_kappa": "(p_o - p_e) / (1 - p_e), p_e = sum_k p_ref(k) p_cmp(k); undefined when "
    "there are no pairs or p_e = 1",
    "gwet_ac1": "(p_o - p_e) / (1 - p_e), p_e = (1 / (q - 1)) sum_k pi_k (1 - pi_k), "
    "pi_k = (p_ref(k) + p_cmp(k)) / 2, q = number of substantive rubric classes of the "
    "dimension; complementary to kappa, never a replacement",
    "weighted_kappa": {
        "dimensions": [d.value for d in Dimension if "ordinal" in RUBRIC[d]],
        "scores": "the rubric's ordinal scores",
        "weights": "linear agreement weights w_ij = 1 - |s_i - s_j| / (s_max - s_min)",
        "nominal_dimensions": "not defined; refused",
    },
    "human_human": {
        "decisions": "each reviewer's current, non-withdrawn, non-adjudication entry",
        "pair_policy": "the headline uses one fixed reviewer pair for every unit: the pair "
        "given explicitly, otherwise the pair sharing the most independently reviewed units "
        "(ties: lexicographic order of the pseudonyms); only units both reviewed enter it",
        "pair_identity": "sha256 of the sorted pseudonym pair, 12 hex; pseudonyms stay private",
        "other_pairs": "every other reviewer pair is a separate population; different pairs "
        "are never pooled into one kappa; pairs share units and are not independent",
        "independence": "asserted by distinct pseudonyms; not verified",
    },
    "human_judge": {
        "reference": "single_reviewed, independently_agreed or adjudicated human label",
        "excluded_references": ["disputed", "withdrawn"],
        "match": "question, run, record digest, rubric, dimension and unit identity",
    },
    "populations": "human_human, human_judge per judge configuration, synthetic_stress per "
    "judge configuration; each split by question partition; synthetic never pooled",
    "bootstrap": {
        "cluster": "original question ID",
        "method": "percentile, linear interpolation between order statistics",
        "seed": 20261125,
        "replicates": 2000,
        "confidence": 0.95,
        "min_clusters": 10,
        "min_pairs": 20,
        "min_valid_fraction": 0.9,
        "undefined_replicates": "counted and dropped",
        "degenerate_interval": "withheld (every replicate equal)",
    },
    "small_sample_pairs": 30,
    "weighting": "unit-level: every paired judgment unit counts once, so questions with more "
    "claims or citations weigh more; the bootstrap resamples whole questions",
    "non_substantive_labels": list(NON_SUBSTANTIVE),
    "class_order": {d.value: list(labels_for(d)) for d in Dimension},
    "ordinal_scores": {d.value: RUBRIC[d]["ordinal"] for d in Dimension if "ordinal" in RUBRIC[d]},
    "min_primary_share": 0.8,
    "category_intervals": "never computed; categories report counts and small-sample flags "
    "(fewer than small_sample_pairs pairs or fewer than min_clusters questions)",
    "partition_rule": "original-question partition from its first ledger entry, withdrawn "
    "history included",
    "target": {
        "statistic": "cohen_kappa",
        "planned_minimum": 0.60,
        "ac1_substitution": "never; AC1 is reported beside kappa, not in place of it",
    },
}


def config_id() -> str:
    canonical = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


class AgreementError(ValueError):
    """An agreement statistic was requested where it is not defined. Codes only."""


# --------------------------------------------------------------------------- #
# Statistics (exact rational arithmetic)
# --------------------------------------------------------------------------- #

LabelPair = tuple[str, str]


def substantive_classes(dimension: Dimension) -> tuple[str, ...]:
    return tuple(label for label in labels_for(dimension) if label not in NON_SUBSTANTIVE)


def _num(value: Fraction | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _marginals(
    pairs: Sequence[LabelPair], classes: Sequence[str]
) -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    n = len(pairs)
    ref, cmp_ = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    return (
        {k: Fraction(ref[k], n) for k in classes},
        {k: Fraction(cmp_[k], n) for k in classes},
    )


def _check_labels(pairs: Sequence[LabelPair], classes: Sequence[str]) -> None:
    allowed = set(classes)
    if any(a not in allowed or b not in allowed for a, b in pairs):
        raise AgreementError("pair label outside the statistic's classes")


def cohen_kappa(pairs: Sequence[LabelPair], classes: Sequence[str]) -> dict[str, Any]:
    _check_labels(pairs, classes)
    if not pairs:
        return {"observed": None, "expected": None, "value": None, "status": "undefined_no_pairs"}
    observed = Fraction(sum(a == b for a, b in pairs), len(pairs))
    p_ref, p_cmp = _marginals(pairs, classes)
    expected = sum((p_ref[k] * p_cmp[k] for k in classes), Fraction(0))
    if expected == 1:
        return {
            "observed": _num(observed),
            "expected": _num(expected),
            "value": None,
            "status": "undefined_degenerate",
        }
    value = (observed - expected) / (1 - expected)
    return {
        "observed": _num(observed),
        "expected": _num(expected),
        "value": _num(value),
        "status": "defined",
    }


def gwet_ac1(pairs: Sequence[LabelPair], classes: Sequence[str]) -> dict[str, Any]:
    _check_labels(pairs, classes)
    q = len(classes)
    if not pairs or q < 2:
        status = "undefined_no_pairs" if not pairs else "undefined_single_class"
        return {"observed": None, "expected": None, "value": None, "status": status}
    observed = Fraction(sum(a == b for a, b in pairs), len(pairs))
    p_ref, p_cmp = _marginals(pairs, classes)
    pi = {k: (p_ref[k] + p_cmp[k]) / 2 for k in classes}
    expected = sum((pi[k] * (1 - pi[k]) for k in classes), Fraction(0)) / (q - 1)
    value = (observed - expected) / (1 - expected)  # expected <= 1/2 < 1 for q >= 2
    return {
        "observed": _num(observed),
        "expected": _num(expected),
        "value": _num(value),
        "status": "defined",
    }


def ordinal_scores(dimension: Dimension) -> dict[str, int]:
    scores = RUBRIC[dimension].get("ordinal")
    if not scores:
        raise AgreementError(f"{dimension.value} is nominal; ordinal weights are not defined")
    return {str(k): int(v) for k, v in scores.items()}


def weighted_kappa(pairs: Sequence[LabelPair], dimension: Dimension) -> dict[str, Any]:
    scores = ordinal_scores(dimension)
    classes = sorted(scores, key=lambda k: scores[k])
    _check_labels(pairs, classes)
    if not pairs:
        return {"observed": None, "expected": None, "value": None, "status": "undefined_no_pairs"}
    span = max(scores.values()) - min(scores.values())

    def w(a: str, b: str) -> Fraction:
        return 1 - Fraction(abs(scores[a] - scores[b]), span)

    n = len(pairs)
    observed = sum((w(a, b) for a, b in pairs), Fraction(0)) / n
    p_ref, p_cmp = _marginals(pairs, classes)
    expected = sum((w(a, b) * p_ref[a] * p_cmp[b] for a in classes for b in classes), Fraction(0))
    if expected == 1:
        return {
            "observed": _num(observed),
            "expected": _num(expected),
            "value": None,
            "status": "undefined_degenerate",
        }
    return {
        "observed": _num(observed),
        "expected": _num(expected),
        "value": _num((observed - expected) / (1 - expected)),
        "status": "defined",
        "weights": "linear",
    }


def confusion(pairs: Sequence[LabelPair], dimension: Dimension) -> dict[str, Any]:
    """All rubric labels as distinct classes, in rubric order (rows: reference)."""
    classes = list(labels_for(dimension))
    _check_labels(pairs, classes)
    counts = Counter(pairs)
    matrix = [[counts[(a, b)] for b in classes] for a in classes]
    return {
        "classes": classes,
        "matrix": matrix,
        "total": len(pairs),
        "reference_marginals": [sum(row) for row in matrix],
        "compared_marginals": [sum(col) for col in zip(*matrix, strict=True)],
        "disagreements_by_reference_class": {
            a: sum(counts[(a, b)] for b in classes if b != a) for a in classes
        },
    }


def raw_agreement(pairs: Sequence[LabelPair]) -> dict[str, Any]:
    agree = sum(a == b for a, b in pairs)
    return {
        "pairs": len(pairs),
        "agree": agree,
        "disagree": len(pairs) - agree,
        "rate": _num(Fraction(agree, len(pairs))) if pairs else None,
    }


# --------------------------------------------------------------------------- #
# Pairs and question-clustered bootstrap
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Pair:
    """One paired decision on one unit (private: holds pseudonyms and unit keys)."""

    question_id: str
    unit_key: str
    dimension: Dimension
    reference: str
    compared: str
    question_category: str
    partition: str
    synthetic: bool = False
    lineage: str | None = None  # synthetic variant ID
    reviewers: tuple[str, str] = ("", "")
    reference_state: str | None = None  # human-judge: review state of the reference

    @property
    def labels(self) -> LabelPair:
        return (self.reference, self.compared)


def _quantile(values: Sequence[float], p: float) -> float:
    h = (len(values) - 1) * p
    lo = int(h)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (h - lo) * (values[hi] - values[lo])


def cluster_bootstrap(
    pairs: Sequence[Pair],
    statistic: Callable[[Sequence[Pair]], float | None],
    config: Mapping[str, Any] = CONFIG["bootstrap"],
) -> dict[str, Any]:
    """Percentile interval from resampling whole original-question clusters."""
    clusters: dict[str, list[Pair]] = {}
    for p in pairs:
        clusters.setdefault(p.question_id, []).append(p)
    keys = sorted(clusters)
    base = {
        "clusters": len(keys),
        "pairs": len(pairs),
        "seed": config["seed"],
        "replicates": config["replicates"],
        "confidence": config["confidence"],
        "valid_replicates": 0,
        "undefined_replicates": 0,
        "interval": None,
    }
    if len(keys) < config["min_clusters"] or len(pairs) < config["min_pairs"]:
        return {**base, "replicates": 0, "status": "insufficient_data"}
    rng = random.Random(config["seed"])
    values: list[float] = []
    for _ in range(config["replicates"]):
        sample = [p for _ in keys for p in clusters[keys[rng.randrange(len(keys))]]]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    values.sort()
    result = {
        **base,
        "valid_replicates": len(values),
        "undefined_replicates": config["replicates"] - len(values),
    }
    if len(values) < config["min_valid_fraction"] * config["replicates"]:
        return {**result, "status": "unstable_too_many_undefined"}
    alpha = 1 - config["confidence"]
    lo, hi = _quantile(values, alpha / 2), _quantile(values, 1 - alpha / 2)
    if lo == hi:
        return {**result, "status": "degenerate_interval"}
    return {**result, "status": "estimated", "interval": [round(lo, 6), round(hi, 6)]}


def _kappa_of(dimension: Dimension) -> Callable[[Sequence[Pair]], float | None]:
    classes = substantive_classes(dimension)

    def stat(sample: Sequence[Pair]) -> float | None:
        value: float | None = cohen_kappa([p.labels for p in sample], classes)["value"]
        return value

    return stat


def target_assessment(kappa: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    target = CONFIG["target"]["planned_minimum"]
    value = kappa.get("value")
    if value is None:
        return {"target": target, "status": "insufficient_information", "interval": None}
    interval = bootstrap.get("interval")
    ci = (
        "not_estimable"
        if interval is None
        else "lower_bound_meets_target"
        if interval[0] >= target
        else "interval_includes_values_below_target"
    )
    return {
        "target": target,
        "statistic": "cohen_kappa",
        "status": "point_meets_target" if value >= target else "point_below_target",
        "interval": ci,
        "caveat": "a point estimate alone does not establish judge reliability",
    }


def analyze_dimension(
    pairs: Sequence[Pair], dimension: Dimension, scheduled: int
) -> dict[str, Any]:
    all_pairs = [p.labels for p in pairs]
    primary = [
        p for p in pairs if p.reference not in NON_SUBSTANTIVE and p.compared not in NON_SUBSTANTIVE
    ]
    mismatch = sum(
        (p.reference in NON_SUBSTANTIVE) != (p.compared in NON_SUBSTANTIVE) for p in pairs
    )
    classes = substantive_classes(dimension)
    labels = [p.labels for p in primary]
    kappa = cohen_kappa(labels, classes)
    bootstrap = cluster_bootstrap(primary, _kappa_of(dimension))
    weighted = None
    if "ordinal" in RUBRIC[dimension]:
        weighted = weighted_kappa(labels, dimension)
    categories: dict[str, dict[str, Any]] = {}
    category_questions: dict[str, set[str]] = {}
    for p in primary:
        row = categories.setdefault(p.question_category, {"pairs": 0, "agree": 0})
        row["pairs"] += 1
        row["agree"] += p.reference == p.compared
        category_questions.setdefault(p.question_category, set()).add(p.question_id)
    small = CONFIG["small_sample_pairs"]
    for category, row in categories.items():
        row["questions"] = len(category_questions[category])
        row["small_sample"] = (
            row["pairs"] < small or row["questions"] < CONFIG["bootstrap"]["min_clusters"]
        )
        row["interval"] = None  # never estimated per category
    per_question: dict[str, dict[str, int]] = {}
    for p in pairs:
        row_q = per_question.setdefault(p.question_id, {"paired_units": 0, "primary_pairs": 0})
        row_q["paired_units"] += 1
        row_q["primary_pairs"] += (
            p.reference not in NON_SUBSTANTIVE and p.compared not in NON_SUBSTANTIVE
        )
    prevalence = Counter(label for pair in labels for label in pair)
    share = Fraction(len(primary), len(pairs)) if pairs else None
    warnings = []
    if len(primary) < small:
        warnings.append("small_sample")
    if share is not None and share < Fraction(str(CONFIG["min_primary_share"])):
        warnings.append("low_primary_share")  # many non-substantive labels left out
    if bootstrap["status"] != "estimated":
        warnings.append(f"interval_{bootstrap['status']}")
    return {
        "estimate_level": "unit",
        "scheduled_units": scheduled,
        "paired_units": len(pairs),
        "coverage": _num(Fraction(len(pairs), scheduled)) if scheduled else None,
        "primary_pairs": len(primary),
        "assessability_mismatch": mismatch,
        "both_non_substantive": len(pairs) - len(primary) - mismatch,
        "reference_non_substantive": sum(p.reference in NON_SUBSTANTIVE for p in pairs),
        "compared_non_substantive": sum(p.compared in NON_SUBSTANTIVE for p in pairs),
        "primary_share": _num(share),
        "questions": len({p.question_id for p in pairs}),
        "clusters": len({p.question_id for p in primary}),
        "per_question": dict(sorted(per_question.items())),
        "raw": raw_agreement(labels),
        "cohen_kappa": kappa,
        "gwet_ac1": gwet_ac1(labels, classes),
        "weighted_kappa": weighted,
        "confusion": confusion(all_pairs, dimension),
        "prevalence": {k: prevalence[k] for k in classes},
        "by_question_category": dict(sorted(categories.items())),
        "bootstrap_kappa": bootstrap,
        "target": target_assessment(kappa, bootstrap),
        "warnings": warnings,
    }


def analyze_population(pairs: Sequence[Pair], scheduled: Mapping[str, int]) -> dict[str, Any]:
    by_dim: dict[Dimension, list[Pair]] = {}
    for p in pairs:
        by_dim.setdefault(p.dimension, []).append(p)
    return {
        "population_id": population_id(pairs),
        "paired_units": len(pairs),
        "clusters": len({p.question_id for p in pairs}),
        "dimensions": {
            d.value: analyze_dimension(
                sorted(by_dim.get(d, []), key=lambda p: p.unit_key), d, scheduled.get(d.value, 0)
            )
            for d in Dimension
            if d in by_dim
        },
    }


def population_id(pairs: Sequence[Pair]) -> str:
    body = sorted(
        (p.unit_key, p.dimension.value, p.reference, p.compared, p.lineage or "") for p in pairs
    )
    return an.digest({"config": config_id(), "pairs": body})[:12]


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #


def _dimension_of(unit_key: str) -> Dimension:
    return Dimension(unit_key.rsplit("#", 1)[1])


def _question_of(unit_key: str) -> str:
    return unit_key.split(":", 1)[1].split("/", 1)[0]


def _valid_human(
    ctx: an.AnnotationContext, entry: an.LedgerEntry
) -> tuple[Judgment | None, str | None]:
    """The entry's judgment if usable as an independent human decision, else a code."""
    if entry.action == "withdrawn":
        return None, "withdrawn"
    if entry.adjudication:
        return None, "adjudication_not_independent"
    if (
        entry.rubric_id != rubric_id()
        or entry.payload.get("provenance", {}).get("rubric_id") != rubric_id()
    ):
        return None, "rubric_mismatch"
    q = ctx.run.question(entry.question_id)
    if q is None or not q.eligible:
        return None, "model_output_failure_or_unknown_question"
    try:
        judgment = Judgment.model_validate(entry.payload)
    except ValidationError:
        return None, "malformed_provenance"
    prov = judgment.provenance
    if prov.source is not Source.HUMAN or prov.annotator_id != entry.annotator_id:
        return None, "malformed_provenance"
    if (judgment.generation_run_id, judgment.generation_record_sha256) != (
        ctx.run.run_id,
        q.record_sha256,
    ) or entry.generation_record_sha256 != q.record_sha256:
        return None, "stale_record"
    return judgment, None


def _pair(
    ctx: an.AnnotationContext,
    unit_key: str,
    reference: str,
    compared: str,
    partition: Role,
    reviewers: tuple[str, str],
    *,
    synthetic: bool = False,
    lineage: str | None = None,
    question_id: str | None = None,
    dimension: Dimension | None = None,
) -> Pair:
    qid = question_id or _question_of(unit_key)
    return Pair(
        question_id=qid,
        unit_key=unit_key,
        dimension=dimension or _dimension_of(unit_key),
        reference=reference,
        compared=compared,
        question_category=ctx.question(qid).category,
        partition=partition.value,
        synthetic=synthetic,
        lineage=lineage,
        reviewers=reviewers,
    )


@dataclass(frozen=True, slots=True)
class Pairing:
    headline: dict[str, list[Pair]]  # partition -> pairs of the one selected reviewer pair
    by_reviewer_pair: dict[tuple[str, str], list[Pair]]
    exclusions: dict[str, int]
    selected_pair: tuple[str, str] | None = None
    selection: str = "none"  # explicit | most_shared_units | none
    headline_accounting: Mapping[str, int] | None = None


def reviewer_pair_id(pair: tuple[str, str]) -> str:
    """A stable, non-revealing identity of an unordered reviewer pair."""
    return an.digest(["reviewer_pair", sorted(pair)])[:12]


def human_human_pairs(
    ctx: an.AnnotationContext,
    ledger: an.Ledger,
    reviewers: tuple[str, str] | None = None,
) -> Pairing:
    """Independent human decisions of one fixed reviewer pair; other pairs kept apart.

    Adjudications, withdrawn and superseded entries never act as either decision.
    """
    if reviewers is not None and (
        reviewers[0] == reviewers[1]
        or not all(re.fullmatch(an.ANNOTATOR_PATTERN, r) for r in reviewers)
    ):
        raise AgreementError("the headline reviewer pair needs two distinct pseudonyms")
    exclusions: Counter[str] = Counter()
    exclusions["superseded_revisions_not_used"] = ledger.superseded()
    partitions = ledger.partitions()
    units: dict[str, list[an.LedgerEntry]] = {}
    for e in ledger.heads().values():
        if e.entry_type == "judgment":
            units.setdefault(e.unit_key, []).append(e)
    decisions: dict[str, dict[str, tuple[Judgment, Role]]] = {}
    for unit_key, heads in sorted(units.items()):
        valid: dict[str, tuple[Judgment, Role]] = {}
        for e in sorted(heads, key=lambda x: x.annotator_id):
            judgment, code = _valid_human(ctx, e)
            if code is not None:
                exclusions[code] += 1
            elif judgment is not None:
                valid[e.annotator_id] = (judgment, e.role)
        roles = {role for _, role in valid.values()}
        if len(roles) > 1 or any(r is not partitions.get(_question_of(unit_key)) for r in roles):
            exclusions["role_conflict"] += 1
            continue
        if len(valid) < 2:
            exclusions["missing_independent_pair"] += 1
        if valid:
            decisions[unit_key] = valid
    reviewer_pairs: dict[tuple[str, str], list[Pair]] = {}
    for unit_key, valid in decisions.items():
        names = sorted(valid)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                (ja, role), (jb, _) = valid[a], valid[b]
                pair = _pair(ctx, unit_key, ja.label, jb.label, role, (a, b))
                reviewer_pairs.setdefault((a, b), []).append(pair)
    if reviewers is not None:
        selected: tuple[str, str] | None = (min(reviewers), max(reviewers))
        selection = "explicit"
    elif reviewer_pairs:
        selected = min(reviewer_pairs, key=lambda k: (-len(reviewer_pairs[k]), k))
        selection = "most_shared_units"
    else:
        selected, selection = None, "none"
    headline: dict[str, list[Pair]] = {}
    accounting: dict[str, int] = {}
    if selected is not None:
        first, second = selected
        for pair in reviewer_pairs.get(selected, []):
            headline.setdefault(pair.partition, []).append(pair)
        accounting = {
            "intersection_units": len(reviewer_pairs.get(selected, [])),
            "missing_second_reviewer_label": sum(
                first in v and second not in v for v in decisions.values()
            ),
            "missing_first_reviewer_label": sum(
                second in v and first not in v for v in decisions.values()
            ),
            "units_reviewed_only_by_other_reviewers": sum(
                first not in v and second not in v for v in decisions.values()
            ),
        }
    return Pairing(
        headline,
        reviewer_pairs,
        dict(sorted(exclusions.items())),
        selected,
        selection,
        accounting,
    )


def human_references(
    ctx: an.AnnotationContext, ledger: an.Ledger
) -> tuple[dict[str, tuple[str, an.ReviewState]], dict[str, int]]:
    """The human reference label per unit; disputed and withdrawn units have none."""
    exclusions: Counter[str] = Counter()
    refs: dict[str, tuple[str, an.ReviewState]] = {}
    for unit_key, (state, decisive) in ledger.resolution("judgment").items():
        if state is an.ReviewState.DISPUTED:
            exclusions["disputed_reference"] += 1
            continue
        if state is an.ReviewState.WITHDRAWN or not decisive:
            exclusions["withdrawn"] += 1
            continue
        entry = decisive[0]
        if entry.adjudication:
            entry = entry.model_copy(update={"adjudication": False})  # reference, not a pair
        judgment, code = _valid_human(ctx, entry)
        if judgment is None:
            exclusions[code or "malformed_provenance"] += 1
            continue
        refs[unit_key] = (judgment.label, state)
    return refs, dict(sorted(exclusions.items()))


def _load_judge_output(raw: Mapping[str, Any]) -> tuple[Judgment | None, str | None]:
    candidate = raw.get("provenance")
    provenance: Mapping[str, Any] = candidate if isinstance(candidate, Mapping) else {}
    if provenance.get("rubric_id") != rubric_id():
        return None, "rubric_mismatch"
    try:
        judgment = Judgment.model_validate(raw)
    except ValidationError:
        return None, "malformed_provenance"
    if judgment.provenance.source is not Source.JUDGE:
        return None, "malformed_provenance"
    return judgment, None


def human_judge_pairs(
    ctx: an.AnnotationContext, ledger: an.Ledger, judge_outputs: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, list[Pair]]], dict[str, int]]:
    """Per judge configuration and partition, human reference vs judge label."""
    refs, ref_exclusions = human_references(ctx, ledger)
    exclusions: Counter[str] = Counter(ref_exclusions)
    partitions = ledger.partitions()
    decompositions = ledger.decompositions()
    outputs: dict[tuple[str, str], list[Judgment]] = {}
    for raw in judge_outputs:
        judgment, code = _load_judge_output(raw)
        if judgment is None:
            exclusions[code or "malformed_provenance"] += 1
            continue
        q = ctx.run.question(judgment.question_id)
        if q is None or not q.eligible:
            exclusions["model_output_failure_or_unknown_question"] += 1
            continue
        if (judgment.generation_run_id, judgment.generation_record_sha256) != (
            ctx.run.run_id,
            q.record_sha256,
        ):
            exclusions["stale_record"] += 1
            continue
        if ji.check_judgment(judgment, ctx.run, ctx.collection, decompositions):
            exclusions["unbound_unit"] += 1
            continue
        key = f"judgment:{judgment.unit_id}#{judgment.dimension.value}"
        config = str(judgment.provenance.judge_config_id)
        outputs.setdefault((config, key), []).append(judgment)
    out: dict[str, dict[str, list[Pair]]] = {}
    configs = sorted({c for c, _ in outputs})
    for (config, key), judged in sorted(outputs.items()):
        if len(judged) > 1:
            exclusions["duplicate_judge_output"] += len(judged)
            continue
        judgment = judged[0]
        if key not in refs:
            exclusions["missing_human_reference"] += 1
            continue
        partition = partitions.get(judgment.question_id)
        if partition is None or judgment.provenance.role is not partition:
            exclusions["role_conflict"] += 1
            continue
        reference, state = refs[key]
        pair = _pair(ctx, key, reference, judgment.label, partition, ("human", config))
        pair = replace(pair, reference_state=state.value)
        out.setdefault(config, {}).setdefault(partition.value, []).append(pair)
    for config in configs:
        judged_keys = {k for c, k in outputs if c == config}
        exclusions["missing_judge_label"] += sum(1 for k in refs if k not in judged_keys)
    return out, dict(sorted(exclusions.items()))


@dataclass(frozen=True, slots=True)
class SyntheticJudgeLabel:
    variant_id: str
    label: str
    judge_config_id: str


def synthetic_pairs(
    ctx: an.AnnotationContext, ledger: an.Ledger, labels: Sequence[SyntheticJudgeLabel]
) -> tuple[dict[str, list[Pair]], dict[str, int]]:
    """Judge labels on synthetic variants vs their human-accepted expected labels."""
    variants = {v.variant_id: v for v in cal.synthetic_variants(ctx, ledger)}
    exclusions: Counter[str] = Counter()
    out: dict[str, list[Pair]] = {}
    seen: Counter[tuple[str, str]] = Counter((x.judge_config_id, x.variant_id) for x in labels)
    for x in sorted(labels, key=lambda x: (x.judge_config_id, x.variant_id)):
        variant = variants.get(x.variant_id)
        if variant is None:
            exclusions["unknown_variant"] += 1
            continue
        if seen[(x.judge_config_id, x.variant_id)] > 1:
            exclusions["duplicate_judge_output"] += 1
            continue
        if variant.review != "accepted":
            exclusions["synthetic_label_not_human_accepted"] += 1
            continue
        dimension = Dimension(str(variant.expected["dimension"]))
        if x.label not in labels_for(dimension):
            exclusions["invalid_label"] += 1
            continue
        pair = _pair(
            ctx,
            f"synthetic:{variant.variant_id}#{dimension.value}",
            str(variant.expected["label"]),
            x.label,
            Role.CALIBRATION,  # variants inherit their calibration source question
            ("expected", x.judge_config_id),
            synthetic=True,
            lineage=variant.variant_id,
            question_id=variant.question_id,
            dimension=dimension,
        )
        out.setdefault(x.judge_config_id, []).append(pair)
    return out, dict(sorted(exclusions.items()))


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def scheduled_units(ctx: an.AnnotationContext) -> dict[str, int]:
    counts: Counter[str] = Counter(_dimension_of(u).value for u in an.template_judgment_units(ctx))
    return dict(sorted(counts.items()))


def build_report(
    ctx: an.AnnotationContext,
    ledger: an.Ledger,
    judge_outputs: Sequence[Mapping[str, Any]] = (),
    synthetic_labels: Sequence[SyntheticJudgeLabel] = (),
    reviewers: tuple[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(public aggregate report, private detail). Deterministic; no timestamps."""
    scheduled = scheduled_units(ctx)
    hh = human_human_pairs(ctx, ledger, reviewers)
    hj, hj_exclusions = human_judge_pairs(ctx, ledger, judge_outputs)
    syn, syn_exclusions = synthetic_pairs(ctx, ledger, synthetic_labels)
    populations: dict[str, dict[str, Any]] = {}
    detail_pairs: dict[str, list[dict[str, Any]]] = {}

    def add(name: str, kind: str, partition: str, pairs: list[Pair], **meta: Any) -> None:
        populations[name] = {
            "kind": kind,
            "partition": partition,
            "synthetic": kind == "synthetic_stress",
            **meta,
            **analyze_population(pairs, scheduled),
        }
        detail_pairs[name] = [
            {
                "question_id": p.question_id,
                "unit_key": p.unit_key,
                "dimension": p.dimension.value,
                "reference": p.reference,
                "compared": p.compared,
                "reviewers": list(p.reviewers),
                "lineage": p.lineage,
                "reference_state": p.reference_state,
            }
            for p in pairs
        ]

    selected_id = None if hh.selected_pair is None else reviewer_pair_id(hh.selected_pair)
    for partition, pairs in sorted(hh.headline.items()):
        add(
            f"human_human:{partition}",
            "human_human",
            partition,
            pairs,
            reviewer_pair_id=selected_id,
            pair_selection=hh.selection,
            headline_accounting=dict(hh.headline_accounting or {}),
        )
    for pair_key, pairs in sorted(hh.by_reviewer_pair.items()):
        if pair_key == hh.selected_pair:
            continue
        by_partition: dict[str, list[Pair]] = {}
        for p in pairs:
            by_partition.setdefault(p.partition, []).append(p)
        for partition, part in sorted(by_partition.items()):
            add(
                f"human_human_reviewer_pair:{reviewer_pair_id(pair_key)}:{partition}",
                "human_human_reviewer_pair",
                partition,
                part,
                reviewer_pair_id=reviewer_pair_id(pair_key),
                dependence="reviewer pairs share units; not independent of the headline",
            )
    for config, by_partition in sorted(hj.items()):
        for partition, pairs in sorted(by_partition.items()):
            states = Counter(str(p.reference_state) for p in pairs)
            add(f"human_judge:{config}:{partition}", "human_judge", partition, pairs,
                judge_config_id=config, reference_states=dict(sorted(states.items())),
                reference_note="agreement with the human reference, not factual truth")  # fmt: skip
    for config, pairs in sorted(syn.items()):
        add(f"synthetic_stress:{config}", "synthetic_stress", "calibration", pairs,
            judge_config_id=config)  # fmt: skip
    independent = ledger.independently_reviewed_units()
    report = {
        "agreement_version": AGREEMENT_VERSION,
        "config_id": config_id(),
        "run_id": ctx.run.run_id,
        **an.annotation_identity(),
        "scheduled_units": scheduled,
        "independently_reviewed_units": len(independent),
        "human_human_paired_units": sum(len(p) for p in hh.headline.values()),
        "human_human_pair_selection": hh.selection,
        "human_judge_paired_units": sum(len(p) for b in hj.values() for p in b.values()),
        "synthetic_paired_units": sum(len(p) for p in syn.values()),
        "exclusions": {
            "human_human": hh.exclusions,
            "human_judge": hj_exclusions,
            "synthetic": syn_exclusions,
        },
        "populations": populations,
        "measurable": bool(populations),
        "interpretation": "agreement is rater consistency, not factual truth",
    }
    report["report_id"] = an.digest(report)[:12]
    detail = {"report_id": report["report_id"], "config": CONFIG, "pairs": detail_pairs}
    return an.ensure_public(report), detail


def load_judge_outputs(path: Path, paths: an.AnnotationPaths) -> list[dict[str, Any]]:
    if not path.resolve().is_relative_to(paths.root.resolve()):
        raise an.AnnotationError(["judge outputs must live in the private annotation directory"])
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError):
        raise an.AnnotationError(["cannot read the judge outputs as JSON lines"]) from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.agreement",
        description="Offline agreement statistics over the private annotation ledger.",
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--judge", default=None, help="private JSONL of judge judgments")
    parser.add_argument("--write", action="store_true", help="save the private detail")
    parser.add_argument(
        "--reviewers", nargs=2, default=None, help="the fixed headline reviewer pair (pseudonyms)"
    )
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        paths = an.annotation_paths(settings, args.run)
        ctx = an.load_context(settings, args.run)
        ledger = an.load_ledger(paths.ledger)
        judge = load_judge_outputs(Path(args.judge), paths) if args.judge else []
        pair = None if args.reviewers is None else (args.reviewers[0], args.reviewers[1])
        report, detail = build_report(ctx, ledger, judge, reviewers=pair)
        if args.write:
            an.write_json(paths.root / "agreement" / f"{report['report_id']}.json", detail)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (
        an.AnnotationError,
        AgreementError,
        ji.JudgingInputError,
        ge.EvaluationRunError,
        EvaluationError,
        RetrievalError,
    ) as exc:
        print(f"Agreement analysis FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
