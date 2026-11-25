"""Agreement statistics and pairing (Commit 7.4), offline and synthetic.

Expected statistic values below were computed by hand from the stated tables
(observed agreement, marginals and chance agreement written out in comments),
not by calling the implementation.
"""

from __future__ import annotations

import copy
import json
import pathlib
import random
import socket
from collections.abc import Sequence
from typing import Any

import pytest

from founderlens.evaluation import agreement as ag
from founderlens.evaluation import annotation as an
from founderlens.evaluation import calibration as cal
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation.judgments import Dimension, Role, rubric_id
from tests.test_evaluation_annotation import (
    Q4,
    _labelled_ledger,
    commit,
    find,
    only,
    private_texts,
    sheet,
    span,
    supported,
)
from tests.test_evaluation_generation_eval import SPLIT
from tests.test_evaluation_judging_input import CLEAN, _run, _verify
from tests.test_guardrails_response_checks import COLLECTION, VOCAB


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(ge, "_git_state", lambda root=None: dict(CLEAN))


@pytest.fixture
def ctx(tmp_path: pathlib.Path) -> an.AnnotationContext:
    inputs = _run(tmp_path / "run")
    return an.make_context(_verify(inputs), COLLECTION, SPLIT, inputs["records"], VOCAB)


FAITH = Dimension.CLAIM_FAITHFULNESS
CLAIM1 = "judgment:dev-003/claim:1#claim_faithfulness"
CLAIM2 = "judgment:dev-003/claim:2#claim_faithfulness"
JUDGE = "abcdef123456"


def table(cells: dict[tuple[str, str], int]) -> list[tuple[str, str]]:
    return [pair for pair, n in cells.items() for _ in range(n)]


def approx(value: float | None, expected: float) -> bool:
    return value is not None and abs(value - expected) < 1e-6


# --- statistics: known values ----------------------------------------------------------------

AB = ("A", "B")


def test_perfect_agreement() -> None:
    pairs = table({("A", "A"): 5, ("B", "B"): 5})
    assert ag.cohen_kappa(pairs, AB)["value"] == 1.0
    assert ag.gwet_ac1(pairs, AB)["value"] == 1.0
    assert ag.raw_agreement(pairs) == {"pairs": 10, "agree": 10, "disagree": 0, "rate": 1.0}


def test_complete_disagreement() -> None:
    # p_o = 0; marginals 0.5/0.5 each: p_e(kappa) = 0.5, p_e(AC1) = 2 * 0.5 * 0.5 = 0.5
    pairs = table({("A", "B"): 5, ("B", "A"): 5})
    assert ag.cohen_kappa(pairs, AB)["value"] == -1.0
    assert ag.gwet_ac1(pairs, AB)["value"] == -1.0


def test_partial_agreement_known_values() -> None:
    # [[20, 5], [10, 15]], n = 50: p_o = 0.7; ref 0.5/0.5, cmp 0.6/0.4
    # kappa: p_e = 0.5*0.6 + 0.5*0.4 = 0.5 -> (0.7 - 0.5) / 0.5 = 0.4
    # AC1: pi = 0.55/0.45, p_e = 2*0.55*0.45 = 0.495 -> 0.205 / 0.505 = 0.405941
    pairs = table({("A", "A"): 20, ("A", "B"): 5, ("B", "A"): 10, ("B", "B"): 15})
    kappa = ag.cohen_kappa(pairs, AB)
    assert (kappa["observed"], kappa["expected"], kappa["value"]) == (0.7, 0.5, 0.4)
    ac1 = ag.gwet_ac1(pairs, AB)
    assert approx(ac1["expected"], 0.495)
    assert approx(ac1["value"], 0.405941)


def test_class_imbalance_kappa_paradox() -> None:
    # [[90, 5], [5, 0]]: p_o = 0.9; both raters 0.95/0.05
    # kappa: p_e = 0.9025 + 0.0025 = 0.905 -> -0.005 / 0.095 = -1/19 = -0.052632
    # AC1: p_e = 2*0.95*0.05 = 0.095 -> 0.805 / 0.905 = 0.889503
    pairs = table({("A", "A"): 90, ("A", "B"): 5, ("B", "A"): 5})
    assert approx(ag.cohen_kappa(pairs, AB)["value"], -0.052632)
    assert approx(ag.gwet_ac1(pairs, AB)["value"], 0.889503)


def test_three_class_kappa_and_ac1() -> None:
    # supported/unsupported/contradicted: (s,s)6 (s,u)2 (u,u)1 (c,c)1, n = 10, p_o = 0.8
    # ref .8/.1/.1, cmp .6/.3/.1: kappa p_e = .48+.03+.01 = .52 -> .28/.48 = 0.583333
    # pi = .7/.2/.1: AC1 p_e = (.21+.16+.09)/2 = .23 -> .57/.77 = 0.740260
    classes = ag.substantive_classes(FAITH)
    assert classes == ("supported", "unsupported", "contradicted")
    pairs = table(
        {
            ("supported", "supported"): 6,
            ("supported", "unsupported"): 2,
            ("unsupported", "unsupported"): 1,
            ("contradicted", "contradicted"): 1,
        }
    )
    assert approx(ag.cohen_kappa(pairs, classes)["value"], 0.583333)
    assert approx(ag.gwet_ac1(pairs, classes)["value"], 0.740260)


def test_one_class_degeneracy_and_empty_populations() -> None:
    same = [("A", "A")] * 8
    kappa = ag.cohen_kappa(same, AB)
    assert (kappa["value"], kappa["status"]) == (None, "undefined_degenerate")
    assert ag.gwet_ac1(same, AB)["value"] == 1.0  # pi = 1/0: p_e = 0
    for result in (ag.cohen_kappa([], AB), ag.gwet_ac1([], AB)):
        assert (result["value"], result["status"]) == (None, "undefined_no_pairs")
    assert ag.raw_agreement([])["rate"] is None
    assert ag.weighted_kappa([], Dimension.ANSWER_RELEVANCE)["status"] == "undefined_no_pairs"


def test_linear_weighted_relevance_kappa() -> None:
    # scores off_topic 0, partially_relevant 1, directly_answers 2; rows ref, cols cmp:
    # [[2,1,0],[0,2,1],[0,0,2]], n = 8; weights 1 / 0.5 / 0
    # p_o_w = 6/8 + 0.5*2/8 = 0.875; ref 3,3,2 / cmp 2,3,3 (/8)
    # p_e_w = (21 + 0.5*30) / 64 = 0.5625 -> 0.3125 / 0.4375 = 5/7 = 0.714286
    # unweighted: p_o = .75, p_e = 21/64 -> 27/43 = 0.627907
    o, p, d = "off_topic", "partially_relevant", "directly_answers"
    pairs = table({(o, o): 2, (o, p): 1, (p, p): 2, (p, d): 1, (d, d): 2})
    weighted = ag.weighted_kappa(pairs, Dimension.ANSWER_RELEVANCE)
    assert (weighted["observed"], weighted["expected"]) == (0.875, 0.5625)
    assert approx(weighted["value"], 0.714286)
    classes = ag.substantive_classes(Dimension.ANSWER_RELEVANCE)
    assert approx(ag.cohen_kappa(pairs, classes)["value"], 0.627907)


@pytest.mark.parametrize(
    "dimension",
    [d for d in Dimension if d is not Dimension.ANSWER_RELEVANCE],
)
def test_nominal_dimensions_refuse_ordinal_weights(dimension: Dimension) -> None:
    with pytest.raises(ag.AgreementError, match="nominal"):
        ag.weighted_kappa([], dimension)


def test_non_substantive_labels_are_never_merged_into_statistics() -> None:
    classes = ag.substantive_classes(FAITH)
    with pytest.raises(ag.AgreementError, match="outside"):
        ag.cohen_kappa([("not_assessable", "unsupported")], classes)
    pairs = [("not_assessable", "unsupported"), ("unsupported", "unsupported")]
    matrix = ag.confusion(pairs, FAITH)
    assert matrix["classes"] == ["supported", "unsupported", "contradicted", "not_assessable"]
    assert matrix["matrix"][3][1] == 1  # kept distinct from the substantive negative
    assert matrix["total"] == sum(map(sum, matrix["matrix"])) == 2
    assert matrix["reference_marginals"] == [0, 1, 0, 1]
    assert matrix["compared_marginals"] == [0, 2, 0, 0]
    assert matrix["disagreements_by_reference_class"]["not_assessable"] == 1


# --- bootstrap -------------------------------------------------------------------------------


def pairs_for(clusters: dict[str, list[tuple[str, str]]]) -> list[ag.Pair]:
    return [
        ag.Pair(
            qid, f"judgment:{qid}/claim:{i}#claim_faithfulness", FAITH, a, b, "cat", "calibration"
        )
        for qid, labels in clusters.items()
        for i, (a, b) in enumerate(labels, start=1)
    ]


def stat(sample: Sequence[ag.Pair]) -> float | None:
    value: float | None = ag.cohen_kappa([p.labels for p in sample], ag.substantive_classes(FAITH))[
        "value"
    ]
    return value


S, U = "supported", "unsupported"


def test_too_few_question_clusters_give_no_interval() -> None:
    many = pairs_for({f"dev-{i:03d}": [(S, S), (U, U), (S, U), (U, U), (S, S)] for i in range(9)})
    result = ag.cluster_bootstrap(many, stat)
    assert (result["status"], result["interval"], result["clusters"]) == (
        "insufficient_data",
        None,
        9,
    )


def test_bootstrap_resamples_whole_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    clusters = {f"dev-{i:03d}": [(S, S), (U, U)] for i in range(10)}
    clusters["dev-000"] = [(S, S), (U, U), (S, U), (U, S), (S, S)]  # one question, five claims
    seen: list[list[ag.Pair]] = []

    def recording(sample: Sequence[ag.Pair]) -> float | None:
        seen.append(list(sample))
        return stat(sample)

    ag.cluster_bootstrap(pairs_for(clusters), recording)
    assert len(seen) == ag.CONFIG["bootstrap"]["replicates"]
    for sample in seen:
        assert sum(p.question_id == "dev-000" for p in sample) % 5 == 0  # never split
        assert len({p.question_id for p in sample}) <= 11


def test_fixed_seed_reproducibility_and_order_independence() -> None:
    rng = random.Random(7)
    clusters = {
        f"dev-{i:03d}": [(rng.choice((S, U)), rng.choice((S, U))) for _ in range(3)]
        for i in range(12)
    }
    pairs = pairs_for(clusters)
    first = ag.cluster_bootstrap(pairs, stat)
    shuffled = list(pairs)
    random.Random(3).shuffle(shuffled)
    assert first == ag.cluster_bootstrap(shuffled, stat)
    assert first["seed"] == 20261125
    assert first["status"] == "estimated"
    assert first["valid_replicates"] + first["undefined_replicates"] == 2000


def test_undefined_replicates_are_counted_and_destabilize() -> None:
    clusters = {f"dev-{i:03d}": [(S, S), (S, S)] for i in range(9)}
    clusters["dev-009"] = [(S, U), (U, U)]  # resamples without it are all one class
    result = ag.cluster_bootstrap(pairs_for(clusters), stat)
    assert result["undefined_replicates"] > 0
    assert result["valid_replicates"] + result["undefined_replicates"] == 2000
    assert (result["status"], result["interval"]) == ("unstable_too_many_undefined", None)


def test_all_matching_labels_never_give_a_perfect_interval() -> None:
    clusters = {f"dev-{i:03d}": [(S, S), (U, U)] for i in range(10)}
    result = ag.cluster_bootstrap(pairs_for(clusters), stat)
    assert (result["status"], result["interval"]) == ("degenerate_interval", None)
    tiny = ag.cluster_bootstrap(pairs_for({"dev-001": [(S, S), (U, U)]}), stat)
    assert tiny["status"] == "insufficient_data"


def test_target_is_never_passed_on_undefined_kappa() -> None:
    assert ag.target_assessment({"value": None}, {})["status"] == "insufficient_information"
    result = ag.target_assessment({"value": 0.65}, {"interval": None})
    assert (result["status"], result["interval"]) == ("point_meets_target", "not_estimable")
    wide = ag.target_assessment({"value": 0.65}, {"interval": [0.3, 0.9]})
    assert wide["interval"] == "interval_includes_values_below_target"
    assert ag.CONFIG["target"]["statistic"] == "cohen_kappa"
    assert ag.CONFIG["target"]["planned_minimum"] == 0.60
    assert ag.CONFIG["target"]["ac1_substitution"].startswith("never")
    # AC1 can be high while kappa is undefined: the target still reports no information
    assert result["statistic"] == "cohen_kappa"


# --- pairing over the annotation ledger ------------------------------------------------------


def label(ws: dict[str, Any], dimension: str, value: str, **unit: Any) -> dict[str, Any]:
    entry = copy.deepcopy(find(ws, dimension, **unit))
    entry.update(review_action="confirmed", label=value)
    if value == "not_assessable":
        entry["not_assessable_reason"] = "annotator_uncertain"
    return entry


def two_reviewers(ctx: an.AnnotationContext) -> an.Ledger:
    a1, c3 = sheet(ctx), sheet(ctx, annotator="ann-c3")
    ledger = commit(ctx, an.Ledger(), only(a1, supported(a1), supported(a1, 2, "E03")))
    disagree = label(c3, "claim_faithfulness", "unsupported", claim_position=2)
    return commit(ctx, ledger, only(c3, supported(c3), disagree))


def judge_output(ctx: an.AnnotationContext, ledger: an.Ledger, unit_key: str, value: str,
                 **over: Any) -> dict[str, Any]:  # fmt: skip
    entry = next(e for e in ledger.entries if e.unit_key == unit_key)
    payload = copy.deepcopy(entry.payload)
    payload.update(label=value, evidence=payload["evidence"] if value == "supported" else [])
    payload["provenance"] = {
        "source": "judge",
        "role": "calibration",
        "judge_config_id": JUDGE,
        "rubric_id": rubric_id(),
    }
    payload.update(over)
    return payload


def test_human_human_pairs_compare_actual_decisions(ctx: an.AnnotationContext) -> None:
    ledger = two_reviewers(ctx)
    pairing = ag.human_human_pairs(ctx, ledger)
    pairs = pairing.headline["calibration"]
    assert [(p.unit_key, p.reference, p.compared) for p in pairs] == [
        (CLAIM1, "supported", "supported"),
        (CLAIM2, "supported", "unsupported"),
    ]
    report, _ = ag.build_report(ctx, ledger)
    dim = report["populations"]["human_human:calibration"]["dimensions"]["claim_faithfulness"]
    assert dim["raw"] == {"pairs": 2, "agree": 1, "disagree": 1, "rate": 0.5}
    # ref s,s; cmp s,u: p_e = 1*0.5 = 0.5, p_o = 0.5 -> kappa 0 (computed, not from states)
    assert dim["cohen_kappa"]["value"] == 0.0
    assert dim["clusters"] == 1  # two claims of one question are one cluster
    assert dim["bootstrap_kappa"]["status"] == "insufficient_data"
    assert dim["target"]["status"] == "point_below_target"
    assert "small_sample" in dim["warnings"]
    assert dim["scheduled_units"] == 4
    assert dim["coverage"] == 0.5  # 2 of 4 claims


def test_single_reviewer_gives_no_human_pair_but_supports_judge_calibration(
    ctx: an.AnnotationContext,
) -> None:
    ws = sheet(ctx)
    ledger = commit(ctx, an.Ledger(), only(ws, supported(ws), supported(ws, 2, "E03")))
    pairing = ag.human_human_pairs(ctx, ledger)
    assert pairing.headline == {}
    assert pairing.exclusions["missing_independent_pair"] == 2
    outputs = [
        judge_output(ctx, ledger, CLAIM1, "supported"),
        judge_output(ctx, ledger, CLAIM2, "unsupported"),
    ]
    by_config, exclusions = ag.human_judge_pairs(ctx, ledger, outputs)
    pairs = by_config[JUDGE]["calibration"]
    assert [(p.reference, p.compared) for p in pairs] == [
        ("supported", "supported"),
        ("supported", "unsupported"),
    ]
    assert exclusions.get("missing_judge_label", 0) == 0
    report, _ = ag.build_report(ctx, ledger, outputs)
    assert report["human_human_paired_units"] == 0
    assert report["human_judge_paired_units"] == 2
    assert f"human_judge:{JUDGE}:calibration" in report["populations"]


def test_judge_outputs_are_bound_and_excluded_precisely(ctx: an.AnnotationContext) -> None:
    ws = sheet(ctx)
    ledger = commit(ctx, an.Ledger(), only(ws, supported(ws), supported(ws, 2, "E03")))
    good = judge_output(ctx, ledger, CLAIM1, "supported")
    stale = judge_output(ctx, ledger, CLAIM2, "supported", generation_record_sha256="c" * 64)
    foreign_rubric = judge_output(ctx, ledger, CLAIM2, "supported")
    foreign_rubric["provenance"] = {**foreign_rubric["provenance"], "rubric_id": "0" * 12}
    unbound = judge_output(ctx, ledger, CLAIM2, "supported")
    unbound["unit"] = {"kind": "claim", "claim_position": 9}
    human_source = judge_output(ctx, ledger, CLAIM2, "supported")
    human_source["provenance"] = {"source": "human", "role": "calibration",
                                  "annotator_id": "ann-x1", "rubric_id": rubric_id()}  # fmt: skip
    wrong_role = judge_output(ctx, ledger, CLAIM2, "supported")
    wrong_role["provenance"] = {**wrong_role["provenance"], "role": "evaluation"}
    _, exclusions = ag.human_judge_pairs(
        ctx, ledger, [good, stale, foreign_rubric, unbound, human_source, wrong_role]
    )
    assert exclusions == {
        "malformed_provenance": 1,
        "missing_judge_label": 0,
        "role_conflict": 1,
        "rubric_mismatch": 1,
        "stale_record": 1,
        "unbound_unit": 1,
    }
    duplicates = [good, judge_output(ctx, ledger, CLAIM1, "unsupported")]
    by_config, exclusions = ag.human_judge_pairs(ctx, ledger, duplicates)
    assert by_config == {}
    assert exclusions["duplicate_judge_output"] == 2
    assert exclusions["missing_judge_label"] == 1  # claim 2; claim 1 counted as duplicate
    failed = judge_output(ctx, ledger, CLAIM1, "supported", question_id="dev-001")
    _, exclusions = ag.human_judge_pairs(ctx, ledger, [failed])
    assert exclusions["model_output_failure_or_unknown_question"] == 1


def test_adjudication_never_substitutes_for_an_independent_decision(
    ctx: an.AnnotationContext,
) -> None:
    ledger = two_reviewers(ctx)
    d4 = sheet(ctx, annotator="ann-d4")
    adjudicated = label(d4, "claim_faithfulness", "supported", claim_position=2)
    adjudicated["evidence"] = [span(Q4, "$39.3 billion", "selected_context", "E03")]
    adjudicated["adjudication"] = True
    ledger = commit(ctx, ledger, only(d4, adjudicated))
    pairing = ag.human_human_pairs(ctx, ledger)
    by_unit = {p.unit_key: p for p in pairing.headline["calibration"]}
    assert (by_unit[CLAIM2].reference, by_unit[CLAIM2].compared) == ("supported", "unsupported")
    assert by_unit[CLAIM2].reviewers == ("ann-a1", "ann-c3")
    assert pairing.exclusions["adjudication_not_independent"] == 1
    refs, _ = ag.human_references(ctx, ledger)
    assert refs[CLAIM2] == ("supported", an.ReviewState.ADJUDICATED)
    assert refs[CLAIM1] == ("supported", an.ReviewState.INDEPENDENTLY_AGREED)


def test_disputed_units_have_no_judge_reference(ctx: an.AnnotationContext) -> None:
    ledger = two_reviewers(ctx)
    refs, exclusions = ag.human_references(ctx, ledger)
    assert CLAIM2 not in refs
    assert exclusions == {"disputed_reference": 1}


def test_superseded_and_withdrawn_entries(ctx: an.AnnotationContext) -> None:
    ledger = two_reviewers(ctx)
    c3 = sheet(ctx, annotator="ann-c3")
    head = ledger.heads()[("ann-c3", CLAIM2)]
    revised = label(c3, "claim_faithfulness", "supported", claim_position=2)
    revised.update(
        evidence=[span(Q4, "$39.3 billion", "selected_context", "E03")],
        supersedes=head.entry_id,
        revision_reason="correction",
    )
    ledger = commit(ctx, ledger, only(c3, revised))
    pairing = ag.human_human_pairs(ctx, ledger)
    assert {p.compared for p in pairing.headline["calibration"]} == {"supported"}  # current only
    assert pairing.exclusions["superseded_revisions_not_used"] == 1
    head = ledger.heads()[("ann-c3", CLAIM1)]
    withdraw = {**supported(sheet(ctx, annotator="ann-c3")), "review_action": "withdrawn",
                "supersedes": head.entry_id, "revision_reason": "entry_error"}  # fmt: skip
    ledger = commit(ctx, ledger, only(c3, withdraw))
    pairing = ag.human_human_pairs(ctx, ledger)
    assert [p.unit_key for p in pairing.headline["calibration"]] == [CLAIM2]
    assert pairing.exclusions["withdrawn"] == 1
    assert pairing.exclusions["missing_independent_pair"] == 1


def test_more_than_two_reviewers(ctx: an.AnnotationContext) -> None:
    ledger = two_reviewers(ctx)
    d4 = sheet(ctx, annotator="ann-d4")
    ledger = commit(ctx, ledger, only(d4, supported(d4)))
    pairing = ag.human_human_pairs(ctx, ledger)
    assert (pairing.selected_pair, pairing.selection) == (("ann-a1", "ann-c3"), "most_shared_units")
    assert {p.reviewers for p in pairing.headline["calibration"]} == {("ann-a1", "ann-c3")}
    assert sorted(pairing.by_reviewer_pair) == [
        ("ann-a1", "ann-c3"),
        ("ann-a1", "ann-d4"),
        ("ann-c3", "ann-d4"),
    ]
    assert pairing.headline_accounting == {
        "intersection_units": 2,
        "missing_first_reviewer_label": 0,
        "missing_second_reviewer_label": 0,
        "units_reviewed_only_by_other_reviewers": 0,
    }
    report, _ = ag.build_report(ctx, ledger)
    pair_populations = [p for p in report["populations"] if p.startswith("human_human_reviewer")]
    assert len(pair_populations) == 2  # the two non-headline pairs, each on its own
    assert all("dependence" in report["populations"][p] for p in pair_populations)


def test_rubric_and_provenance_mismatches_are_excluded(ctx: an.AnnotationContext) -> None:
    ledger = two_reviewers(ctx)
    changed = []
    for e in ledger.entries:
        if (e.annotator_id, e.unit_key) == ("ann-c3", CLAIM1):
            e = e.model_copy(update={"rubric_id": "0" * 12})
        if (e.annotator_id, e.unit_key) == ("ann-c3", CLAIM2):
            payload = copy.deepcopy(e.payload)
            payload["provenance"]["annotator_id"] = "ann-zz"
            e = e.model_copy(update={"payload": payload})
        changed.append(e)
    pairing = ag.human_human_pairs(ctx, an.Ledger(tuple(changed)))
    assert pairing.headline == {}
    assert pairing.exclusions["rubric_mismatch"] == 1
    assert pairing.exclusions["malformed_provenance"] == 1


def test_non_assessable_mismatches_stay_visible(ctx: an.AnnotationContext) -> None:
    a1, c3 = sheet(ctx), sheet(ctx, annotator="ann-c3")
    relevance_a = label(a1, "answer_relevance", "not_assessable")
    ledger = commit(ctx, an.Ledger(), only(a1, supported(a1), relevance_a))
    faith_c = label(c3, "claim_faithfulness", "not_assessable", claim_position=1)
    relevance_c = label(c3, "answer_relevance", "not_assessable")
    ledger = commit(ctx, ledger, only(c3, faith_c, relevance_c))
    report, _ = ag.build_report(ctx, ledger)
    dims = report["populations"]["human_human:calibration"]["dimensions"]
    faith, relevance = dims["claim_faithfulness"], dims["answer_relevance"]
    assert (faith["primary_pairs"], faith["assessability_mismatch"]) == (0, 1)
    assert faith["cohen_kappa"]["status"] == "undefined_no_pairs"
    assert faith["target"]["status"] == "insufficient_information"
    assert faith["confusion"]["matrix"][0][3] == 1  # supported vs not_assessable
    assert (relevance["both_non_substantive"], relevance["primary_pairs"]) == (1, 0)
    assert relevance["weighted_kappa"]["status"] == "undefined_no_pairs"


def test_partitions_never_mix(ctx: an.AnnotationContext) -> None:
    ledger = two_reviewers(ctx)
    for annotator in ("ann-a1", "ann-c3"):
        ws = sheet(ctx, "dev-002", annotator=annotator, role=Role.EVALUATION)
        entry = label(ws, "claim_faithfulness", "supported", claim_position=1)
        entry["evidence"] = [span(Q4, "$39.3 billion", "selected_context", "E03")]
        ledger = commit(ctx, ledger, only(ws, entry))
    pairing = ag.human_human_pairs(ctx, ledger)
    assert {p.question_id for p in pairing.headline["calibration"]} == {"dev-003"}
    assert {p.question_id for p in pairing.headline["evaluation"]} == {"dev-002"}
    report, _ = ag.build_report(ctx, ledger)
    assert report["populations"]["human_human:evaluation"]["partition"] == "evaluation"
    leaking = judge_output(ctx, ledger, CLAIM1, "supported")
    leaking["provenance"] = {**leaking["provenance"], "role": "evaluation"}
    by_config, exclusions = ag.human_judge_pairs(ctx, ledger, [leaking])
    assert by_config == {}
    assert exclusions["role_conflict"] == 1


def test_synthetic_stress_is_separate_from_natural_agreement(ctx: an.AnnotationContext) -> None:
    ledger = _labelled_ledger(ctx)
    variants = cal.synthetic_variants(ctx, ledger)
    accepted, pending = variants[0], variants[1]
    review = {
        "entry_type": "synthetic_review",
        "review_action": "confirmed",
        "variant_id": accepted.variant_id,
        "decision": "accept_expected_label",
    }
    ledger = commit(ctx, ledger, only(sheet(ctx, accepted.question_id), review))
    natural_before, _ = ag.build_report(ctx, ledger)
    labels = [
        ag.SyntheticJudgeLabel(accepted.variant_id, str(accepted.expected["label"]), JUDGE),
        ag.SyntheticJudgeLabel(pending.variant_id, str(pending.expected["label"]), JUDGE),
        ag.SyntheticJudgeLabel("0" * 16, "supported", JUDGE),
    ]
    report, detail = ag.build_report(ctx, ledger, synthetic_labels=labels)
    population = report["populations"][f"synthetic_stress:{JUDGE}"]
    assert (population["synthetic"], population["partition"]) == (True, "calibration")
    assert report["synthetic_paired_units"] == 1
    assert report["exclusions"]["synthetic"] == {
        "synthetic_label_not_human_accepted": 1,
        "unknown_variant": 1,
    }
    assert detail["pairs"][f"synthetic_stress:{JUDGE}"][0]["lineage"] == accepted.variant_id
    natural = {k: v for k, v in report["populations"].items() if not v["synthetic"]}
    assert natural == natural_before["populations"]  # never pooled into natural figures
    assert report["human_judge_paired_units"] == 0


def test_reports_are_deterministic_and_public_safe(
    ctx: an.AnnotationContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    for name in ("read_bytes", "read_text", "open"):
        original = getattr(pathlib.Path, name)

        def spy(self: pathlib.Path, *args: Any, _original: Any = original, **kw: Any) -> Any:
            opened.append(self.name)
            if self.name == "questions.test.jsonl":
                raise AssertionError("held-out file opened")
            return _original(self, *args, **kw)

        monkeypatch.setattr(pathlib.Path, name, spy)
    ws = sheet(ctx)
    entry = supported(ws)
    entry["rationale"] = "SENTINEL-RATIONALE"
    ledger = commit(ctx, an.Ledger(), only(ws, entry))
    c3 = sheet(ctx, annotator="ann-c3")
    ledger = commit(ctx, ledger, only(c3, supported(c3)))
    outputs = [judge_output(ctx, ledger, CLAIM1, "supported")]
    first, detail = ag.build_report(ctx, ledger, outputs)
    assert (first, detail) == ag.build_report(ctx, ledger, outputs)
    blob = json.dumps(first)
    for text in [*private_texts(ctx), "SENTINEL-RATIONALE", "ann-a1", "ann-c3", "unit_key"]:
        assert text not in blob
    assert "imported_at" not in blob
    assert "timestamp" not in blob
    assert first["config_id"] == ag.config_id()
    assert detail["pairs"]["human_human:calibration"][0]["reviewers"] == ["ann-a1", "ann-c3"]
    assert "questions.test.jsonl" not in opened
    assert ag.main(["--run", "test-run"]) == 1  # not a live development run


def test_empty_ledger_reports_nothing_measurable(ctx: an.AnnotationContext) -> None:
    report, detail = ag.build_report(ctx, an.Ledger())
    assert (report["measurable"], report["populations"], detail["pairs"]) == (False, {}, {})
    assert report["human_human_paired_units"] == report["human_judge_paired_units"] == 0
    assert report["scheduled_units"]["citation_support"] == 4


def test_distinct_units_are_never_collapsed_or_cross_matched(ctx: an.AnnotationContext) -> None:
    a1, c3 = sheet(ctx), sheet(ctx, annotator="ann-c3")
    entries = []
    for ws in (a1, c3):
        first = label(ws, "citation_support", "does_not_support", claim_position=1)
        second = label(ws, "citation_support", "does_not_support", claim_position=2)
        entries.append(only(ws, first, second))
    ledger = commit(ctx, commit(ctx, an.Ledger(), entries[0]), entries[1])
    pairs = ag.human_human_pairs(ctx, ledger).headline["calibration"]
    assert [p.unit_key for p in pairs] == [
        "judgment:dev-003/claim:1/citation:1:E02#citation_support",
        "judgment:dev-003/claim:2/citation:2:E03#citation_support",
    ]
    # a judge label on another unit of the same claim never pairs with this reference
    ws = sheet(ctx)
    ledger = commit(ctx, an.Ledger(), only(ws, supported(ws)))
    other_unit = judge_output(ctx, ledger, CLAIM1, "supported")
    other_unit.update(
        dimension="citation_support",
        unit={"kind": "claim_citation", "claim_position": 1, "citation_number": 1,
              "evidence_id": "E02"},
        label="does_not_support",
        evidence=[],
    )  # fmt: skip
    by_config, exclusions = ag.human_judge_pairs(ctx, ledger, [other_unit])
    assert by_config == {}
    assert exclusions["missing_human_reference"] == 1
    assert exclusions["missing_judge_label"] == 1


# --- final verification ------------------------------------------------------------------------

RELEVANCE = "judgment:dev-003/answer#answer_relevance"


def three_reviewers_on_different_units(ctx: an.AnnotationContext) -> an.Ledger:
    """a1: claim 1; c3: claim 1, claim 2, relevance; d4: claim 2, relevance."""
    a1, c3, d4 = (sheet(ctx, annotator=a) for a in ("ann-a1", "ann-c3", "ann-d4"))
    ledger = commit(ctx, an.Ledger(), only(a1, supported(a1)))
    c3_entries = (
        supported(c3),
        supported(c3, 2, "E03"),
        label(c3, "answer_relevance", "directly_answers"),
    )
    ledger = commit(ctx, ledger, only(c3, *c3_entries))
    d4_entries = (
        label(d4, "claim_faithfulness", "unsupported", claim_position=2),
        label(d4, "answer_relevance", "partially_relevant"),
    )
    return commit(ctx, ledger, only(d4, *d4_entries))


def test_headline_uses_one_fixed_reviewer_pair_and_never_pools_pairs(
    ctx: an.AnnotationContext,
) -> None:
    ledger = three_reviewers_on_different_units(ctx)
    auto = ag.human_human_pairs(ctx, ledger)
    # (c3, d4) share 2 units, (a1, c3) only 1: a per-unit "first two reviewers" rule
    # would have pooled a1-c3 on claim 1 with c3-d4 on claim 2 and relevance
    assert (auto.selected_pair, auto.selection) == (("ann-c3", "ann-d4"), "most_shared_units")
    assert {p.unit_key for p in auto.headline["calibration"]} == {CLAIM2, RELEVANCE}
    assert {p.reviewers for p in auto.headline["calibration"]} == {("ann-c3", "ann-d4")}
    assert auto.headline_accounting == {
        "intersection_units": 2,
        "missing_first_reviewer_label": 0,
        "missing_second_reviewer_label": 1,  # claim 1: c3 only
        "units_reviewed_only_by_other_reviewers": 0,
    }
    explicit = ag.human_human_pairs(ctx, ledger, ("ann-c3", "ann-a1"))
    assert (explicit.selected_pair, explicit.selection) == (("ann-a1", "ann-c3"), "explicit")
    assert [p.unit_key for p in explicit.headline["calibration"]] == [CLAIM1]
    assert explicit.headline_accounting is not None
    assert explicit.headline_accounting["missing_first_reviewer_label"] == 2
    absent = ag.human_human_pairs(ctx, ledger, ("ann-a1", "ann-zz"))
    assert absent.headline == {}
    for bad in (("ann-a1", "ann-a1"), ("ann-a1", "Jane")):
        with pytest.raises(ag.AgreementError, match="two distinct pseudonyms"):
            ag.human_human_pairs(ctx, ledger, bad)
    report, _detail = ag.build_report(ctx, ledger)
    headline = report["populations"]["human_human:calibration"]
    assert headline["reviewer_pair_id"] == ag.reviewer_pair_id(("ann-d4", "ann-c3"))
    assert headline["pair_selection"] == "most_shared_units"
    other = f"human_human_reviewer_pair:{ag.reviewer_pair_id(('ann-a1', 'ann-c3'))}:calibration"
    assert report["populations"][other]["paired_units"] == 1  # reported separately
    blob = json.dumps(report)
    assert "ann-a1" not in blob
    assert "ann-c3" not in blob
    assert "ann-d4" not in blob


def test_human_judge_reference_states_are_disclosed(ctx: an.AnnotationContext) -> None:
    ledger = three_reviewers_on_different_units(ctx)  # claim 1 agreed, claim 2 disputed
    e5 = sheet(ctx, annotator="ann-e5")
    adjudicated = label(e5, "claim_faithfulness", "unsupported", claim_position=2)
    adjudicated["adjudication"] = True
    ledger = commit(ctx, ledger, only(e5, adjudicated))
    outputs = [
        judge_output(ctx, ledger, CLAIM1, "supported"),
        judge_output(ctx, ledger, CLAIM2, "unsupported"),
    ]
    report, _detail = ag.build_report(ctx, ledger, outputs)
    population = report["populations"][f"human_judge:{JUDGE}:calibration"]
    assert population["reference_states"] == {"adjudicated": 1, "independently_agreed": 1}
    assert "not factual truth" in population["reference_note"]
    single = sheet(ctx)
    lone = commit(ctx, an.Ledger(), only(single, supported(single)))
    report, _ = ag.build_report(ctx, lone, [judge_output(ctx, lone, CLAIM1, "supported")])
    states = report["populations"][f"human_judge:{JUDGE}:calibration"]["reference_states"]
    assert states == {"single_reviewed": 1}


def test_non_substantive_judges_cannot_look_reliable_on_easy_pairs(
    ctx: an.AnnotationContext,
) -> None:
    ws = sheet(ctx)
    ledger = commit(ctx, an.Ledger(), only(ws, supported(ws), supported(ws, 2, "E03")))
    outputs = [
        judge_output(ctx, ledger, CLAIM1, "supported"),
        judge_output(
            ctx, ledger, CLAIM2, "not_assessable", not_assessable_reason="evidence_missing"
        ),
    ]
    report, _ = ag.build_report(ctx, ledger, outputs)
    dim = report["populations"][f"human_judge:{JUDGE}:calibration"]["dimensions"][
        "claim_faithfulness"
    ]
    assert dim["raw"]["rate"] == 1.0  # the only primary pair agrees ...
    assert (dim["primary_share"], dim["compared_non_substantive"]) == (0.5, 1)
    assert "low_primary_share" in dim["warnings"]  # ... and that is disclosed
    assert dim["assessability_mismatch"] == 1


def test_unverified_against_substantive_is_an_assessability_mismatch(
    ctx: an.AnnotationContext,
) -> None:
    a1, c3 = sheet(ctx), sheet(ctx, annotator="ann-c3")
    unit = {"claim_position": 1, "aspect": "company"}
    a_label = label(a1, "financial_attribution", "unverified", **unit)
    c_label = label(c3, "financial_attribution", "not_applicable", **unit)
    ledger = commit(ctx, an.Ledger(), only(a1, a_label))
    ledger = commit(ctx, ledger, only(c3, c_label))
    report, _ = ag.build_report(ctx, ledger)
    dim = report["populations"]["human_human:calibration"]["dimensions"]["financial_attribution"]
    assert (dim["both_non_substantive"], dim["primary_pairs"]) == (1, 0)
    assert dim["confusion"]["classes"] == [
        "correct",
        "incorrect",
        "unverified",
        "not_applicable",
        "not_assessable",
    ]
    assert dim["confusion"]["matrix"][2][3] == 1  # unverified vs not_applicable, distinct


def test_unit_level_weighting_and_per_question_accounting(ctx: an.AnnotationContext) -> None:
    ledger = two_reviewers(ctx)
    report, _ = ag.build_report(ctx, ledger)
    dim = report["populations"]["human_human:calibration"]["dimensions"]["claim_faithfulness"]
    assert dim["estimate_level"] == "unit"
    assert (dim["questions"], dim["paired_units"]) == (1, 2)
    assert dim["per_question"] == {"dev-003": {"paired_units": 2, "primary_pairs": 2}}
    category = dim["by_question_category"]["temporal_multi_evidence"]
    assert (category["questions"], category["small_sample"], category["interval"]) == (
        1,
        True,
        None,
    )
    assert "unit-level" in ag.CONFIG["weighting"]


def test_configuration_identity_covers_normative_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    base = ag.config_id()
    for key in ("non_substantive_labels", "class_order", "ordinal_scores", "weighting",
                "min_primary_share", "human_human", "bootstrap", "target"):  # fmt: skip
        assert key in ag.CONFIG
    monkeypatch.setitem(ag.CONFIG["bootstrap"], "min_clusters", 11)
    assert ag.config_id() != base
    monkeypatch.setitem(ag.CONFIG["bootstrap"], "min_clusters", 10)
    monkeypatch.setitem(ag.CONFIG, "min_primary_share", 0.5)
    assert ag.config_id() != base
