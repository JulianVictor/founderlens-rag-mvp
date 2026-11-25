"""Judge request-budget accounting (post-7.5 audit), offline with a scripted fake provider.

Pins down what the 7.5 injection test's "budget untouched" meant: the injected text
could not change the plan or the cap; it did not mean zero provider attempts. Every
invoked request is one attempt, a repair is another, and nothing rejected before the
provider is called consumes an attempt.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import pytest

from founderlens.evaluation import annotation as an
from founderlens.evaluation import judge as jd
from founderlens.evaluation.judgments import Role
from founderlens.generation.schema import TokenUsage
from tests.test_evaluation_judge import (
    INJECTION,
    NOW,
    FakeJudge,
    calibration_ledger,
    ctx,  # noqa: F401  (fixture)
    default_results,
    offline,  # noqa: F401  (fixture)
    with_text,
)
from tests.test_evaluation_judging_input import CLEAN

USAGE = TokenUsage(input_tokens=500, cached_input_tokens=0, output_tokens=40,
                   reasoning_tokens=0, total_tokens=540)  # fmt: skip


def extra_unit(payload: dict[str, Any], call: int) -> list[dict[str, Any]]:
    results = default_results(payload)
    return [*results, {**results[0], "unit_ref": "u99"}]


def execute(
    context: an.AnnotationContext,
    plan: jd.Plan,
    tmp_path: pathlib.Path,
    provider: FakeJudge,
    cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs = jd.run_judge(context, plan, lambda: provider, tmp_path, mode="mock",
                           max_requests=cap, git=CLEAN, now=lambda: NOW)  # fmt: skip
    return jd.read_records(outputs.records_path), dict(outputs.summary)


def attempts(records: list[dict[str, Any]]) -> int:
    return sum(len(r["attempts"]) for r in records)


def test_rejected_before_the_provider_consumes_no_attempt(
    ctx: an.AnnotationContext,  # noqa: F811
    tmp_path: pathlib.Path,
) -> None:
    plan = jd.plan_jobs(ctx, calibration_ledger(ctx), Role.CALIBRATION)
    tampered = dataclasses.replace(
        plan, batches=tuple(dataclasses.replace(b, input_sha256="0" * 64) for b in plan.batches)
    )
    provider = FakeJudge()
    records, summary = execute(ctx, tampered, tmp_path, provider, cap=3)
    assert provider.calls == 0
    assert attempts(records) == summary["attempts"] == 0
    assert {r["outcome"] for r in records} == {"not_run_integrity"}


def test_invalid_responses_count_once_each_and_yield_nothing(
    ctx: an.AnnotationContext,  # noqa: F811
    tmp_path: pathlib.Path,
) -> None:
    ledger = calibration_ledger(ctx)
    injected = with_text(ctx, " " + INJECTION)
    plan = jd.plan_jobs(injected, ledger, Role.CALIBRATION)
    provider = FakeJudge(extra_unit, usage=USAGE)
    records, summary = execute(injected, plan, tmp_path, provider, cap=len(plan.batches) * 2)
    # each batch: the invoked request (1) plus its one structured-output repair (1)
    assert provider.calls == attempts(records) == summary["attempts"] == 2 * len(plan.batches)
    assert summary["repair_attempts"] == len(plan.batches)
    assert all(r["judgments"] == [] for r in records)  # zero accepted judgments
    for r in records:
        assert [a["category"] for a in r["attempts"]] == ["structured_output_invalid"] * 2
        assert all(a["usage"] == USAGE.to_dict() for a in r["attempts"])  # usage kept
    assert summary["usage"]["total_tokens"] == 540 * provider.calls
    assert summary["planned_requests"] == len(plan.batches)  # injection changed no plan
    assert "999" not in json.dumps(summary)


def test_repairs_count_and_a_blocked_repair_does_not(
    ctx: an.AnnotationContext,  # noqa: F811
    tmp_path: pathlib.Path,
) -> None:
    ledger = calibration_ledger(ctx)
    plan = jd.plan_jobs(ctx, ledger, Role.CALIBRATION)
    repaired = FakeJudge(lambda p, call: [] if call == 1 else default_results(p))
    records, _ = execute(ctx, plan, tmp_path / "repaired", repaired, cap=50)
    assert [a["repair"] for a in records[0]["attempts"]] == [False, True]
    assert repaired.calls == attempts(records) == len(plan.batches) + 1
    blocked = FakeJudge(lambda p, call: [])
    records, summary = execute(ctx, plan, tmp_path / "blocked", blocked, cap=1)
    assert blocked.calls == attempts(records) == summary["attempts"] == 1
    assert records[0]["failure"]["code"] == "repair_blocked_by_budget"
    assert len(records[0]["attempts"]) == 1  # the blocked repair was never attempted


@pytest.mark.parametrize("cap", [1, 2, 3, 5, 8, 13])
def test_attempts_never_exceed_the_cap_or_reset(
    ctx: an.AnnotationContext,  # noqa: F811
    tmp_path: pathlib.Path,
    cap: int,
) -> None:
    ledger = calibration_ledger(ctx)
    plan = jd.plan_jobs(ctx, ledger, Role.CALIBRATION)

    def alternating(payload: dict[str, Any], call: int) -> list[dict[str, Any]]:
        results = default_results(payload)
        if call % 3 == 0:
            return [{**r, "label": "bogus"} for r in results]  # contract failure, no repair
        if call % 3 == 1:
            return extra_unit(payload, call)  # repairable mapping failure
        return results

    provider = FakeJudge(alternating, usage=USAGE)
    records, summary = execute(ctx, plan, tmp_path, provider, cap=cap)
    assert provider.calls == attempts(records) == summary["attempts"]
    assert provider.calls <= cap
    running = 0
    for r in records:
        running += len(r["attempts"])
        assert running <= cap  # cumulative: a contract failure never resets the counter
        if r["outcome"] == "not_run_budget":
            assert r["attempts"] == []
