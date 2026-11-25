"""Opt-in judge planning smoke on the frozen baseline (Commit 7.5).

Skipped unless ``FOUNDERLENS_REAL_CORPUS=1`` and the git-ignored records of
``live-20261124T045739024562Z-8665d379`` exist locally. Plans jobs and requests
offline only: no provider is constructed, no request is made and no file is
written. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_evaluation_judge_real.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import socket
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.evaluation import annotation as an
from founderlens.evaluation import judge as jd
from founderlens.retrieval.policy import PolicyRetriever

RUN = "live-20261124T045739024562Z-8665d379"
_DIR = load_settings().reports_dir / "generation_eval"

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1"
    or not (_DIR / f"{RUN}.records.jsonl").is_file(),
    reason=f"set FOUNDERLENS_REAL_CORPUS=1 with the local records of {RUN}",
)


@pytest.fixture
def guarded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import founderlens.generation.openai_provider as op

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network, retrieval, provider or held-out access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(PolicyRetriever, "search", blocked)
    monkeypatch.setattr(op, "OpenAIProvider", blocked)
    opened: list[str] = []
    for name in ("read_bytes", "read_text", "open"):
        original = getattr(pathlib.Path, name)

        def spy(self: pathlib.Path, *args: Any, _original: Any = original, **kw: Any) -> Any:
            opened.append(self.name)
            if self.name == "questions.test.jsonl":
                raise AssertionError("held-out file opened")
            return _original(self, *args, **kw)

        monkeypatch.setattr(pathlib.Path, name, spy)
    return opened


def _digests() -> dict[str, str]:
    names = [f"{RUN}.records.jsonl", f"{RUN}.json", "run_log.jsonl"]
    return {n: hashlib.sha256((_DIR / n).read_bytes()).hexdigest() for n in names}


def test_baseline_judge_plan_is_offline_and_deterministic(guarded: list[str]) -> None:
    settings = load_settings()
    before = _digests()
    judge_dir = settings.reports_dir / jd.REPORT_SUBDIR
    existed = judge_dir.exists()
    ctx = an.load_context(settings, RUN)
    ledger = an.load_ledger(an.annotation_paths(settings, RUN).ledger)
    counts = an.run_counts(ctx)
    assert (counts["scheduled_outcomes"], counts["eligible_answers"], counts["claims"]) == (
        24,
        22,
        36,
    )
    assert (counts["claim_evidence_associations"], counts["distinct_citations"]) == (51, 33)
    assert counts["financial_quantities"] == 27
    plan = jd.plan_jobs(ctx, ledger)
    assert plan == jd.plan_jobs(ctx, ledger)
    summary = jd.plan_summary(ctx, plan)
    assert summary["exclusions"]["model_output_failure"] == 2
    assert {"dev-016", "dev-021"}.isdisjoint({j.question_id for j in plan.jobs})
    assert summary["jobs_by_dimension"]["citation_support"] == 51
    assert summary["requests_by_group"]["citation_support"] == 33
    assert summary["jobs"] == len({j.job_id for j in plan.jobs})
    assert summary["requests"] < summary["jobs"]  # batched, not one request per unit
    assert summary["provider_calls"] == 0
    assert summary["judge_config_id"] == jd.judge_config_id()
    if not ledger.entries:  # no partition assigned yet: a live run would be refused
        problems = jd.live_authorization_problems(
            live=True, environ={}, max_requests=1, git={}, plan=plan
        )
        assert any("partition" in p for p in problems)
    assert "SENTINEL" not in json.dumps(summary)
    assert _digests() == before
    assert judge_dir.exists() is existed
    assert "questions.test.jsonl" not in guarded
