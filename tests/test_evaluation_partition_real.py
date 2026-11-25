"""Opt-in partition smoke on the frozen baseline (Commit 7.6A).

Skipped unless ``FOUNDERLENS_REAL_CORPUS=1`` and the git-ignored records of
``live-20261124T045739024562Z-8665d379`` exist locally. Verifies the committed
manifest against the frozen inputs and reconciles unit, fact and request counts.
Read-only: no worksheet, ledger, judge record or provider is created. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_evaluation_partition_real.py -q
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import socket
from collections import Counter
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.evaluation import annotation as an
from founderlens.evaluation import judge as jd
from founderlens.evaluation import partition as pt
from founderlens.evaluation.judgments import Role
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


def _digests(paths: an.AnnotationPaths) -> dict[str, str]:
    files = [_DIR / f"{RUN}.records.jsonl", _DIR / f"{RUN}.json", _DIR / "run_log.jsonl"]
    files += sorted(paths.worksheets.glob("*.json")) if paths.worksheets.is_dir() else []
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def test_committed_allocation_reconciles_with_the_frozen_run(guarded: list[str]) -> None:
    settings = load_settings()
    paths = an.annotation_paths(settings, RUN)
    before = _digests(paths)
    ledger_existed = paths.ledger.exists()
    judge_existed = (settings.reports_dir / jd.REPORT_SUBDIR).exists()
    ctx = an.load_context(settings, RUN)
    manifest = pt.load_manifest(settings, ctx)
    assert manifest.policy_id == pt.APPROVED_POLICY_ID
    calibration, evaluation = (
        manifest.questions(Role.CALIBRATION),
        manifest.questions(Role.EVALUATION),
    )
    assert (len(calibration), len(evaluation), manifest.excluded) == (
        11,
        11,
        ("dev-016", "dev-021"),
    )
    assert len(ctx.run.questions) == 24
    units = Counter(u.split(":", 1)[1].split("/", 1)[0] for u in an.template_judgment_units(ctx))
    assert sum(units[q] for q in calibration) == 167
    assert sum(units[q] for q in evaluation) == 158
    facts = {r: sum(len(ctx.question(q).required_fact_ids) for q in ids)
             for r, ids in (("calibration", calibration), ("evaluation", evaluation))}  # fmt: skip
    assert facts == {"calibration": 17, "evaluation": 20}
    ledger = an.load_ledger(paths.ledger)
    plans = {r: jd.plan_jobs(ctx, ledger, r, manifest=manifest) for r in Role}
    assert len(plans[Role.CALIBRATION].batches) == len(plans[Role.EVALUATION].batches) == 54
    assert plans[Role.CALIBRATION].questions == calibration
    report = pt.preflight(manifest, paths, ledger)
    for row in report["worksheets"]:
        if row.get("question_id") == "dev-003" and row.get("role") == "calibration":
            assert row["status"] == "incompatible"  # the pre-existing pending worksheet
    assert report["changed_files"] == 0
    assert _digests(paths) == before
    assert paths.ledger.exists() is ledger_existed
    assert (settings.reports_dir / jd.REPORT_SUBDIR).exists() is judge_existed
    assert "questions.test.jsonl" not in guarded
