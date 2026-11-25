"""Opt-in agreement smoke on the frozen baseline (Commit 7.4).

Skipped unless ``FOUNDERLENS_REAL_CORPUS=1`` and the git-ignored records of
``live-20261124T045739024562Z-8665d379`` exist locally. Reads the private
annotation ledger if one exists (none is created or changed). No judge outputs
exist, so no human-judge pair can exist. No network, retrieval, provider or
held-out file. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_evaluation_agreement_real.py -q
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
from founderlens.evaluation import agreement as ag
from founderlens.evaluation import annotation as an
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
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network, retrieval or held-out access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(PolicyRetriever, "search", blocked)
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
    if paths.ledger.is_file():
        files.append(paths.ledger)
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def test_baseline_agreement_reports_only_what_exists(guarded: list[str]) -> None:
    settings = load_settings()
    paths = an.annotation_paths(settings, RUN)
    existed = paths.root.exists()
    before = _digests(paths)
    ctx = an.load_context(settings, RUN)
    ledger = an.load_ledger(paths.ledger)
    counts = an.run_counts(ctx)
    assert (counts["scheduled_outcomes"], counts["eligible_answers"], counts["claims"]) == (
        24,
        22,
        36,
    )
    assert counts["ineligible"] == ["dev-016", "dev-021"]
    report, detail = ag.build_report(ctx, ledger)
    assert report["scheduled_units"]["citation_support"] == 51
    assert report["scheduled_units"]["claim_faithfulness"] == 36
    assert report["human_judge_paired_units"] == 0  # no judge outputs exist
    expected_pairs = len(ag.human_human_pairs(ctx, ledger).headline.get("calibration", [])) + len(
        ag.human_human_pairs(ctx, ledger).headline.get("evaluation", [])
    )
    assert report["human_human_paired_units"] == expected_pairs
    if not ledger.entries:
        assert expected_pairs == 0
        assert (report["measurable"], report["populations"]) == (False, {})
    assert report == ag.build_report(ctx, ledger)[0]  # deterministic replay
    assert "SENTINEL" not in json.dumps(report)
    assert _digests(paths) == before
    assert paths.root.exists() is existed  # nothing created
    assert "questions.test.jsonl" not in guarded
    assert detail["report_id"] == report["report_id"]
