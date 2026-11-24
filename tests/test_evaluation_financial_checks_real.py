"""Opt-in read-only financial-check smoke on the frozen baseline (Commit 7.2).

Skipped unless ``FOUNDERLENS_REAL_CORPUS=1`` and the git-ignored records of
``live-20261124T045739024562Z-8665d379`` exist locally. No verified reference
facts exist for real data, so every financial value must stay unverified and
no judgment may be created. No network, retrieval, provider or held-out file.
Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_evaluation_financial_checks_real.py -q
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
from founderlens.evaluation import financial_checks as fc
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


def _digests() -> dict[str, str]:
    names = [f"{RUN}.records.jsonl", f"{RUN}.json", "run_log.jsonl"]
    return {n: hashlib.sha256((_DIR / n).read_bytes()).hexdigest() for n in names}


def test_baseline_values_are_indexed_and_stay_unverified(guarded: list[str]) -> None:
    before = _digests()
    summary = fc.baseline_summary(RUN)
    assert (summary["scheduled"], summary["eligible"]) == (24, 22)
    assert summary["ineligible"] == ["dev-016", "dev-021"]
    assert summary["deterministic_and_indexed"] is True
    assert (summary["judgments_created"], summary["verified_reference_facts"]) == (0, 0)
    financial = summary["occurrence_kinds"].get("financial_quantity", 0)
    assert summary["check_outcomes"].get("unverified", 0) == financial
    assert set(summary["check_outcomes"]) <= {"unverified", "not_applicable"}
    assert summary["reason_codes"].get("no_verified_association", 0) == financial
    assert summary["rubric_id"] == "e8b840e4f4af"
    assert _digests() == before  # frozen answers and records unchanged
    assert "questions.test.jsonl" not in guarded
    payload = json.dumps(summary)
    assert "$" not in payload  # counts and codes only, no claim text
