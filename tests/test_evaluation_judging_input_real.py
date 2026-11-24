"""Opt-in verification of the frozen Stage 5 baseline as judging input (Commit 7.1).

Skipped unless ``FOUNDERLENS_REAL_CORPUS=1`` and the git-ignored records of
``live-20261124T045739024562Z-8665d379`` exist locally. Reads the frozen run,
the development split and the V2 collection only: no network, no retrieval, no
provider and no held-out file. Prints nothing private. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_evaluation_judging_input_real.py -q
"""

from __future__ import annotations

import json
import os
import pathlib
import socket
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.evaluation import judging_input as ji
from founderlens.retrieval.policy import PolicyRetriever

RUN = "live-20261124T045739024562Z-8665d379"
_RECORDS = load_settings().reports_dir / "generation_eval" / f"{RUN}.records.jsonl"

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1" or not _RECORDS.is_file(),
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


def test_frozen_baseline_verifies_as_22_valid_answers_and_two_failures(guarded: list[str]) -> None:
    run, collection = ji.load_verified_run(load_settings(), RUN)
    assert len(run.questions) == 24
    assert run.counts() == {"model_output_failure": 2, "valid_answer": 22}
    assert sum(q.eligible for q in run.questions) == 22
    ineligible = {q.question_id: q.failure_codes for q in run.questions if not q.eligible}
    assert ineligible == {
        "dev-016": ("answered_has_limitations",),
        "dev-021": ("answered_has_limitations",),
    }
    assert run.generation_commit.startswith("7c63bbc")
    assert (run.generation_protocol, run.completion_state) == ("1.3", "execution_complete")
    assert run.collection_artifact_sha256 == collection.artifact_sha256
    summary = json.dumps(run.public_summary())
    for question in run.questions:
        for claim in question.claims:
            assert claim.text not in summary
    assert "questions.test.jsonl" not in guarded
