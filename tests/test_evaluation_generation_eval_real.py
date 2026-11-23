"""Opt-in dry run of the diagnostic generation evaluation on the real corpus (Commit 6.3).

Skipped unless ``FOUNDERLENS_REAL_TOKENIZER=1`` (prepared asset) and
``FOUNDERLENS_REAL_CORPUS=1``. The frozen V2 BM25 policy, context builder,
service, resolver and guardrails run over the real development split; the
answers come from the SYNTHETIC scripted provider. Nothing is sent to OpenAI,
no key is needed, the network is blocked, reports go to ``tmp_path`` and the
held-out file is never opened. Run locally::

    $env:FOUNDERLENS_REAL_TOKENIZER = "1"; $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_evaluation_generation_eval_real.py -q
"""

from __future__ import annotations

import os
import pathlib
import socket
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation.benchmark import load_dev_split

pytestmark = pytest.mark.skipif(
    os.environ.get("FOUNDERLENS_REAL_TOKENIZER") != "1"
    or os.environ.get("FOUNDERLENS_REAL_CORPUS") != "1",
    reason="set FOUNDERLENS_REAL_TOKENIZER=1 and FOUNDERLENS_REAL_CORPUS=1",
)


@pytest.fixture
def guarded(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
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


def test_real_dry_run_accounts_for_every_development_question(
    tmp_path: pathlib.Path, guarded: list[str]
) -> None:
    settings = load_settings()
    split = load_dev_split(settings.evaluation_dir, settings.corpus_manifest_path)
    pipeline = ge.load_pipeline(settings)
    outputs = ge.execute(
        split,
        pipeline,
        tmp_path,
        mode="dry_run",
        provider_factory=ge._provider_factory("dry_run", {}),
    )
    report = outputs.report
    records = ge.load_records(outputs.records_path)
    assert len(records) == 24
    assert sorted(r["question_id"] for r in records) == sorted(
        q.question_id for q in split.questions
    )
    assert report["dataset"]["category_counts"] == {
        "single_doc_factual": 7,
        "single_doc_conceptual": 5,
        "cross_doc_comparison": 5,
        "temporal_multi_evidence": 2,
        "unanswerable": 5,
    }
    outcomes = report["overall"]["execution_outcomes"]
    assert sum(outcomes.values()) == 24
    assert sum(c["questions"] for c in report["by_category"].values()) == 24
    local = [r for r in records if r["origin"] == "local"]
    assert all(not r["provider_attempted"] for r in local)
    called = [r for r in records if r["provider_attempted"]]
    assert report["request_budget"]["attempted_requests"] == len(called) == 24 - len(local)
    assert report["synthetic"] is True
    assert report["provenance_notes"] == [ge.SYNTHETIC_NOTE]
    assert report["origin_mode"] == "dry_run"
    assert report["reportability"]["reportable_as_hosted_model_diagnostic"] is False
    assert report["identities"] == dict(ge.EXPECTED_IDENTITIES)
    assert report["overall"]["provider_usage"]["input_tokens"]["reported_n"] == 0
    assert report["overall"]["local_token_estimates"]["n"] == len(called)
    assert report["overall"]["citation_structural_invariants"]["violations"] == 0
    assert "questions.dev.jsonl" in guarded
    assert "questions.test.jsonl" not in guarded
    assert all(p.is_relative_to(tmp_path) for p in tmp_path.rglob("*"))
