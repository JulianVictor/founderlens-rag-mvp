"""Opt-in annotation smoke on the frozen baseline (Commit 7.3).

Skipped unless ``FOUNDERLENS_REAL_CORPUS=1`` and the git-ignored records of
``live-20261124T045739024562Z-8665d379`` exist locally. Packets and templates
are built through the verified judging-input gate and written only under
``tmp_path``; no human judgment, reference fact or association is created. No
network, retrieval, provider or held-out file. Run locally::

    $env:FOUNDERLENS_REAL_CORPUS = "1"
    python -m pytest tests/test_evaluation_annotation_real.py -q
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
from founderlens.evaluation import calibration as cal
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


def test_baseline_packets_and_templates_without_any_labels(
    guarded: list[str], tmp_path: pathlib.Path
) -> None:
    before = _digests()
    ctx = an.load_context(load_settings(), RUN)
    assert an.run_counts(ctx) == {
        "scheduled_outcomes": 24,
        "eligible_answers": 22,
        "answers_with_claims": 17,
        "answers_without_claims": 5,
        "claims": 36,
        "claim_evidence_associations": 51,
        "distinct_citations": 33,
        "financial_quantities": 27,
        "ineligible": ["dev-016", "dev-021"],
    }
    for qid in ("dev-016", "dev-021"):
        with pytest.raises(an.AnnotationError, match="ineligible"):
            an.build_packet(ctx, qid)
    units = 0
    for q in ctx.run.questions:
        if not q.eligible:
            continue
        packet = an.build_packet(ctx, q.question_id)
        assert packet == an.build_packet(ctx, q.question_id)
        template = an.worksheet_template(packet, "ann-smoke", Role.CALIBRATION)
        an.Worksheet.model_validate(template)
        assert all(e["review_action"] == "pending" for e in template["entries"])
        assert all(e.get("label") is None for e in template["entries"])
        units += len(template["entries"])
    paths = an.AnnotationPaths(tmp_path / "annotation" / RUN)
    summary = an.write_packets(ctx, paths)
    assert (summary["packets"], summary["template_units"]) == (22, units)
    assert units == 362  # 325 pending judgments + 37 pending reference facts
    assert (summary["human_judgments_created"], summary["verified_reference_facts_created"]) == (
        0,
        0,
    )
    assert len(list(paths.packets.glob("dev-*.json"))) == 22
    assert not paths.ledger.exists()
    progress = an.progress(ctx, an.Ledger())
    assert (progress["ledger_entries"], progress["verified_reference_facts"]) == (0, 0)
    assert progress["fact_associations"] == 0
    citation_units = [u for u in an.template_judgment_units(ctx) if u.endswith("#citation_support")]
    assert len(citation_units) == len(set(citation_units)) == 51  # one per claim x citation
    assert progress["pending_citation_support_units"] == 51
    states = progress["judgment_review_states"]
    assert states["unlabeled"] == 325
    assert sum(states.values()) == 325  # nothing single-reviewed, agreed or adjudicated
    assert progress["independently_reviewed_units"] == 0
    calibration = cal.build_calibration_set(ctx, an.Ledger())
    assert (calibration["natural"], calibration["synthetic"]) == ([], [])
    assert "SENTINEL" not in json.dumps(summary)
    assert _digests() == before  # frozen records, report and run log unchanged
    assert "questions.test.jsonl" not in guarded
