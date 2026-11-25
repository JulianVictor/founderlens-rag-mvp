"""Frozen calibration/evaluation allocation (Commit 7.6A), offline and synthetic.

The synthetic run has six questions: dev-001 is a model-output failure, dev-003 is
allocated to calibration and dev-002, dev-004, dev-005 and dev-006 to evaluation
(see ``tests.test_evaluation_annotation.SYNTHETIC_POLICY``).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import pathlib
from datetime import UTC, datetime
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.evaluation import annotation as an
from founderlens.evaluation import calibration as cal
from founderlens.evaluation import judge as jd
from founderlens.evaluation import partition as pt
from founderlens.evaluation.judgments import Role
from tests.test_evaluation_annotation import (
    SYNTHETIC_POLICY,
    commit,
    ctx,  # noqa: F401  (fixture)
    find,
    inputs,  # noqa: F401  (fixture)
    manifest,
    offline,  # noqa: F401  (fixture)
    only,
    paths,  # noqa: F401  (fixture)
    private_texts,
    sheet,
    span,
    supported,
)
from tests.test_evaluation_judge import FakeJudge, calibration_ledger
from tests.test_evaluation_judging_input import CLEAN

NOW = datetime(2026, 11, 25, 10, 0, tzinfo=UTC)
Q4 = "nvda-fy2025-q4:p0001:c0000:0c2f87411627"


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the allocation itself -------------------------------------------------------------------


def test_approved_policy_reproduces_its_identity() -> None:
    assert pt.policy_id(pt.POLICY_V1) == pt.APPROVED_POLICY_ID == "f0c429d4ecc4"
    changed = {**pt.POLICY_V1, "seed": "other"}
    assert pt.policy_id(changed) != pt.APPROVED_POLICY_ID


def test_allocation_covers_every_outcome_exactly_once(
    ctx: an.AnnotationContext,  # noqa: F811
) -> None:
    built = pt.build_manifest(ctx, SYNTHETIC_POLICY)
    assert built == pt.build_manifest(ctx, SYNTHETIC_POLICY)  # deterministic
    calibration, evaluation = set(built["calibration"]), set(built["evaluation"])
    excluded = set(built["excluded_model_output_failures"])
    assert (calibration, excluded) == ({"dev-003"}, {"dev-001"})
    assert evaluation == {"dev-002", "dev-004", "dev-005", "dev-006"}
    assert not calibration & evaluation
    assert calibration | evaluation | excluded == {q.question_id for q in ctx.run.questions}
    assert built["scheduled_outcomes"] == len(ctx.run.questions)
    verified = pt.verify_manifest(ctx, built)
    assert verified.manifest_id == built["allocation_manifest_id"]
    assert verified.role_of("dev-001") is None
    assert verified.excluded == ("dev-001",)


def test_tampered_or_foreign_manifests_fail_closed(
    ctx: an.AnnotationContext,  # noqa: F811
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = pt.build_manifest(ctx, SYNTHETIC_POLICY)
    edited = copy.deepcopy(built)
    edited["calibration"] = ["dev-002", "dev-003"]
    with pytest.raises(pt.PartitionError, match="digest does not match"):
        pt.verify_manifest(ctx, edited)
    body = {k: v for k, v in edited.items() if k != "allocation_manifest_id"}
    edited["allocation_manifest_id"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    with pytest.raises(pt.PartitionError, match="calibration does not reproduce"):
        pt.verify_manifest(ctx, edited)  # consistent digest, wrong assignments
    foreign = {**built, "source_run_id": "live-other"}
    with pytest.raises(pt.PartitionError, match="source_run_id"):
        pt.verify_manifest(ctx, foreign)
    with pytest.raises(pt.PartitionError, match="no policy"):
        pt.verify_manifest(ctx, {k: v for k, v in built.items() if k != "policy"})
    settings = load_settings()
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(pt, "manifest_path", lambda s, run_id: missing)
    with pytest.raises(pt.PartitionError, match="fail closed"):
        pt.load_manifest(settings, ctx)
    missing.write_text(json.dumps(built), encoding="utf-8")  # valid, but not the approved policy
    with pytest.raises(pt.PartitionError, match="approved policy"):
        pt.load_manifest(settings, ctx)


def test_manifest_holds_safe_metadata_only(ctx: an.AnnotationContext) -> None:  # noqa: F811
    blob = json.dumps(pt.build_manifest(ctx, SYNTHETIC_POLICY))
    for text in private_texts(ctx):
        assert text not in blob
    for key in ("rationale", "label", "annotator", "statement"):
        assert key not in blob


# --- annotation enforcement --------------------------------------------------------------------


def test_worksheet_roles_come_from_the_manifest(
    ctx: an.AnnotationContext,  # noqa: F811
    paths: an.AnnotationPaths,  # noqa: F811
) -> None:
    m = manifest(ctx)
    created = an.create_worksheet(ctx, paths, "dev-002", "ann-a1", None, manifest=m)
    assert json.loads(created.read_text(encoding="utf-8"))["role"] == "evaluation"
    with pytest.raises(an.AnnotationError, match="allocated to evaluation"):
        an.create_worksheet(ctx, paths, "dev-004", "ann-a1", Role.CALIBRATION, manifest=m)
    assert not (paths.worksheets / "dev-004.ann-a1.json").exists()
    with pytest.raises(an.AnnotationError, match="model-output failure"):
        an.create_worksheet(ctx, paths, "dev-001", "ann-a1", None, manifest=m)
    calibration = an.create_worksheet(ctx, paths, "dev-003", "ann-a1", None, manifest=m)
    assert json.loads(calibration.read_text(encoding="utf-8"))["role"] == "calibration"


def test_stale_wrong_role_worksheet_is_flagged_never_imported_or_changed(
    ctx: an.AnnotationContext,  # noqa: F811
    paths: an.AnnotationPaths,  # noqa: F811
) -> None:
    m = manifest(ctx)
    stale = paths.worksheets / "dev-002.ann-owner.json"  # like the real dev-003 worksheet
    an.write_json(stale, sheet(ctx, "dev-002", annotator="ann-owner", role=Role.CALIBRATION))
    before = digest(stale)
    report = pt.preflight(m, paths, an.Ledger())
    (row,) = report["worksheets"]
    assert (row["status"], row["role"], row["allocated"]) == (
        "incompatible",
        "calibration",
        "evaluation",
    )
    assert report["changed_files"] == 0
    assert set(row["review_actions"]) == {"pending"}
    with pytest.raises(an.AnnotationError, match="allocated to evaluation"):
        an.import_worksheet(ctx, paths, stale, manifest=m, now=NOW)
    assert digest(stale) == before
    assert not paths.ledger.exists()
    filled = sheet(ctx, "dev-002", annotator="ann-owner", role=Role.CALIBRATION)
    entry = find(filled, "claim_faithfulness", claim_position=1)
    entry.update(review_action="confirmed", label="supported",
                 evidence=[span(Q4, "$39.3 billion", "selected_context", "E03")])  # fmt: skip
    an.write_json(stale, only(filled, entry))
    with pytest.raises(an.AnnotationError, match="partition check failed"):
        an.import_worksheet(ctx, paths, stale, manifest=m, now=NOW)
    assert not paths.ledger.exists()


def test_ledger_history_must_agree_with_the_manifest(
    ctx: an.AnnotationContext,  # noqa: F811
    paths: an.AnnotationPaths,  # noqa: F811
) -> None:
    m = manifest(ctx)
    # a legacy ledger written without the manifest puts evaluation question dev-002 in
    # calibration (plan_import alone does not consult the manifest)
    legacy = sheet(ctx, "dev-002", role=Role.CALIBRATION)
    entry = find(legacy, "claim_faithfulness", claim_position=1)
    entry.update(review_action="confirmed", label="supported",
                 evidence=[span(Q4, "$39.3 billion", "selected_context", "E03")])  # fmt: skip
    ledger = commit(ctx, an.Ledger(), only(legacy, entry))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.ledger.write_text("".join(e.model_dump_json() + "\n" for e in ledger.entries),
                            encoding="utf-8")  # fmt: skip
    assert pt.ledger_conflicts(m, ledger) == [
        "dev-002: ledger history is calibration, manifest is evaluation"
    ]
    before = digest(paths.ledger)
    good = sheet(ctx)
    path = paths.worksheets / "dev-003.ann-a1.json"
    an.write_json(path, only(good, supported(good)))
    with pytest.raises(an.AnnotationError, match="ledger history is calibration"):
        an.import_worksheet(ctx, paths, path, manifest=m, now=NOW)  # nothing is imported
    assert digest(paths.ledger) == before  # history never rewritten
    assert pt.preflight(m, paths, ledger)["ledger_conflicts"] == 1
    with pytest.raises(pt.PartitionError, match="ledger history"):
        jd.plan_jobs(ctx, ledger, Role.CALIBRATION, manifest=m)


def test_withdrawals_and_revisions_never_change_a_partition(
    ctx: an.AnnotationContext,  # noqa: F811
    paths: an.AnnotationPaths,  # noqa: F811
) -> None:
    m = manifest(ctx)
    ws = sheet(ctx)
    path = paths.worksheets / "dev-003.ann-a1.json"
    an.write_json(path, only(ws, supported(ws)))
    an.import_worksheet(ctx, paths, path, manifest=m, now=NOW)
    head = an.load_ledger(paths.ledger).entries[0]
    withdraw = {**supported(sheet(ctx)), "review_action": "withdrawn",
                "supersedes": head.entry_id, "revision_reason": "entry_error"}  # fmt: skip
    an.write_json(path, only(ws, withdraw))
    an.import_worksheet(ctx, paths, path, manifest=m, now=NOW)
    ledger = an.load_ledger(paths.ledger)
    assert ledger.partitions() == {"dev-003": Role.CALIBRATION}  # history keeps it
    other = sheet(ctx, annotator="ann-c3", role=Role.EVALUATION)
    evaluation_path = paths.worksheets / "dev-003.ann-c3.json"
    an.write_json(evaluation_path, only(other, supported(other)))
    with pytest.raises(an.AnnotationError, match="allocated to calibration"):
        an.import_worksheet(ctx, paths, evaluation_path, manifest=m, now=NOW)


def test_synthetic_variants_inherit_the_source_partition(
    ctx: an.AnnotationContext,  # noqa: F811
) -> None:
    m = manifest(ctx)
    ledger = calibration_ledger(ctx)
    for variant in cal.synthetic_variants(ctx, ledger):
        assert m.role_of(variant.question_id) is Role.CALIBRATION
        assert variant.record(private=False)["partition"] == "calibration"
    with pytest.raises(an.AnnotationError, match="synthetic-variant judging"):
        jd.plan_jobs(ctx, ledger, Role.CALIBRATION, lineage="synthetic", manifest=m)


# --- judge enforcement -------------------------------------------------------------------------


def test_judge_plans_follow_the_manifest(ctx: an.AnnotationContext) -> None:  # noqa: F811
    m = manifest(ctx)
    calibration = jd.plan_jobs(ctx, an.Ledger(), Role.CALIBRATION, manifest=m)
    evaluation = jd.plan_jobs(ctx, an.Ledger(), Role.EVALUATION, manifest=m)
    assert calibration.questions == m.questions(Role.CALIBRATION) == ("dev-003",)
    assert evaluation.questions == m.questions(Role.EVALUATION)
    assert not set(calibration.questions) & set(evaluation.questions)
    assert {j.partition for j in evaluation.jobs} == {"evaluation"}
    assert calibration.allocation_manifest_id == m.manifest_id
    with pytest.raises(pt.PartitionError, match="needs the frozen partition manifest"):
        jd.plan_jobs(ctx, an.Ledger(), Role.CALIBRATION)
    assert "dev-001" not in calibration.questions + evaluation.questions


def test_questions_cannot_move_between_planning_and_execution(
    ctx: an.AnnotationContext,  # noqa: F811
    tmp_path: pathlib.Path,
) -> None:
    m = manifest(ctx)
    ledger = calibration_ledger(ctx)
    plan = jd.plan_jobs(ctx, ledger, Role.CALIBRATION, manifest=m)
    other_policy = {**SYNTHETIC_POLICY, "quotas": {"cross_doc_comparison/answered": 1}}
    other = pt.verify_manifest(ctx, pt.build_manifest(ctx, other_policy))
    provider = FakeJudge()
    with pytest.raises(pt.PartitionError, match="another partition allocation"):
        jd.run_judge(ctx, plan, lambda: provider, tmp_path, mode="mock", max_requests=5,
                     git=CLEAN, manifest=other, ledger=ledger, now=lambda: NOW)  # fmt: skip
    moved = dataclasses.replace(
        plan, batches=(dataclasses.replace(plan.batches[0], question_id="dev-002"),)
    )
    with pytest.raises(pt.PartitionError, match="not in the selected partition"):
        jd.run_judge(ctx, moved, lambda: provider, tmp_path, mode="mock", max_requests=5,
                     git=CLEAN, manifest=m, ledger=ledger, now=lambda: NOW)  # fmt: skip
    assert provider.calls == 0
    assert not (tmp_path / "judge").exists()  # refused before pre-registration


def test_replay_verifies_the_partition_identity(
    ctx: an.AnnotationContext,  # noqa: F811
    tmp_path: pathlib.Path,
) -> None:
    m = manifest(ctx)
    ledger = calibration_ledger(ctx)
    plan = jd.plan_jobs(ctx, ledger, Role.CALIBRATION, manifest=m)
    outputs = jd.run_judge(ctx, plan, lambda: FakeJudge(), tmp_path, mode="mock",
                           max_requests=20, git=CLEAN, manifest=m, ledger=ledger,
                           now=lambda: NOW)  # fmt: skip
    log = (tmp_path / "judge" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(log[0])["allocation_manifest_id"] == m.manifest_id  # pre-registered
    assert outputs.summary["allocation_manifest_id"] == m.manifest_id
    records = jd.read_records(outputs.records_path)
    assert {r["allocation_manifest_id"] for r in records} == {m.manifest_id}
    assert jd.replay(ctx, ledger, outputs.records_path, outputs.summary_path, m)[
        "equivalent_to_original"
    ]
    other_policy = {**SYNTHETIC_POLICY, "seed": "another-seed"}
    other = pt.verify_manifest(ctx, pt.build_manifest(ctx, other_policy))
    assert other.manifest_id != m.manifest_id
    with pytest.raises(an.AnnotationError, match="another partition allocation"):
        jd.replay(ctx, ledger, outputs.records_path, outputs.summary_path, other)


def test_partition_tools_never_open_held_out_or_call_a_provider(
    ctx: an.AnnotationContext,  # noqa: F811
    paths: an.AnnotationPaths,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import founderlens.generation.openai_provider as op

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("provider constructed")

    monkeypatch.setattr(op, "OpenAIProvider", forbidden)
    opened: list[str] = []
    for name in ("read_bytes", "read_text", "open"):
        original = getattr(pathlib.Path, name)

        def spy(self: pathlib.Path, *args: Any, _original: Any = original, **kw: Any) -> Any:
            opened.append(self.name)
            if self.name == "questions.test.jsonl":
                raise AssertionError("held-out file opened")
            return _original(self, *args, **kw)

        monkeypatch.setattr(pathlib.Path, name, spy)
    m = manifest(ctx)
    pt.preflight(m, paths, an.Ledger())
    jd.plan_jobs(ctx, an.Ledger(), Role.EVALUATION, manifest=m)
    assert "questions.test.jsonl" not in opened
    assert pt.main(["verify", "--run", "test-run"]) == 1  # not a live development run
