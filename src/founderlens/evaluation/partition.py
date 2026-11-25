"""Frozen question-level calibration/evaluation allocation (Stage 6, Commit 7.6A; §20).

Usage (offline)::

    python -m founderlens.evaluation.partition verify    --run <run_id>
    python -m founderlens.evaluation.partition freeze    --run <run_id>   # refuses to overwrite
    python -m founderlens.evaluation.partition preflight --run <run_id>

A partition manifest assigns every *eligible* development answer of one frozen
generation run to exactly one of ``calibration`` or ``evaluation``; model-output
failures are listed as reliability outcomes and get no semantic judgments. The
manifest holds identifiers, categories, statuses and counts only (no question,
answer or passage text) and is committed under ``evaluation/partitions/``.

The allocation is recomputed from the frozen inputs on every load: the policy
(:data:`POLICY_V1`) groups eligible questions by category and abstention status,
takes each group's quota in ``sha256(seed|question_id)`` order, and assigns the
rest to evaluation. A manifest whose content digest, policy ID, source run,
dataset or generation identities, eligible set or assignments do not reproduce is
refused (fail closed). Annotation import, worksheet creation and judge planning
read the manifest, so no question can cross the boundary silently. This is an
internal development split, not a sealed held-out test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.evaluation import annotation as an
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation import judging_input as ji
from founderlens.evaluation.benchmark import EvaluationError
from founderlens.evaluation.judgments import Role
from founderlens.retrieval.schema import RetrievalError

MANIFEST_SCHEMA_VERSION: Final = 1
MANIFEST_SUBDIR: Final = "partitions"
ALLOCATION_ALGORITHM: Final = (
    "per stratum key '<category>/<answered|abstained>' (abstained: insufficient_evidence or a "
    "local abstention; partially_answered counts as answered), the first quota questions in "
    "sha256(seed|question_id) ascending order are calibration; all other eligible questions, "
    "including strata without a quota, are evaluation; ineligible questions are excluded"
)

# The approved development policy. Its canonical JSON hashes to f0c429d4ecc4.
POLICY_V1: Final[dict[str, Any]] = {
    "version": 1,
    "seed": "founderlens-partition-v1",
    "strata": "(question category, response bucket: answered|abstained; partially_answered "
    "counts as answered)",
    "quotas": {
        "single_doc_factual/answered": 2,
        "single_doc_factual/abstained": 1,
        "single_doc_conceptual/answered": 2,
        "cross_doc_comparison/answered": 2,
        "temporal_multi_evidence/answered": 1,
        "unanswerable/abstained": 2,
        "unanswerable/answered": 1,
    },
    "order": "sha256(seed|question_id) ascending within stratum",
    "excluded": "model-output failures",
    "inputs": "frozen dev split metadata and verified response status only",
}
APPROVED_POLICY_ID: Final = "f0c429d4ecc4"

DISCLOSURE: Final = (
    "Internal development calibration/evaluation split of one frozen generation run. It is "
    "not an independent or sealed held-out test; both partitions come from the development "
    "split, are small, and the judge shares the generator's model."
)


class PartitionError(an.AnnotationError):
    """The allocation is missing, invalid or contradicted. Codes and IDs only."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def policy_id(policy: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(policy).encode()).hexdigest()[:12]


def stratum(q: ji.VerifiedQuestion) -> str:
    abstained = q.status == "insufficient_evidence" or q.execution_outcome == "local_abstention"
    return f"{q.category}/{'abstained' if abstained else 'answered'}"


def _rank(seed: str, question_id: str) -> str:
    return hashlib.sha256(f"{seed}|{question_id}".encode()).hexdigest()


def allocate(run: ji.VerifiedRun, policy: Mapping[str, Any]) -> dict[str, list[str]]:
    """Deterministic allocation of the run's eligible questions under ``policy``."""
    groups: dict[str, list[str]] = {}
    for q in run.questions:
        if q.eligible:
            groups.setdefault(stratum(q), []).append(q.question_id)
    calibration: list[str] = []
    evaluation: list[str] = []
    quotas = policy["quotas"]
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda qid: _rank(policy["seed"], qid))
        take = int(quotas.get(key, 0))
        calibration += ordered[:take]
        evaluation += ordered[take:]
    return {
        "calibration": sorted(calibration),
        "evaluation": sorted(evaluation),
        "excluded_model_output_failures": sorted(
            q.question_id for q in run.questions if not q.eligible
        ),
    }


def _eligible_digest(run: ji.VerifiedRun) -> str:
    rows = sorted((q.question_id, q.record_sha256) for q in run.questions if q.eligible)
    return hashlib.sha256(_canonical(rows).encode()).hexdigest()


def _coverage(ctx: an.AnnotationContext, ids: Sequence[str]) -> dict[str, Any]:
    strata = Counter(stratum(ctx.question(qid)) for qid in ids)
    return {
        "questions": len(ids),
        "questions_with_claims": sum(bool(ctx.question(qid).claims) for qid in ids),
        "required_facts": sum(len(ctx.question(qid).required_fact_ids) for qid in ids),
        "by_stratum": dict(sorted(strata.items())),
    }


def build_manifest(
    ctx: an.AnnotationContext, policy: Mapping[str, Any] = POLICY_V1
) -> dict[str, Any]:
    """The manifest of ``ctx``'s run under ``policy`` (safe metadata only)."""
    allocation = allocate(ctx.run, policy)
    body: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "source_run_id": ctx.run.run_id,
        "dataset_sha256": ctx.run.dataset_sha256,
        "split_sha256": ctx.run.split_sha256,
        "dataset_version": ctx.split.dataset_version,
        "collection_artifact_sha256": ctx.run.collection_artifact_sha256,
        "generation_identities": dict(sorted(ctx.run.identities.items())),
        "eligible_question_set_sha256": _eligible_digest(ctx.run),
        "scheduled_outcomes": len(ctx.run.questions),
        "policy": dict(policy),
        "policy_id": policy_id(policy),
        "allocation_algorithm": ALLOCATION_ALGORITHM,
        **allocation,
        "coverage": {
            "calibration": _coverage(ctx, allocation["calibration"]),
            "evaluation": _coverage(ctx, allocation["evaluation"]),
        },
        "disclosure": DISCLOSURE,
    }
    body["allocation_manifest_id"] = hashlib.sha256(_canonical(body).encode()).hexdigest()[:12]
    return body


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    """A verified allocation: question ID -> role, plus its identity."""

    manifest_id: str
    policy_id: str
    source_run_id: str
    roles: Mapping[str, Role]
    excluded: tuple[str, ...]

    def role_of(self, question_id: str) -> Role | None:
        return self.roles.get(question_id)

    def questions(self, role: Role) -> tuple[str, ...]:
        return tuple(sorted(q for q, r in self.roles.items() if r is role))


def verify_manifest(
    ctx: an.AnnotationContext,
    data: Mapping[str, Any],
    *,
    approved_policy_id: str | None = None,
) -> PartitionManifest:
    """Recompute everything from the frozen inputs; refuse any difference."""
    problems = []
    if data.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        problems.append("unsupported manifest schema version")
    policy = data.get("policy")
    if not isinstance(policy, Mapping):
        raise PartitionError(["manifest has no policy"])
    expected = build_manifest(ctx, policy)
    body = {k: v for k, v in data.items() if k != "allocation_manifest_id"}
    if hashlib.sha256(_canonical(body).encode()).hexdigest()[:12] != data.get(
        "allocation_manifest_id"
    ):
        problems.append("manifest content digest does not match (edited or corrupted)")
    for name in (
        "source_run_id",
        "dataset_sha256",
        "split_sha256",
        "collection_artifact_sha256",
        "generation_identities",
        "eligible_question_set_sha256",
        "policy_id",
        "calibration",
        "evaluation",
        "excluded_model_output_failures",
    ):
        if data.get(name) != expected[name]:
            problems.append(f"manifest {name} does not reproduce from the frozen inputs")
    if approved_policy_id is not None and data.get("policy_id") != approved_policy_id:
        problems.append("manifest policy is not the approved policy")
    calibration, evaluation = set(expected["calibration"]), set(expected["evaluation"])
    if calibration & evaluation:
        problems.append("a question is in both partitions")
    if problems:
        raise PartitionError(problems)
    roles = dict.fromkeys(calibration, Role.CALIBRATION)
    roles |= dict.fromkeys(evaluation, Role.EVALUATION)
    return PartitionManifest(
        manifest_id=str(data["allocation_manifest_id"]),
        policy_id=str(data["policy_id"]),
        source_run_id=ctx.run.run_id,
        roles=roles,
        excluded=tuple(expected["excluded_model_output_failures"]),
    )


def manifest_path(settings: Settings, run_id: str) -> Path:
    an.annotation_paths(settings, run_id)  # validates the run ID
    return settings.evaluation_dir / MANIFEST_SUBDIR / f"{run_id}.json"


def load_manifest(settings: Settings, ctx: an.AnnotationContext) -> PartitionManifest:
    """The committed manifest of ``ctx``'s run, verified; missing or invalid fails closed."""
    path = manifest_path(settings, ctx.run.run_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PartitionError(
            ["no readable partition manifest for this run (fail closed)"]
        ) from None
    if not isinstance(data, dict):
        raise PartitionError(["partition manifest must be a JSON object"])
    return verify_manifest(ctx, data, approved_policy_id=APPROVED_POLICY_ID)


# --------------------------------------------------------------------------- #
# Enforcement helpers
# --------------------------------------------------------------------------- #


def ledger_conflicts(manifest: PartitionManifest, ledger: an.Ledger) -> list[str]:
    """Questions whose ledger history (withdrawn entries included) contradicts the manifest."""
    problems = []
    for qid, role in sorted(ledger.partitions().items()):
        assigned = manifest.role_of(qid)
        if assigned is None:
            problems.append(f"{qid}: ledger entries for a question outside the allocation")
        elif assigned is not role:
            problems.append(f"{qid}: ledger history is {role.value}, manifest is {assigned.value}")
        roles = {e.role for e in ledger.entries if e.question_id == qid}
        if len(roles) > 1:
            problems.append(f"{qid}: ledger history mixes calibration and evaluation")
    return problems


def check_worksheet(manifest: PartitionManifest, question_id: str, role: Role) -> list[str]:
    assigned = manifest.role_of(question_id)
    if assigned is None:
        if question_id in manifest.excluded:
            return [f"{question_id} is a model-output failure; no semantic annotation"]
        return [f"{question_id} is not in the approved allocation"]
    if assigned is not role:
        return [f"{question_id} is allocated to {assigned.value}; worksheet role {role.value}"]
    return []


def preflight(
    manifest: PartitionManifest, paths: an.AnnotationPaths, ledger: an.Ledger
) -> dict[str, Any]:
    """Read-only diagnostics of existing worksheets and ledger history (IDs and codes)."""
    worksheets: list[dict[str, Any]] = []
    if paths.worksheets.is_dir():
        for path in sorted(paths.worksheets.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                qid, role = str(data["question_id"]), Role(data["role"])
                actions = Counter(str(e.get("review_action")) for e in data.get("entries", []))
            except (OSError, ValueError, KeyError, TypeError):
                worksheets.append({"worksheet": path.name, "status": "unreadable"})
                continue
            problems = check_worksheet(manifest, qid, role)
            allocated = manifest.role_of(qid)
            worksheets.append(
                {
                    "worksheet": path.name,
                    "question_id": qid,
                    "role": role.value,
                    "allocated": None if allocated is None else allocated.value,
                    "status": "compatible" if not problems else "incompatible",
                    "review_actions": dict(sorted(actions.items())),
                }
            )
    return an.ensure_public(
        {
            "allocation_manifest_id": manifest.manifest_id,
            "worksheets": worksheets,
            "incompatible_worksheets": sum(w["status"] != "compatible" for w in worksheets),
            "ledger_conflicts": len(ledger_conflicts(manifest, ledger)),
            "ledger_entries": len(ledger.entries),
            "changed_files": 0,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.partition",
        description="Frozen calibration/evaluation question allocation (offline).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "freeze", "preflight"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        ctx = an.load_context(settings, args.run)
        output: object
        if args.command == "freeze":
            manifest = build_manifest(ctx)
            if manifest["policy_id"] != APPROVED_POLICY_ID:
                raise PartitionError(["the policy does not hash to the approved policy ID"])
            path = manifest_path(settings, args.run)
            if path.exists():
                raise PartitionError(["a manifest already exists; it is never overwritten"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            output = {"allocation_manifest_id": manifest["allocation_manifest_id"], "written": True}
        else:
            verified = load_manifest(settings, ctx)
            if args.command == "verify":
                output = {
                    "allocation_manifest_id": verified.manifest_id,
                    "policy_id": verified.policy_id,
                    "calibration": list(verified.questions(Role.CALIBRATION)),
                    "evaluation": list(verified.questions(Role.EVALUATION)),
                    "excluded_model_output_failures": list(verified.excluded),
                }
            else:
                paths = an.annotation_paths(settings, args.run)
                output = preflight(verified, paths, an.load_ledger(paths.ledger))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (
        an.AnnotationError,
        ji.JudgingInputError,
        ge.EvaluationRunError,
        EvaluationError,
        RetrievalError,
    ) as exc:
        print(f"Partition {args.command} FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
