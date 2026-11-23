"""Protocol v1.3 execution outcomes, completion states and the run log (Commit 6.3 follow-up).

Protocol v1.3 (``docs/evaluation_protocol.md`` §14) is **prospective**. It was
introduced after the v1.2 pilot ``live-20261123T111430692539Z-8665d379``
(24 attempts, 23 valid outcomes, one ``response_contract`` failure) showed that
requiring zero failures would make a run reportable only if every answer was
valid. Repeating runs until one is clean selects for success. Runs whose
records carry protocol 1.2 (record schemas 1 and 2) keep v1.2 semantics.

**Failure taxonomy.** A failure is classified from its *stage and category*
together. Only listed combinations are model-output or infrastructure failures.
Everything else, including unknown combinations, is an integrity failure,
which is conservative: it makes the run incomplete.

**Execution outcomes.** Every scheduled question ends as exactly one of
``valid_answer``, ``local_abstention``, ``model_output_failure``,
``infrastructure_failure``, ``integrity_failure`` or ``not_run``.

**Completion states** (mutually exclusive):

- ``answer_complete``: every question has a valid answer or local abstention,
  with every guardrail check completed;
- ``execution_complete``: every question reached a terminal outcome, some of
  them model-output failures, and nothing else went wrong;
- ``incomplete``: any infrastructure or integrity failure, guardrail failure,
  unrun question, interruption, abort or structural problem.

**Run log.** ``run_log.jsonl`` is append-only: a ``preregistered`` entry before
any live provider exists, then a ``finished`` entry. A crash leaves the first
entry alone, which reads as unfinished. Entries are flushed and fsynced, never
rewritten. The log is a local file, not tamper-proof.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

PROTOCOL_V12: Final = "1.2"
PROTOCOL_V13: Final = "1.3"
SELECTION_RULE: Final = (
    "report the first execution-complete live run under protocol v1.3; disclose every other run"
)
RUN_LOG_FILENAME: Final = "run_log.jsonl"


class FailureClass(StrEnum):
    MODEL_OUTPUT = "model_output"
    INFRASTRUCTURE = "infrastructure"
    INTEGRITY = "integrity"


class ExecutionOutcome(StrEnum):
    VALID_ANSWER = "valid_answer"
    LOCAL_ABSTENTION = "local_abstention"
    MODEL_OUTPUT_FAILURE = "model_output_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    INTEGRITY_FAILURE = "integrity_failure"
    NOT_RUN = "not_run"


class CompletionState(StrEnum):
    ANSWER_COMPLETE = "answer_complete"
    EXECUTION_COMPLETE = "execution_complete"
    INCOMPLETE = "incomplete"


# (stage, category) pairs produced by the model's own output. The response contract counts only
# at answer_validation: the same category at citation_resolution is an application inconsistency.
MODEL_OUTPUT_FAILURES: Final = frozenset(
    {
        ("answer_validation", "response_contract"),
        ("generation", "refusal"),
        ("generation", "incomplete"),
        ("generation", "structured_output_invalid"),
    }
)
INFRASTRUCTURE_CATEGORIES: Final = frozenset(
    {
        "timeout",
        "connection",
        "rate_limited",
        "provider_server_error",
        "authentication",
        "permission_denied",
        "quota_exceeded",
        "model_unavailable",
        "configuration",
    }
)
FAILURE_TO_OUTCOME: Final = {
    FailureClass.MODEL_OUTPUT: ExecutionOutcome.MODEL_OUTPUT_FAILURE,
    FailureClass.INFRASTRUCTURE: ExecutionOutcome.INFRASTRUCTURE_FAILURE,
    FailureClass.INTEGRITY: ExecutionOutcome.INTEGRITY_FAILURE,
}
VALID: Final = frozenset({ExecutionOutcome.VALID_ANSWER, ExecutionOutcome.LOCAL_ABSTENTION})
TERMINAL: Final = VALID | {ExecutionOutcome.MODEL_OUTPUT_FAILURE}


def classify_failure(stage: object, category: object) -> FailureClass:
    """Model-output or infrastructure only for listed combinations; anything else is integrity."""
    if (stage, category) in MODEL_OUTPUT_FAILURES:
        return FailureClass.MODEL_OUTPUT
    if stage == "generation" and category in INFRASTRUCTURE_CATEGORIES:
        return FailureClass.INFRASTRUCTURE
    return FailureClass.INTEGRITY


def classification_table() -> dict[str, Any]:
    """The exact mapping, for reports and documentation."""
    return {
        "model_output": sorted(f"{stage}/{category}" for stage, category in MODEL_OUTPUT_FAILURES),
        "infrastructure": sorted(f"generation/{c}" for c in INFRASTRUCTURE_CATEGORIES),
        "integrity": "every other stage/category combination, including unknown ones",
    }


def record_protocol(record: Mapping[str, Any]) -> str:
    """Protocol of a record; schemas 1 and 2 predate the field and mean protocol 1.2."""
    return str(record.get("protocol_version") or PROTOCOL_V12)


def execution_outcome(record: Mapping[str, Any]) -> ExecutionOutcome:
    """The terminal outcome of one record (a replay keeps its source's failure outcome)."""
    outcome = record.get("outcome")
    if outcome == "completed":
        if record.get("origin") == "local":
            return ExecutionOutcome.LOCAL_ABSTENTION
        return ExecutionOutcome.VALID_ANSWER
    if outcome == "failed":
        failure = record.get("failure") or {}
        return FAILURE_TO_OUTCOME[
            classify_failure(failure.get("stage"), failure.get("error_category"))
        ]
    if outcome == "not_replayable":
        source = (record.get("replay_of") or {}).get("execution_outcome")
        if source in (
            ExecutionOutcome.MODEL_OUTPUT_FAILURE,
            ExecutionOutcome.INFRASTRUCTURE_FAILURE,
            ExecutionOutcome.INTEGRITY_FAILURE,
        ):
            return ExecutionOutcome(source)
        return ExecutionOutcome.NOT_RUN
    return ExecutionOutcome.NOT_RUN


def completion_state(
    records: Sequence[Mapping[str, Any]], structural_problems: Sequence[str]
) -> tuple[CompletionState, list[str]]:
    """Protocol v1.3 state and the reasons it is not ``answer_complete``."""
    reasons = list(structural_problems)
    counts: dict[ExecutionOutcome, int] = {}
    for record in records:
        outcome = execution_outcome(record)
        counts[outcome] = counts.get(outcome, 0) + 1
    for outcome in (
        ExecutionOutcome.INFRASTRUCTURE_FAILURE,
        ExecutionOutcome.INTEGRITY_FAILURE,
        ExecutionOutcome.NOT_RUN,
    ):
        if counts.get(outcome):
            reasons.append(f"{counts[outcome]} question(s) with outcome {outcome.value}")
    guardrail_failures = sum(
        1
        for r in records
        if execution_outcome(r) in VALID
        and (r.get("guardrails") or {}).get("status") != "completed"
    )
    if guardrail_failures:
        reasons.append(f"{guardrail_failures} valid answer(s) without completed guardrail checks")
    if reasons:
        return CompletionState.INCOMPLETE, reasons
    failures = counts.get(ExecutionOutcome.MODEL_OUTPUT_FAILURE, 0)
    if failures:
        return CompletionState.EXECUTION_COMPLETE, [f"{failures} model-output failure(s), counted"]
    return CompletionState.ANSWER_COMPLETE, []


# --------------------------------------------------------------------------- #
# Reliability and unconditional metrics
# --------------------------------------------------------------------------- #

NOT_APPLICABLE: Final = "not_applicable"


def _ratio(numerator: int, denominator: int, population: str, **extra: Any) -> dict[str, Any]:
    return {
        "population": population,
        "numerator": numerator,
        "denominator": denominator,
        "rate": NOT_APPLICABLE if denominator == 0 else round(numerator / denominator, 4),
        **extra,
    }


def _excluded(
    rows: Sequence[Mapping[str, Any]], keep: Callable[[ExecutionOutcome], bool]
) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        outcome = execution_outcome(r)
        if not keep(outcome):
            out[outcome.value] = out.get(outcome.value, 0) + 1
    return dict(sorted(out.items()))


def exclusions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Scheduled questions of a population left out of answer-dependent metrics, by outcome."""
    excluded = _excluded(rows, lambda o: o in VALID)
    return {
        "scheduled": len(rows),
        "with_valid_outcome": len(rows) - sum(excluded.values()),
        "excluded_by_outcome": excluded,
    }


def reliability_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Unconditional outcome views over scheduled questions (protocol v1.3 §14)."""
    outcomes = [execution_outcome(r) for r in records]
    distribution = {o.value: 0 for o in ExecutionOutcome}
    statuses: dict[str, int] = {}
    for r, o in zip(records, outcomes, strict=True):
        distribution[o.value] += 1
        if o is ExecutionOutcome.VALID_ANSWER:
            key = f"valid_{r.get('status')}"
            statuses[key] = statuses.get(key, 0) + 1
    terminal = [(r, o) for r, o in zip(records, outcomes, strict=True) if o in TERMINAL]
    provider = [(r, o) for r, o in terminal if o is not ExecutionOutcome.LOCAL_ABSTENTION]
    failed: dict[str, dict[str, int]] = {}
    for r, o in zip(records, outcomes, strict=True):
        if o in (
            ExecutionOutcome.MODEL_OUTPUT_FAILURE,
            ExecutionOutcome.INFRASTRUCTURE_FAILURE,
            ExecutionOutcome.INTEGRITY_FAILURE,
        ):
            category = str(
                r.get("error_category") or (r.get("replay_of") or {}).get("error_category")
            )
            failed.setdefault(o.value, {}).setdefault(category, 0)
            failed[o.value][category] += 1
    non_terminal = len(records) - len(terminal)

    def flag(
        answerable: bool, wanted: Callable[[Mapping[str, Any], ExecutionOutcome], bool]
    ) -> dict[str, Any]:
        rows = [(r, o) for r, o in terminal if bool(r.get("answerable")) is answerable]
        population = (
            "terminal answerable questions" if answerable else "terminal unanswerable questions"
        )
        return _ratio(
            sum(1 for r, o in rows if wanted(r, o)),
            len(rows),
            population + " (model-output failures included as separate outcomes)",
            model_output_failures=sum(
                1 for _, o in rows if o is ExecutionOutcome.MODEL_OUTPUT_FAILURE
            ),
            excluded_non_terminal=sum(
                1
                for r, o in zip(records, outcomes, strict=True)
                if bool(r.get("answerable")) is answerable and o not in TERMINAL
            ),
        )

    def status_in(*statuses_wanted: str) -> Callable[[Mapping[str, Any], ExecutionOutcome], bool]:
        def check(r: Mapping[str, Any], o: ExecutionOutcome) -> bool:
            if o is ExecutionOutcome.LOCAL_ABSTENTION:
                return "local_abstention" in statuses_wanted
            return o is ExecutionOutcome.VALID_ANSWER and r.get("status") in statuses_wanted

        return check

    abstained = status_in("insufficient_evidence", "local_abstention")
    return {
        "scheduled": len(records),
        "unconditional_outcome_distribution": distribution,
        "valid_answer_status_counts": dict(sorted(statuses.items())),
        "contract_valid_outcome_rate": _ratio(
            sum(1 for _, o in terminal if o in VALID),
            len(terminal),
            "scheduled questions with a terminal outcome",
            excluded_non_terminal=non_terminal,
            note="structurally valid under the response contract; not factual correctness",
        ),
        "model_output_failure_rate": _ratio(
            sum(1 for _, o in provider if o is ExecutionOutcome.MODEL_OUTPUT_FAILURE),
            len(provider),
            "terminal questions answered by the provider (valid or model-output failure)",
            excluded_non_terminal=non_terminal,
        ),
        "failure_counts": {k: dict(sorted(v.items())) for k, v in sorted(failed.items())},
        "valid_answer_attempt_rate_unconditional": flag(
            True, status_in("answered", "partially_answered")
        ),
        "abstention_on_answerable_rate_unconditional": flag(True, abstained),
        "abstention_on_unanswerable_rate_unconditional": flag(False, abstained),
        "answered_on_unanswerable_rate_unconditional": flag(False, status_in("answered")),
        "partially_answered_on_unanswerable_rate_unconditional": flag(
            False, status_in("partially_answered")
        ),
    }


# --------------------------------------------------------------------------- #
# Append-only run log
# --------------------------------------------------------------------------- #


def append_run_log(path: Path, entry: Mapping[str, Any]) -> None:
    """Append one JSON line, flushed and fsynced. Existing entries are never rewritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(entry), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_run_log(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Valid entries in order, and the number of unreadable (e.g. truncated) lines skipped."""
    if not path.exists():
        return [], 0
    entries: list[dict[str, Any]] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(value, dict):
            entries.append(value)
        else:
            skipped += 1
    return entries, skipped


def run_log_status(entries: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per run: the preregistration and its finish, or ``unfinished`` when none was logged."""
    runs: dict[str, dict[str, Any]] = {}
    for entry in entries:
        run = runs.setdefault(str(entry.get("run_id")), {"preregistered": False, "finished": None})
        if entry.get("event") == "preregistered":
            run["preregistered"] = True
        elif entry.get("event") == "finished":
            run["finished"] = entry.get("completion_state")
    for run in runs.values():
        if run["finished"] is None:
            run["finished"] = "unfinished"
    return runs
