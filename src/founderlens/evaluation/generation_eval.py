"""Development-split diagnostic generation evaluation (Commit 6.3, protocol v1.2 §13).

Usage::

    python -m founderlens.evaluation.generation_eval                      # offline, SYNTHETIC
    python -m founderlens.evaluation.generation_eval --replay <records>   # offline replay
    # paid, never automatic (see docs/evaluation_protocol.md §13):
    $env:FOUNDERLENS_RUN_PAID_EVAL = "1"
    python -m founderlens.evaluation.generation_eval --live --max-requests 24

``evaluation_kind: diagnostic_deterministic``. These diagnostics do not
establish answer correctness, faithfulness, claim-level citation support or
semantic abstention appropriateness. Formal evaluation remains pending Stage 6.

Flow per development question (the held-out split is never opened): the frozen
service answers once, the resolver resolves citations once, and the
deterministic guardrails check that same response. Nothing is regenerated,
repaired, judged or rewritten. Labels (required facts and accepted evidence)
are used only here, in scoring; the pipeline receives the question text only.
Reference answers and expected abstention reasons are never read.

Modes:

- ``dry_run`` (default): the real pipeline with a SYNTHETIC scripted provider.
  Its answers are placeholders, so status and coverage metrics describe the
  script, not a model.
- ``live``: the pinned OpenAI model with SDK retries disabled, a hard request
  budget enforced before sending, and every authorization check passed first.
- ``replay``: rebuilds each request through the frozen pipeline, requires the
  exact stored request hash, revalidates the stored answer through the
  unchanged service and resolver, checks the citations, and recomputes the
  guardrails. It reproduces neither model sampling nor provider latency, and
  records no new usage or latency.

Provenance: every record and report keeps its *origin* (the mode and run ID
that produced the answers). A replay keeps its source's origin, so ``synthetic``
and ``reportability`` follow the origin, never the execution mode: only a
complete live-origin run (or its replay) is reportable as a hosted-model
diagnostic. Record SHA-256 digests detect accidental modification, not
deliberate forgery. ``accepted_page_citation_overlap`` is a reference-alignment
diagnostic, not semantic citation support.

Failures (record schema 2) keep a ``failure`` block: stage, category, invariant
codes with positions, the sanitized provider request ID and status code. A
rejected answer's provider usage is kept; its text never is. Schema 1 records
replay, with their missing diagnostics reported as not recorded.

Outputs (git-ignored ``reports/generation_eval/``): ``<run_id>.records.jsonl``
(private: answers and citations, no question text), ``<run_id>.json`` and
``<run_id>.md`` (codes, IDs and numbers only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import IO, Any, Final, Literal

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.corpus.manifest import ManifestError, load_manifest
from founderlens.evaluation import metrics as m
from founderlens.evaluation.benchmark import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEV_SPLIT,
    DevSplit,
    EvaluationError,
    _git_state,
    gold_evidence,
    load_dev_split,
)
from founderlens.evaluation.schema import Category, EvalQuestion
from founderlens.generation.answer import (
    AnswerStatus,
    ContractProblem,
    ResponseContractError,
    answer_schema_id,
)
from founderlens.generation.citations import (
    CitationResolver,
    GroundedResponse,
    response_contract_id,
)
from founderlens.generation.config import GenerationConfig
from founderlens.generation.context import DEFAULT_CONTEXT_CONFIG, ContextConfig
from founderlens.generation.errors import GenerationError
from founderlens.generation.openai_provider import sdk_version
from founderlens.generation.prompts import prompt_id
from founderlens.generation.schema import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ProviderIdentity,
    SchemaT,
    TokenUsage,
)
from founderlens.generation.service import (
    AnswerOrigin,
    GenerationRunResult,
    GroundedGenerationService,
)
from founderlens.generation.token_budget import TokenCounter
from founderlens.guardrails.response_checks import (
    GuardrailError,
    check_response,
    guardrail_policy_id,
)
from founderlens.guardrails.scope import CorpusVocabulary, analyzer_id
from founderlens.retrieval.policy import PolicyRetriever

EVALUATION_KIND: Final = "diagnostic_deterministic"
PROTOCOL_VERSION: Final = "1.2"
# 2: failure diagnostics (stage, problem codes, request ID) and usage of rejected answers.
RECORD_SCHEMA_VERSION: Final = 2
REPLAYABLE_RECORD_SCHEMA_VERSIONS: Final = (1, 2)  # 1 is legacy: no failure diagnostics
REPORT_TYPE: Final = "generation_diagnostic_evaluation"
CAVEAT: Final = (
    "These diagnostics do not establish answer correctness, faithfulness, claim-level "
    "citation support or semantic abstention appropriateness. Formal evaluation remains "
    "pending Stage 6."
)
SYNTHETIC_NOTE: Final = (
    "SYNTHETIC: answers come from a scripted placeholder provider, not a model. Status, "
    "abstention and citation metrics describe the script, not measured model performance."
)
REPLAY_NOTE: Final = (
    "REPLAY: stored answers revalidated through the frozen pipeline. Usage shown is the "
    "original run's; replay reproduces neither model sampling nor provider latency."
)
PAID_ENV_VAR: Final = "FOUNDERLENS_RUN_PAID_EVAL"
API_KEY_ENV_VAR: Final = "OPENAI_API_KEY"
HARD_REQUEST_CEILING: Final = 24
SMALL_CATEGORY_N: Final = 5  # categories with fewer dataset questions report counts only
USAGE_KEYS: Final = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
REPORT_SUBDIR: Final = "generation_eval"
NOT_APPLICABLE: Final = "not_applicable"

# The frozen system this harness may measure. A mismatch aborts before any provider exists.
EXPECTED_IDENTITIES: Final[Mapping[str, str]] = {
    "retrieval_policy_id": "f20913f1c1ec",
    "collection_artifact_sha256": (
        "f9a8f12a8d1c61aec3580f7f95dd7fd44a26860ff751d85d5a778693f59c55a8"
    ),
    "context_config_id": "0e7fd58314ff",
    "prompt_id": "2a45c8422c20",
    "generation_config_id": "cbc4e415c848",
    "model": "gpt-5.4-mini-2026-03-17",
    "answer_schema_id": "e3ec802e2267",
    "response_contract_id": "debe65096147",
    "vocabulary_id": "50057f9259f6",
    "analyzer_id": "b447ce1aee09",
    "guardrail_policy_id": "3f202ebc205a",
    "tokenizer": "tiktoken-0.14.0/o200k_base",
    "corpus_version": "1.0.0",
}

# Failures that make every later request pointless: the live run stops.
SYSTEMIC_CATEGORIES: Final = frozenset(
    {"authentication", "permission_denied", "quota_exceeded", "model_unavailable", "configuration"}
)

Mode = Literal["dry_run", "live", "replay"]
OriginMode = Literal["dry_run", "live"]
COMPLETED = "completed"
FAILED = "failed"
NOT_RUN_BUDGET = "not_run_budget"
NOT_RUN_ABORTED = "not_run_aborted"
NOT_RUN_INTERRUPTED = "not_run_interrupted"
NOT_REPLAYABLE = "not_replayable"
OUTCOMES: Final = (
    COMPLETED,
    FAILED,
    NOT_RUN_BUDGET,
    NOT_RUN_ABORTED,
    NOT_RUN_INTERRUPTED,
    NOT_REPLAYABLE,
)
LOCAL_ABSTENTION = "local_abstention"
STATUSES: Final = (
    AnswerStatus.ANSWERED.value,
    AnswerStatus.PARTIALLY_ANSWERED.value,
    AnswerStatus.INSUFFICIENT_EVIDENCE.value,
    LOCAL_ABSTENTION,
)
ATTEMPTED = frozenset({AnswerStatus.ANSWERED.value, AnswerStatus.PARTIALLY_ANSWERED.value})
ABSTAINED = frozenset({AnswerStatus.INSUFFICIENT_EVIDENCE.value, LOCAL_ABSTENTION})

# Retries are operational, so this has the same config_id as the default configuration.
EVALUATION_GENERATION_CONFIG: Final = GenerationConfig(max_retries=0)


class EvaluationRunError(RuntimeError):
    """The run must not start or continue. ``problems`` says why; never quotes content."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class RequestBudgetExhausted(GenerationError):
    category = "request_budget_exhausted"


class ReplayMismatchError(GenerationError):
    category = "replay_mismatch"


# --------------------------------------------------------------------------- #
# Frozen pipeline and identities
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Pipeline:
    """The frozen components; the harness only composes them."""

    retriever: PolicyRetriever
    counter: TokenCounter
    resolver: CitationResolver
    vocabulary: CorpusVocabulary
    corpus_version: str
    context_config: ContextConfig = DEFAULT_CONTEXT_CONFIG
    generation_config: GenerationConfig = EVALUATION_GENERATION_CONFIG

    def service(self, provider: LLMProvider) -> GroundedGenerationService:
        return GroundedGenerationService(
            self.retriever,
            self.counter,
            provider,
            context_config=self.context_config,
            generation_config=self.generation_config,
        )

    def identities(self) -> dict[str, str]:
        retrieval = self.retriever.identities()
        return {
            "retrieval_policy_id": str(retrieval["policy_id"]),
            "collection_artifact_sha256": str(retrieval["collection_artifact_sha256"]),
            "context_config_id": self.context_config.config_id,
            "prompt_id": prompt_id(),
            "generation_config_id": self.generation_config.config_id,
            "model": self.generation_config.model,
            "answer_schema_id": answer_schema_id(),
            "response_contract_id": response_contract_id(),
            "vocabulary_id": self.vocabulary.vocabulary_id,
            "analyzer_id": analyzer_id(),
            "guardrail_policy_id": guardrail_policy_id(self.vocabulary),
            "tokenizer": self.counter.identity,
            "corpus_version": self.corpus_version,
        }


def verify_identities(actual: Mapping[str, str], expected: Mapping[str, str]) -> None:
    problems = [
        f"identity {key} is {actual.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]
    if set(actual) != set(expected):
        problems.append("identity keys differ from the frozen set")
    if problems:
        raise EvaluationRunError(problems)


def _check_run_identities(run: GenerationRunResult, preflight: Mapping[str, str]) -> None:
    ids = run.prepared.identities()
    observed = {
        "retrieval_policy_id": ids["retrieval"]["policy_id"],
        "context_config_id": ids["context_config_id"],
        "prompt_id": ids["prompt_id"],
        "generation_config_id": ids["generation_config_id"],
        "tokenizer": ids["tokenizer"],
    }
    if any(preflight[k] != v for k, v in observed.items()):
        raise EvaluationRunError(["a run used identities different from the pre-flight check"])


# --------------------------------------------------------------------------- #
# Providers: request budget and replay
# --------------------------------------------------------------------------- #


def check_max_requests(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationRunError(["--max-requests must be an integer"])
    if not 1 <= value <= HARD_REQUEST_CEILING:
        raise EvaluationRunError([f"--max-requests must be in [1, {HARD_REQUEST_CEILING}]"])
    return value


class BudgetedProvider:
    """Counts attempted provider requests and refuses, before sending, past the budget.

    An attempt counts even when it fails, so a failing provider cannot exceed the cap.
    """

    def __init__(self, inner: LLMProvider, max_requests: int) -> None:
        self._inner = inner
        self.max_requests = check_max_requests(max_requests)
        self.attempts = 0

    @property
    def identity(self) -> ProviderIdentity:
        return self._inner.identity

    def generate_text(self, request: GenerationRequest) -> GenerationResult[None]:
        raise GenerationError("the evaluation uses structured generation only")

    def generate_structured(
        self, request: GenerationRequest, schema: type[SchemaT]
    ) -> GenerationResult[SchemaT]:
        if self.attempts >= self.max_requests:
            raise RequestBudgetExhausted("request budget exhausted; nothing was sent")
        self.attempts += 1
        return self._inner.generate_structured(request, schema)

    def close(self) -> None:
        self._inner.close()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ReplayProvider:
    """Returns one stored answer, only for the exact model-facing request it was made for."""

    def __init__(self, record: Mapping[str, Any], config: GenerationConfig) -> None:
        self._record = record
        self._identity = ProviderIdentity("replay", "none", config.model, config.config_id)
        self.calls = 0

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def generate_text(self, request: GenerationRequest) -> GenerationResult[None]:
        raise ReplayMismatchError("replay supports structured generation only")

    def generate_structured(
        self, request: GenerationRequest, schema: type[SchemaT]
    ) -> GenerationResult[SchemaT]:
        self.calls += 1
        stored = self._record.get("request") or {}
        if self.calls > 1:
            raise ReplayMismatchError("replay answered more than one request")
        if _sha256(request.input_text) != stored.get("input_sha256") or _sha256(
            request.instructions
        ) != stored.get("instructions_sha256"):
            raise ReplayMismatchError("the rebuilt request differs from the recorded request")
        payload = json.dumps(self._record["answer"], sort_keys=True)
        return GenerationResult(
            text=payload,
            parsed=schema.model_validate_json(payload),
            status="completed",
            requested_model=self._identity.model,
            response_model="replay (no model call)",
            response_id="replay",
            request_id=None,
            usage=TokenUsage(),
            latency_ms=0.0,
            provider=self._identity,
        )

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Live authorization (checked before any provider is constructed)
# --------------------------------------------------------------------------- #


def live_authorization_problems(
    *,
    max_requests: int | None,
    environ: Mapping[str, str],
    git: Mapping[str, Any],
) -> list[str]:
    problems: list[str] = []
    if environ.get(PAID_ENV_VAR) != "1":
        problems.append(f"set {PAID_ENV_VAR}=1 to authorize paid requests")
    if max_requests is None:
        problems.append("--max-requests is required for --live (there is no default)")
    else:
        try:
            check_max_requests(max_requests)
        except EvaluationRunError as exc:
            problems.extend(exc.problems)
    if git.get("working_tree_clean") is not True or not git.get("commit"):
        problems.append("live evaluation requires a clean Git working tree at a known commit")
    if not environ.get(API_KEY_ENV_VAR, "").strip():
        problems.append(f"{API_KEY_ENV_VAR} is not configured")  # the value is never read out
    return problems


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def seal(record: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return {**body, "record_sha256": _sha256(_canonical(body))}


def is_sealed(record: Mapping[str, Any]) -> bool:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return record.get("record_sha256") == _sha256(_canonical(body))


@dataclass(frozen=True, slots=True)
class Origin:
    """Where the answers came from. A replay keeps its source run's origin."""

    mode: OriginMode
    run_id: str

    @property
    def synthetic(self) -> bool:
        return self.mode == "dry_run"


def _base_record(
    run_id: str,
    mode: Mode,
    question: EvalQuestion,
    identities: Mapping[str, str],
    origin: Origin | None = None,
) -> dict[str, Any]:
    source = origin or Origin("live" if mode == "live" else "dry_run", run_id)
    return {
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "origin_mode": source.mode,
        "origin_run_id": source.run_id,
        "synthetic": source.synthetic,
        "question_id": question.question_id,
        "category": question.category.value,
        "answerable": question.answerable,
        "outcome": None,
        "error_category": None,
        "origin": None,
        "local_abstention_reason": None,
        "status": None,
        "provider_attempted": False,
        "request": None,
        "answer": None,
        "selected_evidence": [],
        "claims": [],
        "citations": [],
        "guardrails": {"status": "not_run", "report": None},
        "usage": None,
        "provider": None,
        "timings": None,
        "local_token_estimate": None,
        "identities": dict(identities),
        "replay_of": None,
        "failure": None,
    }


def _run_fields(run: GenerationRunResult) -> dict[str, Any]:
    prepared = run.prepared
    result = run.provider_result
    return {
        "origin": run.origin.value,
        "local_abstention_reason": None
        if run.local_abstention_reason is None
        else run.local_abstention_reason.value,
        "status": LOCAL_ABSTENTION if run.origin is AnswerOrigin.LOCAL else run.answer.status.value,
        "request": {
            "input_sha256": _sha256(prepared.request.input_text),
            "instructions_sha256": _sha256(prepared.request.instructions),
        },
        "answer": run.answer.model_dump(mode="json"),
        "selected_evidence": [
            {
                k: r.provenance()[k]
                for k in ("evidence_id", "rank", "chunk_id", "document_id", "page_number")
            }
            for r in prepared.evidence
        ],
        "usage": None if result is None else result.usage.to_dict(),
        "provider": None
        if result is None
        else {**result.provider.to_dict(), "response_model": result.response_model},
        "timings": {
            "context_ms": round(run.context_ms, 3),
            "provider_ms": None if run.provider_ms is None else round(run.provider_ms, 3),
            "total_ms": round(run.total_ms, 3),
        },
        "local_token_estimate": prepared.budget.to_dict(),
    }


def _response_fields(response: GroundedResponse) -> dict[str, Any]:
    return {
        "claims": [
            {
                "position": c.position,
                "claim_type": c.claim_type.value,
                "evidence_ids": list(c.evidence_ids),
                "citation_numbers": list(c.citation_numbers),
            }
            for c in response.claims
        ],
        "citations": [
            {
                "number": c.number,
                "evidence_id": c.evidence_id,
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "page_number": c.page_number,
            }
            for c in response.citations
        ],
    }


class RecordSink:
    """Appends one sealed record per line and flushes it to disk immediately."""

    def __init__(self, handle: IO[str]) -> None:
        self._handle = handle
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        sealed = seal(record)
        self._handle.write(_canonical(sealed) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.records.append(sealed)


# --------------------------------------------------------------------------- #
# Run loop
# --------------------------------------------------------------------------- #

Answerer = Callable[[EvalQuestion], GenerationRunResult]
AttemptCounter = Callable[[], int]


def _no_attempts() -> int:
    return 0


@dataclass(slots=True)
class RunState:
    aborted: bool = False
    interrupted: bool = False
    systemic_category: str | None = None


STAGE_GENERATION = "generation"  # context preparation or the provider request
STAGE_ANSWER_VALIDATION = "answer_validation"  # local invariants on a parsed provider answer
STAGE_CITATION_RESOLUTION = "citation_resolution"
STAGE_REPLAY = "replay_verification"


def _failure_block(
    category: str,
    stage: str,
    *,
    problems: Sequence[Mapping[str, Any]] = (),
    request_id: str | None = None,
    status_code: int | None = None,
    usage_recorded: bool = False,
) -> dict[str, Any]:
    """Safe failure diagnostics: codes, positions and sanitized IDs only, never text."""
    return {
        "stage": stage,
        "error_category": category,
        "problems": [dict(p) for p in problems],
        "request_id": request_id,
        "status_code": status_code,
        "usage_recorded": usage_recorded,
    }


def _fail(record: dict[str, Any], error: GenerationError, stage: str) -> None:
    """Mark a failure. A rejected provider answer keeps its provider-reported usage."""
    record["outcome"] = FAILED
    record["error_category"] = error.category
    details: tuple[ContractProblem, ...] = ()
    if isinstance(error, ResponseContractError):
        details = error.details
        if error.usage is not None and record.get("usage") is None:
            record["usage"] = error.usage.to_dict()
    record["failure"] = _failure_block(
        error.category,
        stage,
        problems=[d.to_dict() for d in details],
        request_id=error.request_id,  # already sanitized by GenerationError (safe_token)
        status_code=error.status_code,
        usage_recorded=_has_usage(record),
    )


def evaluate_question(
    question: EvalQuestion,
    answer: Answerer,
    pipeline: Pipeline,
    preflight: Mapping[str, str],
    record: dict[str, Any],
    state: RunState,
    *,
    attempts: AttemptCounter = _no_attempts,
    expected_citations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """One question: one generation, one resolution, one guardrail check."""
    before = attempts()
    try:
        run = answer(question)
    except RequestBudgetExhausted:
        record["outcome"] = NOT_RUN_BUDGET
        return record
    except GenerationError as error:
        stage = (
            STAGE_ANSWER_VALIDATION
            if isinstance(error, ResponseContractError)
            else STAGE_GENERATION
        )
        _fail(record, error, stage)
        record["provider_attempted"] = attempts() > before
        if error.category in SYSTEMIC_CATEGORIES:
            state.aborted = True
            state.systemic_category = error.category
        return record
    record["provider_attempted"] = attempts() > before
    _check_run_identities(run, preflight)
    record.update(_run_fields(run))
    try:
        response = pipeline.resolver.resolve(run)
    except GenerationError as error:
        _fail(record, error, STAGE_CITATION_RESOLUTION)
        return record
    record.update(_response_fields(response))
    if expected_citations is not None and record["citations"] != list(expected_citations):
        record["outcome"] = FAILED
        record["error_category"] = ReplayMismatchError.category
        record["failure"] = _failure_block(ReplayMismatchError.category, STAGE_REPLAY)
        return record
    record["outcome"] = COMPLETED
    try:
        report = check_response(response, pipeline.vocabulary)
    except GuardrailError:
        record["guardrails"] = {"status": "failed", "report": None}
    else:
        record["guardrails"] = {"status": "completed", "report": report.metadata()}
    return record


def run_questions(
    questions: Sequence[EvalQuestion],
    answer: Answerer,
    pipeline: Pipeline,
    preflight: Mapping[str, str],
    sink: RecordSink,
    *,
    run_id: str,
    mode: Mode,
    attempts: AttemptCounter = _no_attempts,
    expected: Mapping[str, Mapping[str, Any]] | None = None,
    origin: Origin | None = None,
) -> RunState:
    state = RunState()
    pending = list(questions)
    try:
        while pending:
            question = pending[0]
            record = _base_record(run_id, mode, question, preflight, origin)
            source = None if expected is None else expected.get(question.question_id)
            if state.aborted:
                record["outcome"] = NOT_RUN_ABORTED
            elif expected is not None and (source is None or source.get("outcome") != COMPLETED):
                record["outcome"] = NOT_REPLAYABLE
                record["replay_of"] = _replay_origin(source)
            else:
                record = evaluate_question(
                    question,
                    answer,
                    pipeline,
                    preflight,
                    record,
                    state,
                    attempts=attempts,
                    expected_citations=None if source is None else source["citations"],
                )
                if source is not None:
                    record["replay_of"] = _replay_origin(source)
                    # Replay creates no provider-usage or latency observation.
                    record.update(usage=None, timings=None, provider=None)
                    if record.get("failure"):
                        record["failure"]["usage_recorded"] = False
            sink.write(record)
            pending.pop(0)
    except KeyboardInterrupt:
        state.interrupted = True
        for question in pending:
            record = _base_record(run_id, mode, question, preflight, origin)
            record["outcome"] = NOT_RUN_INTERRUPTED
            sink.write(record)
    return state


def _replay_origin(source: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if source is None:
        return None
    version = source.get("record_schema_version")
    failed = source.get("outcome") == FAILED
    return {
        "run_id": source.get("run_id"),
        "mode": source.get("mode"),
        "record_schema_version": version,
        "outcome": source.get("outcome"),
        "error_category": source.get("error_category"),
        "provider_attempted": source.get("provider_attempted"),
        "usage": source.get("usage"),
        # Schema 1 never recorded failure diagnostics; they are reported as missing, not invented.
        "failure": source.get("failure") if version != 1 else None,
        "failure_diagnostics": None
        if not failed
        else "recorded"
        if version != 1
        else "not_recorded_in_record_schema_1",
        "record_sha256": source.get("record_sha256"),
    }


# --------------------------------------------------------------------------- #
# Aggregation (deterministic from records)
# --------------------------------------------------------------------------- #

PageId = tuple[str, int]


def _pages(items: Iterable[Mapping[str, Any]]) -> frozenset[PageId]:
    return frozenset((str(i["document_id"]), int(i["page_number"])) for i in items)


def _ratio(
    numerator: int, denominator: int, population: str, *, counts_only: bool, values: Sequence[int]
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "population": population,
        "numerator": numerator,
        "denominator": denominator,
    }
    if denominator == 0:
        out["rate"] = NOT_APPLICABLE
    elif counts_only:
        out["rate"] = None
        out["note"] = f"counts only: category has fewer than {SMALL_CATEGORY_N} questions"
    else:
        out["rate"] = round(numerator / denominator, 4)
        interval = m.bootstrap_interval(
            [float(v) for v in values], resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
        )
        out["ci95"] = None if interval is None else [round(x, 4) for x in interval]
        if denominator < 10:
            out["small_sample_warning"] = True
    return out


def _flag(values: Sequence[int], population: str, counts_only: bool) -> dict[str, Any]:
    return _ratio(sum(values), len(values), population, counts_only=counts_only, values=values)


def _mean(values: Sequence[float], population: str, counts_only: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"population": population, "n": len(values)}
    if not values:
        out["mean"] = NOT_APPLICABLE
    elif counts_only:
        out["mean"] = None
        out["sum"] = round(math.fsum(values), 4)
        out["note"] = f"counts only: category has fewer than {SMALL_CATEGORY_N} questions"
    else:
        out["mean"] = round(math.fsum(values) / len(values), 4)
        interval = m.bootstrap_interval(
            list(values), resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
        )
        out["ci95"] = None if interval is None else [round(x, 4) for x in interval]
        if len(values) < 10:
            out["small_sample_warning"] = True
    return out


def _has_usage(record: Mapping[str, Any]) -> bool:
    usage = record.get("usage")
    return isinstance(usage, dict) and any(usage.get(k) is not None for k in USAGE_KEYS)


def failure_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Failed questions by stage and invariant code; codes and IDs only."""
    failed = [r for r in records if r["outcome"] == FAILED]
    stages: dict[str, int] = {}
    codes: dict[str, int] = {}
    for r in failed:
        block = r.get("failure")
        if not block:
            continue
        stages[block["stage"]] = stages.get(block["stage"], 0) + 1
        for problem in block["problems"]:
            codes[problem["code"]] = codes.get(problem["code"], 0) + 1
    return {
        "failed": len(failed),
        "with_diagnostics": sum(1 for r in failed if r.get("failure")),
        "by_stage": dict(sorted(stages.items())),
        "problem_codes": dict(sorted(codes.items())),
        "questions": [
            {
                "question_id": r["question_id"],
                "category": r["category"],
                "error_category": r["error_category"],
                "stage": (r.get("failure") or {}).get("stage"),
                "problem_codes": [p["code"] for p in (r.get("failure") or {}).get("problems", [])],
                "usage_recorded": _has_usage(r),
            }
            for r in sorted(failed, key=lambda r: r["question_id"])
        ],
    }


def provider_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Provider-reported usage over every attempted request, failures included.

    Totals cover only requests with recorded usage. When some attempted request has
    none (unknown, never zero), the totals understate spend and say so.
    """
    attempted = [r for r in records if r.get("provider_attempted")]
    with_usage = sum(1 for r in attempted if _has_usage(r))
    complete = bool(attempted) and with_usage == len(attempted)
    return {
        "population": "questions with an attempted provider request (failures included)",
        "attempted_requests": len(attempted),
        "requests_with_usage": with_usage,
        "failed_requests_with_usage": sum(
            1 for r in attempted if r["outcome"] == FAILED and _has_usage(r)
        ),
        "usage_complete": complete,
        "note": None
        if complete or not attempted
        else "usage unknown for some attempted requests: totals understate spend",
        **{key: _sum_usage(attempted, key) for key in USAGE_KEYS},
    }


def _sum_usage(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    reported = [
        r["usage"][key] for r in records if r.get("usage") and r["usage"].get(key) is not None
    ]
    return {"reported_n": len(reported), "total": sum(reported) if reported else None}


def category_metrics(
    records: Sequence[Mapping[str, Any]],
    questions: Mapping[str, EvalQuestion],
    *,
    counts_only: bool,
) -> dict[str, Any]:
    outcomes = {o: sum(1 for r in records if r["outcome"] == o) for o in OUTCOMES}
    errors: dict[str, int] = {}
    for r in records:
        if r["error_category"]:
            errors[r["error_category"]] = errors.get(r["error_category"], 0) + 1
    completed = [r for r in records if r["outcome"] == COMPLETED]
    answerable = [r for r in completed if r["answerable"]]
    unanswerable = [r for r in completed if not r["answerable"]]
    attempted = [r for r in answerable if r["status"] in ATTEMPTED]

    def flags(rows: Sequence[Mapping[str, Any]], statuses: frozenset[str]) -> list[int]:
        return [int(r["status"] in statuses) for r in rows]

    selected_recall: list[int] = []
    selected_cov: list[float] = []
    selected_complete: list[int] = []
    cited_recall: list[int] = []
    cited_cov: list[float] = []
    cited_complete: list[int] = []
    overlap_num = overlap_den = 0
    attempted_ids = {r["question_id"] for r in attempted}
    for r in answerable:
        gold = gold_evidence(questions[r["question_id"]])
        if gold is None:  # pragma: no cover - answerable questions always have gold
            continue
        selected = _pages(r["selected_evidence"])
        facts = m.facts_covered_by_pages(selected, gold)
        selected_recall.append(int(bool(selected & gold.pages)))
        selected_cov.append(len(facts) / len(gold.facts))
        if len(gold.facts) >= 2:
            selected_complete.append(int(len(facts) == len(gold.facts)))
        if r["question_id"] in attempted_ids:
            cited = _pages(r["citations"])
            cited_facts = m.facts_covered_by_pages(cited, gold)
            cited_recall.append(int(bool(cited & gold.pages)))
            cited_cov.append(len(cited_facts) / len(gold.facts))
            if len(gold.facts) >= 2:
                cited_complete.append(int(len(cited_facts) == len(gold.facts)))
            overlap_num += len(cited & gold.pages)
            overlap_den += len(cited)

    citations = [c for r in completed for c in r["citations"]]
    violations = 0
    for r in completed:
        selected_ids = {e["evidence_id"]: e for e in r["selected_evidence"]}
        for c in r["citations"]:
            e = selected_ids.get(c["evidence_id"])
            if e is None or (e["chunk_id"], e["document_id"], e["page_number"]) != (
                c["chunk_id"],
                c["document_id"],
                c["page_number"],
            ):
                violations += 1

    guarded = [r for r in completed if r["guardrails"]["status"] == "completed"]
    findings: dict[str, dict[str, int]] = {}
    numeric: dict[str, int] = {}
    with_warning = 0
    for r in guarded:
        report = r["guardrails"]["report"]
        warned = False
        for f in report["findings"]:
            findings.setdefault(f["category"], {}).setdefault(f["severity"], 0)
            findings[f["category"]][f["severity"]] += 1
            warned = warned or f["severity"] == "warning"
        with_warning += int(warned)
        for key, value in report["diagnostics"].items():
            numeric[key] = numeric.get(key, 0) + int(value)

    provider_called = [r for r in completed if r["origin"] == AnswerOrigin.MODEL.value]
    estimates = [
        int(r["local_token_estimate"]["local_total_tokens"])
        for r in provider_called
        if r.get("local_token_estimate")
    ]
    return {
        "questions": len(records),
        "execution_outcomes": outcomes,
        "error_categories": dict(sorted(errors.items())),
        "status_distribution": {
            "population": "completed questions",
            "denominator": len(completed),
            "counts": {s: sum(1 for r in completed if r["status"] == s) for s in STATUSES},
        },
        "answer_attempt_rate": _flag(
            flags(answerable, ATTEMPTED), "completed answerable questions", counts_only
        ),
        "abstention_on_answerable_rate": _flag(
            flags(answerable, ABSTAINED), "completed answerable questions", counts_only
        ),
        "abstention_on_unanswerable_rate": _flag(
            flags(unanswerable, ABSTAINED), "completed unanswerable questions", counts_only
        ),
        "answered_on_unanswerable_rate": _flag(
            flags(unanswerable, frozenset({AnswerStatus.ANSWERED.value})),
            "completed unanswerable questions",
            counts_only,
        ),
        "partially_answered_on_unanswerable_rate": _flag(
            flags(unanswerable, frozenset({AnswerStatus.PARTIALLY_ANSWERED.value})),
            "completed unanswerable questions",
            counts_only,
        ),
        "selected_evidence_page_recall": _flag(
            selected_recall, "completed answerable questions", counts_only
        ),
        "selected_evidence_fact_coverage": _mean(
            selected_cov, "completed answerable questions", counts_only
        ),
        "selected_evidence_complete_fact_coverage": _flag(
            selected_complete,
            "completed answerable questions with >= 2 required facts",
            counts_only,
        ),
        "cited_evidence_page_recall": _flag(
            cited_recall, "completed answerable questions with an answer attempt", counts_only
        ),
        "cited_evidence_fact_coverage": _mean(
            cited_cov, "completed answerable questions with an answer attempt", counts_only
        ),
        "cited_evidence_complete_fact_coverage": _flag(
            cited_complete,
            "completed answerable questions with an answer attempt and >= 2 required facts",
            counts_only,
        ),
        "accepted_page_citation_overlap": _ratio(
            overlap_num,
            overlap_den,
            "distinct cited document pages in answerable answer attempts",
            counts_only=counts_only,
            values=[],
        )
        | {"interval": "not computed (pages are not independent questions)"},
        "citation_structural_invariants": {
            "citations": len(citations),
            "distinct_cited_pages": sum(len(_pages(r["citations"])) for r in completed),
            "violations": violations,
            "definition": "each citation's chunk, document and page equal its selected evidence",
        },
        "guardrails": {
            "completed": len(guarded),
            "failed": sum(1 for r in completed if r["guardrails"]["status"] == "failed"),
            "findings": {k: dict(sorted(v.items())) for k, v in sorted(findings.items())},
            "questions_with_warning": _ratio(
                with_warning,
                len(guarded),
                "completed questions with completed guardrail checks",
                counts_only=counts_only,
                values=[
                    int(
                        any(
                            f["severity"] == "warning"
                            for f in r["guardrails"]["report"]["findings"]
                        )
                    )
                    for r in guarded
                ],
            ),
            "numeric_trace": dict(sorted(numeric.items())),
        },
        "provider_usage": provider_usage(records),
        "local_token_estimates": {
            "note": "local tokenizer estimate of instructions + input; not provider usage",
            "n": len(estimates),
            "total": sum(estimates) if estimates else None,
        },
    }


def _latency(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [
        float(r["timings"][key])
        for r in records
        if r.get("timings") and r["timings"].get(key) is not None
    ]
    p50, p95 = m.percentile(values, 50), m.percentile(values, 95)
    return {
        "n": len(values),
        "p50_ms": None if p50 is None else round(p50, 3),
        "p95_ms": None if p95 is None else round(p95, 3),
    }


@dataclass(frozen=True, slots=True)
class Price:
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal
    effective_date: date

    def estimate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows = [
            r["usage"]
            for r in records
            if r.get("usage")
            and r["usage"].get("input_tokens") is not None
            and r["usage"].get("output_tokens") is not None
        ]
        cost = sum(
            (
                Decimal(u["input_tokens"]) * self.input_usd_per_mtok
                + Decimal(u["output_tokens"]) * self.output_usd_per_mtok
            )
            / Decimal(1_000_000)
            for u in rows
        )
        accounting = provider_usage(records)
        return {
            "note": "estimate from provider-reported usage and the user-supplied price",
            "attempted_requests": accounting["attempted_requests"],
            "usage_complete": accounting["usage_complete"],
            "price_effective_date": self.effective_date.isoformat(),
            "input_usd_per_mtok": str(self.input_usd_per_mtok),
            "output_usd_per_mtok": str(self.output_usd_per_mtok),
            "requests_with_usage": len(rows),
            "estimated_usd": str(round(Decimal(cost), 6)) if rows else None,
        }


def aggregate(
    records: Sequence[Mapping[str, Any]],
    split: DevSplit,
    *,
    run_id: str,
    mode: Mode,
    state: RunState,
    max_requests: int | None,
    attempted_requests: int,
    price: Price | None = None,
    replay_of: str | None = None,
) -> dict[str, Any]:
    """Deterministic from the records and split metadata; ``operational`` is added later."""
    questions = {q.question_id: q for q in split.questions}
    ids = [r["question_id"] for r in records]
    problems: list[str] = []
    if sorted(ids) != sorted(questions) or len(set(ids)) != len(ids):
        problems.append("records do not cover each development question exactly once")
    not_completed = [r for r in records if r["outcome"] != COMPLETED]
    if not_completed:
        problems.append(f"{len(not_completed)} question(s) not completed")
    if any(r["guardrails"]["status"] == "failed" for r in records):
        problems.append("guardrail diagnostics failed for at least one question")
    if state.interrupted:
        problems.append("the run was interrupted")
    if state.aborted:
        problems.append(f"the run was aborted after a {state.systemic_category} failure")
    dataset_counts = {c.value: sum(1 for q in split.questions if q.category is c) for c in Category}
    by_category = {
        c.value: category_metrics(
            [r for r in records if r["category"] == c.value],
            questions,
            counts_only=dataset_counts[c.value] < SMALL_CATEGORY_N,
        )
        for c in Category
    }
    identities = records[0]["identities"] if records else {}
    origins = {(r.get("origin_mode"), r.get("origin_run_id")) for r in records}
    if len(origins) != 1:
        problems.append("records do not share one origin run")
    origin_mode, origin_run_id = next(iter(origins)) if len(origins) == 1 else (None, None)
    notes = [SYNTHETIC_NOTE] if origin_mode != "live" else []
    if mode == "replay":
        notes.append(REPLAY_NOTE)
    report: dict[str, Any] = {
        "report_type": REPORT_TYPE,
        "evaluation_kind": EVALUATION_KIND,
        "protocol_version": PROTOCOL_VERSION,
        "caveat": CAVEAT,
        "mode": mode,
        "origin_mode": origin_mode,
        "origin_run_id": origin_run_id,
        "synthetic": origin_mode != "live",
        "provenance_notes": notes,
        "reportability": _reportability(origin_mode, problems),
        "run_id": run_id,
        "replay_of": replay_of,
        "split": DEV_SPLIT,
        "dataset": {
            "dataset_version": split.dataset_version,
            "dataset_sha256": split.dataset_sha256,
            "split_sha256": split.split_sha256,
            "question_count": len(split.questions),
            "category_counts": dataset_counts,
        },
        "identities": identities,
        "request_budget": {
            "max_requests": max_requests,
            "hard_ceiling": HARD_REQUEST_CEILING,
            "attempted_requests": attempted_requests,
            "counts": "attempted provider requests, including failed attempts",
        },
        "run_complete": not problems,
        "completion_problems": problems,
        "overall": category_metrics(records, questions, counts_only=False),
        "by_category": by_category,
        "questions": [
            {
                "question_id": r["question_id"],
                "category": r["category"],
                "outcome": r["outcome"],
                "error_category": r["error_category"],
                "status": r["status"],
                "citations": len(r["citations"]),
                "distinct_cited_pages": len(_pages(r["citations"])),
                "guardrail_status": r["guardrails"]["status"],
                "guardrail_warnings": sum(
                    1
                    for f in (r["guardrails"]["report"] or {}).get("findings", [])
                    if f["severity"] == "warning"
                ),
            }
            for r in sorted(records, key=lambda r: r["question_id"])
        ],
        "cost_estimate": None if price is None else price.estimate(records),
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "failure_diagnostics": failure_diagnostics(records),
    }
    if mode == "replay":
        originals = [r["replay_of"] for r in records if r.get("replay_of")]
        report["original_run_usage"] = {
            **provider_usage(originals),
            "note": "provider usage recorded by the original run; replay sent no request",
        }
    return report


def _reportability(origin_mode: object, problems: Sequence[str]) -> dict[str, Any]:
    """Decided by the origin run's provenance, never by the current execution mode."""
    if origin_mode != "live":
        reason = "synthetic origin: never evidence of hosted-model performance"
    elif problems:
        reason = "incomplete run: not a completed 24-question diagnostic result"
    else:
        reason = "complete live-origin run: reportable as a Stage 5 diagnostic (protocol §13)"
    return {
        "reportable_as_hosted_model_diagnostic": origin_mode == "live" and not problems,
        "reason": reason,
    }


def operational(records: Sequence[Mapping[str, Any]], started_at: str) -> dict[str, Any]:
    completed = [r for r in records if r["outcome"] == COMPLETED]
    return {
        "note": "Varies between runs and machines; excluded from deterministic comparisons.",
        "started_at_utc": started_at,
        "latency": {k: _latency(completed, k) for k in ("context_ms", "provider_ms", "total_ms")},
        "git": _git_state(),
        "sdk_version": sdk_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def deterministic_view(report: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k != "operational"}


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def _fmt_rate(metric: Mapping[str, Any]) -> str:
    num = metric.get("numerator", metric.get("sum"))
    den = metric.get("denominator", metric.get("n"))
    value = metric.get("rate", metric.get("mean"))
    if value == NOT_APPLICABLE:
        return "n/a (0 eligible)"
    if value is None:
        return f"{num} / {den} (counts only)"
    ci = metric.get("ci95")
    ci_text = "" if not ci else f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    count = f"{num}/{den}" if "numerator" in metric else f"n={den}"
    return f"{value:.3f} ({count}){ci_text}"


ROWS: Final = (
    "answer_attempt_rate",
    "abstention_on_answerable_rate",
    "abstention_on_unanswerable_rate",
    "answered_on_unanswerable_rate",
    "partially_answered_on_unanswerable_rate",
    "selected_evidence_page_recall",
    "selected_evidence_fact_coverage",
    "selected_evidence_complete_fact_coverage",
    "cited_evidence_page_recall",
    "cited_evidence_fact_coverage",
    "cited_evidence_complete_fact_coverage",
    "accepted_page_citation_overlap",
)


def _error_suffix(category: object) -> str:
    return f" ({category})" if category else ""


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Generation diagnostic evaluation — {report['run_id']}",
        "",
        f"`evaluation_kind: {report['evaluation_kind']}` · protocol v{report['protocol_version']}"
        f" · mode `{report['mode']}` · split `{report['split']}`",
        "",
        f"> {report['caveat']}",
    ]
    lines += ["", f"Origin: `{report['origin_mode']}` run `{report['origin_run_id']}`."]
    lines += [f"> **{note}**" for note in report["provenance_notes"]]
    lines += ["", f"Reportability: {report['reportability']['reason']}."]
    if not report["run_complete"]:
        lines += ["", "> **INCOMPLETE RUN: not a completed 24-question diagnostic result.**"]
    lines += [
        "",
        f"Run complete: **{report['run_complete']}**"
        + (
            f" ({'; '.join(report['completion_problems'])})"
            if report["completion_problems"]
            else ""
        ),
        "",
        "| Metric | Overall | " + " | ".join(report["by_category"]) + " |",
        "|---|---|" + "---|" * len(report["by_category"]),
    ]
    for name in ROWS:
        cells = [_fmt_rate(report["overall"][name])]
        cells += [_fmt_rate(report["by_category"][c][name]) for c in report["by_category"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    overall = report["overall"]
    lines += [
        "",
        "## Execution",
        "",
        "| Outcome | Count |",
        "|---|---|",
        *(f"| {k} | {v} |" for k, v in overall["execution_outcomes"].items()),
        "",
        f"Status distribution over {overall['status_distribution']['denominator']} completed: "
        + ", ".join(f"{k} {v}" for k, v in overall["status_distribution"]["counts"].items()),
        "",
        f"Error categories: {overall['error_categories'] or 'none'}",
        "",
        "Failure diagnostics (codes only): "
        + (
            "; ".join(
                f"{q['question_id']} {q['stage']} {','.join(q['problem_codes']) or '-'}"
                for q in report["failure_diagnostics"]["questions"]
            )
            or "none"
        ),
        "",
        f"Citation invariants: {overall['citation_structural_invariants']['citations']} citations, "
        f"{overall['citation_structural_invariants']['violations']} violations.",
        "",
        "## Guardrail findings",
        "",
        f"Findings: {overall['guardrails']['findings'] or 'none'}",
        f"Questions with a warning: {_fmt_rate(overall['guardrails']['questions_with_warning'])}",
        f"Numeric trace: {overall['guardrails']['numeric_trace']}",
        "",
        "## Usage",
        "",
        f"Attempted provider requests: {overall['provider_usage']['attempted_requests']} "
        f"(cap {report['request_budget']['max_requests']}); with recorded usage: "
        f"{overall['provider_usage']['requests_with_usage']}"
        + ("" if overall["provider_usage"]["usage_complete"] else " (totals understate spend)"),
        *(
            f"- {k}: {v['total']} (reported for {v['reported_n']})"
            for k, v in overall["provider_usage"].items()
            if isinstance(v, dict)
        ),
        f"- local token estimate (not provider usage): {overall['local_token_estimates']['total']}",
        "",
        "## Questions",
        "",
        "| ID | Category | Outcome | Status | Citations | Pages | Guardrails | Warnings |",
        "|---|---|---|---|---|---|---|---|",
        *(
            f"| {q['question_id']} | {q['category']} | {q['outcome']}"
            f"{_error_suffix(q['error_category'])} | {q['status']} |"
            f" {q['citations']} | {q['distinct_cited_pages']} | {q['guardrail_status']} |"
            f" {q['guardrail_warnings']} |"
            for q in report["questions"]
        ),
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def new_run_id(mode: Mode, identities: Mapping[str, str], now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{mode}-{stamp}-{_sha256(_canonical(dict(identities)))[:8]}"


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationRunError([f"cannot read replay records: {type(exc).__name__}"]) from None
    return records


def check_replay_records(
    records: Sequence[Mapping[str, Any]], identities: Mapping[str, str], split: DevSplit
) -> dict[str, Mapping[str, Any]]:
    problems: list[str] = []
    known = {q.question_id: q for q in split.questions}
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        where = f"record {index}"
        if record.get("record_schema_version") not in REPLAYABLE_RECORD_SCHEMA_VERSIONS:
            problems.append(f"{where}: unsupported record schema")
            continue
        if not is_sealed(record):
            problems.append(f"{where}: record digest mismatch (modified or truncated)")
            continue
        if record.get("mode") not in ("dry_run", "live"):
            problems.append(f"{where}: only dry-run or live records can be replayed")
        elif (record.get("origin_mode"), record.get("origin_run_id")) != (
            record.get("mode"),
            record.get("run_id"),
        ):
            problems.append(f"{where}: origin provenance does not match its run")
        if record.get("synthetic") is not (record.get("mode") == "dry_run"):
            problems.append(f"{where}: synthetic flag does not match its origin")
        if record.get("identities") != dict(identities):
            problems.append(f"{where}: frozen identities differ from the current system")
        qid = record.get("question_id")
        question = known.get(str(qid))
        if question is None:
            problems.append(f"{where}: not a development question ID")
        elif record.get("category") != question.category.value:
            problems.append(f"{where}: category differs from the frozen split")
        if qid in by_id:
            problems.append(f"{where}: duplicate question ID")
        by_id[str(qid)] = record
    if len({r.get("run_id") for r in records}) > 1:
        problems.append("records come from more than one run; runs are never merged")
    if len({r.get("record_schema_version") for r in records}) > 1:
        problems.append("records mix record schema versions")
    if problems:
        raise EvaluationRunError(problems)
    return by_id


@dataclass(frozen=True, slots=True)
class RunOutputs:
    report: dict[str, Any]
    records_path: Path
    json_path: Path
    markdown_path: Path
    interrupted: bool


def execute(
    split: DevSplit,
    pipeline: Pipeline,
    reports_dir: Path,
    *,
    mode: Mode,
    provider_factory: Callable[[], LLMProvider],
    max_requests: int | None = None,
    expected_identities: Mapping[str, str] = EXPECTED_IDENTITIES,
    replay_records: Sequence[Mapping[str, Any]] | None = None,
    price: Price | None = None,
    now: datetime | None = None,
) -> RunOutputs:
    """Verify identities, then (and only then) build the provider and run every question."""
    preflight = pipeline.identities()
    verify_identities(preflight, expected_identities)
    expected = None
    replay_of = None
    origin: Origin | None = None
    if mode == "replay":
        if not replay_records:
            raise EvaluationRunError(["replay needs records"])
        expected = check_replay_records(replay_records, preflight, split)
        first = replay_records[0]
        replay_of = str(first["run_id"])
        origin = Origin("live" if first["mode"] == "live" else "dry_run", replay_of)
    run_id = new_run_id(mode, preflight, now)
    started_at = (now or datetime.now(UTC)).isoformat()
    directory = reports_dir / REPORT_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    records_path = directory / f"{run_id}.records.jsonl"
    budget: BudgetedProvider | None = None
    answer: Answerer
    if expected is not None:
        replayed = expected

        def answer(question: EvalQuestion) -> GenerationRunResult:
            provider = ReplayProvider(replayed[question.question_id], pipeline.generation_config)
            return pipeline.service(provider).answer(question.question)

        def attempts() -> int:
            return 0  # replay never sends a provider request

    else:
        active = BudgetedProvider(provider_factory(), max_requests or HARD_REQUEST_CEILING)
        budget = active
        service = pipeline.service(active)

        def answer(question: EvalQuestion) -> GenerationRunResult:
            return service.answer(question.question)

        def attempts() -> int:
            return active.attempts

    try:
        with records_path.open("x", encoding="utf-8", newline="\n") as handle:
            sink = RecordSink(handle)
            state = run_questions(
                split.questions,
                answer,
                pipeline,
                preflight,
                sink,
                run_id=run_id,
                mode=mode,
                attempts=attempts,
                expected=expected,
                origin=origin,
            )
    finally:
        if budget is not None:
            budget.close()
    report = aggregate(
        sink.records,
        split,
        run_id=run_id,
        mode=mode,
        state=state,
        max_requests=max_requests,
        attempted_requests=0 if budget is None else budget.attempts,
        price=price,
        replay_of=replay_of,
    )
    report["operational"] = operational(sink.records, started_at)
    json_path = directory / f"{run_id}.json"
    markdown_path = directory / f"{run_id}.md"
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with markdown_path.open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    return RunOutputs(report, records_path, json_path, markdown_path, state.interrupted)


def load_pipeline(settings: Settings) -> Pipeline:
    from founderlens.generation.token_budget import load_generation_tokenizer
    from founderlens.retrieval.policy import load_policy_retriever

    manifest = load_manifest(settings.corpus_manifest_path)
    retriever = load_policy_retriever(settings)
    return Pipeline(
        retriever=retriever,
        counter=load_generation_tokenizer(settings),
        resolver=CitationResolver(retriever.collection, manifest),
        vocabulary=CorpusVocabulary.from_manifest(manifest),
        corpus_version=manifest.corpus_version,
    )


def _parse_price(args: argparse.Namespace) -> Price | None:
    given = [args.price_input_per_mtok, args.price_output_per_mtok, args.price_date]
    if all(v is None for v in given):
        return None
    if any(v is None for v in given):
        raise EvaluationRunError(
            ["give all of --price-input-per-mtok, --price-output-per-mtok, --price-date"]
        )
    try:
        return Price(
            Decimal(args.price_input_per_mtok),
            Decimal(args.price_output_per_mtok),
            date.fromisoformat(args.price_date),
        )
    except (InvalidOperation, ValueError):
        raise EvaluationRunError(["invalid price or price date"]) from None


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.evaluation.generation_eval",
        description="Diagnostic (deterministic) generation evaluation on the development split.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="paid run; see protocol §13")
    group.add_argument("--replay", type=Path, default=None, help="records JSONL to replay offline")
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--split", default=DEV_SPLIT, help="Only 'dev' is allowed.")
    parser.add_argument("--price-input-per-mtok", default=None)
    parser.add_argument("--price-output-per-mtok", default=None)
    parser.add_argument(
        "--price-date", default=None, help="effective date of the price, YYYY-MM-DD"
    )
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    env = os.environ if environ is None else environ
    if args.split != DEV_SPLIT:
        parser.error(
            f"only the development split may be evaluated, got {args.split!r}; "
            "held-out evaluation is not authorized"
        )
    mode: Mode = "live" if args.live else "replay" if args.replay is not None else "dry_run"
    try:
        price = _parse_price(args)
        if args.max_requests is not None:
            check_max_requests(args.max_requests)
        settings = load_settings(args.project_root)
        if mode == "live":  # authorization first: nothing is loaded or built without it
            problems = live_authorization_problems(
                max_requests=args.max_requests, environ=env, git=_git_state(settings.project_root)
            )
            if problems:
                raise EvaluationRunError(problems)
        split = load_dev_split(settings.evaluation_dir, settings.corpus_manifest_path)
        pipeline = load_pipeline(settings)
        verify_identities(pipeline.identities(), EXPECTED_IDENTITIES)
        factory = _provider_factory(mode, env)
        replay = None if args.replay is None else load_records(args.replay)
        outputs = execute(
            split,
            pipeline,
            settings.reports_dir,
            mode=mode,
            provider_factory=factory,
            max_requests=args.max_requests,
            expected_identities=EXPECTED_IDENTITIES,
            replay_records=replay,
            price=price,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (EvaluationRunError, EvaluationError, ManifestError, GenerationError) as exc:
        print(f"Generation evaluation refused or FAILED: {exc}", file=sys.stderr)
        return 1
    report = outputs.report
    print(f"evaluation_kind {report['evaluation_kind']}; mode {report['mode']}")
    print(f"run {report['run_id']}")
    print(f"origin {report['origin_mode']} run {report['origin_run_id']}")
    for note in report["provenance_notes"]:
        print(note)
    print(f"reportability: {report['reportability']['reason']}")
    outcomes = report["overall"]["execution_outcomes"]
    print("outcomes: " + ", ".join(f"{k} {v}" for k, v in outcomes.items() if v))
    print(f"attempted provider requests {report['request_budget']['attempted_requests']}")
    print(f"run complete: {report['run_complete']} {report['completion_problems'] or ''}".rstrip())
    print(f"records  {outputs.records_path}")
    print(f"report   {outputs.json_path}")
    print(f"summary  {outputs.markdown_path}")
    return 130 if outputs.interrupted else 0


def _provider_factory(mode: Mode, environ: Mapping[str, str]) -> Callable[[], LLMProvider]:
    if mode == "live":

        def live() -> LLMProvider:
            from founderlens.generation.openai_provider import OpenAIProvider

            return OpenAIProvider(EVALUATION_GENERATION_CONFIG, environ=environ)

        return live

    def synthetic() -> LLMProvider:
        from founderlens.generation.demo import SyntheticDemoProvider

        return SyntheticDemoProvider(AnswerStatus.ANSWERED, EVALUATION_GENERATION_CONFIG)

    return synthetic


if __name__ == "__main__":
    raise SystemExit(main())
