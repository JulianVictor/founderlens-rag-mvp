"""Deterministic evidence context for grounded generation (Commit 5.2).

Usage::

    python -m founderlens.generation.context --smoke   # writes reports/context_smoke_report.json

This module builds generation *input*; it never calls a provider.

1. **Question validation** (:func:`validate_question`). The original text is
   kept unchanged. It is rejected if it is not a string, is blank, has no
   lexical token under the retrieval tokenizer (punctuation-only queries
   retrieve nothing), exceeds 8,192 characters (a coarse guard applied before
   tokenizing) or exceeds 512 local tokens. It is never truncated or
   rewritten.
2. **Candidate validation** (:func:`validate_candidates`). At most ten
   :class:`~founderlens.retrieval.schema.RetrievalResult` from the frozen
   policy are accepted, with:
   - the policy's method;
   - one-based contiguous ranks;
   - unique chunk IDs;
   - finite, non-increasing scores;
   - every chunk exactly equal (text, offsets, provenance) to the record with
     that ID in the verified V2 collection.

   Anything else is a :class:`CandidateValidationError`, never silently
   repaired. Distinct chunks from one page are separate, valid candidates.
3. **Selection**: a rank-ordered prefix of whole chunks. Each next candidate is
   added only if the trusted instructions plus the *re-serialized* input still
   fit ``input_token_limit`` (6,144 local tokens). The first candidate that does
   not fit is excluded as ``exceeds_input_budget``. Every later one is excluded
   as ``after_budget_cutoff``: nothing is skipped over, truncated, merged,
   reordered, rewritten or thresholded. The final rendered input is counted
   again before it is returned.
4. **Evidence IDs** ``E01`` to ``E10`` are assigned in rank order to *included*
   evidence only. They are request-scoped labels, not global chunk identifiers,
   and excluded candidates never get one, so they cannot become citation
   targets.

Outcomes: ``evidence_selected``, ``empty_retrieval`` (no candidates) and
``no_evidence_fits`` (candidates, but not even the first fits). The last two
are ordinary prepared states with zero evidence, not failures and not
evidence that the corpus cannot answer. Invalid questions, incompatible
artifacts or configurations, malformed candidates and an over-budget question
are explicit errors.

Token figures are local ``o200k_base`` estimates. Instruction and input
counts are measured separately because tokenization is not additive across
strings. The 1,024-token schema and framing allowance is provisional until the
answer schema exists (Commit 5.3). None of this is the provider's actual or
billed usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.generation.config import DEFAULT_GENERATION_CONFIG, GenerationConfig
from founderlens.generation.errors import GenerationError, safe_token
from founderlens.generation.prompts import (
    GROUNDED_INSTRUCTIONS,
    PROMPT_VERSION,
    input_sha256,
    prompt_id,
    prompt_sha256,
    serialize_input,
)
from founderlens.generation.schema import GenerationRequest
from founderlens.generation.token_budget import TokenCounter
from founderlens.ingestion.chunk import ChunkRecord
from founderlens.ingestion.extract import publish_atomic
from founderlens.retrieval.bm25 import SMOKE_QUERIES
from founderlens.retrieval.policy import (
    SELECTED_POLICY,
    PolicyRetriever,
    RetrievalPolicy,
    load_policy_retriever,
)
from founderlens.retrieval.schema import ChunkCollection, RetrievalError, RetrievalResult
from founderlens.retrieval.tokenize import tokenize

CONTEXT_SCHEMA_VERSION = 1
SELECTION_ALGORITHM = "rank_ordered_whole_chunk_prefix"
SELECTION_ALGORITHM_VERSION = 1
DEFAULT_TOKENIZER = "tiktoken-0.14.0/o200k_base"
SMOKE_REPORT_FILENAME = "context_smoke_report.json"
EXCEEDS_INPUT_BUDGET = "exceeds_input_budget"
AFTER_BUDGET_CUTOFF = "after_budget_cutoff"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class InvalidQuestionError(GenerationError):
    """The question is unusable as given. Messages never contain the question text."""

    category = "invalid_question"


class CandidateValidationError(GenerationError):
    """Retrieval candidates are malformed or do not match the verified collection."""

    category = "invalid_candidates"

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("; ".join(self.problems[:5]))


class ContextCompatibilityError(GenerationError):
    """Tokenizer, generation configuration, policy or collection identities disagree."""

    category = "context_incompatible"


class ContextBudgetError(GenerationError):
    """The trusted instructions and the question alone exceed the input limit."""

    category = "context_budget"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Application token budgets (not model context-window limits). Hashed into ``config_id``."""

    tokenizer: str = DEFAULT_TOKENIZER
    max_candidates: int = 10
    total_token_target: int = 8192
    input_token_limit: int = 6144
    framing_allowance_tokens: int = 1024
    reserved_output_tokens: int = 1024
    max_question_tokens: int = 512
    max_question_characters: int = 8192

    def __post_init__(self) -> None:
        problems: list[str] = []
        tokenizer: object = self.tokenizer
        if not isinstance(tokenizer, str) or not tokenizer.strip():
            problems.append("tokenizer must name the tokenizer identity")
        for name in (
            "max_candidates",
            "total_token_target",
            "input_token_limit",
            "framing_allowance_tokens",
            "reserved_output_tokens",
            "max_question_tokens",
            "max_question_characters",
        ):
            if not _positive_int(getattr(self, name)):
                problems.append(f"{name} must be a positive integer")
        if not problems:
            if self.max_candidates > 99:
                problems.append("max_candidates must be at most 99 (evidence IDs E01-E99)")
            allocated = (
                self.input_token_limit + self.framing_allowance_tokens + self.reserved_output_tokens
            )
            if allocated > self.total_token_target:
                problems.append(
                    f"input ({self.input_token_limit}) + framing "
                    f"({self.framing_allowance_tokens}) + output "
                    f"({self.reserved_output_tokens}) = {allocated} exceeds the total target "
                    f"{self.total_token_target}"
                )
            if self.max_question_tokens >= self.input_token_limit:
                problems.append("max_question_tokens must be below input_token_limit")
        if problems:
            raise ContextCompatibilityError("invalid context configuration: " + "; ".join(problems))

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "context_schema_version": CONTEXT_SCHEMA_VERSION,
            "tokenizer": self.tokenizer,
            "selection_algorithm": SELECTION_ALGORITHM,
            "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
            "max_candidates": self.max_candidates,
            "total_token_target": self.total_token_target,
            "input_token_limit": self.input_token_limit,
            "framing_allowance_tokens": self.framing_allowance_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "max_question_tokens": self.max_question_tokens,
            "max_question_characters": self.max_question_characters,
            "evidence_id_format": "E{:02d}",
            "chunk_truncation": False,
            "page_deduplication": False,
            "evidence_rewriting": False,
            "score_threshold": None,
        }

    @property
    def config_id(self) -> str:
        canonical = json.dumps(self.semantic_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


DEFAULT_CONTEXT_CONFIG = ContextConfig()


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


def evidence_id(position: int) -> str:
    """``E01`` for the first included record, ``E02`` for the second, and so on."""
    return f"E{position:02d}"


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """An included evidence record wrapping the original, unmodified chunk."""

    evidence_id: str
    rank: int
    score: float
    chunk: ChunkRecord = field(repr=False)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text

    def model_view(self) -> dict[str, Any]:
        """The only fields the model sees, in the fixed serialization order."""
        c = self.chunk
        return {
            "evidence_id": self.evidence_id,
            "document_id": c.document_id,
            "company": c.company,
            "reporting_period": c.fiscal_period,
            "page_number": c.page_number,
            "text": c.text,
        }

    def provenance(self) -> dict[str, Any]:
        """Trusted metadata for logs and later citation checks (no text)."""
        c = self.chunk
        return {
            "evidence_id": self.evidence_id,
            "rank": self.rank,
            "score": self.score,
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "document_sha256": c.document_sha256,
            "page_id": c.page_id,
            "page_number": c.page_number,
            "company": c.company,
            "reporting_period": c.fiscal_period,
            "source_url": c.source_url,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "chunking_config_id": c.chunking_config_id,
        }


class EvidenceRegistry:
    """Immutable, request-scoped mapping of evidence IDs to included evidence."""

    __slots__ = ("_by_id", "_records")
    _records: tuple[EvidenceRecord, ...]
    _by_id: Mapping[str, EvidenceRecord]

    def __init__(self, records: Sequence[EvidenceRecord] = ()) -> None:
        records = tuple(records)
        expected = [evidence_id(i) for i in range(1, len(records) + 1)]
        if [r.evidence_id for r in records] != expected:
            raise ValueError("evidence IDs must be E01, E02, ... in order")
        by_id: Mapping[str, EvidenceRecord] = MappingProxyType({r.evidence_id: r for r in records})
        object.__setattr__(self, "_records", records)
        object.__setattr__(self, "_by_id", by_id)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("EvidenceRegistry is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("EvidenceRegistry is immutable")

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return self._records

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._by_id.get(evidence_id)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._by_id

    def __iter__(self) -> Iterator[EvidenceRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"EvidenceRegistry({list(self._by_id)})"


@dataclass(frozen=True, slots=True)
class ExcludedCandidate:
    chunk_id: str
    rank: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "rank": self.rank, "reason": self.reason}


class ContextStatus(StrEnum):
    EVIDENCE_SELECTED = "evidence_selected"
    EMPTY_RETRIEVAL = "empty_retrieval"
    NO_EVIDENCE_FITS = "no_evidence_fits"


@dataclass(frozen=True, slots=True)
class TokenBudgetReport:
    """Local token estimates. ``question_tokens`` is standalone and not additive."""

    tokenizer: str
    question_tokens: int
    instruction_tokens: int
    input_tokens: int
    input_token_limit: int
    framing_allowance_tokens: int
    reserved_output_tokens: int
    total_token_target: int

    @property
    def local_total_tokens(self) -> int:
        """Instructions + serialized input, each counted separately."""
        return self.instruction_tokens + self.input_tokens

    @property
    def remaining_input_tokens(self) -> int:
        return self.input_token_limit - self.local_total_tokens

    @property
    def estimated_total_allocation(self) -> int:
        """Local total + provisional framing allowance + reserved output."""
        return self.local_total_tokens + self.framing_allowance_tokens + self.reserved_output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": "local tokenizer count, not provider usage",
            "tokenizer": self.tokenizer,
            "question_tokens_standalone": self.question_tokens,
            "instruction_tokens": self.instruction_tokens,
            "input_tokens": self.input_tokens,
            "local_total_tokens": self.local_total_tokens,
            "input_token_limit": self.input_token_limit,
            "remaining_input_tokens": self.remaining_input_tokens,
            "framing_allowance_tokens_provisional": self.framing_allowance_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "estimated_total_allocation": self.estimated_total_allocation,
            "total_token_target": self.total_token_target,
        }


@dataclass(frozen=True, slots=True)
class PreparedGenerationInput:
    """Everything the generation service needs; built once, never rebuilt downstream."""

    status: ContextStatus
    question: str = field(repr=False)
    request: GenerationRequest
    evidence: EvidenceRegistry
    excluded: tuple[ExcludedCandidate, ...]
    candidate_count: int
    budget: TokenBudgetReport
    input_sha256: str
    _identities_json: str = field(repr=False)

    def identities(self) -> dict[str, Any]:
        """A fresh copy of the retrieval, context, prompt and generation identities."""
        data: dict[str, Any] = json.loads(self._identities_json)
        return data

    def metadata(self) -> dict[str, Any]:
        """Diagnostics without question or evidence text."""
        return {
            "status": self.status.value,
            "candidate_count": self.candidate_count,
            "selected_count": len(self.evidence),
            "excluded_count": len(self.excluded),
            "evidence": [r.provenance() for r in self.evidence],
            "excluded": [e.to_dict() for e in self.excluded],
            "budget": self.budget.to_dict(),
            "input_sha256": self.input_sha256,
            "identities": self.identities(),
        }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_question(
    question: object, counter: TokenCounter, config: ContextConfig = DEFAULT_CONTEXT_CONFIG
) -> int:
    """The question's local token count, or :class:`InvalidQuestionError`."""
    if not isinstance(question, str):
        raise InvalidQuestionError(f"question must be a string, got {type(question).__name__}")
    if not question.strip():
        raise InvalidQuestionError("question is empty")
    if len(question) > config.max_question_characters:
        raise InvalidQuestionError(
            f"question has {len(question)} characters; at most "
            f"{config.max_question_characters} are accepted"
        )
    if not tokenize(question):
        raise InvalidQuestionError(
            "question has no letters or digits (punctuation-only questions retrieve nothing)"
        )
    tokens = counter.count(question)
    if tokens > config.max_question_tokens:
        raise InvalidQuestionError(
            f"question has {tokens} tokens; at most {config.max_question_tokens} are accepted "
            "(it is not truncated)"
        )
    return tokens


def validate_candidates(
    candidates: Sequence[RetrievalResult],
    collection: ChunkCollection,
    policy: RetrievalPolicy = SELECTED_POLICY,
    config: ContextConfig = DEFAULT_CONTEXT_CONFIG,
) -> tuple[RetrievalResult, ...]:
    """The candidates unchanged, or :class:`CandidateValidationError` listing every problem."""
    items: tuple[object, ...] = tuple(candidates)
    limit = min(config.max_candidates, policy.candidate_top_k)
    problems: list[str] = []
    if len(items) > limit:
        problems.append(f"{len(items)} candidates; at most {limit} are accepted")
    trusted = {c.chunk_id: c for c in collection.chunks}
    seen: set[str] = set()
    previous_score = math.inf
    results: list[RetrievalResult] = []
    for position, item in enumerate(items, start=1):
        if not isinstance(item, RetrievalResult):
            problems.append(f"candidate {position} is not a RetrievalResult")
            continue
        results.append(item)
        chunk = item.chunk
        label = f"rank {item.rank} ({chunk.chunk_id})"
        if item.method != policy.method:
            problems.append(f"{label}: method {safe_token(item.method)}, expected {policy.method}")
        if item.rank != position:
            problems.append(f"{label}: rank must be {position} (one-based, contiguous)")
        if chunk.chunk_id in seen:
            problems.append(f"{label}: duplicate chunk_id")
        seen.add(chunk.chunk_id)
        score: object = item.score
        if not isinstance(score, float) or not math.isfinite(score):
            problems.append(f"{label}: score must be a finite float")
        elif score > previous_score:
            problems.append(f"{label}: score is higher than the previous rank's")
        else:
            previous_score = score
        if chunk.chunking_config_id != policy.collection_config_id:
            problems.append(f"{label}: chunk is not from collection {policy.collection_config_id}")
        if trusted.get(chunk.chunk_id) != chunk:
            problems.append(f"{label}: chunk text or provenance differs from the collection")
    if problems:
        raise CandidateValidationError(problems)
    return tuple(results)


def _check_compatibility(
    counter: TokenCounter,
    collection: ChunkCollection,
    retrieval_identities: Mapping[str, Any],
    policy: RetrievalPolicy,
    config: ContextConfig,
    generation_config: GenerationConfig,
) -> None:
    problems: list[str] = []
    if counter.identity != config.tokenizer:
        problems.append(
            f"tokenizer {counter.identity!r}, context config expects {config.tokenizer!r}"
        )
    if config.reserved_output_tokens != generation_config.max_output_tokens:
        problems.append(
            f"reserved_output_tokens {config.reserved_output_tokens} differs from the generation "
            f"config's max_output_tokens {generation_config.max_output_tokens}"
        )
    if collection.chunking_config_id != policy.collection_config_id:
        problems.append(
            f"collection {collection.chunking_config_id} is not the policy's "
            f"{policy.collection_config_id}"
        )
    if retrieval_identities.get("policy_id") != policy.policy_id:
        problems.append("retrieval identities do not belong to this policy")
    if retrieval_identities.get("collection_artifact_sha256") != collection.artifact_sha256:
        problems.append("retrieval identities do not belong to this collection artifact")
    if problems:
        raise ContextCompatibilityError("; ".join(problems))


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def build_generation_input(
    question: str,
    candidates: Sequence[RetrievalResult],
    *,
    collection: ChunkCollection,
    retrieval_identities: Mapping[str, Any],
    counter: TokenCounter,
    policy: RetrievalPolicy = SELECTED_POLICY,
    config: ContextConfig = DEFAULT_CONTEXT_CONFIG,
    generation_config: GenerationConfig = DEFAULT_GENERATION_CONFIG,
) -> PreparedGenerationInput:
    """Pure, deterministic construction from already-retrieved candidates."""
    _check_compatibility(
        counter, collection, retrieval_identities, policy, config, generation_config
    )
    question_tokens = validate_question(question, counter, config)
    validated = validate_candidates(candidates, collection, policy, config)
    limit = config.input_token_limit
    instruction_tokens = counter.count(GROUNDED_INSTRUCTIONS)
    input_text = serialize_input(question, [])
    input_tokens = counter.count(input_text)
    if instruction_tokens + input_tokens > limit:
        raise ContextBudgetError(
            f"instructions ({instruction_tokens}) and question input ({input_tokens}) alone "
            f"exceed the input limit {limit}; no request was built"
        )
    selected: list[EvidenceRecord] = []
    excluded: list[ExcludedCandidate] = []
    for candidate in validated:
        if excluded:
            excluded.append(
                ExcludedCandidate(candidate.chunk.chunk_id, candidate.rank, AFTER_BUDGET_CUTOFF)
            )
            continue
        record = EvidenceRecord(
            evidence_id(len(selected) + 1), candidate.rank, candidate.score, candidate.chunk
        )
        trial = serialize_input(question, [r.model_view() for r in (*selected, record)])
        trial_tokens = counter.count(trial)
        if instruction_tokens + trial_tokens <= limit:
            selected.append(record)
            input_text, input_tokens = trial, trial_tokens
        else:
            excluded.append(
                ExcludedCandidate(candidate.chunk.chunk_id, candidate.rank, EXCEEDS_INPUT_BUDGET)
            )
    # Verify the final rendered input with the same counter before returning it.
    final_tokens = counter.count(input_text)
    if final_tokens != input_tokens or instruction_tokens + final_tokens > limit:
        raise ContextBudgetError("final rendered input failed the budget re-check")
    if not validated:
        status = ContextStatus.EMPTY_RETRIEVAL
    elif not selected:
        status = ContextStatus.NO_EVIDENCE_FITS
    else:
        status = ContextStatus.EVIDENCE_SELECTED
    identities = {
        "retrieval": dict(retrieval_identities),
        "context_config_id": config.config_id,
        "context_config": config.semantic_dict(),
        "prompt_version": PROMPT_VERSION,
        "prompt_id": prompt_id(),
        "prompt_sha256": prompt_sha256(),
        "generation_config_id": generation_config.config_id,
        "tokenizer": counter.identity,
    }
    return PreparedGenerationInput(
        status=status,
        question=question,
        request=GenerationRequest(GROUNDED_INSTRUCTIONS, input_text),
        evidence=EvidenceRegistry(selected),
        excluded=tuple(excluded),
        candidate_count=len(validated),
        budget=TokenBudgetReport(
            tokenizer=counter.identity,
            question_tokens=question_tokens,
            instruction_tokens=instruction_tokens,
            input_tokens=input_tokens,
            input_token_limit=limit,
            framing_allowance_tokens=config.framing_allowance_tokens,
            reserved_output_tokens=config.reserved_output_tokens,
            total_token_target=config.total_token_target,
        ),
        input_sha256=input_sha256(input_text),
        _identities_json=json.dumps(identities, sort_keys=True, separators=(",", ":")),
    )


def prepare_generation_input(
    question: str,
    retriever: PolicyRetriever,
    counter: TokenCounter,
    *,
    config: ContextConfig = DEFAULT_CONTEXT_CONFIG,
    generation_config: GenerationConfig = DEFAULT_GENERATION_CONFIG,
) -> PreparedGenerationInput:
    """Validate the question, retrieve with the loaded policy retriever, then build."""
    validate_question(question, counter, config)  # before any retrieval work
    return build_generation_input(
        question,
        retriever.search(question),
        collection=retriever.collection,
        retrieval_identities=retriever.identities(),
        counter=counter,
        policy=retriever.policy,
        config=config,
        generation_config=generation_config,
    )


# --------------------------------------------------------------------------- #
# Smoke report and CLI
# --------------------------------------------------------------------------- #

# Exploratory queries written for smoke tests (not evaluation questions), plus
# one synthetic over-length question.
CONTEXT_SMOKE_QUERIES: tuple[tuple[str, str], ...] = (
    *SMOKE_QUERIES,
    ("oversized synthetic question", "What changed in quarterly revenue and margins? " * 80),
)


def _smoke_entry(
    category: str, query: str, retriever: PolicyRetriever, counter: TokenCounter
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    try:
        prepared = prepare_generation_input(query, retriever, counter)
    except InvalidQuestionError as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "category": category,
            "query_characters": len(query),
            "error": exc.category,
            "detail": exc.detail,
        }, elapsed
    elapsed = (time.perf_counter() - started) * 1000
    meta = prepared.metadata()
    entry = {
        "category": category,
        "query": query,
        "status": meta["status"],
        "candidate_count": meta["candidate_count"],
        "selected_count": meta["selected_count"],
        "excluded_count": meta["excluded_count"],
        "excluded": meta["excluded"],
        "budget": meta["budget"],
        "evidence": [
            {k: e[k] for k in ("evidence_id", "rank", "chunk_id", "document_id", "page_number")}
            for e in meta["evidence"]
        ],
        "input_sha256": meta["input_sha256"],
        "instructions_sha256": _sha(prepared.request.instructions),
    }
    return entry, elapsed


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_smoke_report(settings: Settings) -> dict[str, Any]:
    """Two identical passes over the smoke queries; deterministic content compared exactly."""
    from founderlens.generation.token_budget import load_generation_tokenizer

    started = time.perf_counter()
    retriever = load_policy_retriever(settings)
    retriever_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    counter = load_generation_tokenizer(settings)
    tokenizer_ms = (time.perf_counter() - started) * 1000
    runs: list[list[dict[str, Any]]] = []
    latency: list[list[float]] = []
    for _ in range(2):
        entries, times = [], []
        for category, query in CONTEXT_SMOKE_QUERIES:
            entry, elapsed = _smoke_entry(category, query, retriever, counter)
            entries.append(entry)
            times.append(round(elapsed, 3))
        runs.append(entries)
        latency.append(times)
    identities = retriever.identities()
    return {
        "report_type": "founderlens-context-smoke-report",
        "note": "Functional smoke test with exploratory queries, not evaluation questions. "
        "No LLM is called; token figures are local estimates, not provider usage.",
        "context_config_id": DEFAULT_CONTEXT_CONFIG.config_id,
        "context_config": DEFAULT_CONTEXT_CONFIG.semantic_dict(),
        "prompt_version": PROMPT_VERSION,
        "prompt_id": prompt_id(),
        "generation_config_id": DEFAULT_GENERATION_CONFIG.config_id,
        "retrieval_policy_id": identities["policy_id"],
        "collection_artifact_sha256": identities["collection_artifact_sha256"],
        "tokenizer": counter.identity,
        "queries": runs[0],
        "reproducible": runs[0] == runs[1],
        "operational": {
            "note": "Varies between runs; not part of the deterministic results.",
            "retriever_load_ms": round(retriever_ms, 3),
            "tokenizer_cold_load_ms": round(tokenizer_ms, 3),
            "context_ms_run1": latency[0],
            "context_ms_run2": latency[1],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.generation.context",
        description="Evidence context construction (no LLM call).",
    )
    parser.add_argument(
        "--smoke", action="store_true", required=True, help="Write the context smoke report."
    )
    parser.add_argument("--project-root", default=None, help="Absolute project root override.")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        report = build_smoke_report(settings)
        path = settings.reports_dir / SMOKE_REPORT_FILENAME
        data = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        publish_atomic(path, data.encode("utf-8"))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (RetrievalError, GenerationError) as exc:
        print(f"Context smoke FAILED: {exc}", file=sys.stderr)
        return 1
    for entry in report["queries"]:
        if "error" in entry:
            print(f"  {entry['category']:<34} rejected: {entry['error']}")
        else:
            b = entry["budget"]
            print(
                f"  {entry['category']:<34} {entry['status']:<17} candidates "
                f"{entry['candidate_count']:>2} selected {entry['selected_count']:>2} "
                f"tokens {b['local_total_tokens']:>5}/{b['input_token_limit']}"
            )
    print(f"Reproducible across two runs: {report['reproducible']}; report {path.name}")
    return 0 if report["reproducible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
