"""Citation resolution and the final grounded response (Commit 5.4).

:class:`CitationResolver` turns a :class:`GenerationRunResult` into a
:class:`GroundedResponse`. It makes no LLM call and no network request.

**Boundary checks.** Before resolving anything, the resolver re-checks the run
rather than trusting it because it is a frozen model:

- the answer is re-parsed from its JSON form through the strict
  ``GroundedAnswer`` schema, which catches unvalidated ``model_copy`` or
  ``model_construct`` objects;
- the response invariants are re-checked against the run's own evidence
  registry (:class:`ResponseContractError`);
- the run's internal consistency is checked
  (:class:`CitationResolutionError`):
  - the registry equals the evidence serialized into the request the
    provider received;
  - the request hash and the committed prompt match;
  - the origin matches the preparation status and the presence of a provider
    result;
  - a model answer is the provider's parsed answer;
  - the configuration and answer-schema identities agree;
  - the run belongs to this resolver's collection artifact.

**Resolution.** Each cited ID is looked up only in *this run's* registry; a
missing, malformed or excluded ID is an error and is never dropped. The
record's chunk must equal the trusted collection's chunk of that ID, and its
company, reporting period, source URL and document hash must match the corpus
manifest, which also supplies the document title. Every citation field comes
from those trusted records, never from model text. URLs are not fetched.

**Numbering.** Display numbers start at 1 in order of first appearance,
claim by claim and then reference by reference. A repeated evidence ID reuses
its number, distinct chunks of one page get distinct numbers, and only cited
evidence becomes a citation. Uncited model-visible evidence stays in the run.
Claim and reference order are never changed.

Offsets are Python code-point offsets into the page's *normalized* text
(start inclusive, end exclusive). They are not PDF byte offsets or page
coordinates.

These checks are structural. They show that every citation points to evidence
the model was given; they do not show that the evidence supports the claim,
that a number is right, or that a deliberately falsified run object
(hostile code in the same process) would be detected.

**Internal trace.** ``GroundedResponse.run`` is the full generation trace:
the question, the exact prompt and serialized evidence, every model-visible
passage, the provider's raw output text and its IDs. It exists for evaluation
and debugging. Never serialize a ``GroundedResponse`` or its ``run`` directly
into a public API response; a future API needs an explicit response model with
chosen fields. ``metadata()``, ``repr()``, the log line and exception messages
contain no question, prompt, passage, claim or limitation text.

**Display.** ``render()`` produces plain text for terminals and logs the
caller chooses to write. Untrusted strings are JSON-quoted there, which keeps
them from forging structure in that text. It is not HTML escaping: a web
frontend must escape or sanitize claim, limitation and passage text itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from founderlens.corpus.manifest import CorpusManifest, check_https_url
from founderlens.generation.answer import (
    AnswerStatus,
    ClaimType,
    GroundedAnswer,
    ResponseContractError,
    answer_problems,
    answer_schema_id,
)
from founderlens.generation.context import (
    ContextStatus,
    EvidenceRecord,
    EvidenceRegistry,
    PreparedGenerationInput,
)
from founderlens.generation.errors import GenerationError, safe_token
from founderlens.generation.prompts import GROUNDED_INSTRUCTIONS, input_sha256, prompt_id
from founderlens.generation.schema import GenerationRequest, GenerationResult
from founderlens.generation.service import (
    LOCAL_LIMITATIONS,
    AnswerOrigin,
    GenerationRunResult,
    LocalAbstentionReason,
)
from founderlens.ingestion.chunk import ChunkRecord
from founderlens.retrieval.schema import ChunkCollection

RESPONSE_CONTRACT_VERSION = 1
CITATION_NUMBERING = "first_appearance_v1"
RENDERING_FORMAT = "plain_text_v1"
OFFSET_SEMANTICS = (
    "Python code-point offsets into the page's normalized text, start inclusive and end "
    "exclusive; not PDF byte offsets or page coordinates"
)
logger = logging.getLogger(__name__)


class CitationResolutionError(GenerationError):
    """The run cannot be turned into a cited response. Messages never quote text."""

    category = "citation_resolution"

    def __init__(self, problems: Sequence[str], *, request_id: str | None = None) -> None:
        self.problems = tuple(problems)
        super().__init__("; ".join(self.problems[:5]), request_id=request_id)


def response_contract_definition() -> dict[str, Any]:
    return {
        "response_contract_version": RESPONSE_CONTRACT_VERSION,
        "answer_schema_id": answer_schema_id(),
        "citation_numbering": CITATION_NUMBERING,
        "rendering_format": RENDERING_FORMAT,
        "offset_semantics": OFFSET_SEMANTICS,
        "citation_fields": list(ResolvedCitation.__slots__),
        "claim_fields": list(CitedClaim.__slots__),
    }


def response_contract_id() -> str:
    canonical = json.dumps(response_contract_definition(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Response records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResolvedCitation:
    """One cited evidence record. Every field comes from trusted application data."""

    number: int
    evidence_id: str
    chunk_id: str
    document_id: str
    document_title: str
    company: str
    reporting_period: str
    page_id: str
    page_number: int
    source_url: str
    start_char: int
    end_char: int
    retrieval_rank: int
    retrieval_score: float
    text: str = field(repr=False)

    def provenance(self) -> dict[str, Any]:
        """Everything except the passage text."""
        return {name: getattr(self, name) for name in self.__slots__ if name != "text"}


@dataclass(frozen=True, slots=True)
class CitedClaim:
    """A claim exactly as generated, with application-assigned citation numbers."""

    position: int
    claim_type: ClaimType
    evidence_ids: tuple[str, ...]
    citation_numbers: tuple[int, ...]
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class RenderedResponse:
    """Plain-text display form. ``claims`` keeps the authoritative claim-to-number mapping."""

    format: str
    status: str
    claims: tuple[tuple[str, tuple[int, ...]], ...]
    limitations_label: str | None
    limitations: tuple[str, ...]
    sources: tuple[str, ...]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "status": self.status,
            "claims": [{"text": t, "citations": list(n)} for t, n in self.claims],
            "limitations_label": self.limitations_label,
            "limitations": list(self.limitations),
            "sources": list(self.sources),
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class GroundedResponse:
    """The final application response, tied to its generation trace (``run``)."""

    status: AnswerStatus
    claims: tuple[CitedClaim, ...]
    citations: tuple[ResolvedCitation, ...]
    limitations: tuple[str, ...] = field(repr=False)
    origin: AnswerOrigin
    local_abstention_reason: LocalAbstentionReason | None
    corpus_id: str
    corpus_version: str
    response_contract_id: str
    run: GenerationRunResult = field(repr=False)

    @property
    def abstained(self) -> bool:
        return self.status is AnswerStatus.INSUFFICIENT_EVIDENCE

    def citation(self, number: int) -> ResolvedCitation | None:
        return next((c for c in self.citations if c.number == number), None)

    def identities(self) -> dict[str, Any]:
        ids = self.run.identities()
        ids["response_contract_id"] = self.response_contract_id
        ids["response_contract_version"] = RESPONSE_CONTRACT_VERSION
        ids["corpus"] = {"corpus_id": self.corpus_id, "corpus_version": self.corpus_version}
        return ids

    def usage(self) -> dict[str, int | None] | None:
        """Provider-reported usage, or ``None`` when no provider was called."""
        provider = self.run.provider_result
        return None if provider is None else provider.usage.to_dict()

    def timings(self) -> dict[str, float | None]:
        provider = self.run.provider_result
        return {
            "context_ms": round(self.run.context_ms, 3),
            "provider_ms": None if self.run.provider_ms is None else round(self.run.provider_ms, 3),
            "provider_reported_latency_ms": None if provider is None else provider.latency_ms,
            "total_ms": round(self.run.total_ms, 3),
        }

    def metadata(self) -> dict[str, Any]:
        """Diagnostics without question, claim, limitation, prompt or passage text."""
        cited = {c.evidence_id for c in self.citations}
        visible = self.run.prepared.evidence.ids
        return {
            "status": self.status.value,
            "origin": self.origin.value,
            "local_abstention_reason": None
            if self.local_abstention_reason is None
            else self.local_abstention_reason.value,
            "abstained": self.abstained,
            "claim_count": len(self.claims),
            "claim_types": [c.claim_type.value for c in self.claims],
            "claim_citations": [list(c.citation_numbers) for c in self.claims],
            "citations": [c.provenance() for c in self.citations],
            "uncited_evidence_ids": [i for i in visible if i not in cited],
            "limitation_count": len(self.limitations),
            "identities": self.identities(),
            "usage": self.usage(),
            "timings": self.timings(),
        }

    def render(self) -> RenderedResponse:
        """Deterministic plain text built from claims and application citation numbers.

        Untrusted strings (claim text, limitations) are shown as JSON string
        literals, so line breaks, fake ``[n]`` markers, Markdown links or HTML in
        them stay inside one quoted value. The ``Sources`` lines and the numbers
        come only from trusted citation records. Claim text is never rewritten:
        ``claims`` carries it unchanged.
        """
        label: str | None = None
        provider = self.run.provider_result
        if self.limitations:
            label = (
                "Limitation reported by the application (no model call)"
                if provider is None
                else f"Limitations reported by the generator ({provider.provider.provider})"
            )
        sources = tuple(_source_line(c) for c in self.citations)
        lines = [f"Status: {self.status.value}"]
        if self.claims:
            lines.append("Claims:")
            for claim in self.claims:
                lines.append(f"{claim.position}. {_quote(claim.text)}")
                refs = ", ".join(f"[{n}]" for n in claim.citation_numbers)
                lines.append(f"   Sources: {refs}")
        if self.limitations:
            lines.append(f"{label}:")
            lines.extend(f"- {_quote(text)}" for text in self.limitations)
        if sources:
            lines.append("Sources:")
            lines.extend(sources)
        return RenderedResponse(
            format=RENDERING_FORMAT,
            status=self.status.value,
            claims=tuple((c.text, c.citation_numbers) for c in self.claims),
            limitations_label=label,
            limitations=self.limitations,
            sources=sources,
            text="\n".join(lines) + "\n",
        )


def _quote(untrusted: str) -> str:
    return json.dumps(untrusted, ensure_ascii=False)


def _source_line(c: ResolvedCitation) -> str:
    return (
        f"[{c.number}] {c.company}, {c.document_title} ({c.reporting_period}), page "
        f"{c.page_number}, passage {c.chunk_id} [{c.evidence_id}]; {c.source_url}"
    )


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #


class CitationResolver:
    """Resolves runs against one verified collection and the corpus manifest."""

    __slots__ = ("_chunks", "_collection", "_manifest")

    def __init__(self, collection: ChunkCollection, manifest: CorpusManifest) -> None:
        self._collection = collection
        self._manifest = manifest
        self._chunks: dict[str, ChunkRecord] = {c.chunk_id: c for c in collection.chunks}

    def resolve(self, run: GenerationRunResult) -> GroundedResponse:
        if not isinstance(run, GenerationRunResult):
            raise CitationResolutionError(["run must be a GenerationRunResult"])
        _check_run_types(run)
        request_id = None if run.provider_result is None else run.provider_result.request_id
        answer = _revalidated(run.answer, request_id)
        problems = self._association_problems(run, answer)
        if problems:
            raise CitationResolutionError(problems, request_id=request_id)
        invariant_problems = answer_problems(answer, run.prepared.evidence)
        if invariant_problems:
            raise ResponseContractError(invariant_problems, request_id=request_id)
        claims, citations = self._resolve_claims(answer, run, request_id)
        response = GroundedResponse(
            status=answer.status,
            claims=claims,
            citations=citations,
            limitations=answer.limitations,
            origin=run.origin,
            local_abstention_reason=run.local_abstention_reason,
            corpus_id=self._manifest.corpus_id,
            corpus_version=self._manifest.corpus_version,
            response_contract_id=response_contract_id(),
            run=run,
        )
        logger.info(
            "citations resolved status=%s origin=%s claims=%s citations=%s uncited=%s "
            "response_contract_id=%s",
            response.status.value,
            response.origin.value,
            len(claims),
            len(citations),
            len(run.prepared.evidence) - len(citations),
            response.response_contract_id,
        )
        return response

    # ----------------------------------------------------------------------- #

    def _association_problems(self, run: GenerationRunResult, answer: GroundedAnswer) -> list[str]:
        problems: list[str] = []
        prepared = run.prepared
        registry = prepared.evidence
        ids = prepared.identities()
        request = prepared.request
        try:
            payload = json.loads(request.input_text)
        except (TypeError, ValueError):
            payload = None
        if not isinstance(payload, dict):
            problems.append("the prepared request input is not the serialized evidence payload")
        else:
            if payload.get("evidence") != [r.model_view() for r in registry]:
                problems.append("the evidence registry does not match the evidence in the request")
            if payload.get("question") != prepared.question:
                problems.append("the prepared question does not match the request")
        if input_sha256(request.input_text) != prepared.input_sha256:
            problems.append("the request input hash does not match the prepared input")
        if request.instructions != GROUNDED_INSTRUCTIONS or ids.get("prompt_id") != prompt_id():
            problems.append("the request does not use the committed grounded prompt")
        if run.answer_schema_id != answer_schema_id():
            problems.append("the run used a different answer schema")
        retrieval = ids.get("retrieval", {})
        if retrieval.get("collection_artifact_sha256") != self._collection.artifact_sha256:
            problems.append("the run was prepared from a different collection artifact")
        for record in registry:
            if self._chunks.get(record.chunk_id) != record.chunk:
                problems.append(f"{record.evidence_id}: chunk differs from the trusted collection")
        provider = run.provider_result
        origin: object = run.origin  # widened: a malformed run may carry anything
        if origin is AnswerOrigin.MODEL:
            if provider is None:
                problems.append("a model answer has no provider result")
            else:
                if provider.parsed != run.answer:
                    problems.append("the answer is not the provider's parsed answer")
                if provider.provider.config_id != ids.get("generation_config_id"):
                    problems.append("the provider configuration differs from the context's")
            if prepared.status is not ContextStatus.EVIDENCE_SELECTED:
                problems.append("a model answer requires evidence_selected preparation")
            if run.local_abstention_reason is not None:
                problems.append("a model answer cannot have a local abstention reason")
        elif origin is AnswerOrigin.LOCAL:
            reason = run.local_abstention_reason
            expected = {
                ContextStatus.EMPTY_RETRIEVAL: LocalAbstentionReason.EMPTY_RETRIEVAL,
                ContextStatus.NO_EVIDENCE_FITS: LocalAbstentionReason.NO_EVIDENCE_FITS,
            }.get(prepared.status)
            if provider is not None or run.provider_ms is not None:
                problems.append("a local abstention must not carry provider results")
            if reason is None or reason is not expected:
                problems.append("the local abstention reason does not match the preparation")
            elif answer.limitations != (LOCAL_LIMITATIONS[reason],) or answer.claims:
                problems.append("a local abstention must carry only its fixed limitation")
            if answer.status is not AnswerStatus.INSUFFICIENT_EVIDENCE:
                problems.append("a local abstention must be insufficient_evidence")
        else:
            problems.append("unknown answer origin")
        return problems

    def _resolve_claims(
        self, answer: GroundedAnswer, run: GenerationRunResult, request_id: str | None
    ) -> tuple[tuple[CitedClaim, ...], tuple[ResolvedCitation, ...]]:
        registry = run.prepared.evidence
        numbers: dict[str, int] = {}
        citations: list[ResolvedCitation] = []
        claims: list[CitedClaim] = []
        problems: list[str] = []
        for position, claim in enumerate(answer.claims, start=1):
            assigned: list[int] = []
            for evidence_id in claim.evidence_ids:
                if evidence_id not in numbers:
                    record = registry.get(evidence_id)
                    if record is None:  # already rejected by the invariants; kept as a guard
                        shown = safe_token(evidence_id)
                        problems.append(f"claim {position}: {shown} is not in this run")
                        continue
                    citation = self._citation(len(numbers) + 1, record, problems)
                    if citation is None:
                        continue
                    numbers[evidence_id] = citation.number
                    citations.append(citation)
                assigned.append(numbers[evidence_id])
            claims.append(
                CitedClaim(
                    position=position,
                    claim_type=claim.claim_type,
                    evidence_ids=claim.evidence_ids,
                    citation_numbers=tuple(assigned),
                    text=claim.text,
                )
            )
        if problems:
            raise CitationResolutionError(problems, request_id=request_id)
        return tuple(claims), tuple(citations)

    def _citation(
        self, number: int, record: EvidenceRecord, problems: list[str]
    ) -> ResolvedCitation | None:
        chunk = record.chunk
        entry = self._manifest.get(chunk.document_id)
        label = f"{record.evidence_id} ({chunk.chunk_id})"
        if entry is None:
            problems.append(f"{label}: document is not in the corpus manifest")
            return None
        if (chunk.company, chunk.fiscal_period, chunk.source_url, chunk.document_sha256) != (
            entry.company,
            entry.fiscal_period,
            entry.source_url,
            entry.sha256,
        ):
            problems.append(f"{label}: provenance does not match the corpus manifest")
            return None
        if check_https_url(chunk.source_url) is not None:
            problems.append(f"{label}: source URL is not an acceptable HTTPS URL")
            return None
        return ResolvedCitation(
            number=number,
            evidence_id=record.evidence_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title=entry.title,
            company=chunk.company,
            reporting_period=chunk.fiscal_period,
            page_id=chunk.page_id,
            page_number=chunk.page_number,
            source_url=chunk.source_url,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            retrieval_rank=record.rank,
            retrieval_score=record.score,
            text=chunk.text,
        )


def _check_run_types(run: GenerationRunResult) -> None:
    """Typed failure, not AttributeError, for a run assembled from the wrong objects."""
    prepared: object = run.prepared
    provider: object = run.provider_result
    problems = []
    if not isinstance(prepared, PreparedGenerationInput):
        problems.append("run.prepared is not a PreparedGenerationInput")
    else:
        evidence: object = prepared.evidence
        request: object = prepared.request
        if not isinstance(evidence, EvidenceRegistry):
            problems.append("run.prepared.evidence is not an EvidenceRegistry")
        if not isinstance(request, GenerationRequest):
            problems.append("run.prepared.request is not a GenerationRequest")
    if provider is not None and not isinstance(provider, GenerationResult):
        problems.append("run.provider_result is not a GenerationResult")
    if problems:
        raise CitationResolutionError(problems)


def _revalidated(answer: object, request_id: str | None) -> GroundedAnswer:
    """Re-parse through the strict schema (catches unvalidated copies and constructs)."""
    if not isinstance(answer, GroundedAnswer):
        raise CitationResolutionError(["answer is not a GroundedAnswer"], request_id=request_id)
    try:
        return GroundedAnswer.model_validate_json(answer.model_dump_json(warnings=False))
    except Exception as exc:  # pydantic raises several types; never echo their input
        raise CitationResolutionError(
            [f"answer failed revalidation against the schema ({type(exc).__name__})"],
            request_id=request_id,
        ) from None


def load_citation_resolver(
    collection: ChunkCollection, manifest: CorpusManifest
) -> CitationResolver:
    """Convenience constructor (the collection should be the policy retriever's)."""
    return CitationResolver(collection, manifest)
