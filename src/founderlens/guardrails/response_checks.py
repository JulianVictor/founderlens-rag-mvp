"""Deterministic financial response guardrails (Commit 6.2). Diagnostic only.

Usage::

    python -m founderlens.guardrails.response_checks --smoke   # real V2, SYNTHETIC answers

:func:`check_response` inspects a finished :class:`GroundedResponse` and returns a
:class:`GuardrailReport`. It never changes the response: status, claims,
classifications, evidence IDs, citations, limitations, usage, identities and the
generation trace stay exactly as resolved. :class:`GuardedResponse` pairs the two.
:class:`GuardedRunner` composes the existing service and resolver with the check.
Nothing here calls a model, repeats retrieval, alters context or uses the network.

**Findings** are signals for review and evaluation, never verdicts. None of them
means a claim is false or hallucinated, and a clean report does not mean the
answer is correct.

- ``company_attribution_gap`` (warning): a claim names a company (by name,
  alias, ticker, segment or brand, using the Commit 6.1 vocabulary) but none of
  *that claim's* cited records belongs to it. Other claims' citations never
  count. A segment such as Google Cloud is satisfied only by Alphabet evidence,
  and even then the segment itself is not verifiable from metadata.
- ``period_attribution_gap`` (notice): a claim names a manifest reporting
  period, but none of its cited documents is that period. Releases routinely
  contain other periods' figures (prior-year comparatives, quarterly
  references in annual releases), so this is a *suspected* mismatch only.
  Claim periods that are not manifest periods (for example a prior-year
  comparative inside a newer release) are not assessed.
- ``question_coverage_gap`` (warning for ``answered``, otherwise notice): an
  assessable company or period the question names is absent from the selected
  evidence, or was selected but not cited. Uses the Commit 6.1 coverage
  contract unchanged; ``complete`` is never read as "answered".
- ``numeric_trace_gap`` (notice): a ``reported`` claim states a financial
  amount or percentage that does not appear verbatim (after harmless
  normalization) in its cited passages, or a ``calculated`` claim has no
  observable operand in its cited passages.
- ``scope_unresolved`` (notice): one finding per Commit 6.1
  ``question_limitations`` code (unresolved, unassessable, basis-unspecified,
  or company-unspecified fiscal references).

**Numeric trace.**

- Only values carrying a financial marker are extracted: a currency sign, a
  scale word (thousand/million/billion/trillion and short forms), a percent sign
  or word, or thousands separators.
- Spans recognized as periods, dates or years by the Commit 6.1 analyzer are
  masked first. Bare integers, years, evidence IDs and citation numbers are
  never values.
- Comparison is by exact decimal value after removing currency signs and
  thousands separators (``1,234.50`` equals ``1234.5``). Percentages must match
  percentages.
- A claim value with a scale word matches a passage value with the same scale
  word. A passage value with the same digits but no scale word (for example a
  table "in millions") counts as ``number_only``: compatible, but the unit is
  unverified. That is not a finding.
- There is **no unit conversion and no rounding**: ``$35.6 billion`` versus
  ``$35,580 million`` stays not found, so it is reported as a notice
  (uncertain), never as an error.
- Single-letter scales (``k``, ``m``, ``b``, ``t``) count only when attached
  (``$39.3B``), so ``Note 5 b`` is not five billion.
- A currency or scaled amount never matches a percentage.
- Signs and parentheses are ignored, and magnitudes are compared.
- A local abstention never reached the model, so its cited coverage is not
  evaluated (``None``); a model answer with no citations cited nothing.
- There is no arithmetic recomputation and no entailment. A value found in a
  passage does not show that the claim is right.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from founderlens.config import ConfigError, Settings, load_settings
from founderlens.corpus.manifest import ManifestError, load_manifest
from founderlens.generation.answer import AnswerStatus, ClaimType
from founderlens.generation.citations import CitationResolver, GroundedResponse, ResolvedCitation
from founderlens.generation.errors import GenerationError
from founderlens.generation.schema import (
    GenerationRequest,
    GenerationResult,
    ProviderIdentity,
    SchemaT,
    TokenUsage,
)
from founderlens.generation.service import AnswerOrigin, GroundedGenerationService
from founderlens.guardrails.scope import (
    CompanyScope,
    CorpusVocabulary,
    EvidenceCoverage,
    LayerState,
    ManifestStatus,
    QuestionScope,
    analyze_question,
    analyzer_id,
    evidence_coverage,
)

POLICY_VERSION = 1
ENFORCEMENT_MODE = "diagnostic_only"


class GuardrailError(ValueError):
    """The guardrail inputs are malformed. No report is produced. Never quotes text."""


class FindingCategory(StrEnum):
    COMPANY_ATTRIBUTION_GAP = "company_attribution_gap"
    PERIOD_ATTRIBUTION_GAP = "period_attribution_gap"
    QUESTION_COVERAGE_GAP = "question_coverage_gap"
    NUMERIC_TRACE_GAP = "numeric_trace_gap"
    SCOPE_UNRESOLVED = "scope_unresolved"


class FindingSeverity(StrEnum):
    WARNING = "warning"  # a likely attribution or coverage problem worth review
    NOTICE = "notice"  # uncertainty or a weak signal; often legitimate


DETAIL_CODES: Mapping[str, tuple[FindingCategory, str]] = {
    "named_company_not_cited": (FindingCategory.COMPANY_ATTRIBUTION_GAP, "warning"),
    "named_segment_parent_not_cited": (FindingCategory.COMPANY_ATTRIBUTION_GAP, "warning"),
    "named_brand_company_not_cited": (FindingCategory.COMPANY_ATTRIBUTION_GAP, "warning"),
    "named_period_not_cited": (FindingCategory.PERIOD_ATTRIBUTION_GAP, "notice"),
    "basis_unspecified_period_not_cited": (FindingCategory.PERIOD_ATTRIBUTION_GAP, "notice"),
    "absent_from_selected_evidence": (FindingCategory.QUESTION_COVERAGE_GAP, "status-dependent"),
    "selected_but_not_cited": (FindingCategory.QUESTION_COVERAGE_GAP, "status-dependent"),
    "value_not_found_verbatim": (FindingCategory.NUMERIC_TRACE_GAP, "notice"),
    "no_observable_operand": (FindingCategory.NUMERIC_TRACE_GAP, "notice"),
    "unresolved_references": (FindingCategory.SCOPE_UNRESOLVED, "notice"),
    "unassessable_references": (FindingCategory.SCOPE_UNRESOLVED, "notice"),
    "basis_unspecified_periods": (FindingCategory.SCOPE_UNRESOLVED, "notice"),
    "fiscal_periods_without_named_company": (FindingCategory.SCOPE_UNRESOLVED, "notice"),
}


@dataclass(frozen=True, slots=True)
class GuardrailFinding:
    category: FindingCategory
    severity: FindingSeverity
    detail: str  # one of DETAIL_CODES
    claim_position: int | None
    companies: tuple[str, ...]
    periods: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    values: tuple[str, ...]  # normalized numeric values, numeric findings only

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "detail": self.detail,
            "claim_position": self.claim_position,
            "companies": list(self.companies),
            "periods": list(self.periods),
            "evidence_ids": list(self.evidence_ids),
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class GuardrailReport:
    findings: tuple[GuardrailFinding, ...]
    coverage: EvidenceCoverage
    question_scope: QuestionScope
    diagnostics: tuple[tuple[str, int], ...]  # non-finding counts, e.g. number_only values
    policy_id: str
    vocabulary_id: str
    analyzer_id: str
    response_contract_id: str

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            for key in (finding.category.value, finding.severity.value):
                counts[key] = counts.get(key, 0) + 1
        return counts

    def metadata(self) -> dict[str, Any]:
        """Codes, IDs and counts only: no question, claim, limitation or passage text."""
        return {
            "enforcement": ENFORCEMENT_MODE,
            "policy_id": self.policy_id,
            "vocabulary_id": self.vocabulary_id,
            "analyzer_id": self.analyzer_id,
            "response_contract_id": self.response_contract_id,
            "finding_count": len(self.findings),
            "counts": self.counts(),
            "findings": [f.to_dict() for f in self.findings],
            "diagnostics": dict(self.diagnostics),
            "question_scope": self.question_scope.metadata(),
            "coverage": self.coverage.metadata(),
            "caveat": "diagnostic signals only; not verification of factual support",
        }


@dataclass(frozen=True, slots=True)
class GuardedResponse:
    """The unchanged response plus its diagnostic report."""

    response: GroundedResponse
    report: GuardrailReport


# --------------------------------------------------------------------------- #
# Numeric extraction
# --------------------------------------------------------------------------- #

_SCALES = {
    "thousand": "thousand",
    "k": "thousand",
    "million": "million",
    "mn": "million",
    "mm": "million",
    "m": "million",
    "billion": "billion",
    "bn": "billion",
    "b": "billion",
    "trillion": "trillion",
    "tn": "trillion",
    "t": "trillion",
}
_NUMBER = re.compile(
    r"(?<![\w.])(?P<currency>[$€£])?\s?(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<percent>%|percent\b|per\s+cent\b)"
    r"|\s?(?P<scale>thousand|million|billion|trillion|bn|mn|mm|tn)(?!\w)"
    r"|(?P<letter>[kmbt])(?!\w))?",  # single-letter scales only when attached
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _Value:
    value: Decimal
    percent: bool
    scale: str | None
    currency: bool
    financial: bool  # carries a currency, scale, percent or thousands separator

    def key(self) -> str:
        unit = "%" if self.percent else (self.scale or "")
        return f"{self.value.normalize():f}{unit}"


def _values(text: str, masked: Sequence[tuple[int, int]] = ()) -> list[_Value]:
    found: list[_Value] = []
    for match in _NUMBER.finditer(text):
        if any(start < match.end() and match.start() < end for start, end in masked):
            continue
        raw = match.group("number")
        try:
            value = Decimal(raw.replace(",", ""))
        except InvalidOperation:  # pragma: no cover - the pattern only admits digits
            continue
        percent = match.group("percent") is not None
        scale_word = match.group("scale") or match.group("letter")
        scale = None if scale_word is None else _SCALES[scale_word.casefold()]
        currency = match.group("currency") is not None
        financial = bool(currency or percent or scale or "," in raw)
        found.append(_Value(value, percent, scale, currency, financial))
    return found


def _trace(claim_value: _Value, passages: Sequence[str]) -> str:
    """``matched`` | ``number_only`` | ``absent`` for one claim value."""
    number_only = False
    for passage in passages:
        for v in _values(passage):
            if v.value != claim_value.value:
                continue
            if claim_value.percent:
                if v.percent:
                    return "matched"
                if v.scale is None and not v.currency:  # e.g. a table column in percent
                    number_only = True
            elif claim_value.scale is not None:
                if v.scale == claim_value.scale:
                    return "matched"
                if v.scale is None and not v.percent:
                    number_only = True
            elif not v.percent:
                return "matched"
    return "number_only" if number_only else "absent"


# --------------------------------------------------------------------------- #
# Policy identity
# --------------------------------------------------------------------------- #


def policy_definition(vocabulary: CorpusVocabulary) -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "enforcement": ENFORCEMENT_MODE,
        "categories": [c.value for c in FindingCategory],
        "severities": [s.value for s in FindingSeverity],
        "detail_codes": {k: [c.value, s] for k, (c, s) in DETAIL_CODES.items()},
        "numeric_pattern": _NUMBER.pattern,
        "numeric_scales": dict(_SCALES),
        "numeric_semantics": (
            "reported and calculated claims only; financial marker required (currency, "
            "scale word, percent, thousands separator); analyzer period, date and unresolved "
            "spans masked; currency signs and thousands separators removed; exact decimal "
            "equality; percent matches percent, or number_only against a bare number; scale "
            "matches the same scale word, or number_only against a bare number; currency or "
            "scale never matches percent; no unit conversion, rounding or sign; no "
            "recomputation; calculated: a finding only when no value is observable"
        ),
        "attribution_semantics": (
            "per-claim cited records only; segments and brands need the parent company and "
            "a cited segment stays unverifiable; claim periods use manifest periods of the "
            "claim's named companies, basis-unspecified periods their calendar candidates; "
            "periods not in the manifest are not assessed"
        ),
        "coverage_semantics": (
            "Commit 6.1 evidence coverage over selected and cited records; local abstentions "
            "have cited coverage not evaluated; segments skipped; warning only for answered; "
            "one scope_unresolved notice per question limitation"
        ),
        "vocabulary_id": vocabulary.vocabulary_id,
        "analyzer_id": analyzer_id(),
    }


def guardrail_policy_id(vocabulary: CorpusVocabulary) -> str:
    canonical = json.dumps(policy_definition(vocabulary), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _finding(
    detail: str,
    *,
    severity: FindingSeverity | None = None,
    claim: int | None = None,
    companies: Sequence[str] = (),
    periods: Sequence[str] = (),
    evidence_ids: Sequence[str] = (),
    values: Sequence[str] = (),
) -> GuardrailFinding:
    category, default = DETAIL_CODES[detail]
    chosen = severity if severity is not None else FindingSeverity(default)
    return GuardrailFinding(
        category,
        chosen,
        detail,
        claim,
        tuple(companies),
        tuple(periods),
        tuple(evidence_ids),
        tuple(values),
    )


def _claim_findings(
    response: GroundedResponse, vocabulary: CorpusVocabulary, counts: dict[str, int]
) -> list[GuardrailFinding]:
    by_id: dict[str, ResolvedCitation] = {c.evidence_id: c for c in response.citations}
    findings: list[GuardrailFinding] = []
    for claim in response.claims:
        cited = []
        for evidence_id in claim.evidence_ids:
            citation = by_id.get(evidence_id)
            if citation is None:
                raise GuardrailError(f"claim {claim.position} cites an unresolved evidence ID")
            cited.append(citation)
        ids = claim.evidence_ids
        companies = {c.company for c in cited}
        documents = {c.document_id for c in cited}
        scope = analyze_question(claim.text, vocabulary)
        for ref in scope.companies:
            if ref.company in companies:
                if ref.scope is CompanyScope.SEGMENT:
                    counts["segment_attribution_unverifiable"] += 1
                continue
            detail = {
                CompanyScope.COMPANY: "named_company_not_cited",
                CompanyScope.SEGMENT: "named_segment_parent_not_cited",
                CompanyScope.BRAND_AMBIGUOUS: "named_brand_company_not_cited",
            }[ref.scope]
            findings.append(
                _finding(detail, claim=claim.position, companies=[ref.company], evidence_ids=ids)
            )
        for period in scope.periods:
            if period.manifest_status is ManifestStatus.NOT_IN_MANIFEST:
                counts["claim_periods_not_assessed"] += 1
                continue
            candidates = {*period.manifest_documents, *period.calendar_candidates}
            if candidates & documents:
                continue
            detail = (
                "basis_unspecified_period_not_cited"
                if period.manifest_status is ManifestStatus.BASIS_UNSPECIFIED
                else "named_period_not_cited"
            )
            findings.append(
                _finding(detail, claim=claim.position, periods=[period.canonical], evidence_ids=ids)
            )
        masked = [(r.start, r.end) for r in scope.periods]
        masked += [(u.start, u.end) for u in scope.unresolved]
        values = [v for v in _values(claim.text, masked) if v.financial]
        passages = [c.text for c in cited]
        if claim.claim_type is ClaimType.REPORTED:
            missing = []
            for value in values:
                outcome = _trace(value, passages)
                counts[f"values_{outcome}"] += 1
                if outcome == "absent":
                    missing.append(value.key())
            if missing:
                findings.append(
                    _finding(
                        "value_not_found_verbatim",
                        claim=claim.position,
                        evidence_ids=ids,
                        values=missing,
                    )
                )
        elif claim.claim_type is ClaimType.CALCULATED:
            if not values:
                counts["calculated_claims_not_assessed"] += 1
            elif all(_trace(v, passages) == "absent" for v in values):
                findings.append(
                    _finding(
                        "no_observable_operand",
                        claim=claim.position,
                        evidence_ids=ids,
                        values=[v.key() for v in values],
                    )
                )
    return findings


def _coverage_findings(coverage: EvidenceCoverage, status: AnswerStatus) -> list[GuardrailFinding]:
    severity = (
        FindingSeverity.WARNING if status is AnswerStatus.ANSWERED else FindingSeverity.NOTICE
    )
    findings = []
    for entity in coverage.entities:
        if entity.kind == "segment":
            continue
        companies = [entity.canonical] if entity.kind == "company" else list(entity.companies)
        periods = [entity.canonical] if entity.kind == "period" else []
        if entity.selected is LayerState.ABSENT:
            findings.append(
                _finding(
                    "absent_from_selected_evidence",
                    severity=severity,
                    companies=companies,
                    periods=periods,
                )
            )
        elif entity.selected is LayerState.PRESENT and entity.cited is LayerState.ABSENT:
            findings.append(
                _finding(
                    "selected_but_not_cited",
                    severity=severity,
                    companies=companies,
                    periods=periods,
                )
            )
    findings.extend(_finding(code) for code in coverage.question_limitations)
    return findings


def check_response(
    response: GroundedResponse,
    vocabulary: CorpusVocabulary,
    *,
    scope: QuestionScope | None = None,
) -> GuardrailReport:
    """Diagnostic findings for one resolved response. The response is only read."""
    if not isinstance(response, GroundedResponse):
        raise GuardrailError("response must be a GroundedResponse")
    if not isinstance(vocabulary, CorpusVocabulary):
        raise GuardrailError("vocabulary must be a CorpusVocabulary")
    if scope is not None and not isinstance(scope, QuestionScope):
        raise GuardrailError("scope must be a QuestionScope")
    # The scope is always derived from the response's own question. A supplied scope is
    # accepted only if it is exactly that analysis, so it can never describe another
    # question, vocabulary or analyzer version.
    question_scope = analyze_question(response.run.prepared.question, vocabulary)
    if scope is not None:
        if scope.vocabulary_id != vocabulary.vocabulary_id:
            raise GuardrailError("the question scope was built with a different vocabulary")
        if scope.analyzer_id != question_scope.analyzer_id:
            raise GuardrailError("the question scope was built with a different analyzer")
        if scope != question_scope:
            raise GuardrailError("the question scope does not match the response question")
    # A local abstention never reached the model, so its citations were not evaluated (None);
    # a model answer with no citations was evaluated and cited nothing (empty set).
    cited_ids = (
        None
        if response.origin is AnswerOrigin.LOCAL
        else {c.evidence_id for c in response.citations}
    )
    coverage = evidence_coverage(question_scope, response.run.prepared.evidence.records, cited_ids)
    counts = dict.fromkeys(
        (
            "values_matched",
            "values_number_only",
            "values_absent",
            "calculated_claims_not_assessed",
            "claim_periods_not_assessed",
            "segment_attribution_unverifiable",
        ),
        0,
    )
    findings = _claim_findings(response, vocabulary, counts)
    findings += _coverage_findings(coverage, response.status)
    return GuardrailReport(
        findings=tuple(findings),
        coverage=coverage,
        question_scope=question_scope,
        diagnostics=tuple(sorted(counts.items())),
        policy_id=guardrail_policy_id(vocabulary),
        vocabulary_id=vocabulary.vocabulary_id,
        analyzer_id=question_scope.analyzer_id,
        response_contract_id=response.response_contract_id,
    )


class GuardedRunner:
    """Optional composition: frozen service, then resolver, then diagnostics."""

    def __init__(
        self,
        service: GroundedGenerationService,
        resolver: CitationResolver,
        vocabulary: CorpusVocabulary,
    ) -> None:
        self._service = service
        self._resolver = resolver
        self._vocabulary = vocabulary

    def answer(self, question: str) -> GuardedResponse:
        response = self._resolver.resolve(self._service.answer(question))
        return GuardedResponse(response, check_response(response, self._vocabulary))


# --------------------------------------------------------------------------- #
# Offline smoke with a SYNTHETIC scripted provider (no model call)
# --------------------------------------------------------------------------- #

Builder = Callable[[list[dict[str, Any]]], dict[str, Any]]
_FIRST_AMOUNT = re.compile(r"\$\d+(?:\.\d+)?\s(?:billion|million)")


class ScriptedProvider:
    """Builds a scripted answer from the request's evidence. Not a model."""

    def __init__(self, builder: Builder) -> None:
        from founderlens.generation.config import DEFAULT_GENERATION_CONFIG as config

        self._builder = builder
        self._identity = ProviderIdentity(
            "synthetic-scripted", "none", config.model, config.config_id
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def generate_text(self, request: GenerationRequest) -> GenerationResult[None]:
        raise GenerationError("structured generation only")

    def generate_structured(
        self, request: GenerationRequest, schema: type[SchemaT]
    ) -> GenerationResult[SchemaT]:
        evidence = json.loads(request.input_text)["evidence"]
        payload = json.dumps(self._builder(evidence))
        return GenerationResult(
            text=payload,
            parsed=schema.model_validate_json(payload),
            status="completed",
            requested_model=self._identity.model,
            response_model="synthetic-scripted (no model call)",
            response_id="synthetic",
            request_id=None,
            usage=TokenUsage(),
            latency_ms=0.0,
            provider=self._identity,
        )

    def close(self) -> None:
        return None


def _pick(
    evidence: list[dict[str, Any]], company: str, period: str | None = None
) -> dict[str, Any]:
    for record in evidence:
        if record["company"].startswith(company) and (
            period is None or record["reporting_period"] == period
        ):
            return record
    raise GenerationError("the scripted scenario found no matching evidence")


def _amount(record: dict[str, Any]) -> str:
    match = _FIRST_AMOUNT.search(record["text"])
    return match.group(0) if match else "an amount"


def _claim(text: str, kind: str, *records: dict[str, Any]) -> dict[str, Any]:
    return {"text": text, "claim_type": kind, "evidence_ids": [r["evidence_id"] for r in records]}


def _answer(
    status: str, claims: list[dict[str, Any]], limitations: list[str] | None = None
) -> dict[str, Any]:
    return {"status": status, "claims": claims, "limitations": limitations or []}


SMOKE_SCENARIOS: tuple[tuple[str, str, str, Builder], ...] = (
    (
        "single_company_factual",
        "expected: no attribution findings",
        "What revenue did NVIDIA report for the fourth quarter of fiscal 2025?",
        lambda ev: _answer(
            "answered",
            [
                _claim(
                    "SYNTHETIC: NVIDIA reported revenue of "
                    f"{_amount(_pick(ev, 'NVIDIA', 'FY2025 Q4'))} in FY2025 Q4.",
                    "reported",
                    _pick(ev, "NVIDIA", "FY2025 Q4"),
                )
            ],
        ),
    ),
    (
        "cross_company_one_cited",
        "expected: company_attribution_gap for Alphabet",
        "Compare NVIDIA and Alphabet revenue.",
        lambda ev: _answer(
            "answered",
            [
                _claim(
                    "SYNTHETIC: NVIDIA and Alphabet both reported revenue growth.",
                    "synthesis",
                    _pick(ev, "NVIDIA"),
                )
            ],
        ),
    ),
    (
        "cross_company_both_cited",
        "expected: no company attribution findings",
        "Compare NVIDIA and Alphabet revenue.",
        lambda ev: _answer(
            "answered",
            [
                _claim(
                    "SYNTHETIC: NVIDIA and Alphabet both reported revenue growth.",
                    "synthesis",
                    _pick(ev, "NVIDIA"),
                    _pick(ev, "Alphabet"),
                )
            ],
        ),
    ),
    (
        "temporal_comparison",
        "expected: no period findings",
        "Compare NVIDIA revenue in FY2025 Q3 and FY2025 Q4.",
        lambda ev: _answer(
            "answered",
            [
                _claim(
                    "SYNTHETIC: NVIDIA reported FY2025 Q3 revenue.",
                    "reported",
                    _pick(ev, "NVIDIA", "FY2025 Q3"),
                ),
                _claim(
                    "SYNTHETIC: NVIDIA reported FY2025 Q4 revenue.",
                    "reported",
                    _pick(ev, "NVIDIA", "FY2025 Q4"),
                ),
            ],
        ),
    ),
    (
        "unsupported_period",
        "expected: scope_unresolved notices (FY2019 not a manifest period)",
        "What was NVIDIA revenue in FY2019?",
        lambda ev: _answer(
            "insufficient_evidence", [], ["SYNTHETIC: no FY2019 figures in the supplied evidence."]
        ),
    ),
    (
        "ambiguous_period",
        "expected: basis_unspecified notice",
        "What was Alphabet revenue in Q3 2024?",
        lambda ev: _answer(
            "answered",
            [
                _claim(
                    "SYNTHETIC: Alphabet reported Q3 2024 revenue.",
                    "reported",
                    _pick(ev, "Alphabet", "FY2024 Q3"),
                )
            ],
        ),
    ),
    (
        "mismatched_number",
        "expected: numeric_trace_gap for $99.9 billion",
        "What revenue did NVIDIA report for the fourth quarter of fiscal 2025?",
        lambda ev: _answer(
            "answered",
            [
                _claim(
                    "SYNTHETIC: NVIDIA reported revenue of $99.9 billion in FY2025 Q4.",
                    "reported",
                    _pick(ev, "NVIDIA", "FY2025 Q4"),
                )
            ],
        ),
    ),
)


def finding_label(finding: GuardrailFinding) -> str:
    """Compact code label, e.g. ``warning:named_company_not_cited[Alphabet Inc.]``."""
    tags = finding.companies + finding.periods + finding.values
    return f"{finding.severity.value}:{finding.detail}" + (f"[{','.join(tags)}]" if tags else "")


def run_guardrail_smoke(settings: Settings) -> list[tuple[str, str, GuardedResponse]]:
    """Each SMOKE_SCENARIOS entry through the frozen pipeline and the checks. No LLM."""
    from founderlens.generation.token_budget import load_generation_tokenizer
    from founderlens.retrieval.policy import load_policy_retriever

    manifest = load_manifest(settings.corpus_manifest_path)
    vocabulary = CorpusVocabulary.from_manifest(manifest)
    retriever = load_policy_retriever(settings)
    counter = load_generation_tokenizer(settings)
    resolver = CitationResolver(retriever.collection, manifest)
    results = []
    for name, expectation, question, builder in SMOKE_SCENARIOS:
        service = GroundedGenerationService(retriever, counter, ScriptedProvider(builder))
        results.append(
            (name, expectation, GuardedRunner(service, resolver, vocabulary).answer(question))
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.guardrails.response_checks",
        description="Guardrail diagnostics on SYNTHETIC answers over the real corpus (no LLM).",
    )
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.project_root)
        results = run_guardrail_smoke(settings)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ManifestError, GenerationError, GuardrailError) as exc:
        print(f"Guardrail smoke FAILED: {exc}", file=sys.stderr)
        return 1
    report = results[0][2].report
    print(f"guardrail policy {report.policy_id}; response contract {report.response_contract_id}")
    print("answers are SYNTHETIC (scripted from the retrieved evidence); no model was called")
    for name, expectation, guarded in results:
        row = {
            "scenario": name,
            "expectation": expectation,
            "status": guarded.response.status.value,
            "findings": [finding_label(f) for f in guarded.report.findings],
            "diagnostics": dict(guarded.report.diagnostics),
        }
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
