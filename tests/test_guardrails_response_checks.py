"""Deterministic financial response guardrails (Commit 6.2), offline.

A small synthetic NVIDIA/Alphabet collection (not the real corpus), the real
BM25 policy retriever, context builder, service and citation resolver, and a
scripted provider that cites evidence by document. No model, key, tokenizer
asset, network or report is used.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from datetime import date
from typing import Any

import pytest

from founderlens.corpus.manifest import CorpusManifest
from founderlens.generation.citations import (
    CitationResolutionError,
    CitationResolver,
    GroundedResponse,
    response_contract_id,
)
from founderlens.generation.config import DEFAULT_GENERATION_CONFIG
from founderlens.generation.errors import GenerationError
from founderlens.generation.schema import (
    GenerationRequest,
    GenerationResult,
    ProviderIdentity,
    SchemaT,
    TokenUsage,
)
from founderlens.generation.service import AnswerOrigin, GroundedGenerationService
from founderlens.guardrails import response_checks as rc
from founderlens.guardrails.response_checks import (
    FindingCategory,
    FindingSeverity,
    GuardedRunner,
    GuardrailError,
    check_response,
    guardrail_policy_id,
    policy_definition,
)
from founderlens.guardrails.scope import (
    CorpusVocabulary,
    LayerState,
    QuestionScope,
    analyze_question,
    analyzer_id,
    evidence_coverage,
)
from founderlens.ingestion.chunk import ChunkRecord, chunk_page
from founderlens.ingestion.normalize import NormalizedPageRecord, page_warnings
from founderlens.ingestion.schema import make_page_id, status_for_text
from founderlens.retrieval.bm25 import BM25Config, BM25Retriever
from founderlens.retrieval.collections import CHUNKING_V2_CONFIG
from founderlens.retrieval.policy import SELECTED_POLICY, PolicyRetriever
from founderlens.retrieval.schema import ChunkCollection
from tests.test_generation_context import CHARS, Words
from tests.test_generation_service import SERVICE_CONFIG
from tests.test_guardrails_scope import _doc

N3, N4, G3, G4 = "nvda-fy2025-q3", "nvda-fy2025-q4", "goog-2024-q3", "goog-2024-q4"
NVIDIA, ALPHABET = "NVIDIA Corporation", "Alphabet Inc."
MANIFEST = CorpusManifest(
    "synthetic",
    "0.2.0",
    "synthetic guardrail corpus",
    (
        _doc(G3, ALPHABET, "GOOG", "FY2024 Q3", date(2024, 9, 30)),
        _doc(G4, ALPHABET, "GOOG", "FY2024 Q4", date(2024, 12, 31)),
        _doc(N3, NVIDIA, "NVDA", "FY2025 Q3", date(2024, 10, 27)),
        _doc(N4, NVIDIA, "NVDA", "FY2025 Q4", date(2025, 1, 26)),
    ),
)
VOCAB = CorpusVocabulary.from_manifest(MANIFEST)
# Alphabet pages avoid the words "Alphabet", "NVIDIA" and "and" so that a question naming
# only those words retrieves NVIDIA evidence alone. Every page contains "revenue".
PAGES: dict[tuple[str, int], str] = {
    (N4, 1): "NVIDIA revenue for the fourth quarter was $39.3 billion, up 78% from $22.1 billion "
    "a year ago. Data Center revenue was $35.6 billion. SENTINEL-PASSAGE",
    (N4, 2): "NVIDIA revenue table in millions: 39,331 compared with 22,103.",
    (N3, 1): "NVIDIA revenue for the third quarter was $35.1 billion.",
    (G4, 1): "Consolidated revenue was $96.5 billion. Google Cloud revenue was $12.0 billion.",
    (G3, 1): "Consolidated revenue was $88.3 billion for the quarter.",
}


def _collection() -> ChunkCollection:
    chunks: list[ChunkRecord] = []
    for (document_id, number), text in PAGES.items():
        entry = MANIFEST.get(document_id)
        assert entry is not None
        status = status_for_text(text)
        page = NormalizedPageRecord(
            document_id=document_id,
            document_sha256=entry.sha256,
            company=entry.company,
            fiscal_period=entry.fiscal_period,
            source_url=entry.source_url,
            page_number=number,
            page_id=make_page_id(document_id, number),
            extraction_status=status,
            raw_character_count=len(text),
            normalized_text=text,
            normalized_character_count=len(text),
            tables=(),
            table_text_overlap=False,
            normalization_warnings=page_warnings(status, ()),
        )
        chunks.extend(chunk_page(page, CHUNKING_V2_CONFIG, Words()))
    return ChunkCollection(tuple(chunks), "e" * 64, chunks[0].chunking_config_id, "f" * 64)


COLLECTION = _collection()
RETRIEVER = PolicyRetriever(
    SELECTED_POLICY, COLLECTION, BM25Retriever(COLLECTION.chunks, BM25Config(top_k=10))
)
RESOLVER = CitationResolver(COLLECTION, MANIFEST)
Claim = tuple[str, str, list[str]]  # text, claim type, "document_id[:page]" (page 1 default)


class Scripted:
    """Cites evidence by document (and optionally page); counts provider calls."""

    def __init__(self, status: str, claims: list[Claim], limitations: list[str] | None = None):
        self.status, self.claims, self.limitations = status, claims, limitations or []
        self.calls = 0
        self._identity = ProviderIdentity(
            "fake", "0", DEFAULT_GENERATION_CONFIG.model, DEFAULT_GENERATION_CONFIG.config_id
        )

    @property
    def identity(self) -> ProviderIdentity:
        return self._identity

    def generate_text(self, request: GenerationRequest) -> GenerationResult[None]:
        raise AssertionError("not used")

    def generate_structured(
        self, request: GenerationRequest, schema: type[SchemaT]
    ) -> GenerationResult[SchemaT]:
        self.calls += 1
        evidence = json.loads(request.input_text)["evidence"]

        def ids(refs: list[str]) -> list[str]:
            found = []
            for ref in refs:
                doc, _, page = ref.partition(":")
                found.append(
                    next(
                        e["evidence_id"]
                        for e in evidence
                        if e["document_id"] == doc and e["page_number"] == int(page or 1)
                    )
                )
            return found

        payload = json.dumps(
            {
                "status": self.status,
                "claims": [
                    {"text": t, "claim_type": k, "evidence_ids": ids(r)} for t, k, r in self.claims
                ],
                "limitations": self.limitations,
            }
        )
        return GenerationResult(
            text=payload,
            parsed=schema.model_validate_json(payload),
            status="completed",
            requested_model=self._identity.model,
            response_model="fake",
            response_id="fake",
            request_id=None,
            usage=TokenUsage(),
            latency_ms=0.0,
            provider=self._identity,
        )

    def close(self) -> None:
        return None


def _service(provider: Scripted) -> GroundedGenerationService:
    return GroundedGenerationService(RETRIEVER, CHARS, provider, context_config=SERVICE_CONFIG)


def _respond(
    question: str,
    claims: list[Claim],
    status: str = "answered",
    limitations: list[str] | None = None,
) -> GroundedResponse:
    return RESOLVER.resolve(_service(Scripted(status, claims, limitations)).answer(question))


def _labels(
    question: str, claims: list[Claim], status: str = "answered", **kwargs: Any
) -> list[str]:
    report = check_response(_respond(question, claims, status, **kwargs), VOCAB)
    return [rc.finding_label(f) for f in report.findings]


def _report(question: str, claims: list[Claim], status: str = "answered") -> rc.GuardrailReport:
    return check_response(_respond(question, claims, status), VOCAB)


def report_claim_ids(report: rc.GuardrailReport) -> tuple[str, ...]:
    """Evidence IDs the findings name are request-scoped labels such as E01."""
    ids = {i for f in report.findings for i in f.evidence_ids}
    assert ids
    assert all(i.startswith("E") for i in ids)
    assert all(len(i) == 3 for i in ids)
    return tuple(sorted(ids))


Q = "What was NVIDIA revenue?"


# --- company attribution --------------------------------------------------------------------


def test_correct_single_company_claim_has_no_findings() -> None:
    assert _labels(Q, [("NVIDIA revenue was $39.3 billion.", "reported", [N4])]) == []


def test_wrong_company_citation_is_a_warning_for_that_claim() -> None:
    report = _report(
        "What was Alphabet revenue?",
        [("Alphabet revenue was $39.3 billion.", "reported", [N4])],
    )
    attribution = [
        f for f in report.findings if f.category is FindingCategory.COMPANY_ATTRIBUTION_GAP
    ]
    assert [(f.detail, f.severity, f.claim_position, f.companies) for f in attribution] == [
        ("named_company_not_cited", FindingSeverity.WARNING, 1, (ALPHABET,))
    ]
    assert attribution[0].evidence_ids == report_claim_ids(report)


def test_attribution_uses_each_claims_own_citations() -> None:
    labels = _labels(
        "Compare NVIDIA and Alphabet revenue.",
        [
            ("NVIDIA revenue was $39.3 billion.", "reported", [N4]),
            ("Alphabet revenue was $96.5 billion.", "reported", [N4]),  # cites the wrong one
            ("Consolidated revenue was $96.5 billion.", "reported", [G4]),
        ],
    )
    assert "warning:named_company_not_cited[Alphabet Inc.]" in labels


def test_two_company_claim_with_both_companies_cited() -> None:
    claim = "NVIDIA and Alphabet reported revenue of $39.3 billion and $96.5 billion."
    assert _labels("Compare NVIDIA and Alphabet revenue.", [(claim, "reported", [N4, G4])]) == []
    one = _labels("Compare NVIDIA and Alphabet revenue.", [(claim, "reported", [N4])])
    assert "warning:named_company_not_cited[Alphabet Inc.]" in one


def test_segment_needs_its_parent_and_is_never_the_consolidated_company() -> None:
    ok = _report(
        "What was Google Cloud revenue?",
        [("Google Cloud revenue was $12.0 billion.", "reported", [G4])],
    )
    assert [rc.finding_label(f) for f in ok.findings] == ["notice:unassessable_references"]
    assert dict(ok.diagnostics)["segment_attribution_unverifiable"] == 1
    wrong = _labels(Q, [("Google Cloud revenue was $35.6 billion.", "reported", [N4])])
    assert "warning:named_segment_parent_not_cited[Alphabet Inc.]" in wrong
    brand = _labels(Q, [("Google revenue was $35.6 billion.", "reported", [N4])])
    assert "warning:named_brand_company_not_cited[Alphabet Inc.]" in brand


def test_generic_terms_are_never_companies() -> None:
    claim = "The company and its segments reported revenue of $39.3 billion."
    assert _labels(Q, [(claim, "synthesis", [N4])]) == []


# --- period attribution ---------------------------------------------------------------------


def test_correct_period_has_no_finding() -> None:
    claims: list[Claim] = [
        ("NVIDIA revenue in FY2025 Q4 was $39.3 billion.", "reported", [N4]),
        ("NVIDIA revenue in FY2025 Q3 was $35.1 billion.", "reported", [N3]),
    ]
    assert _labels("Compare NVIDIA revenue in FY2025 Q3 and FY2025 Q4.", claims) == []


def test_suspected_period_mismatch_is_only_a_notice() -> None:
    report = _report(Q, [("NVIDIA revenue in FY2025 Q3 was $39.3 billion.", "reported", [N4])])
    period = [f for f in report.findings if f.category is FindingCategory.PERIOD_ATTRIBUTION_GAP]
    assert [(f.detail, f.severity, f.periods) for f in period] == [
        ("named_period_not_cited", FindingSeverity.NOTICE, ("FY2025 Q3",))
    ]


def test_prior_year_comparative_inside_a_newer_release_is_not_assessed() -> None:
    claim = "NVIDIA revenue in FY2024 Q4 was $22.1 billion."  # comparative in the FY2025 Q4 release
    report = _report(Q, [(claim, "reported", [N4])])
    assert report.findings == ()
    assert dict(report.diagnostics)["claim_periods_not_assessed"] == 1


def test_ambiguous_fiscal_or_calendar_period_uses_calendar_candidates() -> None:
    ok = _labels(
        "What was Alphabet revenue in Q4 2024?",
        [("Alphabet revenue in Q4 2024 was $96.5 billion.", "reported", [G4])],
    )
    assert ok == ["notice:basis_unspecified_periods"]
    off = _labels(
        "What was Alphabet revenue?",
        [("Alphabet revenue in Q4 2024 was $88.3 billion.", "reported", [G3])],
    )
    assert off == ["notice:basis_unspecified_period_not_cited[Q4 2024 (basis unspecified)]"]


# --- question coverage ----------------------------------------------------------------------


def test_two_company_question_with_one_company_missing() -> None:
    absent = _report(
        "Compare NVIDIA and Alphabet.",
        [("NVIDIA revenue was $39.3 billion.", "reported", [N4])],
    )
    assert [rc.finding_label(f) for f in absent.findings] == [
        "warning:absent_from_selected_evidence[Alphabet Inc.]"
    ]
    uncited = _labels(
        "Compare NVIDIA and Alphabet revenue.",
        [("NVIDIA revenue was $39.3 billion.", "reported", [N4])],
        status="partially_answered",
        limitations=["Alphabet figures not used."],
    )
    assert uncited == ["notice:selected_but_not_cited[Alphabet Inc.]"]


def test_coverage_never_changes_the_status() -> None:
    response = _respond(
        "Compare NVIDIA and Alphabet.", [("NVIDIA revenue was $39.3 billion.", "reported", [N4])]
    )
    check_response(response, VOCAB)
    assert response.status.value == "answered"


def test_empty_citations_differ_from_not_evaluated() -> None:
    model = _respond(Q, [], status="insufficient_evidence", limitations=["Not stated."])
    report = check_response(model, VOCAB)
    assert model.origin is AnswerOrigin.MODEL
    assert report.coverage.cited_count == 0
    assert {e.cited for e in report.coverage.entities} == {LayerState.ABSENT}
    assert [rc.finding_label(f) for f in report.findings] == [
        "notice:selected_but_not_cited[NVIDIA Corporation]"
    ]
    local = RESOLVER.resolve(_service(Scripted("answered", [])).answer("Alphabet zyzzyva"))
    local_report = check_response(local, VOCAB)
    assert local.origin is AnswerOrigin.LOCAL
    assert local_report.coverage.cited_count is None
    assert {e.cited for e in local_report.coverage.entities} == {LayerState.NOT_EVALUATED}
    assert [rc.finding_label(f) for f in local_report.findings] == [
        "notice:absent_from_selected_evidence[Alphabet Inc.]"
    ]


def test_no_entities_and_unresolved_scope() -> None:
    assert _labels("What was revenue?", [("Revenue was $39.3 billion.", "reported", [N4])]) == []
    unresolved = _labels(
        "What was NVIDIA revenue last quarter?",
        [("NVIDIA revenue was $39.3 billion.", "reported", [N4])],
    )
    assert unresolved == ["notice:unresolved_references"]
    unknown = _labels(
        "What was NVIDIA revenue in FY2019?",
        [],
        status="insufficient_evidence",
        limitations=["Not stated."],
    )
    assert "notice:unassessable_references" in unknown


# --- numeric trace --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "claim",
    [
        "NVIDIA revenue was $39.3 billion.",
        "NVIDIA revenue was $39.30 billion.",
        "NVIDIA revenue was 39.3 billion dollars.",
        "NVIDIA revenue was $39.3bn.",
        "NVIDIA revenue was $39.3B.",
        "NVIDIA revenue grew 78 percent.",
        "NVIDIA revenue grew 78.0%.",
    ],
)
def test_harmless_formatting_differences_match(claim: str) -> None:
    report = _report(Q, [(claim, "reported", [N4])])
    assert report.findings == ()
    assert dict(report.diagnostics)["values_matched"] == 1


def test_thousands_separators_match_table_values_without_verifying_the_unit() -> None:
    report = _report(Q, [("NVIDIA revenue was $39,331 million.", "reported", [f"{N4}:2"])])
    assert report.findings == ()
    assert dict(report.diagnostics)["values_number_only"] == 1
    plain = _report(Q, [("NVIDIA revenue was 39,331.00 in the table.", "reported", [f"{N4}:2"])])
    assert dict(plain.diagnostics)["values_matched"] == 1


def test_missing_value_is_a_notice_with_the_normalized_value() -> None:
    report = _report(Q, [("NVIDIA revenue was $99.9 billion.", "reported", [N4])])
    assert [(f.detail, f.severity, f.values) for f in report.findings] == [
        ("value_not_found_verbatim", FindingSeverity.NOTICE, ("99.9billion",))
    ]


@pytest.mark.parametrize(
    "claim",
    [
        "NVIDIA revenue was $39 billion.",  # rounded
        "NVIDIA revenue was $39,300 million.",  # unit-converted
        "NVIDIA revenue was 39.3%.",  # percent never equals an amount
    ],
)
def test_rounded_or_converted_values_stay_uncertain_notices(claim: str) -> None:
    findings = _report(Q, [(claim, "reported", [N4])]).findings
    assert [(f.detail, f.severity) for f in findings] == [
        ("value_not_found_verbatim", FindingSeverity.NOTICE)
    ]


def test_years_dates_pages_and_citation_numbers_are_not_values() -> None:
    claim = "In fiscal 2025, ended January 26, 2025, NVIDIA (page 2, E01, [1]) grew in FY2025 Q4."
    report = _report(Q, [(claim, "reported", [N4])])
    assert report.findings == ()
    assert sum(v for k, v in report.diagnostics if k.startswith("values_")) == 0


def test_detached_single_letters_are_not_scales() -> None:
    claim = "NVIDIA revenue is described in note 5 b of the release."
    report = _report(Q, [(claim, "reported", [N4])])
    assert report.findings == ()
    assert sum(v for k, v in report.diagnostics if k.startswith("values_")) == 0


def test_only_the_claims_own_passages_are_searched() -> None:
    labels = _labels(Q, [("NVIDIA revenue was $35.1 billion.", "reported", [N4])])
    assert labels == ["notice:value_not_found_verbatim[35.1billion]"]  # $35.1bn is on N3 only


def test_calculated_claims_report_only_missing_observable_operands() -> None:
    with_operands = _labels(
        Q, [("$39.3 billion versus $22.1 billion is growth of 77.8%.", "calculated", [N4])]
    )
    assert with_operands == []
    none = _report(Q, [("NVIDIA revenue grew 12.5% over two years.", "calculated", [N4])])
    assert [(f.detail, f.values) for f in none.findings] == [("no_observable_operand", ("12.5%",))]
    unassessed = _report(Q, [("NVIDIA revenue grew strongly.", "calculated", [N4])])
    assert unassessed.findings == ()
    assert dict(unassessed.diagnostics)["calculated_claims_not_assessed"] == 1


def test_synthesis_claims_are_not_numerically_traced() -> None:
    assert _labels(Q, [("NVIDIA revenue was about $40 billion.", "synthesis", [N4])]) == []


# --- preservation, identity and contracts ---------------------------------------------------


def test_report_preserves_the_response_and_adds_no_provider_call() -> None:
    provider = Scripted("answered", [("Alphabet revenue was $99.9 billion.", "reported", [N4])])
    guarded = GuardedRunner(_service(provider), RESOLVER, VOCAB).answer(Q)
    assert provider.calls == 1
    baseline = RESOLVER.resolve(_service(Scripted("answered", provider.claims)).answer(Q))
    response = guarded.response
    for name in (
        "status",
        "claims",
        "citations",
        "limitations",
        "origin",
        "local_abstention_reason",
    ):
        assert getattr(response, name) == getattr(baseline, name)
    assert response.render() == baseline.render()

    def timeless(metadata: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in metadata.items() if k != "timings"}  # wall-clock only

    assert timeless(response.metadata()) == timeless(baseline.metadata())
    assert response.usage() == baseline.usage()
    assert response.identities() == baseline.identities()
    assert guarded.report.response_contract_id == response_contract_id() == "debe65096147"
    assert guarded.report.findings  # the problems were reported, not repaired


def test_reports_are_deterministic_immutable_and_text_free() -> None:
    question = "Compare NVIDIA and Alphabet revenue SENTINEL-QUESTION last quarter."
    claims: list[Claim] = [("SENTINEL-CLAIM Alphabet revenue was $99.9 billion.", "reported", [N4])]
    response = _respond(question, claims, "partially_answered", limitations=["SENTINEL-LIMITATION"])
    first, second = check_response(response, VOCAB), check_response(response, VOCAB)
    assert first == second
    assert first.metadata() == second.metadata()
    text = json.dumps(first.metadata())
    for sentinel in ("SENTINEL", "last quarter", "Compare", "revenue was"):
        assert sentinel not in text
    assert first.metadata()["enforcement"] == "diagnostic_only"
    with pytest.raises(FrozenInstanceError):
        first.findings[0].detail = "x"  # type: ignore[misc]  # frozen
    with pytest.raises(FrozenInstanceError):
        first.policy_id = "x"  # type: ignore[misc]  # frozen
    assert isinstance(first.findings, tuple)
    assert "hallucinat" not in text
    assert "fail" not in text


def test_policy_identity_is_versioned_and_separate() -> None:
    definition = policy_definition(VOCAB)
    assert definition["enforcement"] == "diagnostic_only"
    assert definition["vocabulary_id"] == VOCAB.vocabulary_id
    assert definition["analyzer_id"] == analyzer_id()
    policy = guardrail_policy_id(VOCAB)
    assert policy == guardrail_policy_id(VOCAB)
    assert len(policy) == 12
    assert policy != response_contract_id()
    other = CorpusVocabulary.from_manifest(replace(MANIFEST, corpus_version="0.3.0"))
    assert guardrail_policy_id(other) != policy
    text = json.dumps(definition)
    for forbidden in ("Users", "founderlens-ai-rag-mvp", ".jsonl", "SENTINEL", "202611"):
        assert forbidden not in text  # no paths, files, questions or timestamps


def test_malformed_inputs_raise_explicit_errors() -> None:
    response = _respond(Q, [("NVIDIA revenue was $39.3 billion.", "reported", [N4])])
    with pytest.raises(GuardrailError, match="GroundedResponse"):
        check_response(object(), VOCAB)  # type: ignore[arg-type]  # deliberately malformed
    with pytest.raises(GuardrailError, match="CorpusVocabulary"):
        check_response(response, object())  # type: ignore[arg-type]  # deliberately malformed
    other = CorpusVocabulary.from_manifest(replace(MANIFEST, corpus_version="0.3.0"))
    with pytest.raises(GuardrailError, match="QuestionScope"):
        check_response(response, VOCAB, scope=object())  # type: ignore[arg-type]  # malformed
    with pytest.raises(GuardrailError, match="different vocabulary"):
        check_response(response, VOCAB, scope=analyze_question(Q, other))
    orphan = replace(response, citations=())
    with pytest.raises(GuardrailError, match="unresolved evidence ID"):
        check_response(orphan, VOCAB)


def test_supplied_scope_must_be_the_responses_own_analysis() -> None:
    response = _respond(Q, [("NVIDIA revenue was $39.3 billion.", "reported", [N4])])
    own = analyze_question(response.run.prepared.question, VOCAB)
    assert check_response(response, VOCAB, scope=own) == check_response(response, VOCAB)
    unrelated = analyze_question("Compare NVIDIA and Alphabet in FY2025 Q3.", VOCAB)
    with pytest.raises(GuardrailError, match="does not match the response question") as info:
        check_response(response, VOCAB, scope=unrelated)
    assert "NVIDIA" not in str(info.value)  # no question text in the error
    same_entities = analyze_question("NVIDIA revenue?", VOCAB)  # same company, other spans
    with pytest.raises(GuardrailError, match="does not match"):
        check_response(response, VOCAB, scope=same_entities)
    other_analyzer = QuestionScope(
        own.companies, own.periods, own.unresolved, own.vocabulary_id, "000000000000"
    )
    with pytest.raises(GuardrailError, match="different analyzer"):
        check_response(response, VOCAB, scope=other_analyzer)
    other_vocab = CorpusVocabulary.from_manifest(replace(MANIFEST, corpus_version="0.3.0"))
    with pytest.raises(GuardrailError, match="different vocabulary"):
        check_response(response, VOCAB, scope=analyze_question(Q, other_vocab))
    assert check_response(response, VOCAB).question_scope == own


class _Counting:
    """Wraps a service or resolver and counts the calls it forwards."""

    def __init__(self, inner: Any, method: str) -> None:
        self.inner, self.method, self.calls = inner, method, 0

    def __getattr__(self, name: str) -> Any:
        if name != self.method:
            return getattr(self.inner, name)

        def forward(*args: Any) -> Any:
            self.calls += 1
            return getattr(self.inner, name)(*args)

        return forward


def test_runner_calls_each_stage_once_and_keeps_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searches: list[str] = []
    original = PolicyRetriever.search

    def counted(self: PolicyRetriever, query: str) -> Any:
        searches.append(query)
        return original(self, query)

    monkeypatch.setattr(PolicyRetriever, "search", counted)
    claims: list[Claim] = [("NVIDIA revenue was $39.3 billion.", "reported", [N4])]
    provider = Scripted("answered", claims)
    service = _Counting(_service(provider), "answer")
    resolver = _Counting(RESOLVER, "resolve")
    guarded = GuardedRunner(service, resolver, VOCAB).answer(Q)  # type: ignore[arg-type]  # spies
    assert (service.calls, resolver.calls, provider.calls, len(searches)) == (1, 1, 1, 1)
    assert guarded.report == check_response(guarded.response, VOCAB)
    assert guarded.report.question_scope == analyze_question(Q, VOCAB)
    assert guarded.response.claims[0].text == claims[0][0]
    assert len(searches) == 1  # the check itself did not retrieve


class _Failing(Scripted):
    def generate_structured(self, request: GenerationRequest, schema: type[SchemaT]) -> Any:
        raise GenerationError("synthetic provider failure")


def test_runner_propagates_failures_and_never_fabricates_a_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = GuardedRunner(_service(_Failing("answered", [])), RESOLVER, VOCAB)
    with pytest.raises(GenerationError):
        runner.answer(Q)
    other = CorpusManifest("other", "9.9.9", "other", MANIFEST.documents)
    bad_resolver = CitationResolver(COLLECTION, replace(other, documents=MANIFEST.documents[:1]))
    claims: list[Claim] = [("NVIDIA revenue was $39.3 billion.", "reported", [N4])]
    with pytest.raises(CitationResolutionError, match="not in the corpus manifest"):
        GuardedRunner(_service(Scripted("answered", claims)), bad_resolver, VOCAB).answer(Q)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise GuardrailError("synthetic guardrail failure")

    monkeypatch.setattr(rc, "check_response", broken)
    with pytest.raises(GuardrailError):
        GuardedRunner(_service(Scripted("answered", claims)), RESOLVER, VOCAB).answer(Q)


def test_findings_reference_original_claims_and_current_evidence() -> None:
    claims: list[Claim] = [
        ("NVIDIA revenue was $39.3 billion.", "reported", [N4]),
        ("Alphabet revenue was $99.9 billion.", "reported", [N4]),
        ("NVIDIA revenue in FY2025 Q3 was $35.1 billion.", "reported", [N3]),
    ]
    response = _respond("Compare NVIDIA and Alphabet revenue.", claims)
    report = check_response(response, VOCAB)
    by_position = {c.position: c for c in response.claims}
    known = {c.company for c in VOCAB.companies}
    current = set(response.run.prepared.evidence.ids)
    claim_findings = [f for f in report.findings if f.claim_position is not None]
    assert {f.claim_position for f in claim_findings} == {2}
    for finding in claim_findings:
        assert finding.claim_position is not None
        assert finding.evidence_ids == by_position[finding.claim_position].evidence_ids
    for finding in report.findings:
        assert finding.detail in rc.DETAIL_CODES
        assert finding.category is rc.DETAIL_CODES[finding.detail][0]
        assert set(finding.companies) <= known
        assert set(finding.evidence_ids) <= current
    assert dict(report.diagnostics)["values_matched"] == 2


def test_number_only_is_reported_separately_from_matches() -> None:
    table = _report(Q, [("NVIDIA revenue was $39,331 million.", "reported", [f"{N4}:2"])])
    exact = _report(Q, [("NVIDIA revenue was $39.3 billion.", "reported", [N4])])
    assert (
        dict(table.diagnostics)["values_number_only"],
        dict(table.diagnostics)["values_matched"],
    ) == (1, 0)
    assert (
        dict(exact.diagnostics)["values_number_only"],
        dict(exact.diagnostics)["values_matched"],
    ) == (0, 1)
    assert table.findings == exact.findings == ()
    caveat = table.metadata()["caveat"]
    assert "not verification" in caveat


def test_explicit_scope_matches_the_default() -> None:
    response = _respond(Q, [("NVIDIA revenue was $39.3 billion.", "reported", [N4])])
    scope = analyze_question(response.run.prepared.question, VOCAB)
    assert check_response(response, VOCAB, scope=scope) == check_response(response, VOCAB)
    expected = evidence_coverage(
        scope, response.run.prepared.evidence.records, {c.evidence_id for c in response.citations}
    )
    assert check_response(response, VOCAB).coverage == expected
