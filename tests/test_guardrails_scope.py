"""Corpus vocabulary, question scope and evidence coverage (Commit 6.1), offline.

Synthetic manifests and synthetic questions only; evidence records come from
the synthetic V2 collection used by the generation tests. No model, network,
tokenizer asset or evaluation file.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from founderlens.corpus.manifest import CorpusManifest, DocumentEntry
from founderlens.guardrails import scope as sc
from founderlens.guardrails.scope import (
    CompanyScope,
    CorpusVocabulary,
    CoverageStatus,
    LayerState,
    ManifestStatus,
    ScopeAnalysisError,
    analyze_question,
    analyzer_id,
    evidence_coverage,
)
from tests.test_generation_context import COLLECTION, _build, _candidates


def _doc(document_id: str, company: str, ticker: str, period: str, end: date) -> DocumentEntry:
    return DocumentEntry(
        document_id=document_id,
        company=company,
        ticker=ticker,
        title=f"{company} {period} release",
        document_type="earnings_release",
        fiscal_period=period,
        period_end=end,
        publication_date=end,
        source_url=f"https://example.com/{document_id}.pdf",
        download_url=None,
        acquisition_date=end,
        filename=f"{document_id}.pdf",
        sha256="a" * 64,
        size_bytes=1,
        page_count=1,
    )


# Synthetic metadata using the approved-alias company names (not the real manifest file).
MANIFEST = CorpusManifest(
    "synthetic",
    "0.1.0",
    "synthetic",
    (
        _doc("goog-q3", "Alphabet Inc.", "GOOG", "FY2024 Q3", date(2024, 9, 30)),
        _doc("goog-q4", "Alphabet Inc.", "GOOG", "FY2024 Q4", date(2024, 12, 31)),
        _doc("nvda-q3", "NVIDIA Corporation", "NVDA", "FY2025 Q3", date(2024, 10, 27)),
        _doc("nvda-q4", "NVIDIA Corporation", "NVDA", "FY2025 Q4", date(2025, 1, 26)),
    ),
)
VOCAB = CorpusVocabulary.from_manifest(MANIFEST)
FULL_WIDTH_NVIDIA = "".join(chr(ord(c) + 0xFEE0) for c in "Nvidia")  # Unicode full-width


def _scope(question: str) -> sc.QuestionScope:
    return analyze_question(question, VOCAB)


def _companies(question: str) -> list[tuple[str, str, str | None]]:
    return [(c.company, c.scope.value, c.segment) for c in _scope(question).companies]


def _periods(question: str) -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (p.canonical, p.manifest_status.value, p.manifest_documents)
        for p in _scope(question).periods
    ]


def _unresolved(question: str) -> list[str]:
    return [u.kind for u in _scope(question).unresolved]


# --- vocabulary -------------------------------------------------------------------------------


def test_vocabulary_comes_from_the_manifest_plus_the_approved_table() -> None:
    nvda = VOCAB.company("NVIDIA Corporation")
    goog = VOCAB.company("Alphabet Inc.")
    assert nvda is not None
    assert goog is not None
    assert nvda.tickers == ("NVDA",)
    assert goog.tickers == ("GOOG", "GOOGL")
    assert goog.segments == ("Google Cloud", "Google Services")
    assert nvda.document_ids == ("nvda-q3", "nvda-q4")
    assert [(p.document_id, p.fiscal_year, p.fiscal_quarter) for p in VOCAB.periods] == [
        ("goog-q3", 2024, 3),
        ("goog-q4", 2024, 4),
        ("nvda-q3", 2025, 3),
        ("nvda-q4", 2025, 4),
    ]
    with pytest.raises(FrozenInstanceError):
        VOCAB.companies = ()  # type: ignore[misc]  # frozen
    assert isinstance(VOCAB.companies, tuple)


def test_identities_are_stable_and_track_content() -> None:
    assert VOCAB.vocabulary_id == CorpusVocabulary.from_manifest(MANIFEST).vocabulary_id
    other = CorpusVocabulary.from_manifest(
        replace(MANIFEST, documents=MANIFEST.documents[:2], corpus_version="0.2.0")
    )
    assert other.vocabulary_id != VOCAB.vocabulary_id
    assert analyzer_id() == analyzer_id()
    scope = _scope("NVIDIA")
    assert (scope.vocabulary_id, scope.analyzer_id) == (VOCAB.vocabulary_id, analyzer_id())


def test_alias_table_changes_change_the_vocabulary_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = VOCAB.vocabulary_id
    monkeypatch.setitem(sc.APPROVED_ALIASES, "NVIDIA Corporation", (("NVIDIA", False),))
    assert CorpusVocabulary.from_manifest(MANIFEST).vocabulary_id != before


def test_rule_changes_change_the_analyzer_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    before = analyzer_id()
    monkeypatch.setattr(sc, "PERIOD_RULES", sc.PERIOD_RULES[:-1])
    assert analyzer_id() != before


def test_an_alias_claimed_by_two_companies_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sc.APPROVED_ALIASES, "NVIDIA Corporation", (("Alphabet", True),))
    with pytest.raises(ScopeAnalysisError, match="more than one company"):
        CorpusVocabulary.from_manifest(MANIFEST)


# --- companies --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("NVIDIA Corporation results", [("NVIDIA Corporation", "company", None)]),
        ("nvidia results", [("NVIDIA Corporation", "company", None)]),
        ("What did NVIDIA's data center do?", [("NVIDIA Corporation", "company", None)]),
        (
            "NVDA and GOOGL",
            [("NVIDIA Corporation", "company", None), ("Alphabet Inc.", "company", None)],
        ),
        ("Alphabet Inc. revenue", [("Alphabet Inc.", "company", None)]),
        (FULL_WIDTH_NVIDIA + " results", [("NVIDIA Corporation", "company", None)]),  # NFKC
        (
            "Compare NVIDIA and Alphabet",
            [("NVIDIA Corporation", "company", None), ("Alphabet Inc.", "company", None)],
        ),
    ],
)
def test_company_recognition(question: str, expected: list[tuple[str, str, str | None]]) -> None:
    assert _companies(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "What was cloud revenue?",
        "How did the data center business and AI demand grow?",
        "Is revenue up?",
        "the alphabet of financial metrics",  # lower-case common noun
        "nvda and goog in lower case",  # tickers must be upper case
        "NVIDIAN Googleplex",  # token boundaries
        "",
    ],
)
def test_generic_terms_and_partial_tokens_are_not_companies(question: str) -> None:
    assert _scope(question).companies == ()


def test_google_segments_are_not_the_consolidated_company() -> None:
    assert _companies("What was Google Cloud revenue?") == [
        ("Alphabet Inc.", "segment", "Google Cloud")
    ]
    assert _companies("Google Services operating income") == [
        ("Alphabet Inc.", "segment", "Google Services")
    ]
    scope = _scope("How is Google doing?")
    assert [(c.company, c.scope) for c in scope.companies] == [
        ("Alphabet Inc.", CompanyScope.BRAND_AMBIGUOUS)
    ]
    assert scope.ambiguous == scope.companies
    assert _companies("a Google-owned unit") == [("Alphabet Inc.", "brand_ambiguous", None)]


# --- periods ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "canonical", "documents"),
    [
        ("FY2025 Q4 revenue", "FY2025 Q4", ("nvda-q4",)),
        ("Q4 FY2025 revenue", "FY2025 Q4", ("nvda-q4",)),
        ("revenue in fiscal 2025 fourth quarter", "FY2025 Q4", ("nvda-q4",)),
        ("the fourth quarter of fiscal 2025", "FY2025 Q4", ("nvda-q4",)),
        ("Q3 of fiscal year 2025", "FY2025 Q3", ("nvda-q3",)),
        ("FY2025", "FY2025", ("nvda-q3", "nvda-q4")),
        ("fiscal year 2024", "FY2024", ("goog-q3", "goog-q4")),
    ],
)
def test_fiscal_periods(question: str, canonical: str, documents: tuple[str, ...]) -> None:
    assert _periods(question) == [(canonical, "in_manifest_reporting_periods", documents)]


def test_calendar_periods_match_only_exact_period_ends() -> None:
    assert _periods("calendar Q3 2024") == [
        ("CY2024 Q3", "in_manifest_reporting_periods", ("goog-q3",))
    ]
    # NVIDIA's fiscal Q3 FY2025 ended 2024-10-27, so it is not calendar Q3 2024.
    assert _periods("NVIDIA calendar Q3 2024") == [
        ("CY2024 Q3", "not_in_manifest_reporting_periods", ())
    ]
    assert _periods("CY2024") == [("CY2024", "in_manifest_reporting_periods", ("goog-q4",))]
    assert _periods("fiscal Q4 2025") == [
        ("FY2025 Q4", "in_manifest_reporting_periods", ("nvda-q4",))
    ]


def test_period_end_dates() -> None:
    assert _periods("three months ended October 27, 2024") == [
        ("period ended 2024-10-27 (three_months)", "in_manifest_reporting_periods", ("nvda-q3",))
    ]
    assert _periods("quarter ended 2024-09-30") == [
        ("period ended 2024-09-30 (quarter)", "in_manifest_reporting_periods", ("goog-q3",))
    ]
    assert _periods("nine months ended September 30, 2024")[0][1] == "in_manifest_reporting_periods"
    assert _periods("quarter ended February 30, 2024") == []  # invalid date is not recognized
    assert _unresolved("as of September 30, 2024") == ["date_without_period_context"]


def test_quarter_without_basis_is_not_guessed() -> None:
    scope = _scope("Q3 2024 results")
    (period,) = scope.periods
    assert period.manifest_status is ManifestStatus.BASIS_UNSPECIFIED
    assert period.manifest_documents == ("goog-q3",)  # fiscal-label reading
    assert period.calendar_candidates == ("goog-q3",)  # calendar period-end reading
    nvda = _scope("NVIDIA third quarter 2025").periods[0]
    assert nvda.manifest_status is ManifestStatus.BASIS_UNSPECIFIED
    assert (nvda.manifest_documents, nvda.calendar_candidates) == (("nvda-q3",), ())


@pytest.mark.parametrize(
    ("question", "kinds"),
    [
        ("What happened last quarter?", ["relative_period"]),
        ("How did this year compare?", ["relative_period"]),
        ("What changed recently?", ["relative_period"]),
        ("Revenue a year ago", ["relative_period"]),
        ("FY25 results", ["two_digit_fiscal_year"]),
        ("fourth quarter revenue", ["quarter_without_year"]),
        ("Q2 revenue", ["quarter_without_year"]),
        ("What happened in 2019?", ["bare_year"]),
    ],
)
def test_ambiguous_mentions_stay_unresolved(question: str, kinds: list[str]) -> None:
    assert _unresolved(question) == kinds
    assert _scope(question).periods == ()


def test_periods_missing_from_the_manifest_are_not_out_of_corpus() -> None:
    assert _periods("What happened in FY2019?") == [
        ("FY2019", "not_in_manifest_reporting_periods", ())
    ]
    # NVIDIA's FY2024 is not a manifest period, though its FY2025 releases carry FY2024
    # comparatives: the status names manifest periods, not document contents.
    assert _periods("NVIDIA FY2024 revenue") == [
        ("FY2024", "not_in_manifest_reporting_periods", ())
    ]
    assert not hasattr(ManifestStatus, "OUT_OF_CORPUS")


def test_multiple_companies_and_periods_restrict_labels_to_named_companies() -> None:
    question = "Compare NVIDIA FY2025 Q4 with Alphabet FY2024 Q4 and FY2025 Q3"
    assert _periods(question) == [
        ("FY2025 Q4", "in_manifest_reporting_periods", ("nvda-q4",)),
        ("FY2024 Q4", "in_manifest_reporting_periods", ("goog-q4",)),
        ("FY2025 Q3", "in_manifest_reporting_periods", ("nvda-q3",)),
    ]
    assert _periods("Compare FY2024 with FY2025") == [
        ("FY2024", "in_manifest_reporting_periods", ("goog-q3", "goog-q4")),
        ("FY2025", "in_manifest_reporting_periods", ("nvda-q3", "nvda-q4")),
    ]


def test_analysis_is_deterministic_immutable_and_does_not_rewrite() -> None:
    question = "Compare NVIDIA FY2025 Q4 with Google Cloud last quarter in 2019"
    first, second = _scope(question), _scope(question)
    assert first == second
    spans = [(c.start, c.end, c.matched_text) for c in first.companies]
    spans += [(p.start, p.end, p.matched_text) for p in first.periods]
    spans += [(u.start, u.end, u.matched_text) for u in first.unresolved]
    assert all(question[start:end] == text for start, end, text in spans)
    with pytest.raises(FrozenInstanceError):
        first.periods[0].canonical = "x"  # type: ignore[misc]  # frozen
    metadata = str(first.metadata())
    assert "last quarter" not in metadata
    assert "Google Cloud" in metadata  # canonical segment name, not question text
    with pytest.raises(ScopeAnalysisError, match="must be a string"):
        analyze_question(None, VOCAB)


# --- coverage ---------------------------------------------------------------------------------

EVIDENCE_MANIFEST = CorpusManifest(
    "synthetic",
    "0.1.0",
    "synthetic",
    (
        _doc("aaa-2024-q1", "Alpha", "ALPH", "FY2024 Q1", date(2024, 3, 31)),
        _doc("bbb-2024-q4", "Beta", "BETA", "FY2024 Q4", date(2024, 12, 31)),
    ),
)
EVIDENCE_VOCAB = CorpusVocabulary.from_manifest(EVIDENCE_MANIFEST)
ALPHA_ONLY = _build(candidates=_candidates(0, 1)).evidence.records  # E01, E02: Alpha pages
BETA = next(i for i, c in enumerate(COLLECTION.chunks) if c.company == "Beta")
BOTH = _build(candidates=_candidates(0, BETA)).evidence.records  # E01 Alpha, E02 Beta


def _coverage(
    question: str, records: tuple[object, ...], cited: set[str] | None = None
) -> sc.EvidenceCoverage:
    return evidence_coverage(analyze_question(question, EVIDENCE_VOCAB), records, cited)  # type: ignore[arg-type]


def test_missing_company_in_selected_evidence_is_reported() -> None:
    coverage = _coverage("Compare Alpha and Beta", ALPHA_ONLY)
    states = {e.canonical: e.selected for e in coverage.entities}
    assert states == {"Alpha": LayerState.PRESENT, "Beta": LayerState.ABSENT}
    assert coverage.selected_reference_status is CoverageStatus.PARTIAL
    assert coverage.selected_question_status is CoverageStatus.PARTIAL
    assert coverage.cited_reference_status is None
    assert coverage.cited_question_status is None
    assert coverage.selected_count == 2


def test_selected_but_not_cited_coverage_is_distinguished() -> None:
    coverage = _coverage("Compare Alpha and Beta FY2024 Q4", BOTH, cited={"E01"})
    by_name = {e.canonical: (e.selected, e.cited) for e in coverage.entities}
    assert by_name["Beta"] == (LayerState.PRESENT, LayerState.ABSENT)
    assert by_name["FY2024 Q4"] == (LayerState.PRESENT, LayerState.ABSENT)
    assert coverage.selected_reference_status is CoverageStatus.COMPLETE
    assert coverage.selected_question_status is CoverageStatus.COMPLETE
    assert coverage.cited_reference_status is CoverageStatus.PARTIAL
    assert coverage.cited_count == 1


def test_coverage_status_edge_cases() -> None:
    nothing = _coverage("What was revenue?", BOTH)
    assert nothing.selected_reference_status is CoverageStatus.NOT_APPLICABLE
    assert nothing.selected_question_status is CoverageStatus.NOT_APPLICABLE
    unknown = _coverage("What happened in FY2019?", BOTH)
    assert unknown.selected_reference_status is CoverageStatus.UNKNOWN
    assert unknown.selected_question_status is CoverageStatus.UNKNOWN
    assert unknown.entities[0].selected is LayerState.NOT_APPLICABLE
    none = _coverage("Beta results", ALPHA_ONLY)
    assert none.selected_question_status is CoverageStatus.NONE
    empty = _coverage("Alpha results", ())
    assert (empty.selected_reference_status, empty.selected_count) == (CoverageStatus.NONE, 0)


def test_coverage_reads_inputs_without_mutating_them() -> None:
    records = tuple(BOTH)
    snapshot = [(r.evidence_id, r.chunk) for r in records]
    first = _coverage("Alpha FY2024 Q1", records, {"E01"})
    assert first == _coverage("Alpha FY2024 Q1", records, {"E01"})
    assert [(r.evidence_id, r.chunk) for r in records] == snapshot
    assert "caveat" in first.metadata()


# --- final 6.1 verification: question-level coverage, cited states, Unicode spans ------------

REAL_STYLE = CorpusVocabulary.from_manifest(MANIFEST)  # NVIDIA / Alphabet synthetic metadata


def _real_style_records(companies: tuple[str, ...], document_ids: tuple[str, ...]) -> Any:
    """Minimal stand-ins exposing only what coverage reads (a ChunkRecord rejects relabelling)."""
    return tuple(
        SimpleNamespace(evidence_id=f"E{i:02d}", chunk=SimpleNamespace(company=c, document_id=d))
        for i, (c, d) in enumerate(zip(companies, document_ids, strict=True), start=1)
    )


def test_unresolved_period_keeps_the_question_status_unknown() -> None:
    scope = analyze_question("Compare NVIDIA FY2025 Q3 and Q4.", REAL_STYLE)
    assert [p.canonical for p in scope.periods] == ["FY2025 Q3"]
    assert [u.kind for u in scope.unresolved] == ["quarter_without_year"]
    coverage = evidence_coverage(scope, ())
    assert coverage.question_limitations == (sc.UNRESOLVED_REFERENCES,)
    with_evidence = _coverage_for(scope, ("NVIDIA Corporation",) * 2, ("nvda-q3",) * 2)
    assert with_evidence.selected_reference_status is CoverageStatus.COMPLETE
    assert with_evidence.selected_question_status is CoverageStatus.UNKNOWN


def test_segment_questions_are_not_complete_at_question_level() -> None:
    scope = analyze_question("What was Google Cloud revenue?", REAL_STYLE)
    coverage = _coverage_for(scope, ("Alphabet Inc.",) * 2, ("goog-q3",) * 2)
    kinds = {e.kind: e.selected for e in coverage.entities}
    assert kinds == {"company": LayerState.PRESENT, "segment": LayerState.NOT_APPLICABLE}
    assert coverage.selected_reference_status is CoverageStatus.COMPLETE
    assert coverage.selected_question_status is CoverageStatus.UNKNOWN
    assert coverage.question_limitations == (sc.UNASSESSABLE_REFERENCES,)


def test_fiscal_labels_without_a_company_do_not_imply_one_company() -> None:
    scope = analyze_question("Compare FY2024 with FY2025", REAL_STYLE)
    coverage = _coverage_for(scope, ("Alphabet Inc.", "NVIDIA Corporation"), ("goog-q3", "nvda-q4"))
    by_period = {e.canonical: e.companies for e in coverage.entities}
    assert by_period == {"FY2024": ("Alphabet Inc.",), "FY2025": ("NVIDIA Corporation",)}
    assert coverage.selected_reference_status is CoverageStatus.COMPLETE
    assert coverage.selected_question_status is CoverageStatus.UNKNOWN
    assert coverage.question_limitations == (sc.FISCAL_PERIODS_WITHOUT_COMPANY,)


def test_basis_unspecified_and_relative_questions_are_unknown() -> None:
    basis = evidence_coverage(analyze_question("Alphabet Q3 2024", REAL_STYLE), ())
    assert sc.BASIS_UNSPECIFIED_PERIODS in basis.question_limitations
    relative = evidence_coverage(analyze_question("What happened last quarter?", REAL_STYLE), ())
    assert relative.selected_reference_status is CoverageStatus.NOT_APPLICABLE
    assert relative.selected_question_status is CoverageStatus.UNKNOWN


def test_missing_and_empty_citation_sets_are_distinct() -> None:
    scope = analyze_question("Alpha FY2024 Q1", EVIDENCE_VOCAB)
    not_evaluated = evidence_coverage(scope, BOTH, None)
    none_cited = evidence_coverage(scope, BOTH, set())
    assert {e.cited for e in not_evaluated.entities} == {LayerState.NOT_EVALUATED}
    assert (not_evaluated.cited_count, not_evaluated.cited_reference_status) == (None, None)
    assert {e.cited for e in none_cited.entities} == {LayerState.ABSENT}
    assert (none_cited.cited_count, none_cited.cited_reference_status) == (0, CoverageStatus.NONE)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (chr(0xFB01) + "rm NVIDIA FY2025 Q4", ["NVIDIA", "FY2025 Q4"]),  # ligature expands
        ("cafe" + chr(0x301) + " NVIDIA FY2025", ["NVIDIA", "FY2025"]),  # combining mark
        (FULL_WIDTH_NVIDIA + " FY2025", [FULL_WIDTH_NVIDIA, "FY2025"]),  # full-width
        (chr(0x301) + "NVIDIA", ["NVIDIA"]),  # a leading mark is its own segment
    ],
)
def test_spans_and_text_refer_to_the_original_question(question: str, expected: list[str]) -> None:
    first = _scope(question)
    found = [(c.start, c.end, c.matched_text) for c in first.companies]
    found += [(p.start, p.end, p.matched_text) for p in first.periods]
    found.sort()
    assert [text for _, _, text in found] == expected
    assert all(question[start:end] == text for start, end, text in found)
    assert [c.company for c in first.companies] == ["NVIDIA Corporation"]
    assert _scope(question) == first


def _coverage_for(
    scope: sc.QuestionScope, companies: tuple[str, ...], document_ids: tuple[str, ...]
) -> sc.EvidenceCoverage:
    return evidence_coverage(scope, _real_style_records(companies, document_ids))
