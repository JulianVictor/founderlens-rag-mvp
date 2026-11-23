"""Corpus vocabulary, question scope and evidence coverage (Commit 6.1). Diagnostic only.

Usage::

    python -m founderlens.guardrails.scope --smoke   # real manifest and V2 evidence, no LLM

**Vocabulary.** :class:`CorpusVocabulary` is built only from the trusted corpus
manifest (``company``, ``ticker``, ``document_id``, ``fiscal_period``,
``period_end``) plus :data:`APPROVED_ALIASES`, a small reviewed code table. It
holds a few extra company aliases and tickers (``NVIDIA``, ``Alphabet``,
``GOOGL``) and two Alphabet segment names (``Google Cloud``,
``Google Services``). Nothing is derived from evaluation files or outside
sources.

**Companies.** Matching is token-bounded, never a substring match. Names and
aliases are case-insensitive, except ``Alphabet``, which must be capitalized
(the common noun is not a company). Tickers must be upper case. Longer forms
win (``Google Cloud`` before ``Google``). Generic terms (cloud, data center,
AI, revenue) are never company names. ``Google Cloud`` / ``Google Services``
are *segment* references of Alphabet, not the consolidated company. A bare
``Google`` maps to Alphabet but is flagged ``brand_ambiguous``, because it may
mean the consolidated company or its Google segments. A recognized alias is
an aid to recognition, never evidence that a value belongs to that company.

**Periods.** Only explicit forms are recognized:

- fiscal quarters (``FY2025 Q4``, ``Q4 FY2025``, ``fourth quarter of fiscal
  2025``, ``fiscal 2025 fourth quarter``);
- fiscal years (``FY2025``, ``fiscal year 2025``);
- calendar quarters and years, only when marked ``calendar``/``CY``;
- period-end dates after ``ended``/``ending`` (``three months ended
  October 27, 2024``).

Fiscal labels are company-relative and are matched only against the manifest's
``fiscal_period`` labels of the companies the question names (all companies
when none is named; a period is not tied to the nearest company). Fiscal
quarters are never assumed to follow calendar quarters. A calendar quarter or
date matches a document only when the document's ``period_end`` equals that
date, and no fiscal quarter is inferred from a date.

A quarter with a year but no basis marker (``Q3 2024``) is recognized as
``basis_unspecified``, with candidate documents under both readings listed
separately. These stay unresolved and are never guessed:

- relative periods (``last quarter``, ``this year``, ``recently``,
  ``a year ago``);
- bare years (``2019``);
- two-digit fiscal years (``FY25``);
- quarters without a year;
- dates without ``ended``/``ending``.

**Manifest coverage is not content coverage.** A release for one period
contains figures for others (prior-year comparatives, full-year totals). A
period missing from the manifest is labelled
``not_in_manifest_reporting_periods``, never "out of corpus": it does not show
that the documents lack the value, and nothing here abstains, blocks or
filters.

**Coverage** (:func:`evidence_coverage`) compares recognized references with
three layers: manifest document metadata, model-visible (selected) evidence
and cited evidence. Metadata representation is not proof that a passage
supports the question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum
from typing import Any, Final

from founderlens.config import ConfigError, load_settings
from founderlens.corpus.manifest import CorpusManifest, ManifestError, load_manifest
from founderlens.generation.context import EvidenceRecord

VOCABULARY_VERSION = 1
ANALYZER_VERSION = 2  # 2: original-text offsets; reference vs question coverage status
NORMALIZATION: Final = "NFKC"

# Reviewed, code-defined additions to the manifest. Keys are manifest company names.
# Entries for companies absent from a manifest are ignored (and not hashed).
APPROVED_ALIASES: dict[str, tuple[tuple[str, bool], ...]] = {
    # (alias, case_sensitive)
    "NVIDIA Corporation": (("NVIDIA Corp", False), ("NVIDIA", False)),
    "Alphabet Inc.": (("Alphabet Inc", False), ("Alphabet", True), ("ALPHABET", True)),
}
APPROVED_TICKERS: dict[str, tuple[str, ...]] = {"Alphabet Inc.": ("GOOGL",)}
APPROVED_SEGMENTS: dict[str, tuple[str, ...]] = {
    "Alphabet Inc.": ("Google Cloud", "Google Services"),
}
# A brand that may mean the consolidated company or its segments.
APPROVED_BRANDS: dict[str, tuple[str, ...]] = {"Alphabet Inc.": ("Google",)}


class ScopeAnalysisError(ValueError):
    """Invalid analyzer input or vocabulary. Messages never quote question text."""


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CompanyEntry:
    company: str  # canonical identity: the manifest ``company`` value
    tickers: tuple[str, ...]
    aliases: tuple[tuple[str, bool], ...]
    segments: tuple[str, ...]
    brands: tuple[str, ...]
    document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentPeriod:
    document_id: str
    company: str
    fiscal_period: str  # manifest label, e.g. "FY2025 Q4"
    fiscal_year: int | None
    fiscal_quarter: int | None
    period_end: date


_LABEL = re.compile(r"FY(\d{4})(?: Q([1-4]))?")


@dataclass(frozen=True, slots=True)
class CorpusVocabulary:
    corpus_id: str
    corpus_version: str
    companies: tuple[CompanyEntry, ...]
    periods: tuple[DocumentPeriod, ...]

    @classmethod
    def from_manifest(cls, manifest: CorpusManifest) -> CorpusVocabulary:
        companies: list[CompanyEntry] = []
        names: list[str] = []
        for doc in manifest.documents:
            if doc.company not in names:
                names.append(doc.company)
        for name in names:
            docs = [d for d in manifest.documents if d.company == name]
            tickers = tuple(
                dict.fromkeys([*(d.ticker for d in docs), *APPROVED_TICKERS.get(name, ())])
            )
            companies.append(
                CompanyEntry(
                    company=name,
                    tickers=tickers,
                    aliases=((name, False), *APPROVED_ALIASES.get(name, ())),
                    segments=APPROVED_SEGMENTS.get(name, ()),
                    brands=APPROVED_BRANDS.get(name, ()),
                    document_ids=tuple(d.document_id for d in docs),
                )
            )
        periods = []
        for doc in manifest.documents:
            match = _LABEL.fullmatch(doc.fiscal_period)
            periods.append(
                DocumentPeriod(
                    document_id=doc.document_id,
                    company=doc.company,
                    fiscal_period=doc.fiscal_period,
                    fiscal_year=int(match.group(1)) if match else None,
                    fiscal_quarter=int(match.group(2)) if match and match.group(2) else None,
                    period_end=doc.period_end,
                )
            )
        vocabulary = cls(
            manifest.corpus_id, manifest.corpus_version, tuple(companies), tuple(periods)
        )
        vocabulary._check_unambiguous()
        return vocabulary

    def _check_unambiguous(self) -> None:
        owners: dict[str, str] = {}
        for entry in self.companies:
            forms = [a for a, _ in entry.aliases] + list(entry.tickers)
            forms += list(entry.segments) + list(entry.brands)
            for form in forms:
                key = form.casefold()
                if owners.setdefault(key, entry.company) != entry.company:
                    raise ScopeAnalysisError(f"alias {form!r} maps to more than one company")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "vocabulary_version": VOCABULARY_VERSION,
            "corpus_id": self.corpus_id,
            "corpus_version": self.corpus_version,
            "companies": [
                {
                    "company": c.company,
                    "tickers": list(c.tickers),
                    "aliases": [[a, s] for a, s in c.aliases],
                    "segments": list(c.segments),
                    "brands": list(c.brands),
                    "document_ids": list(c.document_ids),
                }
                for c in self.companies
            ],
            "periods": [
                {
                    "document_id": p.document_id,
                    "company": p.company,
                    "fiscal_period": p.fiscal_period,
                    "period_end": p.period_end.isoformat(),
                }
                for p in self.periods
            ],
        }

    @property
    def vocabulary_id(self) -> str:
        return _hash(self.canonical_dict())

    def company(self, name: str) -> CompanyEntry | None:
        return next((c for c in self.companies if c.company == name), None)

    def company_of(self, document_id: str) -> str | None:
        return next((p.company for p in self.periods if p.document_id == document_id), None)


def _hash(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Recognition rules
# --------------------------------------------------------------------------- #

_Y = r"((?:19|20)\d{2})"
_QW = r"(first|second|third|fourth|1st|2nd|3rd|4th)"
_QUARTER_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
}
_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH = r"(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DATE = rf"(?:{_MONTH}\s+(\d{{1,2}}),?\s+{_Y}|{_Y}-(\d{{2}})-(\d{{2}}))"
_B = r"(?<!\w)"
_E = r"(?!\w)"
_FISCAL = r"(?:fiscal\s+(?:year\s+)?|FY\s?)"
_CALENDAR = r"(?:calendar\s+(?:year\s+)?|CY\s?)"

# (kind, pattern); order matters: earlier rules consume their spans first.
PERIOD_RULES: tuple[tuple[str, str], ...] = (
    ("period_end_date", rf"{_B}(three|six|nine|twelve)\s+months\s+ended\s+{_DATE}{_E}"),
    ("period_end_date", rf"{_B}(quarter|year|period)\s+end(?:ed|ing)\s+{_DATE}{_E}"),
    ("fiscal_quarter", rf"{_B}FY\s?{_Y}\s*Q([1-4]){_E}"),
    ("fiscal_quarter", rf"{_B}Q([1-4])\s+(?:of\s+)?{_FISCAL}{_Y}{_E}"),
    ("fiscal_quarter", rf"{_B}{_QW}\s+(?:fiscal\s+)?quarter\s+(?:of\s+)?{_FISCAL}{_Y}{_E}"),
    ("fiscal_quarter", rf"{_B}{_FISCAL}{_Y}\s+{_QW}\s+quarter{_E}"),
    ("fiscal_quarter", rf"{_B}fiscal\s+(?:year\s+)?{_Y}\s+Q([1-4]){_E}"),
    ("fiscal_quarter", rf"{_B}(?:fiscal|FY)\s+Q([1-4])\s+(?:of\s+)?{_Y}{_E}"),
    ("calendar_quarter", rf"{_B}CY\s?{_Y}\s*Q([1-4]){_E}"),
    ("calendar_quarter", rf"{_B}calendar\s+(?:year\s+)?{_Y}\s+Q([1-4]){_E}"),
    ("calendar_quarter", rf"{_B}(?:calendar|CY)\s+Q([1-4])\s+(?:of\s+)?{_Y}{_E}"),
    ("calendar_quarter", rf"{_B}Q([1-4])\s+(?:of\s+)?{_CALENDAR}{_Y}{_E}"),
    ("calendar_quarter", rf"{_B}{_QW}\s+quarter\s+(?:of\s+)?{_CALENDAR}{_Y}{_E}"),
    ("fiscal_year", rf"{_B}{_FISCAL}{_Y}{_E}"),
    ("calendar_year", rf"{_B}{_CALENDAR}{_Y}{_E}"),
    ("quarter_basis_unspecified", rf"{_B}Q([1-4])\s+{_Y}{_E}"),
    ("quarter_basis_unspecified", rf"{_B}{_Y}\s+Q([1-4]){_E}"),
    ("quarter_basis_unspecified", rf"{_B}{_QW}\s+quarter\s+(?:of\s+)?{_Y}{_E}"),
)
UNRESOLVED_RULES: tuple[tuple[str, str], ...] = (
    (
        "relative_period",
        rf"{_B}(?:last|previous|prior|this|current|next|latest|most\s+recent|recent)\s+"
        rf"(?:fiscal\s+)?(?:quarter|year|period){_E}",
    ),
    ("relative_period", rf"{_B}(?:recently|a\s+year\s+ago|year\s+ago){_E}"),
    ("two_digit_fiscal_year", rf"{_B}FY\s?'?\d{{2}}{_E}"),
    ("date_without_period_context", rf"{_B}{_DATE}{_E}"),
    ("quarter_without_year", rf"{_B}(?:Q[1-4]|{_QW}\s+quarter){_E}"),
    ("bare_year", rf"{_B}{_Y}{_E}"),
)


def analyzer_definition() -> dict[str, Any]:
    return {
        "analyzer_version": ANALYZER_VERSION,
        "normalization": f"{NORMALIZATION} per base-character segment; original-text spans",
        "coverage_semantics": (
            "reference status over assessable recognized references; question status "
            "unknown when unresolved, unassessable, basis-unspecified or "
            "company-unspecified fiscal references exist"
        ),
        "period_rules": [list(r) for r in PERIOD_RULES],
        "unresolved_rules": [list(r) for r in UNRESOLVED_RULES],
        "company_matching": "token-bounded; longest first; tickers upper case; per-alias case",
        "basis_unspecified_policy": "list fiscal-label and calendar-end candidates separately",
        "calendar_matching": "document period_end equals the calendar period end",
        "period_company_filter": "documents of companies named anywhere in the question",
    }


def analyzer_id() -> str:
    return _hash(analyzer_definition())


# --------------------------------------------------------------------------- #
# Scope result
# --------------------------------------------------------------------------- #


class CompanyScope(StrEnum):
    COMPANY = "company"
    SEGMENT = "segment"
    BRAND_AMBIGUOUS = "brand_ambiguous"


class ManifestStatus(StrEnum):
    IN_MANIFEST = "in_manifest_reporting_periods"
    NOT_IN_MANIFEST = "not_in_manifest_reporting_periods"
    BASIS_UNSPECIFIED = "basis_unspecified"


@dataclass(frozen=True, slots=True)
class CompanyReference:
    company: str
    scope: CompanyScope
    segment: str | None
    match_kind: str  # name | alias | ticker | segment | brand
    matched_text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PeriodReference:
    kind: str
    canonical: str
    year: int | None
    quarter: int | None
    period_end: date | None
    duration: str | None
    manifest_status: ManifestStatus
    manifest_documents: tuple[str, ...]  # fiscal-label or period-end matches
    calendar_candidates: tuple[str, ...]  # basis_unspecified only
    manifest_companies: tuple[str, ...]  # companies owning those documents
    matched_text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class UnresolvedMention:
    kind: str
    matched_text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class QuestionScope:
    """Explicit references in one question. ``matched_text`` comes from the NFKC form."""

    companies: tuple[CompanyReference, ...]
    periods: tuple[PeriodReference, ...]
    unresolved: tuple[UnresolvedMention, ...]
    vocabulary_id: str
    analyzer_id: str

    @property
    def canonical_companies(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.company for c in self.companies))

    @property
    def ambiguous(self) -> tuple[CompanyReference, ...]:
        return tuple(c for c in self.companies if c.scope is CompanyScope.BRAND_AMBIGUOUS)

    def metadata(self) -> dict[str, Any]:
        """Canonical identities and counts; no question fragments."""
        return {
            "vocabulary_id": self.vocabulary_id,
            "analyzer_id": self.analyzer_id,
            "companies": [
                {
                    "company": c.company,
                    "scope": c.scope.value,
                    "segment": c.segment,
                    "match_kind": c.match_kind,
                }
                for c in self.companies
            ],
            "periods": [
                {
                    "kind": p.kind,
                    "canonical": p.canonical,
                    "manifest_status": p.manifest_status.value,
                    "manifest_documents": list(p.manifest_documents),
                    "calendar_candidates": list(p.calendar_candidates),
                    "manifest_companies": list(p.manifest_companies),
                }
                for p in self.periods
            ],
            "unresolved": [u.kind for u in self.unresolved],
        }


def _boundary_pattern(form: str, case_sensitive: bool) -> re.Pattern[str]:
    flags = 0 if case_sensitive else re.IGNORECASE
    words = r"\s+".join(re.escape(part) for part in form.split())
    return re.compile(rf"{_B}{words}(?!\w)", flags)


def _company_forms(
    vocabulary: CorpusVocabulary,
) -> list[tuple[str, re.Pattern[str], CompanyEntry, str, str | None]]:
    forms: list[tuple[str, re.Pattern[str], CompanyEntry, str, str | None]] = []
    for entry in vocabulary.companies:
        for alias, case_sensitive in entry.aliases:
            kind = "name" if alias == entry.company else "alias"
            forms.append((alias, _boundary_pattern(alias, case_sensitive), entry, kind, None))
        for ticker in entry.tickers:
            forms.append((ticker, _boundary_pattern(ticker, True), entry, "ticker", None))
        for segment in entry.segments:
            forms.append((segment, _boundary_pattern(segment, False), entry, "segment", segment))
        for brand in entry.brands:
            forms.append((brand, _boundary_pattern(brand, False), entry, "brand", None))
    forms.sort(key=lambda f: (-len(f[0]), f[0]))  # longest first, deterministic
    return forms


def _free(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return all(end <= s or start >= e for s, e in spans)


def _date_from(groups: Sequence[str | None]) -> date | None:
    month, day, year, iso_year, iso_month, iso_day = groups
    try:
        if month is not None and day is not None and year is not None:
            return date(int(year), _MONTHS[month.casefold()], int(day))
        if iso_year is not None and iso_month is not None and iso_day is not None:
            return date(int(iso_year), int(iso_month), int(iso_day))
    except ValueError:
        return None
    return None


def _calendar_quarter_end(year: int, quarter: int) -> date:
    return {
        1: date(year, 3, 31),
        2: date(year, 6, 30),
        3: date(year, 9, 30),
        4: date(year, 12, 31),
    }[quarter]


def _period(
    kind: str, match: re.Match[str], vocabulary: CorpusVocabulary, named: frozenset[str]
) -> PeriodReference | None:
    groups = list(match.groups())
    text = match.group(0)
    # Fiscal labels are company-relative: restrict to companies the question names.
    periods = tuple(p for p in vocabulary.periods if not named or p.company in named)

    def ref(
        canonical: str,
        year: int | None,
        quarter: int | None,
        end: date | None,
        duration: str | None,
        docs: Sequence[str],
        status: ManifestStatus | None = None,
        calendar: Sequence[str] = (),
    ) -> PeriodReference:
        if status is None:
            status = ManifestStatus.IN_MANIFEST if docs else ManifestStatus.NOT_IN_MANIFEST
        return PeriodReference(
            kind,
            canonical,
            year,
            quarter,
            end,
            duration,
            status,
            tuple(docs),
            tuple(calendar),
            tuple(dict.fromkeys(p.company for p in periods if p.document_id in {*docs, *calendar})),
            text,
            match.start(),
            match.end(),
        )

    if kind == "period_end_date":
        duration = groups[0].casefold()
        end = _date_from(groups[1:7])
        if end is None:
            return None
        docs = [p.document_id for p in periods if p.period_end == end]
        label = {
            "three": "three_months",
            "six": "six_months",
            "nine": "nine_months",
            "twelve": "twelve_months",
        }.get(duration, duration)
        return ref(f"period ended {end.isoformat()} ({label})", None, None, end, label, docs)
    numbers: list[int] = []
    for g in groups:
        if g is None:
            continue
        key = g.casefold()
        numbers.append(_QUARTER_WORDS[key] if key in _QUARTER_WORDS else int(g))
    year = next(n for n in numbers if n > 4)
    quarter = next((n for n in numbers if n <= 4), None)
    if kind == "fiscal_quarter":
        docs = [
            p.document_id for p in periods if (p.fiscal_year, p.fiscal_quarter) == (year, quarter)
        ]
        return ref(f"FY{year} Q{quarter}", year, quarter, None, "fiscal_quarter", docs)
    if kind == "fiscal_year":
        docs = [p.document_id for p in periods if p.fiscal_year == year]
        return ref(f"FY{year}", year, None, None, "fiscal_year", docs)
    if kind == "calendar_quarter":
        assert quarter is not None
        end = _calendar_quarter_end(year, quarter)
        docs = [p.document_id for p in periods if p.period_end == end]
        return ref(f"CY{year} Q{quarter}", year, quarter, end, "calendar_quarter", docs)
    if kind == "calendar_year":
        end = date(year, 12, 31)
        docs = [p.document_id for p in periods if p.period_end == end]
        return ref(f"CY{year}", year, None, end, "calendar_year", docs)
    assert quarter is not None  # quarter_basis_unspecified
    fiscal = [
        p.document_id for p in periods if (p.fiscal_year, p.fiscal_quarter) == (year, quarter)
    ]
    end = _calendar_quarter_end(year, quarter)
    calendar = [p.document_id for p in periods if p.period_end == end]
    return ref(
        f"Q{quarter} {year} (basis unspecified)",
        year,
        quarter,
        None,
        None,
        fiscal,
        ManifestStatus.BASIS_UNSPECIFIED,
        calendar,
    )


def normalize_with_offsets(question: str) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    """NFKC per base-character segment, with a map back to original offsets.

    A segment is one character plus any combining marks after it. Each
    normalized character records the original start and end of its segment, so
    a normalized span ``[a, b)`` maps to ``[starts[a], ends[b - 1])`` in the
    original question, even when NFKC expands (``ﬁ`` to ``fi``) or composes
    (``e`` + combining acute) text.
    """
    parts: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i = 0
    while i < len(question):
        j = i + 1
        while j < len(question) and unicodedata.combining(question[j]):
            j += 1
        normalized = unicodedata.normalize(NORMALIZATION, question[i:j])
        parts.append(normalized)
        starts.extend([i] * len(normalized))
        ends.extend([j] * len(normalized))
        i = j
    return "".join(parts), tuple(starts), tuple(ends)


def analyze_question(question: object, vocabulary: CorpusVocabulary) -> QuestionScope:
    """Explicit company and period references, in text order. Never rewrites the question.

    Matching runs on the segment-wise NFKC form; every reported span and
    ``matched_text`` refers to the *original* question.
    """
    if not isinstance(question, str):
        raise ScopeAnalysisError(f"question must be a string, got {type(question).__name__}")
    text, starts, ends = normalize_with_offsets(question)

    def original(start: int, end: int) -> tuple[str, int, int]:
        a, b = starts[start], ends[end - 1]
        return question[a:b], a, b

    spans: list[tuple[int, int]] = []
    companies: list[CompanyReference] = []
    for _form, pattern, entry, kind, segment in _company_forms(vocabulary):
        for match in pattern.finditer(text):
            if not _free(spans, match.start(), match.end()):
                continue
            spans.append((match.start(), match.end()))
            scope = {"segment": CompanyScope.SEGMENT, "brand": CompanyScope.BRAND_AMBIGUOUS}.get(
                kind, CompanyScope.COMPANY
            )
            companies.append(
                CompanyReference(entry.company, scope, segment, kind, *original(*match.span()))
            )
    named = frozenset(c.company for c in companies)
    periods: list[PeriodReference] = []
    for kind, rule in PERIOD_RULES:
        for match in re.finditer(rule, text, re.IGNORECASE):
            if not _free(spans, match.start(), match.end()):
                continue
            reference = _period(kind, match, vocabulary, named)
            if reference is None:
                continue
            spans.append((match.start(), match.end()))
            matched, a, b = original(*match.span())
            periods.append(replace(reference, matched_text=matched, start=a, end=b))
    unresolved: list[UnresolvedMention] = []
    for kind, rule in UNRESOLVED_RULES:
        for match in re.finditer(rule, text, re.IGNORECASE):
            if _free(spans, match.start(), match.end()):
                spans.append((match.start(), match.end()))
                unresolved.append(UnresolvedMention(kind, *original(*match.span())))
    return QuestionScope(
        companies=tuple(sorted(companies, key=lambda c: c.start)),
        periods=tuple(sorted(periods, key=lambda p: p.start)),
        unresolved=tuple(sorted(unresolved, key=lambda u: u.start)),
        vocabulary_id=vocabulary.vocabulary_id,
        analyzer_id=analyzer_id(),
    )


# --------------------------------------------------------------------------- #
# Evidence coverage
# --------------------------------------------------------------------------- #


class LayerState(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    NOT_APPLICABLE = "not_applicable"  # cannot be decided from metadata
    NOT_EVALUATED = "not_evaluated"  # no cited evidence supplied


class CoverageStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"  # nothing recognized (questions: nothing unresolved either)
    UNKNOWN = "unknown"  # not decidable from metadata
    COMPLETE = "complete"
    PARTIAL = "partial"
    NONE = "none"


# Reasons a question-level status is not "complete" even when every assessable reference is.
UNRESOLVED_REFERENCES = "unresolved_references"
UNASSESSABLE_REFERENCES = "unassessable_references"  # segments, periods not in the manifest
BASIS_UNSPECIFIED_PERIODS = "basis_unspecified_periods"
FISCAL_PERIODS_WITHOUT_COMPANY = "fiscal_periods_without_named_company"


@dataclass(frozen=True, slots=True)
class EntityCoverage:
    kind: str  # company | segment | period
    canonical: str
    manifest: str
    selected: LayerState
    cited: LayerState
    note: str | None
    companies: tuple[str, ...] = ()  # periods: companies owning the matched documents


@dataclass(frozen=True, slots=True)
class EvidenceCoverage:
    entities: tuple[EntityCoverage, ...]
    selected_count: int
    cited_count: int | None
    unresolved_count: int
    # Over assessable recognized references only, never the whole question:
    selected_reference_status: CoverageStatus
    cited_reference_status: CoverageStatus | None
    # The whole question: UNKNOWN whenever ``question_limitations`` is not empty:
    selected_question_status: CoverageStatus
    cited_question_status: CoverageStatus | None
    question_limitations: tuple[str, ...]
    vocabulary_id: str
    analyzer_id: str

    def metadata(self) -> dict[str, Any]:
        def value(status: CoverageStatus | None) -> str | None:
            return None if status is None else status.value

        return {
            "selected_reference_status": self.selected_reference_status.value,
            "selected_question_status": self.selected_question_status.value,
            "cited_reference_status": value(self.cited_reference_status),
            "cited_question_status": value(self.cited_question_status),
            "question_limitations": list(self.question_limitations),
            "selected_count": self.selected_count,
            "cited_count": self.cited_count,
            "unresolved_count": self.unresolved_count,
            "entities": [
                {
                    "kind": e.kind,
                    "canonical": e.canonical,
                    "manifest": e.manifest,
                    "selected": e.selected.value,
                    "cited": e.cited.value,
                    "note": e.note,
                    "companies": list(e.companies),
                }
                for e in self.entities
            ],
            "vocabulary_id": self.vocabulary_id,
            "analyzer_id": self.analyzer_id,
            "caveat": "metadata coverage only; not proof that any passage supports the question",
        }


def _state(present: bool) -> LayerState:
    return LayerState.PRESENT if present else LayerState.ABSENT


def _status(states: Sequence[LayerState], recognized: bool) -> CoverageStatus:
    decidable = [s for s in states if s in (LayerState.PRESENT, LayerState.ABSENT)]
    if not recognized:
        return CoverageStatus.NOT_APPLICABLE
    if not decidable:
        return CoverageStatus.UNKNOWN
    present = sum(s is LayerState.PRESENT for s in decidable)
    if present == len(decidable):
        return CoverageStatus.COMPLETE
    return CoverageStatus.NONE if present == 0 else CoverageStatus.PARTIAL


def _question_status(reference: CoverageStatus, limitations: Sequence[str]) -> CoverageStatus:
    """A gap already found stays visible; otherwise any limitation makes it unknown."""
    if reference in (CoverageStatus.PARTIAL, CoverageStatus.NONE) or not limitations:
        return reference
    return CoverageStatus.UNKNOWN


def evidence_coverage(
    scope: QuestionScope,
    selected: Sequence[EvidenceRecord],
    cited_ids: Collection[str] | None = None,
) -> EvidenceCoverage:
    """Pure comparison of recognized references with selected and (optionally) cited evidence.

    ``selected`` is the model-visible evidence (for example a run's evidence
    registry); ``cited_ids`` are the evidence IDs the answer cites. Inputs are only
    read. A company counts as present when an evidence record's trusted
    ``company`` equals it; a period when the record's document is one of the
    period's manifest documents.
    """
    records = tuple(selected)
    cited = None if cited_ids is None else tuple(r for r in records if r.evidence_id in cited_ids)
    entities: list[EntityCoverage] = []

    def layers(match: Any) -> tuple[LayerState, LayerState]:
        sel = _state(any(match(r) for r in records))
        cit = LayerState.NOT_EVALUATED if cited is None else _state(any(match(r) for r in cited))
        return sel, cit

    for company in scope.canonical_companies:
        sel, cit = layers(lambda r, c=company: r.chunk.company == c)
        entities.append(EntityCoverage("company", company, "in_manifest", sel, cit, None))
    for ref in scope.companies:
        if ref.scope is CompanyScope.SEGMENT:
            entities.append(
                EntityCoverage(
                    "segment",
                    f"{ref.company} / {ref.segment}",
                    "segment_alias",
                    LayerState.NOT_APPLICABLE,
                    LayerState.NOT_APPLICABLE,
                    "segment coverage is not decidable from document metadata",
                )
            )
    for period in scope.periods:
        if period.manifest_status is ManifestStatus.NOT_IN_MANIFEST:
            entities.append(
                EntityCoverage(
                    "period",
                    period.canonical,
                    period.manifest_status.value,
                    LayerState.NOT_APPLICABLE,
                    LayerState.NOT_APPLICABLE,
                    "not a manifest reporting period; other releases may still "
                    "contain figures for it",
                )
            )
            continue
        docs = set(period.manifest_documents) | set(period.calendar_candidates)
        sel, cit = layers(lambda r, d=docs: r.chunk.document_id in d)
        note = (
            "basis unspecified: fiscal and calendar candidates combined"
            if period.manifest_status is ManifestStatus.BASIS_UNSPECIFIED
            else None
        )
        entities.append(
            EntityCoverage(
                "period",
                period.canonical,
                period.manifest_status.value,
                sel,
                cit,
                note,
                period.manifest_companies,
            )
        )
    recognized = bool(entities)
    limitations: list[str] = []
    if scope.unresolved:
        limitations.append(UNRESOLVED_REFERENCES)
    if any(e.selected is LayerState.NOT_APPLICABLE for e in entities):
        limitations.append(UNASSESSABLE_REFERENCES)
    if any(p.manifest_status is ManifestStatus.BASIS_UNSPECIFIED for p in scope.periods):
        limitations.append(BASIS_UNSPECIFIED_PERIODS)
    if not scope.companies and any(
        p.kind in ("fiscal_quarter", "fiscal_year") for p in scope.periods
    ):
        limitations.append(FISCAL_PERIODS_WITHOUT_COMPANY)
    selected_reference = _status([e.selected for e in entities], recognized)
    cited_reference = None if cited is None else _status([e.cited for e in entities], recognized)
    return EvidenceCoverage(
        entities=tuple(entities),
        selected_count=len(records),
        cited_count=None if cited is None else len(cited),
        unresolved_count=len(scope.unresolved),
        selected_reference_status=selected_reference,
        cited_reference_status=cited_reference,
        selected_question_status=_question_status(selected_reference, limitations),
        cited_question_status=None
        if cited_reference is None
        else _question_status(cited_reference, limitations),
        question_limitations=tuple(limitations),
        vocabulary_id=scope.vocabulary_id,
        analyzer_id=scope.analyzer_id,
    )


# --------------------------------------------------------------------------- #
# Smoke CLI (no LLM call)
# --------------------------------------------------------------------------- #

SMOKE_QUESTIONS: tuple[str, ...] = (
    "Compare NVIDIA and Alphabet.",
    "Compare NVIDIA FY2025 Q3 and Q4.",
    "What happened in FY2019?",
    "Compare FY2024 with FY2025.",
    "What was Google Cloud revenue?",
    "What happened last quarter?",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m founderlens.guardrails.scope",
        description="Question scope and evidence coverage diagnostics (no LLM call).",
    )
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    try:
        from founderlens.generation.context import prepare_generation_input
        from founderlens.generation.errors import GenerationError
        from founderlens.generation.token_budget import load_generation_tokenizer
        from founderlens.retrieval.policy import load_policy_retriever

        settings = load_settings(args.project_root)
        vocabulary = CorpusVocabulary.from_manifest(load_manifest(settings.corpus_manifest_path))
        retriever = load_policy_retriever(settings)
        counter = load_generation_tokenizer(settings)
        print(f"vocabulary {vocabulary.vocabulary_id}; analyzer {analyzer_id()}")
        for question in SMOKE_QUESTIONS:
            scope = analyze_question(question, vocabulary)
            prepared = prepare_generation_input(question, retriever, counter)
            coverage = evidence_coverage(scope, prepared.evidence.records)
            print(
                json.dumps(
                    {
                        "question": question,
                        "scope": scope.metadata(),
                        "coverage": coverage.metadata(),
                    },
                    ensure_ascii=False,
                )
            )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ManifestError, ScopeAnalysisError, GenerationError) as exc:
        print(f"Scope smoke FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
