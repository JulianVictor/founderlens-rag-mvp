"""Typed records for the evaluation question set (JSONL, one question per line).

Record shape::

    {
        "question_id": "dev-001",
        "question": "...",
        "category": "single_doc_factual",
        "answerable": true,
        "reference_answer": "...",  # null when unanswerable
        "required_facts": [  # [] when unanswerable
            {
                "fact_id": "dev-001-f1",
                "statement": "...",
                "numeric": {
                    "company": "...",
                    "fiscal_period": "...",
                    "metric": "...",
                    "value": 35082,
                    "unit": "USD millions",
                    "basis": "GAAP",
                },
            }
        ],
        "accepted_evidence": [  # [] when unanswerable
            {"fact_id": "dev-001-f1", "document_id": "nvda-fy2025-q3", "page_number": 1}
        ],
        "expected_abstention_reason": null,  # non-empty string when unanswerable
    }

``page_number`` is the one-based *physical* PDF page, not a printed page label.
Parsing is strict: unknown keys and wrong types are errors.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Category(StrEnum):
    SINGLE_DOC_FACTUAL = "single_doc_factual"
    SINGLE_DOC_CONCEPTUAL = "single_doc_conceptual"
    CROSS_DOC_COMPARISON = "cross_doc_comparison"
    TEMPORAL_MULTI_EVIDENCE = "temporal_multi_evidence"
    UNANSWERABLE = "unanswerable"


class Basis(StrEnum):
    GAAP = "GAAP"
    NON_GAAP = "non-GAAP"
    NOT_APPLICABLE = "not_applicable"


ALLOWED_UNITS = frozenset(
    {"USD", "USD millions", "USD billions", "percent", "percentage points", "count"}
)

_QUESTION_ID_RE = re.compile(r"^(dev|test)-\d{3}$")
_FACT_ID_RE = re.compile(r"^(dev|test)-\d{3}-f\d+$")

_QUESTION_KEYS = frozenset(
    {
        "question_id",
        "question",
        "category",
        "answerable",
        "reference_answer",
        "required_facts",
        "accepted_evidence",
        "expected_abstention_reason",
    }
)
_FACT_KEYS = frozenset({"fact_id", "statement", "numeric"})
_NUMERIC_KEYS = frozenset({"company", "fiscal_period", "metric", "value", "unit", "basis"})
_EVIDENCE_KEYS = frozenset({"fact_id", "document_id", "page_number"})


class RecordError(ValueError):
    """A single JSONL record is structurally invalid."""


@dataclass(frozen=True, slots=True)
class NumericValue:
    company: str
    fiscal_period: str
    metric: str
    value: float
    unit: str
    basis: Basis


@dataclass(frozen=True, slots=True)
class RequiredFact:
    fact_id: str
    statement: str
    numeric: NumericValue | None


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    fact_id: str
    document_id: str
    page_number: int


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    question_id: str
    question: str
    category: Category
    answerable: bool
    reference_answer: str | None
    required_facts: tuple[RequiredFact, ...]
    accepted_evidence: tuple[EvidenceRef, ...]
    expected_abstention_reason: str | None

    @property
    def split(self) -> str:
        return self.question_id.split("-", 1)[0]

    def evidence_for(self, fact_id: str) -> tuple[EvidenceRef, ...]:
        return tuple(e for e in self.accepted_evidence if e.fact_id == fact_id)


def _exact_keys(
    obj: dict[str, Any],
    required: frozenset[str],
    where: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    unknown = sorted(set(obj) - required - optional)
    missing = sorted(required - set(obj))
    if unknown or missing:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        raise RecordError(f"{where}: " + ", ".join(detail))


def _string(obj: dict[str, Any], key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{where}: '{key}' must be a non-empty string")
    return value


def _optional_string(obj: dict[str, Any], key: str, where: str) -> str | None:
    if obj.get(key) is None:
        return None
    return _string(obj, key, where)


def _object(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecordError(f"{where}: must be a JSON object")
    return value


def _list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise RecordError(f"{where}: must be a JSON array")
    return value


def _parse_numeric(value: object, where: str) -> NumericValue:
    obj = _object(value, where)
    _exact_keys(obj, _NUMERIC_KEYS, where)
    number = obj["value"]
    if isinstance(number, bool) or not isinstance(number, int | float):
        raise RecordError(f"{where}: 'value' must be a number")
    unit = _string(obj, "unit", where)
    if unit not in ALLOWED_UNITS:
        raise RecordError(f"{where}: unit {unit!r} not in {sorted(ALLOWED_UNITS)}")
    try:
        basis = Basis(_string(obj, "basis", where))
    except ValueError as exc:
        raise RecordError(f"{where}: invalid basis ({exc})") from exc
    return NumericValue(
        company=_string(obj, "company", where),
        fiscal_period=_string(obj, "fiscal_period", where),
        metric=_string(obj, "metric", where),
        value=float(number),
        unit=unit,
        basis=basis,
    )


def _parse_fact(value: object, where: str) -> RequiredFact:
    obj = _object(value, where)
    _exact_keys(obj, _FACT_KEYS - {"numeric"}, where, optional=frozenset({"numeric"}))
    fact_id = _string(obj, "fact_id", where)
    if not _FACT_ID_RE.fullmatch(fact_id):
        raise RecordError(f"{where}: malformed fact_id {fact_id!r}")
    numeric = obj.get("numeric")
    return RequiredFact(
        fact_id=fact_id,
        statement=_string(obj, "statement", where),
        numeric=None if numeric is None else _parse_numeric(numeric, f"{where}.numeric"),
    )


def _parse_evidence(value: object, where: str) -> EvidenceRef:
    obj = _object(value, where)
    _exact_keys(obj, _EVIDENCE_KEYS, where)
    page = obj["page_number"]
    if isinstance(page, bool) or not isinstance(page, int):
        raise RecordError(f"{where}: 'page_number' must be an integer")
    return EvidenceRef(
        fact_id=_string(obj, "fact_id", where),
        document_id=_string(obj, "document_id", where),
        page_number=page,
    )


def parse_record(data: object) -> EvalQuestion:
    """Parse one decoded JSONL object. Raises :class:`RecordError` on structural problems."""
    obj = _object(data, "record")
    question_id = obj.get("question_id")
    where = f"question {question_id!r}" if isinstance(question_id, str) else "record"
    _exact_keys(obj, _QUESTION_KEYS, where)

    question_id = _string(obj, "question_id", where)
    if not _QUESTION_ID_RE.fullmatch(question_id):
        raise RecordError(f"{where}: malformed question_id (expected dev-NNN or test-NNN)")
    try:
        category = Category(_string(obj, "category", where))
    except ValueError as exc:
        raise RecordError(f"{where}: invalid category ({exc})") from exc
    answerable = obj["answerable"]
    if not isinstance(answerable, bool):
        raise RecordError(f"{where}: 'answerable' must be true or false")

    facts = tuple(
        _parse_fact(item, f"{where}.required_facts[{i}]")
        for i, item in enumerate(_list(obj["required_facts"], f"{where}.required_facts"))
    )
    evidence = tuple(
        _parse_evidence(item, f"{where}.accepted_evidence[{i}]")
        for i, item in enumerate(_list(obj["accepted_evidence"], f"{where}.accepted_evidence"))
    )
    return EvalQuestion(
        question_id=question_id,
        question=_string(obj, "question", where),
        category=category,
        answerable=answerable,
        reference_answer=_optional_string(obj, "reference_answer", where),
        required_facts=facts,
        accepted_evidence=evidence,
        expected_abstention_reason=_optional_string(obj, "expected_abstention_reason", where),
    )


def parse_jsonl(text: str, *, source: str) -> tuple[list[EvalQuestion], list[str]]:
    """Parse JSONL text. Returns parsed records and per-line problems (never raises)."""
    records: list[EvalQuestion] = []
    problems: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            problems.append(f"{source}:{line_no}: blank line")
            continue
        try:
            data: object = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"{source}:{line_no}: invalid JSON ({exc.msg})")
            continue
        try:
            records.append(parse_record(data))
        except RecordError as exc:
            problems.append(f"{source}:{line_no}: {exc}")
    return records, problems
