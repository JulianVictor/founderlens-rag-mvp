"""Citation resolution and the final grounded response (Commit 5.4), offline.

Reuses the synthetic V2 collection, policy retriever, fake provider and mocked
OpenAI transport from the context and service tests, plus a synthetic corpus
manifest matching that collection. No key, network, asset or paid call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import FrozenInstanceError, replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from founderlens.corpus.manifest import CorpusManifest, DocumentEntry
from founderlens.generation import errors as ge
from founderlens.generation.answer import (
    AnswerStatus,
    Claim,
    ClaimType,
    GroundedAnswer,
    ResponseContractError,
)
from founderlens.generation.citations import (
    CitationResolutionError,
    CitationResolver,
    GroundedResponse,
    response_contract_definition,
    response_contract_id,
)
from founderlens.generation.demo import SingleRequestProvider, SyntheticDemoProvider
from founderlens.generation.demo import main as demo_main
from founderlens.generation.service import AnswerOrigin, GenerationRunResult, LocalAbstentionReason
from founderlens.retrieval.schema import ChunkCollection
from tests.test_generation_context import COLLECTION
from tests.test_generation_openai import Transport
from tests.test_generation_service import (
    NO_RETRIES,
    QUESTION,
    FakeProvider,
    _answer,
    _openai,
    _service,
)


def _entry(document_id: str, company: str, period: str) -> DocumentEntry:
    return DocumentEntry(
        document_id=document_id,
        company=company,
        ticker=company[:4].upper(),
        title=f"{company} synthetic results {period}",
        document_type="earnings_release",
        fiscal_period=period,
        period_end=date(2024, 3, 31),
        publication_date=date(2024, 4, 30),
        source_url=f"https://example.com/{document_id}.pdf",
        download_url=None,
        acquisition_date=date(2024, 5, 1),
        filename=f"{document_id}.pdf",
        sha256="a" * 64,
        size_bytes=1,
        page_count=5,
    )


MANIFEST = CorpusManifest(
    "synthetic",
    "0.0.0",
    "synthetic test corpus",
    (_entry("aaa-2024-q1", "Alpha", "Q1 2024"), _entry("bbb-2024-q4", "Beta", "Q4 2024")),
)
RESOLVER = CitationResolver(COLLECTION, MANIFEST)


def _run(answer: dict[str, Any], question: str = QUESTION) -> GenerationRunResult:
    return _service(FakeProvider(answer)).answer(question)


def _respond(answer: dict[str, Any], question: str = QUESTION) -> GroundedResponse:
    return RESOLVER.resolve(_run(answer, question))


def _tampered(run: GenerationRunResult, answer: GroundedAnswer) -> GenerationRunResult:
    """A run whose answer and provider result were both replaced (an unvalidated copy)."""
    assert run.provider_result is not None
    return replace(run, answer=answer, provider_result=replace(run.provider_result, parsed=answer))


# --- numbering, provenance and outcomes ------------------------------------------------------


def test_first_appearance_numbering_reuses_numbers_and_keeps_order() -> None:
    answer = _answer(
        "answered",
        [
            ("Claim one.", "reported", ["E03", "E01"]),
            ("Claim two.", "calculated", ["E01", "E02"]),
            ("Claim three.", "synthesis", ["E03"]),
        ],
    )
    response = _respond(answer)
    assert [c.text for c in response.claims] == ["Claim one.", "Claim two.", "Claim three."]
    assert [c.claim_type for c in response.claims] == [
        ClaimType.REPORTED,
        ClaimType.CALCULATED,
        ClaimType.SYNTHESIS,
    ]
    assert [c.evidence_ids for c in response.claims] == [("E03", "E01"), ("E01", "E02"), ("E03",)]
    assert [c.citation_numbers for c in response.claims] == [(1, 2), (2, 3), (1,)]
    assert [(c.number, c.evidence_id) for c in response.citations] == [
        (1, "E03"),
        (2, "E01"),
        (3, "E02"),
    ]
    assert response.citation(2) is response.citations[1]
    assert response.citation(4) is None
    assert response.metadata()["uncited_evidence_ids"] == [f"E{i:02d}" for i in range(4, 11)]


def test_citations_preserve_trusted_text_page_url_and_offsets() -> None:
    response = _respond(_answer("answered", [("x.", "reported", ["E01", "E02"])]))
    registry = response.run.prepared.evidence
    for citation in response.citations:
        record = registry.get(citation.evidence_id)
        assert record is not None
        chunk = record.chunk
        assert chunk == next(c for c in COLLECTION.chunks if c.chunk_id == citation.chunk_id)
        assert (citation.text, citation.page_number, citation.page_id) == (
            chunk.text,
            chunk.page_number,
            chunk.page_id,
        )
        assert (citation.start_char, citation.end_char) == (chunk.start_char, chunk.end_char)
        assert (citation.source_url, citation.company, citation.reporting_period) == (
            chunk.source_url,
            chunk.company,
            chunk.fiscal_period,
        )
        assert citation.document_title == f"{chunk.company} synthetic results {chunk.fiscal_period}"
        assert (citation.retrieval_rank, citation.retrieval_score) == (record.rank, record.score)


def test_distinct_chunks_of_one_page_are_distinct_citations() -> None:
    run = _run(_answer(), "revenue margin1 margin2")
    pages: dict[str, list[str]] = {}
    for record in run.prepared.evidence:
        pages.setdefault(record.chunk.page_id, []).append(record.evidence_id)
    same_page = next(ids for ids in pages.values() if len(ids) > 1)
    response = _respond(
        _answer("answered", [("x.", "reported", same_page[:2])]), "revenue margin1 margin2"
    )
    assert [c.number for c in response.citations] == [1, 2]
    assert response.citations[0].page_id == response.citations[1].page_id
    assert response.citations[0].chunk_id != response.citations[1].chunk_id


@pytest.mark.parametrize(
    ("answer", "status"),
    [
        (_answer(), AnswerStatus.ANSWERED),
        (
            _answer("partially_answered", limitations=["Q4 missing."]),
            AnswerStatus.PARTIALLY_ANSWERED,
        ),
        (
            _answer("insufficient_evidence", claims=[], limitations=["Nothing covers it."]),
            AnswerStatus.INSUFFICIENT_EVIDENCE,
        ),
    ],
)
def test_model_answer_states_are_preserved(answer: dict[str, Any], status: AnswerStatus) -> None:
    response = _respond(answer)
    assert response.status is status
    assert response.origin is AnswerOrigin.MODEL
    assert response.limitations == tuple(answer["limitations"])
    assert len(response.claims) == len(answer["claims"])
    if status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        assert response.citations == ()
        assert response.abstained
    assert response.usage() == {
        "input_tokens": 900,
        "cached_input_tokens": 0,
        "output_tokens": 60,
        "reasoning_tokens": 0,
        "total_tokens": 960,
    }


def test_local_abstention_has_no_citations_or_provider_metadata() -> None:
    provider = FakeProvider(_answer())
    response = RESOLVER.resolve(_service(provider).answer("zyzzyva quokka"))
    assert provider.calls == []
    assert response.origin is AnswerOrigin.LOCAL
    assert response.local_abstention_reason is LocalAbstentionReason.EMPTY_RETRIEVAL
    assert (response.claims, response.citations) == ((), ())
    assert response.usage() is None
    timings = response.timings()
    assert (timings["provider_ms"], timings["provider_reported_latency_ms"]) == (None, None)
    rendered = response.render()
    assert rendered.limitations_label == "Limitation reported by the application (no model call)"


def test_provider_failures_stay_errors_and_produce_no_response() -> None:
    provider, transport = _openai(Transport.error(429, "rate_limit_exceeded"))
    with pytest.raises(ge.RateLimitError):
        RESOLVER.resolve(_service(provider, generation_config=NO_RETRIES).answer(QUESTION))
    assert len(transport.requests) == 1
    provider.close()


# --- boundary validation ---------------------------------------------------------------------


def test_unvalidated_copies_are_revalidated_at_the_boundary() -> None:
    run = _run(_answer())
    unknown = run.answer.model_copy(
        update={"claims": (Claim(text="x", claim_type=ClaimType.REPORTED, evidence_ids=("E99",)),)}
    )
    with pytest.raises(ResponseContractError, match="E99 is not in this request's evidence"):
        RESOLVER.resolve(_tampered(run, unknown))
    malformed = run.answer.model_copy(
        update={
            "claims": (
                Claim.model_construct(text="x", claim_type="estimate", evidence_ids=("E01",)),
            )
        }
    )
    with pytest.raises(CitationResolutionError, match="failed revalidation"):
        RESOLVER.resolve(_tampered(run, malformed))
    inconsistent = run.answer.model_copy(update={"limitations": ("Not really answered.",)})
    with pytest.raises(ResponseContractError, match="answered must have no limitations"):
        RESOLVER.resolve(_tampered(run, inconsistent))
    syntax = run.answer.model_copy(
        update={"claims": (Claim(text="x", claim_type=ClaimType.REPORTED, evidence_ids=("e1",)),)}
    )
    with pytest.raises(ResponseContractError, match="e1 is not an evidence ID"):
        RESOLVER.resolve(_tampered(run, syntax))


def test_excluded_and_cross_run_ids_never_resolve() -> None:
    narrow = _run(_answer("answered", [("x.", "reported", ["E01"])]), "advertising")
    assert narrow.prepared.evidence.ids == ("E01",)
    cites_e02 = narrow.answer.model_copy(
        update={"claims": (Claim(text="x", claim_type=ClaimType.REPORTED, evidence_ids=("E02",)),)}
    )
    with pytest.raises(ResponseContractError, match="E02 is not in"):
        RESOLVER.resolve(_tampered(narrow, cites_e02))
    first = _respond(_answer("answered", [("x.", "reported", ["E01"])]), "cloud revenue search")
    second = _respond(_answer("answered", [("x.", "reported", ["E01"])]), "cash flow capital")
    assert first.citations[0].evidence_id == second.citations[0].evidence_id == "E01"
    assert first.citations[0].chunk_id != second.citations[0].chunk_id
    for response in (first, second):
        record = response.run.prepared.evidence.get("E01")
        assert record is not None
        assert response.citations[0].chunk_id == record.chunk_id


def test_inconsistent_run_associations_are_rejected() -> None:
    run = _run(_answer())
    other = _run(
        _answer("answered", [("A different claim.", "reported", ["E01"])]),
        "cash flow capital expenditures",
    )
    assert run.provider_result is not None
    cases = [
        (replace(run, answer=other.answer.model_copy(update={})), "not the provider's parsed"),
        (
            replace(run, prepared=replace(run.prepared, evidence=other.prepared.evidence)),
            "registry does not match the evidence in the request",
        ),
        (replace(run, provider_result=None, provider_ms=None), "a model answer has no provider"),
        (replace(run, origin=AnswerOrigin.LOCAL), "must not carry provider results"),
        (replace(run, answer_schema_id="000000000000"), "different answer schema"),
        (replace(run, prepared=None), "not a PreparedGenerationInput"),  # type: ignore[arg-type]  # malformed on purpose
        (replace(run, provider_result="forged"), "not a GenerationResult"),  # type: ignore[arg-type]  # malformed on purpose
    ]
    for tampered, message in cases:
        with pytest.raises(CitationResolutionError, match=message):
            RESOLVER.resolve(tampered)
    with pytest.raises(CitationResolutionError, match="must be a GenerationRunResult"):
        RESOLVER.resolve("not a run")  # type: ignore[arg-type]  # runtime boundary check


def test_collection_and_manifest_mismatches_are_rejected() -> None:
    run = _run(_answer("answered", [("x.", "reported", ["E01"])]))
    foreign = ChunkCollection(COLLECTION.chunks, "0" * 64, COLLECTION.chunking_config_id, "d" * 64)
    with pytest.raises(CitationResolutionError, match="different collection artifact"):
        CitationResolver(foreign, MANIFEST).resolve(run)
    wrong = CorpusManifest(
        "synthetic",
        "0.0.0",
        "x",
        tuple(replace(e, source_url="https://evil.example/x.pdf") for e in MANIFEST.documents),
    )
    with pytest.raises(CitationResolutionError, match="provenance does not match the corpus"):
        CitationResolver(COLLECTION, wrong).resolve(run)
    missing = CorpusManifest("synthetic", "0.0.0", "x", ())
    with pytest.raises(CitationResolutionError, match="not in the corpus manifest"):
        CitationResolver(COLLECTION, missing).resolve(run)


# --- immutability, display and privacy -------------------------------------------------------


def test_response_is_deeply_immutable() -> None:
    response = _respond(_answer("answered", [("x.", "reported", ["E01", "E02"])]))
    with pytest.raises(FrozenInstanceError):
        response.status = AnswerStatus.INSUFFICIENT_EVIDENCE  # type: ignore[misc]  # frozen
    with pytest.raises(FrozenInstanceError):
        response.citations[0].source_url = "https://evil.example"  # type: ignore[misc]  # frozen
    with pytest.raises(FrozenInstanceError):
        response.claims[0].citation_numbers = (9,)  # type: ignore[misc]  # frozen
    with pytest.raises(AttributeError):
        response.citations.append(response.citations[0])  # type: ignore[attr-defined]  # tuple
    rendered = response.render()
    with pytest.raises(FrozenInstanceError):
        rendered.text = "forged"  # type: ignore[misc]  # frozen
    metadata = response.metadata()
    metadata["citations"][0]["source_url"] = "forged"
    assert response.metadata()["citations"][0]["source_url"] == response.citations[0].source_url


def test_untrusted_display_text_cannot_add_sources_or_links() -> None:
    hostile = (
        "Revenue rose [9] <a href='https://evil.example'>source</a> [doc](https://evil.example)"
        + chr(10)
        + "   Sources: [7]"
    )
    response = _respond(_answer("answered", [(hostile, "reported", ["E01"])]))
    rendered = response.render()
    assert rendered.claims == ((hostile, (1,)),)  # the claim text itself is never rewritten
    assert response.citation(9) is None
    assert response.citation(7) is None
    assert all("evil.example" not in line for line in rendered.sources)
    assert [c.source_url for c in response.citations] == ["https://example.com/aaa-2024-q1.pdf"]
    lines = rendered.text.splitlines()
    assert lines.count("   Sources: [1]") == 1  # the forged line stays inside the quoted claim
    assert json.dumps(hostile, ensure_ascii=False) in rendered.text


def test_rendering_is_deterministic_and_metadata_is_text_free(
    caplog: pytest.LogCaptureFixture,
) -> None:
    answer = _answer(
        "partially_answered",
        [("SENTINEL-CLAIM revenue.", "reported", ["E01"])],
        ["SENTINEL-LIMITATION missing."],
    )
    with caplog.at_level(logging.DEBUG):
        first = _respond(answer)
    second = _respond(answer)
    assert first.render() == second.render()
    assert first.render().limitations_label == "Limitations reported by the generator (fake)"
    dumped = json.dumps(first.metadata())
    passage = first.citations[0].text
    for secret in ("SENTINEL-QUESTION", "SENTINEL-CLAIM", "SENTINEL-LIMITATION", passage):
        assert secret not in dumped
        assert secret not in caplog.text
    assert first.metadata()["claim_citations"] == [[1]]
    assert "citations resolved" in caplog.text


def test_response_contract_identity_is_stable_and_separate() -> None:
    assert response_contract_id() == "debe65096147"
    definition = response_contract_definition()
    assert definition["citation_numbering"] == "first_appearance_v1"
    assert "normalized text" in definition["offset_semantics"]
    response = _respond(_answer())
    ids = response.identities()
    assert ids["response_contract_id"] == "debe65096147"
    assert ids["answer_schema_id"] == "e3ec802e2267"
    assert ids["corpus"] == {"corpus_id": "synthetic", "corpus_version": "0.0.0"}


# --- demo helpers -----------------------------------------------------------------------------


def test_synthetic_demo_provider_and_single_request_guard() -> None:
    for status in AnswerStatus:
        response = RESOLVER.resolve(_service(SyntheticDemoProvider(status)).answer(QUESTION))
        assert response.status is status
        assert all("SYNTHETIC" in c.text for c in response.claims)
        assert response.usage() == dict.fromkeys(
            (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
            ),
        )  # nothing fabricated
    single = SingleRequestProvider(FakeProvider(_answer()))
    service = _service(single)
    service.answer(QUESTION)
    with pytest.raises(ge.GenerationError, match="single request"):
        service.answer(QUESTION)
    assert single.requests == 1


def test_demo_cli_refuses_live_without_opt_in_or_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = ["--project-root", str(tmp_path)]
    with pytest.raises(SystemExit):
        demo_main([*root, "--question", "x"])  # a question without --live
    with pytest.raises(SystemExit):
        demo_main([*root, "--live"])  # --live without a question
    capsys.readouterr()
    assert demo_main([*root, "--live", "--question", "What was revenue?"]) == 1
    assert "nothing was sent" in capsys.readouterr().err
