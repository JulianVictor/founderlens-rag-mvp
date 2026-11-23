"""Development-split diagnostic generation evaluation (Commit 6.3), offline.

The run-loop tests use a small synthetic NVIDIA/Alphabet collection, the real
BM25 policy retriever, context builder, service, citation resolver and
guardrails, synthetic evaluation questions, and a scripted provider. No model,
key, network, tokenizer asset or real report is used, and files are written
only under ``tmp_path``. Split-guard tests read the tracked development split
through the authorized loader without printing it; the held-out file is never
opened.
"""

from __future__ import annotations

import json
import pathlib
import socket
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from founderlens.config import load_settings
from founderlens.evaluation import generation_eval as ge
from founderlens.evaluation.benchmark import DevSplit, EvaluationError, load_dev_split
from founderlens.evaluation.schema import Category, EvalQuestion, EvidenceRef, RequiredFact
from founderlens.generation import errors as gerr
from founderlens.generation.config import DEFAULT_GENERATION_CONFIG
from founderlens.generation.openai_provider import OpenAIProvider
from founderlens.generation.schema import (
    GenerationRequest,
    GenerationResult,
    ProviderIdentity,
    SchemaT,
    TokenUsage,
)
from founderlens.guardrails.response_checks import GuardrailError
from founderlens.guardrails.response_checks import check_response as real_check
from tests.test_generation_context import CHARS
from tests.test_generation_service import SERVICE_CONFIG
from tests.test_guardrails_response_checks import G3, G4, N3, N4, RESOLVER, RETRIEVER, VOCAB

Claim = tuple[str, str, list[str]]  # text, claim type, ["document_id[:page]"]


def _question(
    qid: str,
    text: str,
    category: Category,
    facts: Mapping[str, list[tuple[str, int]]] | None = None,
) -> EvalQuestion:
    facts = facts or {}
    return EvalQuestion(
        question_id=qid,
        question=text,
        category=category,
        answerable=category is not Category.UNANSWERABLE,
        reference_answer=None,
        required_facts=tuple(RequiredFact(f, "synthetic fact", None) for f in facts),
        accepted_evidence=tuple(
            EvidenceRef(f, d, p) for f, pages in facts.items() for d, p in pages
        ),
        expected_abstention_reason=None,
    )


QUESTIONS = (
    _question("dev-001", "What was NVIDIA revenue SENTINEL-QUESTION?", Category.SINGLE_DOC_FACTUAL,
              {"dev-001-f1": [(N4, 1)]}),
    _question("dev-002", "Compare NVIDIA and Alphabet revenue.", Category.CROSS_DOC_COMPARISON,
              {"dev-002-f1": [(N4, 1)], "dev-002-f2": [(G4, 1)]}),
    _question("dev-003", "NVIDIA revenue in FY2025 Q3 and FY2025 Q4?",
              Category.TEMPORAL_MULTI_EVIDENCE, {"dev-003-f1": [(N3, 1)], "dev-003-f2": [(N4, 1)]}),
    _question("dev-004", "What was Alphabet revenue from Antarctica?", Category.UNANSWERABLE),
    _question("dev-005", "zyzzyva quokka", Category.UNANSWERABLE),  # empty retrieval: local
    _question("dev-006", "What was Alphabet consolidated revenue?", Category.SINGLE_DOC_CONCEPTUAL,
              {"dev-006-f1": [(G3, 1), (G4, 1)]}),
)  # fmt: skip
SPLIT = DevSplit(QUESTIONS, "test-0", "a" * 64, "b" * 64)
PIPELINE = ge.Pipeline(
    retriever=RETRIEVER,
    counter=CHARS,
    resolver=RESOLVER,
    vocabulary=VOCAB,
    corpus_version="0.2.0",
    context_config=SERVICE_CONFIG,
)
IDS = PIPELINE.identities()
# question text -> (status, claims); every claim cites by document (page 1 unless given)
DEFAULT_SCRIPT: dict[str, tuple[str, list[Claim]]] = {
    QUESTIONS[0].question: ("answered", [("SENTINEL-CLAIM $39.3 billion.", "reported", [N4])]),
    QUESTIONS[1].question: ("partially_answered", [("NVIDIA $39.3 billion.", "reported", [N4])]),
    QUESTIONS[2].question: (
        "answered",
        [("NVIDIA $35.1 billion.", "reported", [N3]), ("NVIDIA $39.3 billion.", "reported", [N4])],
    ),
    QUESTIONS[3].question: ("partially_answered", [("Alphabet $96.5 billion.", "reported", [G4])]),
    QUESTIONS[5].question: ("insufficient_evidence", []),
}


class ScriptProvider:
    """Answers from a script keyed by the question in the request; can fail on demand."""

    def __init__(
        self,
        script: Mapping[str, tuple[str, list[Claim]]] | None = None,
        fail: Mapping[str, BaseException] | None = None,
        on_call: Callable[[int], None] | None = None,
    ) -> None:
        self.script = dict(DEFAULT_SCRIPT if script is None else script)
        self.fail = dict(fail or {})
        self.on_call = on_call
        self.calls = 0
        self.closed = False
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
        if self.on_call is not None:
            self.on_call(self.calls)
        payload_in = json.loads(request.input_text)
        question = payload_in["question"]
        if question in self.fail:
            raise self.fail[question]
        evidence = payload_in["evidence"]
        status, claims = self.script[question]

        def ids(refs: list[str]) -> list[str]:
            out = []
            for ref in refs:
                doc, _, page = ref.partition(":")
                out.append(
                    next(
                        e["evidence_id"]
                        for e in evidence
                        if e["document_id"] == doc and e["page_number"] == int(page or 1)
                    )
                )
            return out

        answer = {
            "status": status,
            "claims": [{"text": t, "claim_type": k, "evidence_ids": ids(r)} for t, k, r in claims],
            "limitations": [] if status == "answered" else ["SENTINEL-LIMITATION"],
        }
        text = json.dumps(answer)
        return GenerationResult(
            text=text,
            parsed=schema.model_validate_json(text),
            status="completed",
            requested_model=self._identity.model,
            response_model="fake",
            response_id="fake",
            request_id=None,
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            latency_ms=1.0,
            provider=self._identity,
        )

    def close(self) -> None:
        self.closed = True


def _execute(
    tmp_path: pathlib.Path,
    provider: ScriptProvider | None = None,
    *,
    mode: ge.Mode = "dry_run",
    max_requests: int | None = None,
    replay: list[dict[str, Any]] | None = None,
    price: ge.Price | None = None,
) -> tuple[ge.RunOutputs, ScriptProvider]:
    used = provider or ScriptProvider()
    outputs = ge.execute(
        SPLIT,
        PIPELINE,
        tmp_path / "reports",
        mode=mode,
        provider_factory=lambda: used,
        max_requests=max_requests,
        expected_identities=IDS,
        replay_records=replay,
        price=price,
    )
    return outputs, used


def _records(outputs: ge.RunOutputs) -> dict[str, dict[str, Any]]:
    return {r["question_id"]: r for r in ge.load_records(outputs.records_path)}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


# --- split, dataset and identity guards -----------------------------------------------------


def test_only_the_development_split_is_accepted(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        ge.main(["--split", "test"])
    assert exit_info.value.code == 2
    assert "held-out evaluation is not authorized" in capsys.readouterr().err


def test_the_held_out_file_is_never_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    for name in ("read_bytes", "read_text", "open"):
        original = getattr(pathlib.Path, name)

        def spy(self: pathlib.Path, *args: Any, _original: Any = original, **kw: Any) -> Any:
            opened.append(self.name)
            if self.name == "questions.test.jsonl":
                raise AssertionError("held-out file opened")
            return _original(self, *args, **kw)

        monkeypatch.setattr(pathlib.Path, name, spy)
    settings = load_settings()
    split = load_dev_split(settings.evaluation_dir, settings.corpus_manifest_path)
    assert len(split.questions) == 24
    assert {q.split for q in split.questions} == {"dev"}
    assert "questions.test.jsonl" not in opened
    assert "questions.dev.jsonl" in opened


def test_dataset_hash_mismatch_is_refused(tmp_path: pathlib.Path) -> None:
    settings = load_settings()
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    for name in ("dataset.json", "questions.dev.jsonl"):
        (evaluation / name).write_bytes((settings.evaluation_dir / name).read_bytes())
    tampered = evaluation / "questions.dev.jsonl"
    tampered.write_bytes(tampered.read_bytes() + b"\n")
    with pytest.raises(EvaluationError, match="SHA-256 does not match"):
        load_dev_split(evaluation, settings.corpus_manifest_path)


def test_identity_mismatch_aborts_before_any_provider_exists(tmp_path: pathlib.Path) -> None:
    def factory() -> Any:
        raise AssertionError("provider constructed despite an identity mismatch")

    wrong = {**IDS, "prompt_id": "000000000000"}
    with pytest.raises(ge.EvaluationRunError, match="prompt_id"):
        ge.execute(SPLIT, PIPELINE, tmp_path, mode="dry_run", provider_factory=factory,
                   expected_identities=wrong)  # fmt: skip
    assert not (tmp_path / ge.REPORT_SUBDIR).exists()
    with pytest.raises(ge.EvaluationRunError, match="keys differ"):
        ge.verify_identities({**IDS, "extra": "x"}, IDS)


def test_frozen_identities_and_retry_free_live_configuration() -> None:
    assert ge.EXPECTED_IDENTITIES["guardrail_policy_id"] == "3f202ebc205a"
    assert ge.EXPECTED_IDENTITIES["retrieval_policy_id"] == "f20913f1c1ec"
    config = ge.EVALUATION_GENERATION_CONFIG
    assert config.max_retries == 0
    assert config.config_id == DEFAULT_GENERATION_CONFIG.config_id == "cbc4e415c848"
    live = ge._provider_factory("live", {"OPENAI_API_KEY": "test-key-not-real"})()
    try:
        assert live.identity.config_id == "cbc4e415c848"
        assert isinstance(live, OpenAIProvider)
        assert live.config.max_retries == 0
    finally:
        live.close()


# --- paid-run authorization -----------------------------------------------------------------

CLEAN = {"commit": "abc", "working_tree_clean": True}
AUTHORIZED = {ge.PAID_ENV_VAR: "1", "OPENAI_API_KEY": "test-key-not-real"}


@pytest.mark.parametrize(
    ("max_requests", "environ", "git", "expected"),
    [
        (5, {"OPENAI_API_KEY": "k"}, CLEAN, ge.PAID_ENV_VAR),
        (None, AUTHORIZED, CLEAN, "--max-requests is required"),
        (25, AUTHORIZED, CLEAN, "must be in [1, 24]"),
        (0, AUTHORIZED, CLEAN, "must be in [1, 24]"),
        (5, AUTHORIZED, {"commit": "abc", "working_tree_clean": False}, "clean Git working tree"),
        (5, AUTHORIZED, {"commit": None, "working_tree_clean": None}, "clean Git working tree"),
        (5, {ge.PAID_ENV_VAR: "1"}, CLEAN, "OPENAI_API_KEY is not configured"),
    ],
)
def test_live_runs_need_every_authorization(
    max_requests: int | None, environ: dict[str, str], git: dict[str, Any], expected: str
) -> None:
    problems = ge.live_authorization_problems(max_requests=max_requests, environ=environ, git=git)
    assert any(expected in p for p in problems)
    assert not any("test-key-not-real" in p for p in problems)


def test_complete_authorization_has_no_problems() -> None:
    assert ge.live_authorization_problems(max_requests=24, environ=AUTHORIZED, git=CLEAN) == []


def test_cli_refuses_live_without_authorization(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pipeline or provider built before authorization")

    monkeypatch.setattr(ge, "load_pipeline", forbidden)
    monkeypatch.setattr(ge, "_provider_factory", forbidden)
    assert ge.main(["--live", "--max-requests", "3"], environ={}) == 1
    assert ge.PAID_ENV_VAR in capsys.readouterr().err
    assert ge.main(["--max-requests", "25"], environ={}) == 1


def _fake_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    provider: ScriptProvider,
    *,
    clean: bool = True,
    expected: Mapping[str, str] = IDS,
) -> list[str]:
    """Run the real CLI path against the synthetic pipeline, a mocked provider and tmp_path."""
    built: list[str] = []
    settings = SimpleNamespace(
        project_root=tmp_path,
        reports_dir=tmp_path / "reports",
        evaluation_dir=tmp_path,
        corpus_manifest_path=tmp_path / "manifest.json",
    )
    monkeypatch.setattr(ge, "load_settings", lambda root=None: settings)
    monkeypatch.setattr(ge, "load_dev_split", lambda *a: SPLIT)
    monkeypatch.setattr(ge, "load_pipeline", lambda s: PIPELINE)
    monkeypatch.setattr(ge, "EXPECTED_IDENTITIES", expected)
    monkeypatch.setattr(
        ge, "_git_state", lambda root=None: {"commit": "c", "working_tree_clean": clean}
    )

    def factory(mode: str, environ: Mapping[str, str]) -> Callable[[], Any]:
        def build() -> Any:
            built.append(mode)
            return provider

        return build

    monkeypatch.setattr(ge, "_provider_factory", factory)
    return built


def test_cli_live_path_with_mocked_provider_respects_every_safeguard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = ScriptProvider()
    built = _fake_cli(monkeypatch, tmp_path, provider)
    assert ge.main(["--live", "--max-requests", "3"], environ=AUTHORIZED) == 0
    out = capsys.readouterr().out
    assert built == ["live"]
    assert provider.calls == 3  # the cap, not the five questions that would reach a provider
    assert "origin live" in out
    reports = list((tmp_path / "reports" / ge.REPORT_SUBDIR).glob("live-*.json"))
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["request_budget"]["attempted_requests"] == 3
    assert report["run_complete"] is False  # budget-exhausted: incomplete, never reportable
    assert report["reportability"]["reportable_as_hosted_model_diagnostic"] is False


@pytest.mark.parametrize(
    ("argv", "environ", "clean", "expected_ids", "message"),
    [
        (["--live", "--max-requests", "3"], AUTHORIZED, False, IDS, "clean Git working tree"),
        (["--live"], AUTHORIZED, True, IDS, "--max-requests is required"),
        (["--live", "--max-requests", "3"], {ge.PAID_ENV_VAR: "1"}, True, IDS, "OPENAI_API_KEY"),
        (["--live", "--max-requests", "3"], AUTHORIZED, True, {**IDS, "model": "other"}, "model"),
    ],
)
def test_cli_live_refusals_never_build_a_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    environ: dict[str, str],
    clean: bool,
    expected_ids: Mapping[str, str],
    message: str,
) -> None:
    provider = ScriptProvider()
    built = _fake_cli(monkeypatch, tmp_path, provider, clean=clean, expected=expected_ids)
    assert ge.main(argv, environ=environ) == 1
    assert message in capsys.readouterr().err
    assert built == []
    assert provider.calls == 0
    assert not (tmp_path / "reports").exists()


# --- execution, budget and single generation ------------------------------------------------


def test_each_question_is_generated_once_and_guardrails_use_that_response(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved: list[Any] = []
    checked: list[Any] = []
    real_resolve = RESOLVER.resolve

    class SpyResolver:
        def resolve(self, run: Any) -> Any:
            response = real_resolve(run)
            resolved.append(response)
            return response

    def spy_check(response: Any, vocabulary: Any) -> Any:
        checked.append(response)
        return real_check(response, vocabulary)

    monkeypatch.setattr(ge, "check_response", spy_check)
    pipeline = replace(PIPELINE, resolver=SpyResolver())  # type: ignore[arg-type]  # spy
    provider = ScriptProvider()
    outputs = ge.execute(SPLIT, pipeline, tmp_path, mode="dry_run",
                         provider_factory=lambda: provider, expected_identities=IDS)  # fmt: skip
    records = _records(outputs)
    assert provider.calls == 5  # dev-005 is a local abstention: no provider call
    assert provider.closed
    assert len(resolved) == len(checked) == 6
    assert all(a is b for a, b in zip(resolved, checked, strict=True))
    assert records["dev-005"]["origin"] == "local"
    assert records["dev-005"]["provider_attempted"] is False
    assert outputs.report["request_budget"]["attempted_requests"] == 5
    assert outputs.report["run_complete"] is True


def test_request_budget_is_enforced_before_sending(tmp_path: pathlib.Path) -> None:
    outputs, provider = _execute(tmp_path, max_requests=2)
    records = _records(outputs)
    assert provider.calls == 2
    assert [records[q]["outcome"] for q in sorted(records)] == [
        "completed", "completed", "not_run_budget", "not_run_budget", "completed",
        "not_run_budget",
    ]  # fmt: skip
    assert outputs.report["request_budget"] == {
        "max_requests": 2,
        "hard_ceiling": 24,
        "attempted_requests": 2,
        "counts": "attempted provider requests, including failed attempts",
    }
    assert outputs.report["run_complete"] is False
    with pytest.raises(ge.EvaluationRunError):
        ge.BudgetedProvider(ScriptProvider(), 25)


def test_failed_attempts_count_against_the_budget(tmp_path: pathlib.Path) -> None:
    fail = {QUESTIONS[0].question: gerr.ProviderTimeoutError("synthetic timeout")}
    outputs, provider = _execute(tmp_path, ScriptProvider(fail=fail), max_requests=1)
    records = _records(outputs)
    assert provider.calls == 1
    assert (records["dev-001"]["outcome"], records["dev-001"]["error_category"]) == (
        "failed",
        "timeout",
    )
    assert records["dev-001"]["provider_attempted"] is True
    assert records["dev-002"]["outcome"] == "not_run_budget"


# --- failures and interruption --------------------------------------------------------------


def test_failures_are_not_abstentions_and_systemic_failures_abort(tmp_path: pathlib.Path) -> None:
    fail = {
        QUESTIONS[0].question: gerr.RateLimitError("synthetic"),
        QUESTIONS[1].question: gerr.ModelRefusalError("synthetic", refusal="SENTINEL-REFUSAL"),
        QUESTIONS[3].question: gerr.AuthenticationError("synthetic"),
    }
    outputs, provider = _execute(tmp_path, ScriptProvider(fail=fail))
    records = _records(outputs)
    assert [(q, records[q]["outcome"], records[q]["error_category"]) for q in sorted(records)] == [
        ("dev-001", "failed", "rate_limited"),
        ("dev-002", "failed", "refusal"),
        ("dev-003", "completed", None),
        ("dev-004", "failed", "authentication"),
        ("dev-005", "not_run_aborted", None),
        ("dev-006", "not_run_aborted", None),
    ]
    assert provider.calls == 4
    overall = outputs.report["overall"]
    assert overall["status_distribution"]["denominator"] == 1
    assert overall["abstention_on_unanswerable_rate"]["denominator"] == 0
    assert overall["abstention_on_unanswerable_rate"]["rate"] == "not_applicable"
    assert overall["error_categories"] == {"authentication": 1, "rate_limited": 1, "refusal": 1}
    assert outputs.report["run_complete"] is False
    assert any("aborted" in p for p in outputs.report["completion_problems"])
    text = outputs.json_path.read_text(encoding="utf-8") + outputs.records_path.read_text("utf-8")
    assert "[rate_limited]" not in text  # no raw exception messages


def test_citation_and_guardrail_failures(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise GuardrailError("synthetic")

    monkeypatch.setattr(ge, "check_response", broken)
    outputs, _ = _execute(tmp_path)
    records = _records(outputs)
    assert records["dev-001"]["outcome"] == "completed"
    assert records["dev-001"]["guardrails"] == {"status": "failed", "report": None}
    assert records["dev-001"]["answer"]["status"] == "answered"  # the response survives
    assert outputs.report["overall"]["guardrails"]["failed"] == 6
    assert outputs.report["run_complete"] is False

    class FailingResolver:
        def resolve(self, run: Any) -> Any:
            raise gerr.GenerationError("synthetic")

    monkeypatch.undo()
    pipeline = replace(PIPELINE, resolver=FailingResolver())  # type: ignore[arg-type]  # fake
    out = ge.execute(SPLIT, pipeline, tmp_path / "b", mode="dry_run",
                     provider_factory=ScriptProvider, expected_identities=IDS)  # fmt: skip
    failed = _records(out)["dev-001"]
    assert (failed["outcome"], failed["error_category"]) == ("failed", "generation_error")
    assert failed["answer"] is not None


def test_interrupted_runs_are_explicitly_incomplete(tmp_path: pathlib.Path) -> None:
    def interrupt(call: int) -> None:
        if call == 3:
            raise KeyboardInterrupt

    outputs, _ = _execute(tmp_path, ScriptProvider(on_call=interrupt))
    records = _records(outputs)
    assert outputs.interrupted
    assert [records[q]["outcome"] for q in sorted(records)] == [
        "completed", "completed", "not_run_interrupted", "not_run_interrupted",
        "not_run_interrupted", "not_run_interrupted",
    ]  # fmt: skip
    assert "the run was interrupted" in outputs.report["completion_problems"]


def test_records_are_persisted_incrementally(tmp_path: pathlib.Path) -> None:
    seen: list[int] = []
    directory = tmp_path / "reports" / ge.REPORT_SUBDIR

    def check(call: int) -> None:
        files = list(directory.glob("*.records.jsonl"))
        seen.append(len(files[0].read_text(encoding="utf-8").splitlines()))

    _execute(tmp_path, ScriptProvider(on_call=check))
    assert seen == [0, 1, 2, 3, 5]  # dev-005 (local abstention) needs no provider call


# --- metrics and denominators ---------------------------------------------------------------


def test_status_and_abstention_denominators(tmp_path: pathlib.Path) -> None:
    outputs, _ = _execute(tmp_path)
    overall = outputs.report["overall"]
    assert overall["status_distribution"]["counts"] == {
        "answered": 2,
        "partially_answered": 2,
        "insufficient_evidence": 1,
        "local_abstention": 1,
    }

    def nd(name: str) -> tuple[int, int]:
        return overall[name]["numerator"], overall[name]["denominator"]

    assert nd("answer_attempt_rate") == (3, 4)
    assert nd("abstention_on_answerable_rate") == (1, 4)
    assert nd("abstention_on_unanswerable_rate") == (1, 2)  # local abstention on dev-005
    assert nd("answered_on_unanswerable_rate") == (0, 2)
    assert nd("partially_answered_on_unanswerable_rate") == (1, 2)  # kept separate
    temporal = outputs.report["by_category"]["temporal_multi_evidence"]["answer_attempt_rate"]
    assert (temporal["numerator"], temporal["denominator"], temporal["rate"]) == (1, 1, None)
    assert "counts only" in temporal["note"]


def test_selected_versus_cited_coverage(tmp_path: pathlib.Path) -> None:
    outputs, _ = _execute(tmp_path)
    overall = outputs.report["overall"]
    # dev-002 retrieved both companies but cited only NVIDIA.
    cross = outputs.report["by_category"]["cross_doc_comparison"]
    assert cross["selected_evidence_complete_fact_coverage"]["numerator"] == 1
    assert cross["cited_evidence_complete_fact_coverage"]["numerator"] == 0
    assert cross["cited_evidence_fact_coverage"]["sum"] == 0.5  # counts only (n = 1)
    assert overall["cited_evidence_page_recall"]["denominator"] == 3  # answer attempts only
    assert overall["citation_structural_invariants"]["violations"] == 0
    warnings = cross["guardrails"]["findings"]["question_coverage_gap"]
    assert warnings == {"notice": 1}  # partially_answered: selected but not cited is a notice


def _record(qid: str, category: str, *, status: str, selected: list[tuple[str, int]],
            cited: list[tuple[str, int]]) -> dict[str, Any]:  # fmt: skip
    question = next(q for q in QUESTIONS if q.question_id == qid)
    record = ge._base_record("r", "dry_run", question, IDS)
    record.update(
        outcome="completed",
        status=status,
        origin="model",
        selected_evidence=[
            {
                "evidence_id": f"E{i:02d}",
                "rank": i,
                "chunk_id": f"c{i}",
                "document_id": d,
                "page_number": p,
            }
            for i, (d, p) in enumerate(selected, start=1)
        ],
        citations=[
            {
                "number": i,
                "evidence_id": f"E{i:02d}",
                "chunk_id": f"c{i}",
                "document_id": d,
                "page_number": p,
            }
            for i, (d, p) in enumerate(cited, start=1)
        ],
    )
    assert record["category"] == category
    return record


def test_duplicate_pages_missing_evidence_and_empty_retrieval() -> None:
    questions = {q.question_id: q for q in QUESTIONS}
    duplicate = _record("dev-001", "single_doc_factual", status="answered",
                        selected=[(N4, 1), (N4, 1)], cited=[(N4, 1), (N4, 1)])  # fmt: skip
    metrics = ge.category_metrics([duplicate], questions, counts_only=False)
    assert metrics["citation_structural_invariants"]["citations"] == 2
    assert metrics["citation_structural_invariants"]["distinct_cited_pages"] == 1
    overlap = metrics["accepted_page_citation_overlap"]
    assert (overlap["numerator"], overlap["denominator"]) == (1, 1)  # not 2/2
    missing = _record("dev-006", "single_doc_conceptual", status="answered",
                      selected=[(N4, 1)], cited=[(N4, 1)])  # fmt: skip
    empty = _record("dev-002", "cross_doc_comparison", status="insufficient_evidence",
                    selected=[], cited=[])  # fmt: skip
    metrics = ge.category_metrics([missing, empty], questions, counts_only=False)
    assert metrics["selected_evidence_page_recall"]["numerator"] == 0
    assert metrics["selected_evidence_page_recall"]["denominator"] == 2
    assert metrics["selected_evidence_fact_coverage"]["mean"] == 0.0
    assert metrics["cited_evidence_page_recall"]["denominator"] == 1  # abstention not counted
    overlap = metrics["accepted_page_citation_overlap"]
    assert (overlap["numerator"], overlap["denominator"], overlap["rate"]) == (0, 1, 0.0)
    nothing = ge.category_metrics([], questions, counts_only=False)
    assert nothing["answer_attempt_rate"]["rate"] == "not_applicable"
    assert nothing["selected_evidence_fact_coverage"]["mean"] == "not_applicable"


def test_structural_invariant_violations_are_counted() -> None:
    questions = {q.question_id: q for q in QUESTIONS}
    record = _record("dev-001", "single_doc_factual", status="answered",
                     selected=[(N4, 1)], cited=[(N4, 1)])  # fmt: skip
    record["citations"][0]["page_number"] = 2
    metrics = ge.category_metrics([record], questions, counts_only=False)
    assert metrics["citation_structural_invariants"]["violations"] == 1


def test_usage_estimates_and_price_are_separate(tmp_path: pathlib.Path) -> None:
    price = ge.Price(Decimal("1.00"), Decimal("4.00"), date(2026, 11, 1))
    outputs, _ = _execute(tmp_path, price=price)
    usage = outputs.report["overall"]["provider_usage"]
    assert (usage["attempted_requests"], usage["requests_with_usage"]) == (5, 5)
    assert usage["usage_complete"] is True
    assert usage["input_tokens"] == {"reported_n": 5, "total": 500}
    assert usage["cached_input_tokens"] == {"reported_n": 0, "total": None}
    estimates = outputs.report["overall"]["local_token_estimates"]
    assert estimates["n"] == 5
    assert "not provider usage" in estimates["note"]
    cost = outputs.report["cost_estimate"]
    assert cost["estimated_usd"] == "0.000900"  # (500 * 1 + 100 * 4) / 1e6
    assert cost["price_effective_date"] == "2026-11-01"
    assert _execute(tmp_path / "x")[0].report["cost_estimate"] is None


# --- replay ---------------------------------------------------------------------------------


def test_replay_revalidates_without_generation(tmp_path: pathlib.Path) -> None:
    original, _ = _execute(tmp_path)
    source = ge.load_records(original.records_path)

    def factory() -> Any:
        raise AssertionError("replay must not build a provider")

    replayed = ge.execute(SPLIT, PIPELINE, tmp_path, mode="replay", provider_factory=factory,
                          expected_identities=IDS, replay_records=source)  # fmt: skip
    again = _records(replayed)
    first = _records(original)
    for qid, record in first.items():
        assert again[qid]["outcome"] == record["outcome"] == "completed"
        for key in ("status", "answer", "citations", "claims", "request", "selected_evidence"):
            assert again[qid][key] == record[key]
        assert again[qid]["guardrails"] == record["guardrails"]
        assert again[qid]["provider_attempted"] is False
        assert again[qid]["replay_of"]["record_sha256"] == record["record_sha256"]
    assert replayed.report["request_budget"]["attempted_requests"] == 0
    assert replayed.report["original_run_usage"]["input_tokens"] == {"reported_n": 5, "total": 500}
    assert replayed.report["overall"]["provider_usage"]["input_tokens"]["reported_n"] == 0
    assert replayed.report["replay_of"] == first["dev-001"]["run_id"]


# --- provenance: synthetic versus live origin -------------------------------------------------


def _replay(tmp_path: pathlib.Path, source: list[dict[str, Any]]) -> ge.RunOutputs:
    def factory() -> Any:
        raise AssertionError("replay must not build a provider")

    return ge.execute(SPLIT, PIPELINE, tmp_path, mode="replay", provider_factory=factory,
                      expected_identities=IDS, replay_records=source)  # fmt: skip


def test_dry_runs_and_their_replays_stay_synthetic(tmp_path: pathlib.Path) -> None:
    original, _ = _execute(tmp_path)
    assert original.report["synthetic"] is True
    assert original.report["origin_mode"] == "dry_run"
    assert original.report["provenance_notes"] == [ge.SYNTHETIC_NOTE]
    assert original.report["reportability"]["reportable_as_hosted_model_diagnostic"] is False
    replayed = _replay(tmp_path, ge.load_records(original.records_path))
    report = replayed.report
    assert (report["mode"], report["origin_mode"], report["synthetic"]) == (
        "replay",
        "dry_run",
        True,
    )
    assert report["origin_run_id"] == original.report["run_id"]
    assert report["provenance_notes"] == [ge.SYNTHETIC_NOTE, ge.REPLAY_NOTE]
    assert report["reportability"] == {
        "reportable_as_hosted_model_diagnostic": False,
        "reason": "synthetic origin: never evidence of hosted-model performance",
    }
    for record in _records(replayed).values():
        assert record["synthetic"] is True
        assert record["origin_mode"] == "dry_run"
        assert (record["usage"], record["timings"], record["provider"]) == (None, None, None)
    assert report["operational"]["latency"]["provider_ms"]["n"] == 0  # no new observations
    assert "SYNTHETIC" in replayed.markdown_path.read_text(encoding="utf-8")


def test_live_origin_replay_keeps_the_live_run_identity(tmp_path: pathlib.Path) -> None:
    live, provider = _execute(tmp_path, mode="live", max_requests=24)  # fake provider only
    assert provider.calls == 5
    assert (live.report["origin_mode"], live.report["synthetic"]) == ("live", False)
    assert live.report["provenance_notes"] == []
    assert live.report["reportability"]["reportable_as_hosted_model_diagnostic"] is True
    replayed = _replay(tmp_path, ge.load_records(live.records_path))
    report = replayed.report
    assert (report["mode"], report["origin_mode"], report["synthetic"]) == ("replay", "live", False)
    assert report["origin_run_id"] == report["replay_of"] == live.report["run_id"]
    assert report["provenance_notes"] == [ge.REPLAY_NOTE]
    assert report["reportability"]["reportable_as_hosted_model_diagnostic"] is True
    assert report["overall"]["provider_usage"]["input_tokens"]["reported_n"] == 0
    assert report["original_run_usage"]["input_tokens"] == {"reported_n": 5, "total": 500}
    for record in _records(replayed).values():
        assert record["origin_run_id"] == live.report["run_id"]
        assert record["synthetic"] is False


def test_incomplete_live_runs_are_never_reportable(tmp_path: pathlib.Path) -> None:
    fail = {QUESTIONS[0].question: gerr.ProviderServerError("synthetic")}
    live, _ = _execute(tmp_path, ScriptProvider(fail=fail), mode="live", max_requests=24)
    assert live.report["run_complete"] is False
    assert live.report["reportability"] == {
        "reportable_as_hosted_model_diagnostic": False,
        "reason": "incomplete run: not a completed 24-question diagnostic result",
    }
    replayed = _replay(tmp_path, ge.load_records(live.records_path))
    assert replayed.report["reportability"]["reportable_as_hosted_model_diagnostic"] is False
    assert "INCOMPLETE" in live.markdown_path.read_text(encoding="utf-8")


def test_replay_rejects_altered_provenance(tmp_path: pathlib.Path) -> None:
    original, _ = _execute(tmp_path)
    source = ge.load_records(original.records_path)
    relabelled = [ge.seal({**r, "synthetic": False}) for r in source]
    with pytest.raises(ge.EvaluationRunError, match="synthetic flag"):
        _replay(tmp_path / "a", relabelled)
    promoted = [ge.seal({**r, "mode": "live", "synthetic": False}) for r in source]
    with pytest.raises(ge.EvaluationRunError, match="origin provenance"):
        _replay(tmp_path / "b", promoted)
    rebadged = [ge.seal({**r, "origin_run_id": "other"}) for r in source]
    with pytest.raises(ge.EvaluationRunError, match="origin provenance"):
        _replay(tmp_path / "c", rebadged)
    with pytest.raises(ge.EvaluationRunError, match="needs records"):
        _replay(tmp_path / "d", [])


def _resealed(records: list[dict[str, Any]], qid: str, **changes: Any) -> list[dict[str, Any]]:
    out = []
    for record in records:
        if record["question_id"] == qid:
            record = ge.seal({**record, **changes})
        out.append(record)
    return out


def test_replay_rejects_tampered_or_incompatible_records(tmp_path: pathlib.Path) -> None:
    original, _ = _execute(tmp_path)
    source = ge.load_records(original.records_path)

    def run(records: list[dict[str, Any]], directory: str) -> ge.RunOutputs:
        return ge.execute(SPLIT, PIPELINE, tmp_path / directory, mode="replay",
                          provider_factory=ScriptProvider, expected_identities=IDS,
                          replay_records=records)  # fmt: skip

    unsealed = [dict(r) for r in source]
    unsealed[0]["status"] = "partially_answered"
    with pytest.raises(ge.EvaluationRunError, match="digest mismatch"):
        run(unsealed, "a")
    other_ids = _resealed(source, "dev-001", identities={**IDS, "prompt_id": "000000000000"})
    with pytest.raises(ge.EvaluationRunError, match="identities differ"):
        run(other_ids, "b")
    mixed = _resealed(source, "dev-002", run_id="another-run")
    with pytest.raises(ge.EvaluationRunError, match="never merged"):
        run(mixed, "c")
    with pytest.raises(ge.EvaluationRunError, match="duplicate"):
        run([*source, source[0]], "d")

    request = {**source[0]["request"], "input_sha256": "0" * 64}
    wrong_request = _resealed(source, "dev-001", request=request)
    first = _records(run(wrong_request, "e"))["dev-001"]
    assert (first["outcome"], first["error_category"]) == ("failed", "replay_mismatch")

    answer = json.loads(json.dumps(source[0]["answer"]))
    answer["claims"][0]["evidence_ids"] = ["E09"]  # not a selected evidence ID
    wrong_answer = _resealed(source, "dev-001", answer=answer)
    second = _records(run(wrong_answer, "f"))["dev-001"]
    assert (second["outcome"], second["error_category"]) == ("failed", "response_contract")

    citations = [{**c, "number": 7} for c in source[0]["citations"]]
    wrong_citations = _resealed(source, "dev-001", citations=citations)
    third = _records(run(wrong_citations, "g"))["dev-001"]
    assert (third["outcome"], third["error_category"]) == ("failed", "replay_mismatch")


def test_unreplayable_originals_stay_unrun(tmp_path: pathlib.Path) -> None:
    fail = {QUESTIONS[0].question: gerr.RateLimitError("synthetic")}
    original, _ = _execute(tmp_path, ScriptProvider(fail=fail))
    source = ge.load_records(original.records_path)
    replayed = ge.execute(SPLIT, PIPELINE, tmp_path, mode="replay", provider_factory=ScriptProvider,
                          expected_identities=IDS, replay_records=source)  # fmt: skip
    first = _records(replayed)["dev-001"]
    assert first["outcome"] == "not_replayable"
    assert first["replay_of"]["outcome"] == "failed"
    assert replayed.report["run_complete"] is False


# --- privacy and determinism ----------------------------------------------------------------


def test_reports_are_private_and_contained(tmp_path: pathlib.Path) -> None:
    outputs, _ = _execute(tmp_path)
    summary = outputs.json_path.read_text(encoding="utf-8") + outputs.markdown_path.read_text(
        encoding="utf-8"
    )
    for sentinel in ("SENTINEL-QUESTION", "SENTINEL-CLAIM", "SENTINEL-LIMITATION", "revenue",
                     "test-key", "sk-"):  # fmt: skip
        assert sentinel not in summary
    records = outputs.records_path.read_text(encoding="utf-8")
    assert "SENTINEL-QUESTION" not in records  # questions are never stored
    assert "SENTINEL-CLAIM" in records  # answers are, privately, for replay and review
    assert all(p.is_relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert outputs.report["evaluation_kind"] == "diagnostic_deterministic"
    assert outputs.report["caveat"] == ge.CAVEAT
    assert outputs.report["synthetic"] is True
    assert "SYNTHETIC" in outputs.markdown_path.read_text(encoding="utf-8")


def test_aggregation_is_deterministic_from_stored_records(tmp_path: pathlib.Path) -> None:
    outputs, _ = _execute(tmp_path)
    records = ge.load_records(outputs.records_path)
    kwargs: dict[str, Any] = {"run_id": outputs.report["run_id"], "mode": "dry_run",
                              "state": ge.RunState(), "max_requests": None,
                              "attempted_requests": 5}  # fmt: skip
    first = ge.aggregate(records, SPLIT, **kwargs)
    second = ge.aggregate(list(reversed(records)), SPLIT, **kwargs)
    assert first == ge.deterministic_view(outputs.report)
    assert json.dumps(first, sort_keys=True) == json.dumps(
        {**second, "questions": first["questions"]}, sort_keys=True
    )
    assert all(ge.is_sealed(r) for r in records)


def test_run_ids_are_unique_and_files_are_never_overwritten(tmp_path: pathlib.Path) -> None:
    now = datetime(2026, 11, 23, tzinfo=UTC)
    first = ge.execute(SPLIT, PIPELINE, tmp_path, mode="dry_run", provider_factory=ScriptProvider,
                       expected_identities=IDS, now=now)  # fmt: skip
    assert first.records_path.name.startswith("dry_run-20261123T000000000000Z-")
    with pytest.raises(FileExistsError):
        ge.execute(SPLIT, PIPELINE, tmp_path, mode="dry_run", provider_factory=ScriptProvider,
                   expected_identities=IDS, now=now)  # fmt: skip
