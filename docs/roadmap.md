# Roadmap

Eight stages, each delivered as small, reviewable commits. A stage is complete
only when its acceptance criteria are met and verified by the quality gates.

| Stage | Title | Status |
|------:|-------|--------|
| 1 | Project Foundation & Dataset | Complete; evaluation dataset human-reviewed and frozen |
| 2 | Document Ingestion Pipeline | Implemented (Commits 2.1–2.3) |
| 3 | Retrieval & Vector Search | Complete (Commits 3.1–3.4, 4.1–4.3; BM25 on V2 selected for generation, top 10) |
| 4 | Grounded RAG Generation | Complete locally (Commits 5.1–5.4, offline-verified); live model behaviour and answer quality unverified |
| 5 | Cross-Document Reasoning & Guardrails | Complete (diagnostic acceptance; Commits 6.1–6.3; 6.4 deferred) |
| 6 | RAG Evaluation & Benchmarking | In progress (Commits 7.1 rubric and judgment contracts, 7.2 deterministic financial checks, 7.3 annotation and calibration workflow, 7.4 agreement statistics) |
| 7 | Streamlit Interactive Demo | Planned |
| 8 | Testing, Deployment & Documentation | Planned |

---

## Stage 1 — Project Foundation & Dataset

**Goal.** A reproducible, quality-gated Python project and a defined, fixed corpus.

**Deliverables.**
- Commit 1.1: package scaffolding, typed configuration, Ruff/mypy/pytest
  configuration, CI workflow, initial documentation. *(implemented)*
- Commit 1.2: versioned corpus manifest (four official earnings releases,
  provenance, SHA-256, sizes, page counts), a verified downloader, a corpus
  validator, 40 candidate evaluation questions with page-level evidence
  (24 dev / 16 held-out), a dataset validator, and the finalized evaluation
  protocol including the LLM-as-a-Judge design. *(implemented)*

**Acceptance criteria.**
- All quality gates pass locally and in CI.
- `python -m founderlens` runs without credentials or network access.
- The corpus is fully specified by the manifest; raw PDFs are not committed.
- All four real PDFs verify against the manifest locally.
- The evaluation dataset passes structural validation with the exact required
  distribution.
- The evaluation protocol is no longer marked preliminary.
- Project-owner review of the annotations is recorded: 40 accepted, 0
  corrections. Dataset v1.0.0 is frozen. *(met)*

## Stage 2 — Document Ingestion Pipeline

**Goal.** Turn the fixed PDF corpus into clean, page-aware, chunked text.

**Deliverables.**
- Commit 2.1: raw page-level PyMuPDF extraction from the verified corpus,
  with a typed page record, stable page IDs, an atomic
  `data/processed/pages.raw.jsonl`, an extraction diagnostics report and
  documented layout observations ([ingestion.md](ingestion.md)).
  *(implemented)*
- Commit 2.2: conservative, idempotent text normalization and
  word-coordinate table candidates (`structured` / `uncertain` / `fallback`)
  in `data/processed/pages.normalized.jsonl`, with a normalization report.
  *(implemented)*
- Commit 2.3: configurable, page-aware word-window chunking (default 120
  words, 24 overlap) of `normalized_text` into
  `data/processed/chunks.jsonl`. Each chunk carries exact source offsets,
  stable IDs and a configuration ID, and the run writes a chunking report.
  *(implemented)*

**Acceptance criteria.** Ingestion is deterministic (same input → identical
chunks); every chunk traces to a document and page(s); tests use synthetic
PDFs/fixtures only. *(met locally: repeated runs give byte-identical
artifacts, each chunk maps to exactly one page by offsets, and ordinary tests
use synthetic PDFs; remote CI not verified)*

**Carried into Stage 3.** Measure real token lengths with the selected
embedding model's tokenizer before embedding, and revise the chunk
configuration on dev data only if any chunk exceeds the model limit (see
[ingestion.md](ingestion.md#words-versus-embedding-model-tokens)).

## Stage 3 — Retrieval & Vector Search

**Goal.** Measurable keyword and semantic retrieval.

**Deliverables.**
- Commit 3.1: BM25 lexical baseline (`rank-bm25` BM25Okapi, in-memory
  index) with a versioned shared tokenizer, a common retrieval result type
  (chunk, raw score, provenance), a query CLI and a smoke report
  ([architecture.md](architecture.md#bm25-lexical-baseline-implemented-in-commit-31)).
  *(implemented; no retrieval metrics yet)*
- Commit 3.2: reproducible local embedding pipeline
  (`all-MiniLM-L6-v2` at a pinned revision, CPU), with a mandatory token-length
  audit, validated L2-normalized float32 artifacts in `artifacts/embeddings/`,
  and audit/build/validate commands
  ([architecture.md](architecture.md#chunk-embeddings-commit-32)).
  The original 120-word collection (V1, 162 chunks) has 9 chunks over the
  256-token limit, so the commit adds **chunking V2**, a separate
  token-budgeted collection of 164 chunks with 0 over the limit.
  - BM25 is rebuilt on V2, and V2 is fully embedded as (164, 384) float32.
  - V1 is kept frozen as the historical baseline.
  - BM25 and dense retrieval are to be compared on V2.

  *(implemented; no retrieval quality measured)*
- Commit 3.3: exact dense retrieval: cosine similarity as a dot product of
  L2-normalized vectors, exhaustive in NumPy, over the validated V2 artifact.
  - Query embedding with the same pinned model; over-limit queries are
    rejected, never truncated.
  - Stable tie-breaking, a shared result type, a CLI defaulting to V2, and a
    dense-vs-BM25 smoke report on V2
    ([architecture.md](architecture.md#exact-dense-retrieval-commit-33)).

  *(implemented; retrieval quality not yet measured)*
- Commit 3.4: deterministic development-split retrieval benchmark of BM25 and
  dense, both on V2.
  - Protocol v1.1 §3 metrics, k ∈ {1, 3, 5, 10} with primary k = 5, and 95%
    bootstrap intervals.
  - Per-question and per-category results, paired comparison, latency and
    error analysis.
  - The held-out split is never opened
    ([retrieval_evaluation.md](retrieval_evaluation.md)).

  *(implemented; results on 19 answerable dev questions only)*
- Hybrid fusion (Commit 4.1).
  - Deterministic reciprocal rank fusion of BM25 and dense, both on V2:
    constant 60, depth 10 each, equal weights, ties in collection order.
  - Per-chunk component ranks, scores and contributions, a CLI defaulting to
    V2, and a smoke report
    ([architecture.md](architecture.md#hybrid-retrieval-reciprocal-rank-fusion-commit-41)).

  *(implemented; evaluated in Commit 4.2; not the default retriever)*
- Comparative hybrid evaluation (Commit 4.2).
  - BM25, dense and hybrid RRF with the same harness and settings.
  - Diagnostics:
    - verified component agreement;
    - an exact baseline reproducibility check;
    - paired comparisons for all three pairs;
    - candidate-pool coverage;
    - fact outcome classification;
    - displaced exclusive evidence;
    - company and period diversity;
    - latency.
  - Separate experiment ID and reports
    ([retrieval_evaluation.md](retrieval_evaluation.md#three-way-comparison-with-hybrid-rrf-commit-42)).

  *(implemented; development split only)*
- Retrieval strategy decision (Commit 4.3).
  - BM25 on collection V2 is selected for the generation pipeline: ten ranked
    candidates, no score threshold, no page deduplication, no fallback.
  - A frozen, typed policy (`founderlens.retrieval.policy`, policy
    `f20913f1c1ec`) and one entry point, `load_policy_retriever`, that
    verifies V2 and the BM25 configuration and loads no embedding model.
  - Decision record with evidence, alternatives and limitations
    ([retrieval_strategy.md](retrieval_strategy.md)). Dense and hybrid stay
    available as evaluated alternatives.

  *(implemented; a development-informed decision, not a general superiority
  claim; answer quality unmeasured)*

**Acceptance criteria.** Index build is reproducible from processed chunks;
retrieval metrics (see evaluation protocol) are reported for BM25 and dense
on the development split, from actual runs; one retrieval path is selected
for generation from recorded evidence. *(met locally: reproducible V2 index
and embeddings; development-split metrics from identical benchmark runs,
Commits 3.4 and 4.2; policy verified against the local V2 artifact,
Commit 4.3. Remote CI not verified.)*

## Stage 4 — Grounded RAG Generation

**Goal.** Answers grounded only in retrieved evidence, with verifiable citations.

**Deliverables.** OpenAI API integration (key read from environment only);
prompt that constrains answers to provided context; structured output with
document/page citations; citation validation against retrieved chunks.
Retrieval comes from `load_policy_retriever` (BM25 V2, up to ten
candidates); context selection and token budgeting from those candidates are
part of this stage. Commits:
- Commit 5.1: typed OpenAI provider (Responses API, SDK `openai==3.55.0`,
  model `gpt-5.4-mini-2026-03-17`, reasoning effort none, 1,024 output
  tokens, `store=False`). Native Structured Outputs, normalized results and
  token usage, sanitized error categories, SDK-only retries (30 s per
  attempt, one retry), metadata-only logging, offline tests and an opt-in
  live smoke test ([architecture.md](architecture.md#llm-provider-commit-51)).
  *(implemented; offline-tested only; live test not run; no retrieval
  integration)*
- Commit 5.2: evidence context and grounded prompts. Covers question and
  candidate validation against the verified V2 collection, rank-ordered
  whole-chunk prefix selection within a 6,144-token local input limit
  (`tiktoken` 0.14.0, `o200k_base`, prepared offline asset), and an
  immutable evidence registry with request-scoped `E01`..`E10`. It also adds
  grounded prompt `grounded-financial-v1` (`2a45c8422c20`), deterministic
  JSON input, context configuration `0e7fd58314ff` and a context smoke
  report ([architecture.md](architecture.md#evidence-context-and-grounded-prompt-commit-52)).
  *(implemented; no LLM call; answer quality unmeasured)*
- Commit 5.3: structured grounded generation.
  - `GroundedAnswer`: answered / partially_answered / insufficient_evidence;
    claims typed reported / calculated / synthesis, each citing evidence
    IDs; limitations.
  - Strict schema `e3ec802e2267`, 481 local tokens of the 1,024-token
    allowance.
  - Local response invariants and request-scoped ID checks.
  - `GroundedGenerationService`: one structured provider call for
    `evidence_selected`; local abstention, without a call, for empty
    retrieval or no fitting evidence. Provider errors are raised, never
    converted, with no retry or repair.
  ([architecture.md](architecture.md#structured-grounded-generation-commit-53))
  *(implemented; offline-verified with fake and mocked providers; the live
  smoke `tests/test_generation_service_live.py` exists but has not been run;
  answer quality unmeasured)*
- Commit 5.4: citation and response validation.
  - `CitationResolver` re-parses each answer and checks the run's
    consistency (registry against request, origin against provider result,
    collection and manifest provenance).
  - It resolves cited IDs only within the run's own registry, numbers
    citations by first appearance, and returns a frozen `GroundedResponse`
    (contract `debe65096147`) with trusted citations and deterministic
    plain-text rendering.
  - Offline demo with a synthetic provider, and an opt-in single-request
    live demo
    ([architecture.md](architecture.md#citation-resolution-and-grounded-response-commit-54)).
  *(implemented; offline-verified; live demo and live tests not run)*

**Acceptance criteria.** Every citation refers to a retrieved chunk; tests mock
the API client; no key needed for the test suite. *(met locally and offline:
citations resolve only to the run's registry of retrieved, model-visible
chunks; provider tests use a mocked transport or fake provider; the default
suite needs no key. Hosted-model behaviour, answer quality and live token
usage remain unverified; remote CI not verified.)*

## Stage 5 — Cross-Document Reasoning & Guardrails

**Goal.** Correct multi-source answers and principled abstention.

**Deliverables.** Retrieval and prompting for questions spanning several
documents; abstention when evidence is insufficient; guardrails against
answering outside the corpus. *Delivered:* corpus scope analysis,
diagnostic-only response guardrails and a reproducible diagnostic
evaluation harness over the unchanged Stage 4 retrieval and prompt; no
retrieval or prompt change was made.

**Acceptance criteria.** Cross-document and unanswerable question categories
are evaluated separately with reported (not invented) results. Under
protocol v1.2 §13 these are deterministic diagnostic results
(`diagnostic_deterministic`) from an authorized real run; formal answer
quality stays in Stage 6. Under protocol v1.3 §14 the reported run is the
first pre-registered execution-complete live run, with every model-output
failure counted; the v1.2 pilot stays incomplete. **Satisfied**: the
cross-document and unanswerable categories were evaluated separately in
the reportable live run `live-20261124T045739024562Z-8665d379` (diagnostic acceptance only; factual
answer quality is unmeasured).

**Stage 5 diagnostic results** (`evaluation_kind: diagnostic_deterministic`,
protocol v1.3 §14). Run `live-20261124T045739024562Z-8665d379`: real live inference, pre-registered at commit
`7c63bbc` on a clean working tree, frozen identities unchanged, completion
state `execution_complete`, reportable with model-output failures counted.

> These diagnostics do not establish answer correctness, faithfulness,
> claim-level citation support or semantic abstention appropriateness. Formal
> evaluation remains pending Stage 6.

| Overall | Result |
|---|---|
| Scheduled questions / attempted provider requests | 24 / 24 |
| Contract-valid outcomes | 22/24 |
| Model-output failures | 2/24, both `answer_validation` / `answered_has_limitations` (`dev-016`, `dev-021`) |
| Selected-evidence page recall (answerable, recorded evidence) | 19/19 |
| Selected complete multi-fact coverage | 12/14 |
| Cited-evidence page recall (valid answer attempts) | 15/16 |
| Cited complete multi-fact coverage (valid answer attempts) | 11/13 (1 excluded: model-output failure) |
| Accepted-page citation overlap (distinct cited pages) | 24/30 (1 question excluded: model-output failure) |
| Provider usage (24/24 requests recorded) | 69,777 input, 2,985 output tokens |

| Category | Scheduled | Valid outcomes | Model-output failures | Valid statuses | Selected complete coverage | Cited complete coverage | Accepted-page overlap |
|---|---|---|---|---|---|---|---|
| single_doc_factual | 7 | 7 | 0 | 5 answered, 2 insufficient_evidence | 2/2 | 2/2 | 6/7 |
| single_doc_conceptual | 5 | 5 | 0 | 5 answered | 5/5 | 5/5 | 8/10 |
| cross_doc_comparison | 5 | 4 | 1 (`dev-016`) | 4 answered | 4/5 | 3/4 | 7/9 |
| temporal_multi_evidence (counts only) | 2 | 2 | 0 | 1 answered, 1 partially_answered | 1/2 | 1/2 | 3/4 |
| unanswerable | 5 | 4 | 1 (`dev-021`) | 1 answered, 3 insufficient_evidence | n/a | n/a | n/a |

- Unanswerable, unconditional over all 5 scheduled: 3 abstentions, 1
  answered, 0 partially answered, 1 model-output failure (operational flags,
  not judged abstention appropriateness).
- Answerable, unconditional over all 19 scheduled: 16 valid answer
  attempts, 2 abstentions, 1 model-output failure.
- Guardrails: 0 of 22 valid answers carry a warning; all findings are notices.
- Contract validity, accepted-page overlap and structural citation validity
  are not factual correctness, faithfulness or semantic citation support.
- Small sample: 24 development questions, one sample per question; temporal
  n = 2 and unanswerable n = 5. No held-out data was used.
- The v1.2 pilot `live-20261123T111430692539Z-8665d379` (23 of 24 valid, one `response_contract` failure)
  remains incomplete and not reportable; it predates the run log.

Commits:
- Commit 6.1: corpus vocabulary (manifest plus a reviewed alias table),
  conservative explicit company and period recognition, and manifest,
  selected and cited evidence coverage. Diagnostic only: there is no
  abstention or blocking, and a period missing from the manifest is
  `not_in_manifest_reporting_periods`, not out of corpus. Vocabulary
  `50057f9259f6`, analyzer `b447ce1aee09`
  ([architecture.md](architecture.md#corpus-scope-analysis-stage-5-commit-61)).
  *(implemented; offline)*
- Commit 6.2: deterministic response guardrails: per-claim company and
  period attribution, question coverage gaps, a conservative numeric trace
  and scope limitations. They are diagnostic only (`enforcement:
  diagnostic_only`): nothing is rewritten, blocked or abstained, and
  responses are unchanged. Policy `3f202ebc205a`
  ([architecture.md](architecture.md#response-guardrails-stage-5-commit-62)).
  *(implemented; offline; any enforcement must be decided from 6.3 and
  versioned)*
- Commit 6.3: development-split diagnostic evaluation harness
  (`founderlens.evaluation.generation_eval`) and protocol v1.2 §13:
  deterministic status, abstention-flag, selected and cited evidence
  coverage, accepted-page overlap, guardrail, usage and failure
  diagnostics with explicit denominators; offline dry run and replay;
  hard-capped, explicitly authorized live mode.
  *(implemented; offline-verified; the paid run needs separate explicit
  authorization; the first authorized live run, `live-20261123T111430692539Z-8665d379`,
  completed 23 of 24 questions with one `response_contract` failure, so it is
  incomplete and not reportable; the first v1.3 live run,
  `live-20261124T045739024562Z-8665d379`, is execution-complete and reportable)*
- Commit 6.4: **deliberately deferred.** The frozen baseline has known
  retrieval and generation limitations (for example the missing-company
  selection behind `dev-016`), but deterministic diagnostics do not establish
  semantic answer correctness. Stage 6 first evaluates factual accuracy,
  faithfulness, numerical correctness, claim support and abstention
  appropriateness on the unchanged baseline. Any later retrieval, prompt or
  generation change must be versioned and labelled development-informed
  tuning. Stage 5 did not change the retrieval strategy or the generation
  prompt.

**Transition to Stage 6.** Formal evaluation of the frozen baseline: a
versioned rubric and typed judgments, deterministic financial checks,
human labels and judge calibration (κ with intervals), a bounded,
explicitly authorized LLM judge over the 22 valid answers (the two
failures stay reliability outcomes), per-category and intention-to-treat
summaries, and regression procedures. The held-out split stays closed.

## Stage 6 — RAG Evaluation & Benchmarking

**Goal.** Reproducible end-to-end evaluation.

**Deliverables.** Evaluation runner implementing the hybrid framework in
`docs/evaluation_protocol.md`: deterministic retrieval and citation metrics,
reference-based checks, LLM-as-a-Judge dimensions, human calibration, and
latency, token and cost capture; versioned reports in `reports/`.

**Acceptance criteria.** One command reproduces a report; the judge's
agreement with human labels is measured on the predefined development
calibration set before its scores are reported; the held-out split is
evaluated only after development tuning is frozen.

Commits:
- Commit 7.1: protocol v1.4 §15, rubric v1 (`rubric_id` `e8b840e4f4af`), typed judgment contracts (units, labels,
  `not_assessable` reasons, evidence spans, provenance, compound-claim
  decompositions) and the verified-input gate over the frozen baseline
  `live-20261124T045739024562Z-8665d379` (24 outcomes, 22 eligible answers, 2
  model-output failures ineligible). *(implemented; offline; no labels or
  scores)*
- Commit 7.2: deterministic financial checks (protocol §16): versioned numeric
  extraction with stable `value_index` and spans (extractor `c6956d52681f`),
  Decimal comparison with stated-precision rounding, explicit verified
  reference facts and associations, conservative company, segment, period and
  basis checks, declared calculations (checker v1 `b86609c129e6`; v2 `c4160f8c38ec` in 7.3 adds an `unverified` reference basis). Baseline smoke:
  every real value stays unverified; no judgments. *(implemented; offline)*
- Commit 7.3: private, offline human-annotation workflow (protocol §17):
  deterministic review packets for the 22 eligible answers, pending
  worksheets, validated all-or-nothing import into an append-only ledger with
  revisions, explicit review states, adjudication and question-level
  calibration/evaluation partitions, human-verified reference facts and
  explicit fact associations, a versioned calibration-set selection and
  human-reviewed synthetic variants. *(implemented; offline; no human labels
  created yet)*
- Commit 7.4: agreement statistics (protocol §18): human–human and human–judge
  pairing over the annotation ledger, Cohen's kappa, Gwet's AC1, linearly
  weighted kappa for ordinal relevance, confusion matrices, explicit
  non-substantive and exclusion accounting, question-clustered bootstrap
  intervals and the planned kappa 0.60 target (configuration `6f37129cddc2`).
  *(implemented; offline; no labels or judge outputs exist, nothing measured)*
- Commit 7.5: frozen judge configuration and mocked judge workflow. *(planned)*
- Commits 7.6–7.8: authorized calibration and evaluation runs, reporting,
  regression procedures. *(planned; paid steps need explicit authorization)*

## Stage 7 — Streamlit Interactive Demo

**Goal.** An interview-ready interactive demonstration.

**Deliverables.** Streamlit app: question input, answer with citations,
retrieved evidence with document/page references, abstention display.

**Acceptance criteria.** Runs locally from documented commands; no
authentication; no arbitrary uploads.

## Stage 8 — Testing, Deployment & Documentation

**Goal.** Public, documented, maintainable release.

**Deliverables.** Hardened tests; public deployment without authentication;
final architecture, evaluation results and limitations documented.

**Acceptance criteria.** Deployed demo reachable publicly; README reflects
actual verified capabilities and measured results only.
