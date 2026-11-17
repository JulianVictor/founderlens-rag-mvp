# Roadmap

Eight stages, each delivered as small, reviewable commits. A stage is complete
only when its acceptance criteria are met and verified by the quality gates.

| Stage | Title | Status |
|------:|-------|--------|
| 1 | Project Foundation & Dataset | Complete; evaluation dataset human-reviewed and frozen |
| 2 | Document Ingestion Pipeline | In progress (Commits 2.1–2.2 implemented) |
| 3 | Retrieval & Vector Search | Planned |
| 4 | Grounded RAG Generation | Planned |
| 5 | Cross-Document Reasoning & Guardrails | Planned |
| 6 | RAG Evaluation & Benchmarking | Planned |
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
- Optional hybrid fusion. *(planned)*

**Acceptance criteria.** Index build is reproducible from processed chunks;
retrieval metrics (see evaluation protocol) are reported for BM25 and dense
on the development split, from actual runs. *(met locally: reproducible V2
index and embeddings; development-split metrics from two identical benchmark
runs, Commit 3.4. Remote CI not verified.)*

## Stage 4 — Grounded RAG Generation

**Goal.** Answers grounded only in retrieved evidence, with verifiable citations.

**Deliverables.** OpenAI API integration (key read from environment only);
prompt that constrains answers to provided context; structured output with
document/page citations; citation validation against retrieved chunks.

**Acceptance criteria.** Every citation refers to a retrieved chunk; tests mock
the API client; no key needed for the test suite.

## Stage 5 — Cross-Document Reasoning & Guardrails

**Goal.** Correct multi-source answers and principled abstention.

**Deliverables.** Retrieval and prompting for questions spanning several
documents; abstention when evidence is insufficient; guardrails against
answering outside the corpus.

**Acceptance criteria.** Cross-document and unanswerable question categories
are evaluated separately with reported (not invented) results.

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
