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
- Later: configurable chunking (size, overlap) that keeps the document ID
  and page range per chunk; processed output in `data/processed/`.
  *(planned)*

**Acceptance criteria.** Ingestion is deterministic (same input → identical
chunks); every chunk traces to a document and page(s); tests use synthetic
PDFs/fixtures only.

## Stage 3 — Retrieval & Vector Search

**Goal.** Measurable keyword and semantic retrieval.

**Deliverables.** BM25 baseline; dense retrieval with Sentence Transformers
embeddings in a FAISS CPU index persisted to `artifacts/`; a common retrieval
result type (chunk, score, citation metadata); optional hybrid fusion.

**Acceptance criteria.** Index build is reproducible from processed chunks;
retrieval metrics (see evaluation protocol) are reported for BM25 and dense
on the development split, from actual runs.

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
