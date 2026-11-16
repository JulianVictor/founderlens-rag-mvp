# Roadmap

Eight stages, each delivered as small, reviewable commits. A stage is complete
only when its acceptance criteria are met and verified by the quality gates.

| Stage | Title | Status |
|------:|-------|--------|
| 1 | Project Foundation & Dataset | In progress — Commit 1.1 done |
| 2 | Document Ingestion Pipeline | Planned |
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
- Commit 1.2: corpus definition (`data/manifest.json` with source URLs,
  checksums and licensing notes, `data/README.md`) and an initial evaluation
  question set; finalize the evaluation protocol. *(planned)*

**Acceptance criteria.**
- All quality gates pass locally and in CI.
- `python -m founderlens` runs without credentials or network access.
- The corpus is fully specified by the manifest; raw PDFs are not committed.
- The evaluation protocol is no longer marked preliminary.

## Stage 2 — Document Ingestion Pipeline

**Goal.** Turn the fixed PDF corpus into clean, page-aware, chunked text.

**Deliverables.** Manifest-driven fetch with checksum verification; PyMuPDF
page-level extraction; text cleaning; configurable chunking (size, overlap)
that preserves document ID and page range per chunk; processed output in
`data/processed/`.

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

**Deliverables.** Evaluation runner for retrieval, answer correctness,
citation correctness, grounding and abstention; latency and token usage
capture; versioned reports in `reports/`.

**Acceptance criteria.** One command reproduces a report; held-out split is
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
