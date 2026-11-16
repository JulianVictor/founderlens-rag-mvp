# FounderLens AI RAG MVP

An evidence-grounded research assistant for a fixed corpus of public financial
documents. It is designed to answer questions with **verifiable document and
page-level citations**, to reason across documents, and to **abstain** when the
evidence is insufficient — with every claim about quality backed by a
reproducible evaluation.

> **Current status: Stage 1, Commit 1.1 — project foundation only.**
> Ingestion, retrieval, generation, evaluation, the demo UI and deployment are
> **not implemented yet**. See [Implementation status](#implementation-status).

This is a standalone portfolio project, separate from the larger FounderLens AI
application.

## Problem statement

Financial filings and reports are long, dense, and spread across many documents.
General-purpose LLM chat answers questions about them fluently but often without
traceable evidence, and sometimes with fabricated figures. For research use, an
answer is only useful if a reader can check it against the exact source page,
and if the system says "not found" instead of guessing.

## Target users

- Analysts, founders and researchers reading public financial documents.
- Technical reviewers and interviewers evaluating practical RAG engineering:
  retrieval quality, grounding, citation integrity and evaluation rigor.

## Intended AI engineering capabilities (planned)

- Page-aware PDF text extraction and cleaning.
- Configurable chunking with page-span metadata.
- BM25 keyword retrieval baseline.
- Dense semantic retrieval (Sentence Transformers) with a FAISS CPU index.
- Grounded answer generation via the OpenAI API.
- Document and page-level citations, validated against retrieved evidence.
- Cross-document reasoning.
- Abstention on insufficient evidence.
- Reproducible retrieval and generation evaluation (development vs held-out).
- Interactive Streamlit demo, publicly deployed without authentication.

## Proposed architecture

Offline: manifest → fetch & verify → page-aware extraction → cleaning →
chunking → BM25 index + embeddings/FAISS index.
Online: question → retrieval → evidence set → grounded generation → citation
validation / abstention → answer.

Details and a Mermaid diagram: [docs/architecture.md](docs/architecture.md).

## Technology stack

| Concern | Choice | Status |
|---------|--------|--------|
| Language | Python 3.12 | In use |
| Packaging | setuptools, `pyproject.toml`, src layout | In use |
| Quality | Ruff (lint + format), mypy (strict), pytest, pytest-cov | In use |
| CI | GitHub Actions (Ubuntu, Python 3.12) | Configured |
| PDF extraction | PyMuPDF | Planned (Stage 2) |
| Keyword retrieval | BM25 library (selected in Stage 3) | Planned |
| Embeddings | Sentence Transformers | Planned (Stage 3) |
| Vector index | FAISS CPU | Planned (Stage 3) |
| Generation | OpenAI API | Planned (Stage 4) |
| Demo | Streamlit | Planned (Stage 7) |

## Local setup (Windows PowerShell)

```powershell
git clone https://github.com/JulianVictor/founderlens-rag-mvp.git
cd founderlens-rag-mvp

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .

python -m founderlens
```

No API key is required in Stage 1. `.env.example` lists future variables;
copy it to `.env` only when a later stage needs credentials.

For a non-editable install, set the project root explicitly:

```powershell
$env:FOUNDERLENS_PROJECT_ROOT = "C:\absolute\path\to\founderlens-rag-mvp"
```

## Development quality commands

```powershell
python -m pip check
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
python -m pytest -q
python -m pytest -q --cov=founderlens --cov-report=term-missing
```

The same gates run in CI on pushes and pull requests to `main`
([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Implementation status

**Implemented (Commit 1.1):**
- Installable `founderlens` package (src layout, version 0.1.0).
- `founderlens.config`: immutable, typed path configuration with project-root
  discovery, `FOUNDERLENS_PROJECT_ROOT` / explicit overrides and validation.
- `python -m founderlens` status entry point (no credentials, no network).
- Ruff, mypy strict, pytest and coverage configuration; unit and smoke tests.
- GitHub Actions CI workflow.
- Initial architecture, preliminary evaluation protocol, and roadmap docs.

**Not implemented:** document corpus, ingestion, chunking, retrieval, vector
indexing, generation, citations, abstention, evaluation runs, Streamlit demo,
deployment. No evaluation results exist yet.

## Roadmap

1. Project Foundation & Dataset — *in progress*
2. Document Ingestion Pipeline
3. Retrieval & Vector Search
4. Grounded RAG Generation
5. Cross-Document Reasoning & Guardrails
6. RAG Evaluation & Benchmarking
7. Streamlit Interactive Demo
8. Testing, Deployment & Documentation

Goals, deliverables and acceptance criteria: [docs/roadmap.md](docs/roadmap.md).
Evaluation design: [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## Limitations

- Fixed, curated corpus only; no user uploads.
- Single-user demo; no authentication, accounts, billing or multi-tenancy.
- No database; local files and a FAISS index are the only storage.
- Generation will depend on a third-party API (OpenAI) and its cost/latency.
- Answers are research aids, not financial advice.
- Evaluation will use a small hand-built question set; results will not
  generalize beyond this corpus.
