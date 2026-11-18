# FounderLens AI RAG MVP

An evidence-grounded research assistant for a fixed corpus of public financial
documents. It is designed to answer questions with **verifiable document and
page-level citations**, to reason across documents, and to **abstain** when the
evidence is insufficient — with every claim about quality backed by a
reproducible evaluation.

> **Current status: Commits 3.1–3.4 and 4.1–4.3.** Stages 1–3
> are complete: the project foundation, a verified four-document corpus, a
> human-reviewed frozen evaluation dataset, page-aware extraction,
> normalization and chunking, and retrieval. Implemented so far:
> - a BM25 lexical retrieval baseline;
> - a local embedding pipeline, with token-budgeted chunking V2 (164 chunks,
>   all within the model's 256-token limit) fully embedded;
> - exact dense (cosine) retrieval over V2;
> - a deterministic **development-split** retrieval benchmark comparing the
>   two;
> - reciprocal-rank-fusion hybrid retrieval over the same V2 collection,
>   compared with both baselines on the development split;
> - a frozen retrieval policy for generation: **BM25 on V2, ten candidates**
>   ([decision record](docs/retrieval_strategy.md)).
>
> Retrieval quality is measured for BM25, dense and hybrid on the 19
> answerable development questions only; the held-out split has not been
> used. Generation (Stage 4, next), answer evaluation, the demo UI and
> deployment are **not implemented yet**. See
> [Implementation status](#implementation-status).

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
| PDF validity checks | PyMuPDF 1.28.2 | In use (page counts, readability) |
| PDF text extraction | PyMuPDF 1.28.2 | In use (raw page-level text, Commit 2.1) |
| Keyword retrieval | rank-bm25 0.2.2 (BM25Okapi) + NumPy 2.5.4 | In use (in-memory baseline, Commit 3.1) |
| Embeddings | sentence-transformers 6.2.0, `all-MiniLM-L6-v2` (pinned revision), PyTorch 2.15.0 CPU | In use (Commit 3.2): chunking V2 fully embedded; no dense search yet |
| Vector search | Exact NumPy dot product (164 × 384); FAISS not needed at this size | In use (Commit 3.3) |
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

# Acquire and verify the corpus (official PDFs into data/raw/, git-ignored)
python -m founderlens.corpus.download
python -m founderlens.corpus.validate
python -m founderlens.evaluation.validate

# Extract raw page-level text (data/processed/pages.raw.jsonl, git-ignored)
python -m founderlens.ingestion.extract
# Normalize text and detect table candidates (pages.normalized.jsonl, git-ignored)
python -m founderlens.ingestion.normalize
# Split normalized pages into retrieval chunks (chunks.jsonl, git-ignored)
python -m founderlens.ingestion.chunk
# Search the chunks with the BM25 baseline (in-memory index, no network)
python -m founderlens.retrieval.bm25 --query "NVIDIA Blackwell" --top-k 5

# Chunking V2 + embeddings: fetch the pinned model once (about 92 MB, user
# Hugging Face cache) while building the token-budgeted collection V2
python -m founderlens.retrieval.embeddings build-collection --allow-download
# Offline afterwards: validate, audit, embed and check V2 (V1 stays frozen)
python -m founderlens.retrieval.embeddings validate-collection --collection v2
python -m founderlens.retrieval.embeddings audit --collection v2
python -m founderlens.retrieval.embeddings build --collection v2
python -m founderlens.retrieval.embeddings validate --collection v2
python -m founderlens.retrieval.bm25 --collection v2 --query "NVIDIA Blackwell"
# Exact dense retrieval (defaults to V2; model loaded from the local cache)
python -m founderlens.retrieval.dense --query "How quickly is the AI chip maker's main business growing?"
python -m founderlens.retrieval.dense --smoke   # dense and BM25 on V2 -> reports/
# Development-split retrieval benchmark (BM25 and dense, both on V2)
python -m founderlens.evaluation.benchmark      # reports/retrieval_benchmark.dev.{json,md}
# Three-way benchmark (adds hybrid RRF). Run the two-method command above first:
# it reads that git-ignored report as its baseline and never rewrites it.
python -m founderlens.evaluation.benchmark --include-hybrid  # reports/retrieval_benchmark_three_way.dev.*
# Hybrid retrieval (RRF of BM25 and dense, both on V2)
python -m founderlens.retrieval.hybrid --query "NVIDIA Blackwell"
python -m founderlens.retrieval.hybrid --smoke  # reports/hybrid_smoke_report.<id>.json
# Selected retrieval policy for generation (BM25 on V2, top 10): verify artifacts
python -m founderlens.retrieval.policy
```

PyTorch is a pinned runtime dependency. On Windows and macOS the PyPI wheel
is CPU-only. On Linux, install the CPU build first, as CI does:
`python -m pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu`.

If the automated download fails, follow the manual steps in
[data/README.md](data/README.md#manual-download-if-automated-download-fails).

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
python -m founderlens.evaluation.validate
python -m founderlens.corpus.validate        # needs the PDFs locally
python -m founderlens.ingestion.extract     # needs the PDFs locally
$env:FOUNDERLENS_REAL_CORPUS = "1"; python -m pytest tests/test_corpus_real.py tests/test_ingestion_real.py -q
```

CI runs the same gates on pushes and pull requests to `main`
([.github/workflows/ci.yml](.github/workflows/ci.yml)), except the two
real-corpus checks: CI never downloads the financial PDFs.

## Implementation status

**Implemented (Commit 1.1):**
- Installable `founderlens` package (src layout, version 0.1.0).
- `founderlens.config`: immutable, typed path configuration with project-root
  discovery, `FOUNDERLENS_PROJECT_ROOT` / explicit overrides and validation.
- `python -m founderlens` status entry point (no credentials, no network).
- Ruff, mypy strict, pytest and coverage configuration; unit and smoke tests.
- GitHub Actions CI workflow.

**Implemented (Commit 1.2):**
- Versioned corpus manifest ([data/manifest.json](data/manifest.json), corpus
  v1.0.0): four official earnings releases, NVIDIA FY2025 Q3/Q4 and Alphabet
  2024 Q3/Q4, with provenance, SHA-256, sizes and page counts.
- `founderlens.corpus`: typed manifest parsing; a host-allowlisted HTTPS
  downloader with checksum verification; a read-only corpus validator.
- `founderlens.evaluation`: typed JSONL schema and a dataset validator.
- 40 evaluation questions (24 development, 16 held-out)
  ([evaluation/README.md](evaluation/README.md)).

**Dataset finalization:**
- The project owner reviewed all 40 annotations and accepted them without
  corrections. Dataset v1.0.0 is frozen, and its question files and hashes
  are unchanged.
- The retrieval metric definitions in the evaluation protocol are made exact
  (protocol v1.1).
- Finalized evaluation protocol, including an LLM-as-a-Judge design
  ([docs/evaluation_protocol.md](docs/evaluation_protocol.md)).

**Implemented (Commit 2.1), see [docs/ingestion.md](docs/ingestion.md):**
- `founderlens.ingestion`: a typed page record and raw page-level extraction
  with PyMuPDF.
  - One record per physical page, with stable IDs
    (`nvda-fy2025-q3:p0001`) and the source metadata and SHA-256 from the
    manifest.
  - The command validates the corpus before extracting.
  - Parser failures fail the run, naming the document and page.
- `data/processed/pages.raw.jsonl` is published atomically and is
  byte-identical across reruns with the same PDFs, PyMuPDF version and
  configuration.
- `reports/extraction_report.json` holds page and document diagnostics and
  automatic heuristic warnings. Manually observed table-layout problems are
  documented separately.

**Implemented (Commit 2.2), see [docs/ingestion.md](docs/ingestion.md#commit-22-layout-aware-normalization):**
- Conservative, idempotent text normalization:
  - LF line endings, NFC, and trailing whitespace removed;
  - blank-line runs collapsed;
  - column-alignment spaces preserved.
- Word-coordinate table candidates with explicit `structured` / `uncertain`
  / `fallback` status.
  - Columns are assigned by position, so empty cells stay in place and period
    headers stay with their columns.
  - Nothing is inferred: no headers, units or zero values are invented.
  - PyMuPDF's `find_tables` was evaluated and rejected for this corpus.
- `data/processed/pages.normalized.jsonl` (one record per page, raw text kept
  separately) and `reports/normalization_report.json`.
  - On corpus v1.0.0 it finds 58 table candidates: 21 structured,
    35 uncertain and 2 fallback.
  - "Structured" means the automatic checks passed, not that the
    associations are verified.

**Implemented (Commit 2.3), see [docs/ingestion.md](docs/ingestion.md#commit-23-page-aware-chunking):**
- Page-aware word-window chunking of `normalized_text`:
  - default 120 words with up to 24 words of overlap, configurable with
    `--chunk-size-words` and `--chunk-overlap-words`;
  - boundaries prefer paragraph breaks, then line breaks, then word
    boundaries.
  - These are whitespace words, not model tokens.
- Every chunk is an exact source slice:
  - text equals `normalized_text[start_char:end_char]`;
  - chunks never cross pages;
  - IDs are stable (`{page_id}:c{index:04d}:{config_id}`).
- Table candidates are not indexed separately.
- `data/processed/chunks.jsonl` and `reports/chunking_report.json`.
  - On corpus v1.0.0 the default configuration gives 162 chunks, validated
    for full source coverage and overlap limits.
  - No retrieval quality is claimed.

**Implemented (Commit 3.1), see [docs/architecture.md](docs/architecture.md#bm25-lexical-baseline-implemented-in-commit-31):**
- BM25 lexical baseline over `chunks.jsonl`: `rank-bm25` BM25Okapi with
  defaults k1 1.5, b 0.75, epsilon 0.25, top-k 5, and an in-memory index
  built once per retriever.
- A versioned tokenizer shared by chunks and queries: casefold plus Unicode
  letter/digit runs, keeping `39,331`, `37.5` and `non-gaap`. No stemming or
  stop words.
- Only chunks containing a query token are returned, ranked by raw score with
  ties in collection order. Empty or out-of-vocabulary queries return
  nothing, and scores are not probabilities.
- `python -m founderlens.retrieval.bm25 --query "..."` (text or `--json`), and
  `--smoke` for `reports/bm25_smoke_report.json`.
- Functional smoke test only: no retrieval metrics have been computed.

**Implemented (Commit 3.2), see [docs/architecture.md](docs/architecture.md#chunk-embeddings-commit-32):**
- Local embeddings with `sentence-transformers/all-MiniLM-L6-v2`:
  - pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
  - CPU, eval mode, safetensors weights, no remote code;
  - no text leaves the machine.
- A mandatory token-length audit with the model's own tokenizer (special
  tokens on, truncation off). Over-limit chunks are never truncated: the
  build stops before encoding.
- L2-normalized float32 vectors (384 dimensions), stored in
  `artifacts/embeddings/<collection id>/` as `embeddings.npy` (no pickle)
  plus `manifest.json`: configuration ID, source chunk hash, matrix hash and
  the ordered row → chunk mapping.
  - The artifact is published atomically and validated on load, and rejected
    with any other collection.
- **Chunking V2.** The original 120-word collection (V1, 162 chunks, kept
  frozen) has 9 chunks over the 256-token limit (max 308), all on dense
  financial tables, where digits and punctuation split into many tokens.
  - V2 keeps the same 120/24-word rule and adds a hard 256-token budget in
    the pinned tokenizer. It shortens only windows that overflow; 149 of
    V1's spans are unchanged.
  - Result: 164 chunks, max 255 tokens, 0 over the limit.
  - Fixed 96-word windows were rejected: they would shrink every chunk to fix
    five pages and leave only 15 tokens of headroom.
- BM25 is rebuilt on V2 (`--collection v2`). V2 is fully embedded: (164,
  384) float32, unit norm within 3e-8, and two builds were byte-identical.
  BM25 and dense retrieval are compared on V2 only.

**Implemented (Commit 3.3), see [docs/architecture.md](docs/architecture.md#exact-dense-retrieval-commit-33):**
- Exact dense retrieval over the validated V2 embedding artifact:
  - query encoded with the same pinned MiniLM model, loaded lazily and
    reused;
  - cosine as a dot product of unit vectors (`matrix @ query`), exhaustive in
    NumPy;
  - descending raw scores with ties in collection order, default top-k 5.
- Empty or punctuation-only queries return nothing without loading the
  model. Queries over 256 tokens are rejected, never truncated.
- Artifacts are validated once at load (hashes, model revision,
  configuration, row order) and never rebuilt implicitly.
- Cosine scores are not relevance probabilities, and there is no threshold:
  the nearest neighbours are returned even for unrelated questions.
- `python -m founderlens.retrieval.dense --query "..."` defaults to V2. BM25
  keeps its V1 default, so comparisons pass `--collection v2` to both.
- Functional smoke test only: retrieval quality has not been measured
  (Commit 3.4).

**Implemented (Commit 3.4), see [docs/retrieval_evaluation.md](docs/retrieval_evaluation.md):**
- `python -m founderlens.evaluation.benchmark` evaluates BM25 and dense, both
  on V2.
  - It runs on the development split only; the held-out file is never
    opened, and `--split test` is refused.
  - Same queries, k ∈ {1, 3, 5, 10} (primary k = 5, fixed before running),
    page-level metrics from protocol v1.1 §3, and 95% bootstrap intervals.
- Development results (19 answerable questions, k = 5):

  | | Recall@5 (hit) | Fact coverage@5 | Complete coverage@5 (n = 14) | MRR@10 |
  |---|---|---|---|---|
  | BM25 | 0.947 [0.842, 1.000] | 0.829 | 0.571 | 0.622 |
  | Dense | 0.737 [0.526, 0.895] | 0.658 | 0.571 | 0.548 |

  - BM25 had higher observed Recall@5, by 21.1 percentage points. The paired
    bootstrap interval prescribed by the protocol, [−0.421, −0.053], excluded
    zero, but 19 questions limit how strong or general this finding is.
    An exact McNemar cross-check on the 4-to-0 discordant pairs gives
    p = 0.125.
  - The other differences were inconclusive, which does not mean the systems
    are equivalent.
  - Both systems hit every question by k = 10.
  - Recall@k here is the protocol's binary accepted-evidence hit, not
    fractional IR recall.
  - These are page-level retrieval numbers on a small set, not answer
    quality, and not held-out results.
- Two full runs gave identical deterministic report content.

**Implemented (Commit 4.1), see [docs/architecture.md](docs/architecture.md#hybrid-retrieval-reciprocal-rank-fusion-commit-41):**
- Hybrid retrieval by Reciprocal Rank Fusion of BM25 and dense, both on the
  same validated V2 collection: `rrf = Σ 1 / (60 + rank)`.
  - Each retriever contributes its top 10 for the original query, with equal
    weight; ties keep collection order; top-k is 1–10, default 5.
  - BM25 and cosine scores are on incomparable scales, so only ranks are
    fused. The raw scores are reported but never combined.
- With these fixed settings, any chunk found by both retrievers outranks
  every chunk found by only one. This can push down useful single-list
  results, and RRF does not diversify across companies or periods.
- Each result keeps the original chunk and its BM25 and dense ranks, scores
  and contributions. Component failures are raised, never replaced by one
  retriever's results.
- `python -m founderlens.retrieval.hybrid --query "..."` (V2 only; `--json`,
  `--smoke`).
- Evaluated on the dev split in Commit 4.2 (below); not selected for
  generation in Commit 4.3, kept as an evaluated alternative.

**Implemented (Commit 4.2), see [docs/retrieval_evaluation.md](docs/retrieval_evaluation.md#three-way-comparison-with-hybrid-rrf-commit-42):**
- `python -m founderlens.evaluation.benchmark --include-hybrid` runs BM25,
  dense and the real hybrid retriever on the same 24 dev questions.
  - It is a separate experiment (`f6982b8eb438`) with separate reports.
  - The two-method experiment `3fd7b205dc81` and its reports are unchanged.
  - The run fails unless the hybrid's component rankings equal the
    standalone ones and the BM25 and dense results reproduce the two-method
    report: identities, chunk IDs, ranks, provenance and metrics exactly, raw
    scores within an absolute 1e-6. On a fresh checkout, run the two-method
    benchmark first; without its report the three-way run exits 1. Both
    checks held in every run, with raw scores identical.
- Development results (19 answerable questions, k = 5):

  | | Recall@5 (hit) | Fact coverage@5 | Complete coverage@5 (n = 14) | MRR@10 |
  |---|---|---|---|---|
  | BM25 | 0.947 [0.842, 1.000] | 0.829 | 0.571 | 0.622 |
  | Dense | 0.737 [0.526, 0.895] | 0.658 | 0.571 | 0.548 |
  | Hybrid RRF | 0.947 [0.842, 1.000] | 0.829 | 0.643 | 0.679 |

  - Hybrid and BM25 hit the same questions at k = 5. Their other differences
    are small and their intervals include or touch zero, which does not make
    them equivalent.
  - Hybrid recovered BM25's four Recall@5 wins over dense (+0.211
    [0.053, 0.421]; exact McNemar cross-check p = 0.125).
  - Of hybrid's 8 missing facts at k = 5, 6 were in the BM25 + dense
    candidate pool but cut off by fusion, and 2 were in neither list.
  - One question (dev-015) lost evidence because a BM25-only rank-1 chunk was
    fused below shared candidates.
  - At k = 10 hybrid trailed BM25 on two questions.
  - Warm hybrid queries took 17.9 / 23.2 ms (p50 / p95) against 1.4 / 2.1 ms
    for BM25 on this machine.
- These are exploratory, page-level, development-only numbers. No strategy
  was selected in Commit 4.2.

**Implemented (Commit 4.3), see [docs/retrieval_strategy.md](docs/retrieval_strategy.md):**
- Decision: generation will retrieve with **BM25 on collection V2**
  (`0c2f87411627`, BM25 config `c21bdd0691f8`), **ten ranked candidates**,
  no score threshold, no page deduplication and no fallback to dense or
  hybrid.
  - Evidence: at k = 5 BM25 matched hybrid on Recall@5 (0.947) and fact
    coverage@5 (0.829). Hybrid's higher complete coverage@5 and MRR@10 were
    inconclusive, and BM25 had the lowest warm latency without a
    transformer model.
  - BM25 at k = 10: Recall 1.000, fact coverage 0.947, complete coverage
    0.857 (development split, page-level).
  - Development-informed, not a general superiority claim. Answer quality is
    unmeasured.
- `founderlens.retrieval.policy`: the frozen `RetrievalPolicy` (policy ID
  `f20913f1c1ec`) and `load_policy_retriever(settings)`, the entry point for
  generation. It verifies V2's configuration and pinned SHA-256 and the BM25
  configuration, builds the index once and returns up to ten ranked chunks
  with their original text and provenance. It needs no PyTorch, embedding
  model or embedding files, and no benchmark reports.
- `python -m founderlens.retrieval.policy [--json]` checks the policy against
  the local artifacts.
- The benchmark's primary k stays 5; ten candidates is the generation input,
  not the final LLM context. Existing CLIs are unchanged (BM25 still
  defaults to V1 and top 5).

**Not implemented:** generation,
citations, abstention, answer-level evaluation and the judge, the Streamlit
demo, deployment. No held-out results exist.

## Roadmap

1. Project Foundation & Dataset — *complete; benchmark frozen*
2. Document Ingestion Pipeline
3. Retrieval & Vector Search — *complete; BM25 V2 selected for generation*
4. Grounded RAG Generation — *next*
5. Cross-Document Reasoning & Guardrails
6. RAG Evaluation & Benchmarking
7. Streamlit Interactive Demo
8. Testing, Deployment & Documentation

Goals, deliverables and acceptance criteria: [docs/roadmap.md](docs/roadmap.md).
Evaluation design: [docs/evaluation_protocol.md](docs/evaluation_protocol.md).

## Licensing note: PyMuPDF

PyMuPDF is dual-licensed: **GNU AGPL-3.0** or a commercial license from
Artifex. Under the AGPL, offering the software to users over a network can
require releasing the complete corresponding source of the application under
the AGPL. Before the public deployment in Stage 8, either keep the whole
project AGPL-compatible and publish its source, obtain a commercial license, or
replace PyMuPDF. This note is not legal advice.

## Limitations

- Table extraction is heuristic and corpus-tuned, not production-grade.
  Most table candidates are `uncertain`. Tables that span pages, wrapped
  cells and labels, and multi-level headers are not reconstructed. Values in
  page text are not reliably linked to their period columns in
  reconciliation tables or in tables that span pages
  ([docs/ingestion.md](docs/ingestion.md#manual-extraction-observations)).
- Fixed, curated corpus only; no user uploads. The source PDFs are not
  redistributed; users download them from the official sites.
- Evaluation labels were written by Claude Code and reviewed by a single
  reviewer, the project owner. They have not been independently verified. The held-out split is a procedural holdout with public labels.
- Single-user demo; no authentication, accounts, billing or multi-tenancy.
- No database; local files and a FAISS index are the only storage.
- Generation will depend on a third-party API (OpenAI) and its cost/latency.
- Answers are research aids, not financial advice.
- Evaluation will use a small hand-built question set; results will not
  generalize beyond this corpus.
