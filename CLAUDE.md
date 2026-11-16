# CLAUDE.md — FounderLens AI RAG MVP

Persistent instructions for Claude Code sessions in this repository.

## Purpose and scope

A small, evidence-grounded RAG system over a **fixed corpus of public financial
PDFs**: page-aware ingestion, BM25 + dense (Sentence Transformers + FAISS CPU)
retrieval, OpenAI-based grounded answers with document/page citations,
abstention, reproducible evaluation, and a Streamlit demo. It is a portfolio
project demonstrating AI engineering depth — not a production SaaS. It is
separate from the larger FounderLens AI application.

See `docs/roadmap.md` for stages and `docs/architecture.md` for module boundaries.

## Out of scope — do not add

- Authentication, multi-tenant workspaces, billing.
- Neo4j or knowledge graphs.
- LangGraph, LangChain agents, or multi-agent systems.
- Arbitrary user uploads (the corpus is fixed).
- Database infrastructure of any kind.
- Premature abstractions (plugin systems, provider registries, base classes
  with one implementation).
- Framework dependencies a task does not strictly need.

## Workflow

- Work on **one authorized commit at a time**; do not start the next commit or
  stage without explicit authorization.
- **Inspect the existing implementation first** (`git status`, relevant files,
  tests) before changing anything. Never overwrite uncommitted user work.
- Report blockers and unresolved issues explicitly. Never mark unfinished work
  as complete, and never weaken or skip a quality gate to make it pass.

## Layout and conventions

- Windows + PowerShell locally; Python 3.12; virtual env at `.venv`.
- src layout: code in `src/founderlens/`, tests in `tests/`.
- Paths come from `founderlens.config.load_settings()` — no hard-coded
  absolute paths, no reliance on the working directory, no import-time side
  effects (no directory creation, secret reads, or network calls on import).
- Full type annotations; immutable config/value objects where practical.

## Quality gates (all must pass before committing)

```powershell
python -m pip check
python -m ruff check .
python -m ruff format --check .
python -m mypy src tests
python -m pytest -q
python -m founderlens.evaluation.validate
python -m founderlens.corpus.validate   # needs local PDFs; not run in CI
```

- mypy is strict; no blanket `ignore_missing_imports`, no unjustified `Any`
  or `# type: ignore`. Any targeted ignore needs an error code and a reason.
- No blanket lint suppressions.

## Testing

- Every behavior change ships with meaningful tests; no `assert True` filler.
- Tests must not need credentials, network access, downloaded models, or the
  real financial corpus. Use small synthetic fixtures; mock the OpenAI client.
- Tests must not write outside `tmp_path`.

## Dependencies

- Runtime deps go in `pyproject.toml` `[project].dependencies`, added only in
  the commit that first uses them. Dev tools are pinned in
  `requirements-dev.txt` (direct + transitive). Never declare a dependency twice.
- Verify Python 3.12 compatibility before pinning.

## Secrets

- Never read, print, log, or commit API keys. `.env` is git-ignored;
  `.env.example` holds placeholders only. Check diffs for secrets before committing.

## Corpus

- The corpus is fixed by `data/manifest.json` (v1.0.0). Never commit PDFs,
  never edit hashes, sizes or page counts to make validation pass, and never
  substitute documents. Any change needs a new corpus version.
- PyMuPDF is used only for validity checks and page counts until Stage 2.

## Evaluation integrity

- Never invent metrics, scores, latencies, or benchmark results. Report only
  numbers produced by a run in this repository, with the command used.
- **Do not read `evaluation/questions.test.jsonl` content** for development,
  debugging or optimization. Running `python -m founderlens.evaluation.validate`,
  which prints only counts and identifiers, is allowed.
- Tune chunking, top-k, embedding model choice and prompts on the dev split
  only. Never index evaluation files or put labels in LLM context.
- Dataset v1.0.0 is **frozen** and **human-reviewed**: the project owner
  accepted all 40 annotations (recorded in `evaluation/README.md`). That was a
  single-reviewer check, so do not call the labels independently verified.
- Held-out questions must not guide preprocessing, chunking, embedding model
  choice, retrieval configuration, generation prompts, or judge prompts and
  calibration.
- Changing the dataset requires bumping `dataset_version`, updating hashes and
  the changelog in `evaluation/dataset.json`, and documenting why.

## Git

- Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`).
- Stage only files relevant to the current commit. Do not push, rewrite
  history, `reset --hard`, or bypass hooks unless explicitly authorized.
