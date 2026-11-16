# Evaluation dataset

**Dataset:** `founderlens-financial-research-eval` **v1.0.0**, built on corpus
v1.0.0. The protocol is in
[`docs/evaluation_protocol.md`](../docs/evaluation_protocol.md).

> **Annotation status: HUMAN-REVIEWED, FROZEN.** Claude Code authored all 40
> questions, reference answers and evidence labels from the original PDFs. The
> project owner reviewed every question and accepted all 40 without
> corrections (see [Project-owner review](#project-owner-review)). The
> benchmark is frozen at dataset v1.0.0. The review was by one person, the
> project owner, so the labels are human-reviewed but not independently
> verified.

## Files

| File | Split | Questions | SHA-256 |
|------|-------|----------:|---------|
| `questions.dev.jsonl` | development | 24 | `12d6c7f5e3161f3aa6848597e3f031005a4e4d6347c140db57d40fee0162a8cb` |
| `questions.test.jsonl` | held-out | 16 | `4248f5b50f8d25421a650d881812d4c7d27dce7a3e8c32431f02cdf765ac7307` |
| `dataset.json` | metadata | — | version, status, file hashes, changelog |

The files are LF-only (enforced by `.gitattributes`), so the hashes are
identical on every platform. `python -m founderlens.evaluation.validate` fails
if either file changes without its hash being updated.

## Distribution

| Category | Dev | Held-out |
|----------|----:|---------:|
| `single_doc_factual` | 7 | 5 |
| `single_doc_conceptual` | 5 | 3 |
| `cross_doc_comparison` | 5 | 3 |
| `temporal_multi_evidence` | 2 | 2 |
| `unanswerable` | 5 | 3 |
| **Total** | **24** | **16** |

The split is stratified by category. Question intent is not duplicated across
the splits: each held-out question asks about a different metric, period or
relationship from every development question.

## Record format

One JSON object per line:

| Field | Answerable | Unanswerable |
|-------|-----------|--------------|
| `question_id` | `dev-NNN` / `test-NNN` | same |
| `question` | text | text |
| `category` | one of the four answerable categories | `unanswerable` |
| `answerable` | `true` | `false` |
| `reference_answer` | non-empty | `null` |
| `required_facts` | ≥ 1 `{fact_id, statement, numeric?}` | `[]` |
| `accepted_evidence` | ≥ 1 `{fact_id, document_id, page_number}` per fact | `[]` |
| `expected_abstention_reason` | `null` | why the corpus cannot answer |

`numeric` (optional) holds `company`, `fiscal_period`, `metric`, `value`,
`unit` (`USD`, `USD millions`, `USD billions`, `percent`, `percentage points`,
`count`) and `basis` (`GAAP`, `non-GAAP`, `not_applicable`).
`page_number` is the **one-based physical PDF page**.

Development example (`dev-001`, abridged):

```json
{"question_id": "dev-001",
 "question": "What was NVIDIA's total revenue for the third quarter of fiscal 2025 (quarter ended October 27, 2024)?",
 "category": "single_doc_factual", "answerable": true,
 "reference_answer": "NVIDIA reported revenue of $35,082 million (about $35.1 billion) for the third quarter of fiscal 2025.",
 "required_facts": [{"fact_id": "dev-001-f1", "statement": "NVIDIA revenue for Q3 FY2025 was $35,082 million (about $35.1 billion).",
   "numeric": {"company": "NVIDIA", "fiscal_period": "FY2025 Q3", "metric": "revenue", "value": 35082, "unit": "USD millions", "basis": "GAAP"}}],
 "accepted_evidence": [{"fact_id": "dev-001-f1", "document_id": "nvda-fy2025-q3", "page_number": 1},
                       {"fact_id": "dev-001-f1", "document_id": "nvda-fy2025-q3", "page_number": 3},
                       {"fact_id": "dev-001-f1", "document_id": "nvda-fy2025-q4", "page_number": 1}],
 "expected_abstention_reason": null}
```

The third evidence page is valid: the Q4 release's summary table repeats the
Q3 FY25 revenue in its comparison column.

## How the labels were produced

1. All 40 pages were read in full from the original PDFs (PyMuPDF page text),
   not taken from memory, summaries or web search.
2. Each fact was written with the pages that state it. Where the same fact
   appears on several pages or in another document (comparison columns,
   reconciliations, repeated segment tables), those pages were added as
   alternatives.
3. An authoring script, kept out of the repository, confirmed that every cited
   page contains the supporting value or phrase. It also searched all pages for
   uncited occurrences of each numeric value. Every match it found was reviewed
   and was a coincidental number with a different meaning, so none was added.
4. Each unanswerable question was checked against the whole corpus, both by
   keyword search and by reading. They are plausible in-domain requests whose
   figure is absent, not obviously off-topic questions. Development examples:
   Google Cloud backlog, YouTube subscription revenue, Blackwell unit
   shipments, China revenue.
5. Cross-document questions are validated so that no single document covers
   every required fact, alternatives included. That is why NVIDIA Q3-versus-Q4
   comparisons, which the Q4 release alone could answer, are not used.

Schema validation proves the files are well-formed. It does **not** prove that
the labels are semantically correct.

## Project-owner review

For each question, the reviewer checked a ten-item checklist in a local,
git-ignored review document generated from the committed files:

- the question's clarity and its answerability label;
- the reference answer, company, period, numbers and units;
- whether the evidence supports each fact, and whether alternative evidence
  pages are valid;
- duplicate intent;
- the cross-document rule, where it applies;
- that unanswerable questions are absent from the whole corpus.

Items that did not apply to a question were marked N/A in advance.

| Date | Reviewer | Scope | Outcome |
|------|----------|-------|---------|
| Not recorded by the reviewer (outcome recorded 2026-11-16) | Julian (project owner) | All 40 questions (24 dev, 16 held-out) | 40 accepted; 0 corrections requested; 0 unresolved. Every applicable checklist item affirmed (176 dev, 119 held-out); no review notes. |

Because no label changed, the dataset stays at **v1.0.0** and both file hashes
are unchanged. The review documents themselves are private working files and
are not committed.

## Benchmark freeze

Dataset v1.0.0 is the **frozen benchmark** for all later stages.

- Later stages may tune the system on **development questions only**.
- Held-out questions must not guide:
  - document preprocessing or text-cleaning decisions;
  - chunk size or overlap;
  - embedding model selection;
  - retrieval configuration (top-k, BM25 parameters, fusion weights);
  - generation prompts;
  - judge prompts or judge calibration.
- Evaluation labels and reference answers are never indexed, chunked or placed
  in retrieval or generation context.
- Any later change to questions, answers or evidence needs a new
  `dataset_version`, new hashes, a changelog entry and a written reason. Held-out
  results produced before such a change are not comparable with results after it.

## Integrity rules

- **Only development questions** may be used for tuning: chunk size, overlap,
  top-k, embedding model choice, fusion weights and prompts.
- Never index evaluation files, and never put labels or reference answers in
  retrieval or generation context.
- Do not use held-out questions as demo defaults.
- The held-out split is a **procedural holdout**. Its labels are public, so it
  is not a cryptographically isolated or independently administered test set.
- Future Claude Code sessions must not read `questions.test.jsonl` content for
  development or optimization. Running the validator, which prints only counts
  and identifiers, is allowed.
- Any modification must be explicit: bump `dataset_version`, update hashes,
  add a changelog entry and explain it here. Do not silently change questions
  after benchmarking.

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-11-16 | Initial 40 candidate questions (24 dev, 16 held-out). |
| 1.0.0 | 2026-11-16 | Metadata only: recorded the project-owner review (40 accepted, 0 corrections) and froze the benchmark. Question files and hashes unchanged. |
