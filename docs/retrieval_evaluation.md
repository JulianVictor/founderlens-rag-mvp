# Retrieval evaluation (Commit 3.4)

How BM25 and dense retrieval are compared on the development split. The
frozen [evaluation protocol v1.1](evaluation_protocol.md) is authoritative.
This document records how it is implemented and, where it is silent, the
decisions taken **before the first benchmark run**. It does not change the
protocol or the dataset.

## Software tests vs effectiveness evaluation

Unit and integration tests check that the code does what it claims: metric
arithmetic, ranking, validation. This benchmark measures something else: how
often each retriever puts the annotated evidence pages near the top for real
development questions. It measures **retrieval only**. It does not judge
answers, citations, groundedness or abstention, which belong to the
end-to-end evaluation (protocol §4–§6).

## Design (fixed before running)

| Item | Decision | Source |
|------|----------|--------|
| Split | Development only (`questions.dev.jsonl`, 24 questions). The held-out file is never opened; `--split test` is refused. | protocol §10 |
| Systems | BM25 (`rank-bm25` BM25Okapi, config `c21bdd0691f8`) and dense (MiniLM, exact cosine, config `088da6b50923`) | Commits 3.1, 3.3 |
| Collection | **V2** (`0c2f87411627`) for both, selected explicitly. Same chunker, size and overlap; only the ranking differs. | protocol §3 |
| Queries | The `question` text exactly as stored, identical for both systems | protocol §3 |
| Budgets | k ∈ {1, 3, 5, 10}; **k = 5 is the primary operating point**. The protocol lists the budgets but names no primary k, so this is fixed here, before any run. | protocol §3; this document |
| Depth | Each system is asked once per question for its top 10. Smaller k are prefixes, which is exact because both systems produce a stable total order and their top k is the first k of their top 10. | this document |
| Relevance unit | Page `(document_id, page_number)`, never page number alone. V2 chunks never cross pages, so `pages(c)` is one page. | protocol §2 |

## Metrics (protocol §3, answerable questions only)

`P_k(q)` is the set of pages of the first k chunks. There is no
deduplication before the cut: a duplicate page consumes a top-k position but
earns no repeated evidence credit, because it counts once inside the set.
`E(f)` is the set of accepted pages for fact `f`; they are **alternatives**
(any one satisfies the fact). `G(q)` is the union of `E(f)`. Required facts
are **jointly** needed.

| Metric | Per question | Notes |
|--------|--------------|-------|
| Evidence-page Recall@k | `1` if `P_k ∩ G ≠ ∅`, else `0` | A binary accepted-evidence hit, as the project protocol defines it. This differs from conventional IR recall (the fraction of relevant items retrieved): gold pages are alternatives, so that fraction would penalize retrieving one of several interchangeable pages. |
| Required-fact coverage@k | `|{f : P_k ∩ E(f) ≠ ∅}| / |F|` | Page-level availability of each fact, not proof that the chunk contains it |
| Complete coverage@k | `1` if every fact is covered, else `0` | Only for questions with ≥ 2 facts (14 of 19); undefined (`null`) otherwise |
| MRR@10 | `1 / rank` of the first chunk on a page in `G`; `0` if none in ranks 1–10 | Truncated at 10 as the protocol says. No full-ranking MRR is computed: BM25 only ranks lexically matching chunks, so a full ranking is not defined for it. |
| Unique pages@k, duplicate-page rate@k | `|P_k|`, `1 − |P_k| / chunks in top k` | Supporting diagnostics; `null` when nothing was returned |

**Aggregation.**
- The macro mean over answerable questions, reported overall and per
  category, always with `n`.
- A metric that is undefined for a question (`null`) is excluded from that
  metric's mean, and its count is reported. Undefined is never converted to
  `0`.

**Uncertainty (protocol §3).**
- 95% percentile bootstrap intervals over questions: 10,000 resamples, fixed
  seed `20261116`.
- Computed for every overall mean and for the paired difference
  (dense − BM25), using the same resampled question sets for both systems.
- Category subsets (2–7 questions) are reported as point estimates with `n`,
  without intervals.

**Paired comparison.** On the same eligible questions, per metric and k: the
mean difference, its interval, and the number of questions where dense is
higher, BM25 is higher, or they tie.

**Unanswerable questions (5).**
- Excluded from every relevance metric; there is no gold evidence (protocol
  §3).
- Reported separately: how many chunks were returned and the top and
  tenth-ranked raw scores, descriptively.
- Returned chunks are not wrong answers, and an empty result is not
  abstention. Abstention is measured end to end later.

**Failures.**
- Invalid dataset, labels or artifacts stop the run before any query.
- A query that raises (including a dense query over the token limit) is
  recorded as an execution error with its message, never as a miss.
- The benchmark is then marked incomplete, aggregates are withheld, and the
  command exits `1`.

**Latency (protocol §3).**
- Indexes and models are loaded first and timed separately (BM25 index
  build; dense artifact load; dense cold start).
- Then one warm-up query: "quarterly revenue growth", written for this
  purpose and not taken from the evaluation set.
- Then per-query wall-clock time on a monotonic clock, reported as p50 and
  p95. For dense, encoding and scoring are also timed separately.
- Latency, timestamps, machine details and the git state are kept in an
  `operational` section, outside the deterministic results.

**Reports.**
- `reports/retrieval_benchmark.dev.json`, which holds everything listed
  above, and `reports/retrieval_benchmark.dev.md`. Both are git-ignored.
- Reports contain question IDs, categories, ranks, chunk IDs, pages and
  scores. They do not contain question or chunk text.

**Deviation from the protocol text.** Protocol §3 names a FAISS CPU exact
inner-product index for the dense baseline. Commit 3.3 implements the same
exact inner-product search with NumPy (`matrix @ query` over all 164 rows). It
is exhaustive and exact in the same sense, without an extra dependency. The
protocol and dataset are unchanged.

## Running it

```powershell
python -m founderlens.evaluation.benchmark                  # dev split, BM25 and dense on V2
python -m founderlens.evaluation.benchmark --split test     # refused (exit 2): held-out not authorized
```

- It needs the V2 collection and its embedding artifact, built with
  `python -m founderlens.retrieval.embeddings build-collection` and `build
  --collection v2`, and the model in the local cache.
- It exits `1` if anything fails or the run is incomplete.
- The opt-in test `tests/test_evaluation_benchmark_real.py` runs the full
  benchmark twice in a temporary project that does not contain the held-out
  file.

## Results: development split (first run)

Evaluation config `3fd7b205dc81`. Run at commit `d8f95a7` with Commit 3.4
uncommitted.

| Identity | Value |
|----------|-------|
| Dataset | 1.0.0; `dataset.json` `c5349999…`; `questions.dev.jsonl` `12d6c7f5…` |
| Collection | V2 `0c2f87411627`, `f9a8f12a…`, 164 chunks |
| BM25 | config `c21bdd0691f8`; rank-bm25 0.2.2; tokenizer v1 |
| Dense | config `088da6b50923`; embedding `cdf57c2283c6`; matrix `ec351632…`; MiniLM `1110a243…` |

24 questions: 19 answerable (7 factual, 5 conceptual, 5 cross-document,
2 temporal) and 5 unanswerable. Both systems ran on all 24 without errors.

### Primary results (k = 5, n = 19 answerable)

| Metric | BM25 (95% CI) | Dense (95% CI) | Dense − BM25 (95% CI) | Dense better / BM25 better / tie |
|--------|---------------|----------------|-----------------------|----------------------------------|
| Recall@5 (hit) | 0.947 [0.842, 1.000] | 0.737 [0.526, 0.895] | −0.211 [−0.421, −0.053] | 0 / 4 / 15 |
| Fact coverage@5 | 0.829 [0.697, 0.947] | 0.658 [0.447, 0.842] | −0.171 [−0.368, 0.000] | 1 / 4 / 14 |
| Complete coverage@5 (n = 14) | 0.571 [0.286, 0.786] | 0.571 [0.286, 0.857] | 0.000 [−0.214, 0.214] | 1 / 1 / 12 |
| MRR@10 | 0.622 [0.473, 0.772] | 0.548 [0.387, 0.711] | −0.074 [−0.251, 0.098] | 6 / 8 / 5 |

**Interpretation.**
- BM25 achieved higher observed Recall@5 on the development set, by 21.1
  percentage points. The paired bootstrap interval prescribed by the
  protocol, [−0.421, −0.053], excluded zero.
- However, the small sample (19 answerable questions) limits the strength
  and generalizability of this finding.
- The remaining metric differences were inconclusive. An interval that
  contains zero does not show the systems are equivalent; it shows these
  data cannot tell them apart.
- **Caveat, not part of the protocol:** an exact two-sided McNemar test on
  the Recall@5 outcomes (4 BM25-only successes, 0 dense-only) gives
  p = 0.125. This shows how weak any strong statistical claim is with only
  four discordant pairs. The bootstrap remains the protocol's procedure; the
  McNemar figure is a hand-computed cross-check, not a benchmark metric.
- Nothing here shows BM25 is better than dense retrieval in general, on
  other questions, or for answer quality.

### All k

| Metric | k | BM25 | Dense | Dense − BM25 (95% CI) |
|--------|---|-----:|------:|-----------------------|
| Recall (hit) | 1 | 0.421 | 0.368 | −0.053 [−0.316, 0.211] |
| | 3 | 0.789 | 0.684 | −0.105 [−0.316, 0.105] |
| | 5 | 0.947 | 0.737 | −0.211 [−0.421, −0.053] |
| | 10 | 1.000 | 1.000 | 0.000 [0.000, 0.000] |
| Fact coverage | 1 | 0.316 | 0.316 | 0.000 [−0.211, 0.211] |
| | 3 | 0.684 | 0.579 | −0.105 [−0.289, 0.026] |
| | 5 | 0.829 | 0.658 | −0.171 [−0.368, 0.000] |
| | 10 | 0.947 | 0.882 | −0.066 [−0.171, 0.000] |
| Complete coverage (n = 14) | 1 | 0.214 | 0.214 | 0.000 [−0.214, 0.214] |
| | 3 | 0.500 | 0.500 | 0.000 [0.000, 0.000] |
| | 5 | 0.571 | 0.571 | 0.000 [−0.214, 0.214] |
| | 10 | 0.857 | 0.714 | −0.143 [−0.357, 0.000] |

### By category (k = 5; point estimates, tiny n)

| Category | n | BM25 R / Cov / Complete / MRR | Dense R / Cov / Complete / MRR |
|----------|--:|-------------------------------|--------------------------------|
| Single-document factual | 7 | 1.00 / 1.00 / 1.00† / 0.54 | 0.57 / 0.57 / 0.50† / 0.51 |
| Single-document conceptual | 5 | 1.00 / 1.00 / 1.00 / 0.67 | 1.00 / 1.00 / 1.00 / 0.67 |
| Cross-document comparison | 5 | 0.80 / 0.55 / 0.20 / 0.68 | 0.60 / 0.50 / 0.40 / 0.40 |
| Temporal / multi-evidence | 2 | 1.00 / 0.50 / 0.00 / 0.67 | 1.00 / 0.50 / 0.00 / 0.75 |

† Over the factual questions that have two or more facts.

Categories have 2–7 questions, so no category-level conclusion is drawn.

### Unanswerable questions (5, descriptive)

- Both systems returned 10 chunks for every unanswerable question.
- Top scores: BM25 13.2–17.5 (answerable questions: 13.6–41.7); dense cosine
  0.63–0.76 (answerable: 0.53–0.87).
- The unanswerable top scores fall inside the range of answerable top scores
  for both systems. On this development sample, no single score threshold
  perfectly separates unanswerable from answerable questions.
- This does not mean retrieval scores can never carry useful answerability
  information. That is untested here, and abstention is measured in the
  end-to-end evaluation.

### Duplicate pages

| | Unique pages in top 5 / 10 | Duplicate-page rate @5 / @10 |
|---|---|---|
| BM25 | 4.00 / 7.00 | 0.200 / 0.300 |
| Dense | 4.32 / 7.68 | 0.137 / 0.232 |

### Latency (this machine, CPU, 12 threads; varies between runs)

| | Index or artifact load | First (cold) query | Warm query p50 / p95 |
|---|---|---|---|
| BM25 | 7.7 ms index build | 0.3 ms | 1.28 / 2.36 ms |
| Dense | 29.4 ms artifact load | 49.4 s (run 2: 39.4 s) | 13.9 / 16.2 ms |

- Dense warm-query breakdown: encoding 13.7 / 16.0 ms; exact scoring and
  ranking 0.15 / 0.21 ms (p50 / p95).
- The dense cold start includes the PyTorch import and model load. It was
  much slower here than the 7.7 s measured in Commit 3.3, on the same
  machine. The cause was not investigated; this is reported as measured.
- Cold and warm figures are not comparable operating conditions.

### Reproducibility

- Two complete runs: the deterministic report content (everything except
  `operational`) was identical.
- Chunk IDs, ranks and scores were identical, with max abs score difference
  0.0.
- This was measured on one machine; cross-platform bitwise equality is not
  claimed.

## Error analysis (development split)

Recall@5 outcomes:
- both hit: 14;
- BM25 only: 4 (dev-005, dev-006, dev-007, dev-015);
- dense only: 0;
- neither: 1 (dev-016).

At k = 10 both systems hit every answerable question.

**Observed** (from retrieved pages against the gold pages):
- **Specific line items in tables or narrative.** BM25 found and dense
  missed, by k = 5, NVIDIA's full-year stock repurchase payments (cash-flow
  table, Q4 p7), Alphabet's employee count (Q3 p2) and NVIDIA's Automotive
  revenue (Q3 p2). Dense's top results for these were page-1 headline chunks
  of the same company.
- **Dense concentrates on page-1 summaries.** Across all questions, dense's
  most frequent top-5 pages were NVIDIA Q3 p1 (18 times), NVIDIA Q4 p1 (15)
  and Alphabet Q4 p1 (12).
- **BM25 concentrates on NVIDIA Q3 p3** (20 times in top-5 lists) and repeats
  pages more often (duplicate rate@10 0.30 vs 0.23).
- **Company coverage in cross-document questions.**
  - Dense's top 5 came from a single company in all five cross-document
    questions; BM25's in two (dev-014, dev-016).
  - In dev-015 dense retrieved only NVIDIA pages, so all four facts were
    missed at k = 5.
  - In dev-013, the opposite: dense covered both Alphabet facts and BM25 one.
- **Multi-fact questions.** Complete coverage@5 is 0.571 for both. The
  temporal questions (dev-018, dev-019) each lost one fact with both systems:
  - dev-018: the Q3 release's outlook page;
  - dev-019: Alphabet's Q4 period page.
- **Fiscal-period mixing.** Both systems often return the other quarter's
  release of the same company: Q3 pages for a Q4 question (dev-016, dev-019)
  and the reverse.

**Hypotheses, not established here.**
- MiniLM may weight general "company results" semantics above specific line
  items and exact figures.
- Exact terms ("repurchases", "employees", "Automotive") favour BM25.
- Both systems may under-represent the second company in two-company queries
  because one company's vocabulary dominates the query.

Testing any of these needs controlled experiments, which this commit does
not run. No retriever was changed after these results.

## Recommendations (not implemented)

- The systems are partly complementary at shallow depth:
  - BM25 wins four Recall@5 cases.
  - Dense wins coverage on dev-013 and has the best first-rank on several
    questions; MRR@10 favours dense on 6 questions and BM25 on 8.
  - A rank-fusion experiment (for example reciprocal rank fusion) on V2 is a
    reasonable next step.
  - It must be evaluated as a third system against both baselines with this
    harness, on the dev split, with its settings fixed before running.
  - No improvement from fusion is claimed.
- Any retriever choice must not use the held-out split. That split is for a
  single final evaluation of a frozen configuration (protocol §10).
- Cross-document questions would benefit from retrieval that does not let one
  company fill the top k. That is a hypothesis to test, not a finding.

## Limitations

- 19 answerable development questions: intervals are wide and category
  subsets are tiny.
- Page-level relevance is a proxy. A hit means a chunk is on an accepted
  page, not that it contains the fact, and nothing here measures answer
  correctness, groundedness or citation quality.
- One small general-purpose embedding model; four documents from two
  companies; one chunking configuration (V2).
- The labels were written by one annotator and accepted by one reviewer
  (protocol §11).
- The PyMuPDF AGPL licensing decision (see README) is still open. It does
  not affect these measurements, but it does affect deployment.
