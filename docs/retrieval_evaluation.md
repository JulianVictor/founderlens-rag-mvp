# Retrieval evaluation (Commits 3.4 and 4.2)

How BM25, dense and hybrid RRF retrieval are compared on the development
split. Commit 3.4 compared BM25 and dense; Commit 4.2 adds hybrid RRF as a
third system ([below](#three-way-comparison-with-hybrid-rrf-commit-42)). The
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

## Three-way comparison with hybrid RRF (Commit 4.2)

The same harness, protocol, split and settings as Commit 3.4, with hybrid RRF
(Commit 4.1, config `b13fa6b0f706`) as a third system. **No retrieval
strategy is selected here**, and nothing was tuned: RRF keeps constant 60,
depth 10 per component and equal weights, and no retriever was changed after
these results.

### Design

| Item | Decision |
|------|----------|
| Systems | BM25 `c21bdd0691f8`, dense `088da6b50923`, hybrid RRF `b13fa6b0f706` (components: the same BM25 and dense configurations) |
| Split, k, bootstrap | Unchanged: 24 dev questions (19 answerable, 14 with ≥ 2 facts), k ∈ {1, 3, 5, 10}, **primary k = 5**, 10,000 resamples, seed `20261116` |
| Hybrid execution | The real `HybridRetriever`, with its own BM25 index, dense artifact load and model instance (nothing shared with the standalone systems). Each question is sent once to each system with identical text, top 10. |
| Component check | For every question, the hybrid's BM25 and dense component rankings must equal the standalone rankings (chunk IDs, ranks; scores within 1e-6). A disagreement fails the run (exit 1). |
| Baseline check | The rerun BM25 and dense results are compared with the two-method report `reports/retrieval_benchmark.dev.json`. **Exact equality** is required for the experiment ID (the two-method configuration with the same settings, `3fd7b205dc81` by default), dataset, collection and retriever identities, chunk IDs, ranks, document/page provenance, per-question metrics (applicability, denominators, relevance outcomes), overall and category aggregates and the dense − BM25 paired comparison. **Raw scores** are compared unrounded with an absolute tolerance of `1e-6` (the repository's `SCORE_TOLERANCE`, also used for the component check) and zero relative tolerance; a missing or non-finite score fails. The report records the tolerance, the maximum absolute difference and, separately, whether all raw scores were identical. A ranking change fails even when scores are within tolerance. Any failure exits `1`. |
| Identity | Experiment `f6982b8eb438`: the two-method configuration plus the method list and the three retriever configuration IDs. The two-method experiment keeps `3fd7b205dc81`, and its command and reports are unchanged. |
| Reports | `reports/retrieval_benchmark_three_way.dev.{json,md}` (git-ignored). The historical two-method reports are read, never written. |

**Diagnostics** (descriptive, defined before running):
- **Candidate pool:** the unordered union of the BM25 and dense top 10 that
  RRF fuses. It is a diagnostic ceiling for fusion at depth 10, not a ranked
  or deployable retriever, so it has no MRR or Recall@k.
- **Fact outcomes for hybrid at each k** (micro counts over all 39 required
  facts, mutually exclusive per fact): *retained* (an accepted page is in the
  hybrid top k); *candidate retrieval failure* (no accepted page anywhere in
  the pool); *fusion/cut-off failure* (in the pool, but not in the hybrid top
  k). The three outcomes are mutually exclusive and together exhaustive at
  every k, so they sum to 39. They are **micro** counts (each fact weighs
  the same), unlike the **macro** fact coverage above (the mean of
  per-question fractions), so the two are not interchangeable.
- All coverage here is **page-level**: a fact counts as covered when a
  retrieved chunk lies on an accepted evidence page. It does not show that
  the chunk text states the fact or that an answer could extract it.
- **Displaced relevant chunks at k = 5:** chunks on an accepted page that are
  in a baseline's top 5 but not in the hybrid top 5, with their candidate
  type (shared or exclusive), their full fused position (from the
  implemented `rrf_fuse`), and whether the fact was lost (not covered by any
  other hybrid top-5 chunk).
- **Diversity at k = 5:** documents and companies in each top 5, and which
  companies and company-periods the evidence hits come from.

### Running it

```powershell
python -m founderlens.evaluation.benchmark --include-hybrid            # three-way, dev only
python -m founderlens.evaluation.benchmark --include-hybrid --split test  # refused (exit 2)
python -m founderlens.evaluation.benchmark                             # unchanged two-method run
```

- It needs everything the two-method run needs. The baseline check also
  needs `reports/retrieval_benchmark.dev.json` in the configured reports
  directory (`--output-dir` changes only where the three-way reports are
  written). Reports are git-ignored, so on a fresh checkout run the
  two-method command first, then `--include-hybrid`. Without the file the
  three-way run still writes its own reports (with
  `baseline_reproducibility: null`) but exits `1` and prints that command.
- The three-way run only reads the two-method report; it never writes or
  regenerates it. A locally regenerated two-method report is a new run of
  experiment `3fd7b205dc81`, not the original Commit 3.4 file, so a passing
  check means "reproduces the local two-method run", and the experiment ID
  must still match exactly.
- The opt-in test `tests/test_evaluation_benchmark_real.py` also runs the
  three-way benchmark twice in a temporary project without the held-out file,
  and checks the baseline against a two-method run in that same project. The
  default tests use synthetic reports built in memory or under `tmp_path`;
  no test reads the developer's `reports/` directory.

### Results: development split

Run twice with `HF_HUB_OFFLINE=1` at commit `c7d02da` with Commit 4.2
uncommitted. Both runs exited 0.
- Baseline reproduced: no differences; raw scores identical (max abs score
  difference 0.0, inside the 1e-6 absolute tolerance).
- After the tolerance-based comparison was introduced, two further runs
  (written outside `reports/`) gave the same results. Their deterministic
  content differs from the first runs only in `baseline_reproducibility`,
  which gained `score_tolerance` and `scores_identical`. Metrics, rankings
  and identities did not change.
- Hybrid components agreed with the standalone rankings for all 24
  questions, max abs score difference 0.0.
- The dense and hybrid systems use the same embedding `cdf57c2283c6` and
  matrix `ec351632…`.

#### Quality (n = 19 answerable; complete coverage n = 14)

| Metric | k | BM25 | Dense | Hybrid |
|--------|---|-----:|------:|-------:|
| Recall (hit) | 1 | 0.421 | 0.368 | 0.474 |
| | 3 | 0.789 | 0.684 | 0.895 |
| | **5** | **0.947** [0.842, 1.000] | **0.737** [0.526, 0.895] | **0.947** [0.842, 1.000] |
| | 10 | 1.000 | 1.000 | 1.000 |
| Fact coverage | 1 | 0.316 | 0.316 | 0.395 |
| | 3 | 0.684 | 0.579 | 0.750 |
| | **5** | **0.829** [0.697, 0.947] | **0.658** [0.447, 0.842] | **0.829** [0.684, 0.947] |
| | 10 | 0.947 | 0.882 | 0.908 |
| Complete coverage | 1 | 0.214 | 0.214 | 0.286 |
| | 3 | 0.500 | 0.500 | 0.500 |
| | **5** | **0.571** [0.286, 0.786] | **0.571** [0.286, 0.857] | **0.643** [0.357, 0.857] |
| | 10 | 0.857 | 0.714 | 0.714 |
| MRR@10 | – | 0.622 [0.473, 0.772] | 0.548 [0.387, 0.711] | 0.679 [0.536, 0.816] |

Unique pages in top 5 / 10: BM25 4.00 / 7.00, dense 4.32 / 7.68, hybrid
4.21 / 7.37. Duplicate-page rate @5 / @10: 0.200 / 0.300, 0.137 / 0.232,
0.158 / 0.263.

#### Paired differences (exploratory)

Candidate − baseline over the same questions; "better" counts are candidate
better / baseline better / tie.

| Pair | Metric | Δ mean | 95% CI | better / worse / tie |
|------|--------|-------:|--------|----------------------|
| Hybrid − BM25 | Recall@5 | 0.000 | [0.000, 0.000] | 0 / 0 / 19 |
| | Fact coverage@5 | 0.000 | [−0.079, 0.079] | 1 / 1 / 17 |
| | Complete coverage@5 | +0.071 | [0.000, 0.214] | 1 / 0 / 13 |
| | MRR@10 | +0.056 | [−0.105, 0.224] | 7 / 3 / 9 |
| | Recall@3 | +0.105 | [0.000, 0.263] | 2 / 0 / 17 |
| | Fact coverage@10 | −0.039 | [−0.105, 0.000] | 0 / 2 / 17 |
| | Complete coverage@10 | −0.143 | [−0.357, 0.000] | 0 / 2 / 12 |
| Hybrid − dense | Recall@5 | +0.211 | [0.053, 0.421] | 4 / 0 / 15 |
| | Fact coverage@5 | +0.171 | [0.013, 0.342] | 4 / 0 / 15 |
| | Complete coverage@5 | +0.071 | [0.000, 0.214] | 1 / 0 / 13 |
| | MRR@10 | +0.131 | [0.006, 0.267] | 8 / 2 / 9 |
| Dense − BM25 | Recall@5 | −0.211 | [−0.421, −0.053] | 0 / 4 / 15 (identical to Commit 3.4) |

All other k are in the JSON report.

**Interpretation.**
- **Hybrid vs BM25.** At the primary k, hybrid and BM25 had the same
  Recall@5 on every question and the same mean fact coverage@5. Hybrid had
  observed higher complete coverage@5 (9 of 14 eligible questions against
  8 of 14), MRR@10 and Recall@3,
  but these intervals contain zero or touch it. With these data the two
  cannot be told apart at k = 5. That is not evidence that they are
  equivalent.
- **Hybrid vs dense.** Hybrid had higher observed Recall@5 (+21.1 points),
  fact coverage@5 and MRR@10, and the protocol's bootstrap intervals
  excluded zero.
  - The Recall@5 difference rests on four discordant questions, all in
    hybrid's favour. An exact two-sided McNemar cross-check on 4-to-0
    discordant pairs gives p = 0.125 (hand-computed, not part of the
    protocol), so the evidence is weak.
  - Hybrid contains the BM25 ranking, so recovering BM25's wins over dense
    is expected rather than surprising.
- **At k = 10 hybrid trailed BM25** on fact and complete coverage (two
  questions each, dev-014 and dev-015). Both intervals touch zero.
- These are three pairwise comparisons across several metrics and k, with no
  multiplicity correction. They are exploratory, on 19 questions, and do not
  establish general superiority of any system.

#### Recall@5 outcomes

| Pair | Both hit | First only | Second only | Neither |
|------|---------:|------------|-------------|---------|
| Hybrid vs BM25 | 18 | hybrid only: none | BM25 only: none | 1 (dev-016) |
| Hybrid vs dense | 14 | hybrid only: 4 (dev-005, dev-006, dev-007, dev-015) | dense only: none | 1 (dev-016) |
| Dense vs BM25 | 14 | dense only: none | BM25 only: 4 (dev-005, dev-006, dev-007, dev-015) | 1 (dev-016) |

Relative to BM25, hybrid recovered and lost no Recall@5 question. Relative
to dense, it recovered exactly BM25's four wins and lost none.

At the fact level (k = 5), relative to BM25:
- hybrid gained dev-013 (both Alphabet facts, 0.50 → 1.00, now complete);
- it lost two of three covered facts in dev-015 (0.75 → 0.25).

#### By category (k = 5; R / Cov / Complete (n) / MRR@10; tiny n, point estimates)

| Category | n | BM25 | Dense | Hybrid |
|----------|--:|------|-------|--------|
| Single-document factual | 7 | 1.00 / 1.00 / 1.00 (2) / 0.54 | 0.57 / 0.57 / 0.50 (2) / 0.51 | 1.00 / 1.00 / 1.00 (2) / 0.65 |
| Single-document conceptual | 5 | 1.00 / 1.00 / 1.00 (5) / 0.67 | 1.00 / 1.00 / 1.00 (5) / 0.67 | 1.00 / 1.00 / 1.00 (5) / 0.80 |
| Cross-document comparison | 5 | 0.80 / 0.55 / 0.20 (5) / 0.68 | 0.60 / 0.50 / 0.40 (5) / 0.40 | 0.80 / 0.55 / 0.40 (5) / 0.56 |
| Temporal / multi-evidence | 2 | 1.00 / 0.50 / 0.00 (2) / 0.67 | 1.00 / 0.50 / 0.00 (2) / 0.75 | 1.00 / 0.50 / 0.00 (2) / 0.75 |

No category-level conclusion is drawn.

#### Candidate pool and fact outcomes

- Pool size: mean 15.7 distinct chunks (min 11, max 19), with 4.3 shared
  between BM25 and dense on average.
- **Pool coverage:** macro fact coverage 0.947 (n = 19); 37 of 39 required
  facts have an accepted page somewhere in the pool.
- **Hybrid fact outcomes** (micro, 39 facts):

  | k | Retained | Candidate retrieval failure | Fusion/cut-off failure |
  |---|---:|---:|---:|
  | 1 | 14 | 2 | 23 |
  | 3 | 28 | 2 | 9 |
  | 5 | 31 | 2 | 6 |
  | 10 | 35 | 2 | 2 |

- **Candidate retrieval failures** (neither retriever's top 10 has the page):
  dev-016-f2 and dev-018-f1.
- **Fusion/cut-off failures at k = 5:**
  - dev-014-f2;
  - dev-015-f1, f2, f3;
  - dev-016-f1;
  - dev-019-f2.

  Only two of these six (dev-015-f2 and f3) were in a baseline's top 5 and
  lost by exclusive-candidate suppression (next section). The other four
  were missing from **both** the BM25 and the dense top 5 as well: their
  evidence sat at ranks 6–10 of at least one list, so they are general
  cut-off failures of the candidate union, not suppression relative to a
  baseline.

  At k = 10 two remain: dev-014-f2 and dev-015-f1. These evidence pages were
  in one list's top 10 but fused below position 10.

#### Displaced exclusive evidence (k = 5)

- 11 (relevant chunk, fact) pairs were in a baseline's top 5 but not in the
  hybrid top 5. All the chunks were **exclusive** candidates (found by only
  one retriever), fused at positions 6–9.
  - 7 came from BM25: dev-001 (two chunks), dev-004, dev-010 (one chunk,
    two facts), dev-015 (one chunk, two facts).
  - 4 came from dense: dev-001, dev-013 (two chunks), dev-014.
- In 9 of the 11, the fact was still covered by another hybrid top-5 chunk:
  losing a particular chunk is not the same as losing all accepted evidence
  for the fact.
- **Evidence was lost in one question:** dev-015. BM25's **rank-1** chunk,
  Alphabet Q4 p8, covered facts f2 and f3. It was BM25-exclusive and fused at
  position 7, behind 6 shared candidates. This is the shared-candidate effect
  described in the architecture document (any shared chunk scores ≥ 2/70,
  any exclusive chunk ≤ 1/61).

#### Cross-document and temporal diversity (k = 5)

- Four answerable questions require evidence from two companies: dev-014,
  dev-015, dev-016 and dev-019. "One or fewer companies with evidence" in the
  top 5:
  - BM25: 3 (dev-014, dev-016, dev-019);
  - dense: 4;
  - hybrid: 4.
- In dev-015, BM25's top 5 had evidence from both Alphabet and NVIDIA, but
  hybrid's only from NVIDIA: the lost Alphabet page above.
- dev-016 had no evidence page in any system's top 5.
- Temporal questions (dev-018, dev-019): all three systems covered one of two
  facts at k = 5. dev-018's missing fact is a candidate retrieval failure.
  dev-019's is a cut-off failure that hybrid recovers by k = 10.

#### Unanswerable questions (5, descriptive)

All three systems returned 10 chunks for every unanswerable question.

| System | Answerable top score | Unanswerable top score | Ranges overlap |
|--------|---------------------|------------------------|----------------|
| BM25 | 13.64–41.69 | 13.22–17.48 | yes |
| Dense (cosine) | 0.534–0.866 | 0.631–0.762 | yes |
| Hybrid (RRF) | 0.03055–0.03279 | 0.03150–0.03279 | yes |

- The three scales are different quantities and must not be compared
  across systems.
- RRF scores depend only on ranks. The maximum, 2/61 ≈ 0.03279, means both
  retrievers ranked the same chunk first, and it occurs for answerable and
  unanswerable questions alike.
- On this sample no single score threshold perfectly separates unanswerable
  from answerable questions for any system. Abstention is measured end to
  end later.

#### Latency (this machine, CPU; same run, not deterministic)

Execution order:
1. Load the systems in sequence (BM25 index build, dense artifact load,
   hybrid initialization).
2. One warm-up ("cold") query per system, in the order BM25, dense, hybrid.
3. Each question, in the order BM25, dense, hybrid.

| Run 1 (`reports/`) | Load | Cold first query | Warm p50 / p95 | Component p50 / p95 |
|--------|------|------------------|----------------|---------------------|
| BM25 | 7.9 ms index build | 0.3 ms | 1.41 / 2.14 ms | – |
| Dense | 17.8 ms artifact load | 41.3 s | 15.42 / 18.69 ms | encode 15.24 / 18.52; score and rank 0.17 / 0.27 ms |
| Hybrid | 36.2 ms (own BM25 index and dense artifact) | 197.6 ms | 17.89 / 23.25 ms | BM25 1.46 / 2.26; dense 16.01 / 21.43; fusion 0.21 / 0.24 ms |

- Run 2 measured the same pattern with different absolute values: dense cold
  start 15.0 s; hybrid warm p50 / p95 12.61 / 14.82 ms; fusion p50 0.18 ms.
- **The hybrid cold start is not a standalone cold start.** It ran after the
  dense warm-up, which had already imported PyTorch and read the model files
  into the OS cache. Only its own model instantiation is included. A fresh
  process serving hybrid alone would pay roughly the dense cold start.
- Hybrid end-to-end time is measured around the whole call. Component times
  are measured inside it, and their medians must not be added (the median of
  a sum is not the sum of medians).
- The dense cold start varied between runs again (41.3 s vs 15.0 s). The cause
  was not investigated.

#### Reproducibility

- The two runs gave identical deterministic report content (everything
  except `operational`).
- In every run, the baseline was reproduced with identical raw scores and
  the hybrid components agreed exactly.
- These checks were made on one machine; cross-platform bitwise equality is
  not claimed.

### Observations, hypotheses and recommendations (Commit 4.2)

**Observed:**
- On 19 development questions, hybrid matched BM25's Recall@5 question for
  question, and recovered all four of BM25's Recall@5 wins over dense.
- Hybrid had the highest observed MRR@10, Recall@1–3 and complete
  coverage@5, all by small margins with intervals including or touching
  zero against BM25.
- Hybrid's top 5 covers 31 of 39 required facts (micro). Its 8 misses at
  k = 5 are mostly fusion/cut-off failures (6), not candidate failures (2).
  The pool covers 37 of 39 facts.
- Only one question (dev-015, facts f2 and f3) lost evidence to the
  shared-candidate effect: a BM25-exclusive rank-1 chunk that fused to
  position 7. The other four cut-off failures were outside both baselines'
  top 5 too.
- Hybrid did not improve company coverage on two-company questions.
- Hybrid trailed BM25 at k = 10 on two questions (dev-014, dev-015).

**Hypotheses, not tested:**
- The fixed shared-before-exclusive ordering costs evidence when one
  retriever finds a decisive page the other misses (dev-015).
- Most remaining failures are positional, so a fusion that weights exclusive
  candidates differently, or a later reranking step, might recover them.
  This could equally hurt questions that are currently retained.
- Company-balanced selection might help two-company questions.

**Recommendations (not implemented; nothing is selected):**
- These results alone do not justify choosing a default. The paired data
  cannot tell hybrid and BM25 apart at the primary k. Hybrid costs about
  10–15× BM25's warm latency, plus a model cold start.
- Any change to RRF (constant, weights, depths) or added steps (reranking,
  diversification, deduplication) must be pre-registered and evaluated with
  this harness on the dev split as a new system. It must not be tuned
  against these numbers question by question.
- The held-out split stays closed until one frozen configuration is chosen
  (protocol §10).

### Limitations (Commit 4.2)

- The same limitations as Commit 3.4 apply: 19 answerable questions;
  page-level relevance only; one embedding model; four documents from two
  companies; one chunking configuration; single-annotator labels accepted
  by one reviewer.
- Many exploratory comparisons without multiplicity correction.
- Latency comes from two runs on one machine, with large cold-start
  variation.
- No answer quality, citation or abstention is measured.
- The PyMuPDF AGPL licensing decision (see README) is still open.
