# Retrieval strategy decision (Commit 4.3)

**Status:** accepted for the initial grounded generation pipeline.
**Decision:** BM25 over collection V2, ten ranked candidates per query.
**Code:** `founderlens.retrieval.policy` (policy `f20913f1c1ec`).

This is a development-informed architecture decision for a first serving
path. It is not a claim that BM25 is generally better than dense or hybrid
retrieval. No benchmark was rerun, and nothing was tuned, to make it.

## Context

Commits 3.4 and 4.2 compared BM25, dense (MiniLM, exact cosine) and hybrid
RRF on the frozen development split: 24 questions, 19 answerable, 39 required
facts, 14 questions with at least two facts. Primary k = 5. Full-precision
results, 95% bootstrap intervals and denominators are in
[retrieval_evaluation.md](retrieval_evaluation.md); the figures below are
copied from there, not remeasured.

| Metric (k = 5, n = 19) | BM25 | Dense | Hybrid RRF |
|---|---:|---:|---:|
| Recall (hit) | 0.947 | 0.737 | 0.947 |
| Required-fact coverage (macro) | 0.829 | 0.658 | 0.829 |
| Complete coverage (n = 14) | 0.571 | 0.571 | 0.643 |
| MRR@10 | 0.622 | 0.548 | 0.679 |

Warm query latency, run 1 of Commit 4.2 (one machine, CPU):

| Method | p50 | p95 |
|---|---:|---:|
| BM25 | 1.4 ms | 2.1 ms |
| Dense | 15.4 ms | 18.7 ms |
| Hybrid | 17.9 ms | 23.2 ms |

BM25 at the two depths that matter here:

| BM25 metric | k = 5 | k = 10 |
|---|---:|---:|
| Recall (hit) | 0.947 | 1.000 |
| Required-fact coverage | 0.829 | 0.947 |
| Complete coverage (n = 14) | 0.571 | 0.857 |

## Decision

| Setting | Value |
|---|---|
| Method | BM25 (`rank-bm25` BM25Okapi, k1 1.5, b 0.75, epsilon 0.25, shared lexical tokenizer), config `c21bdd0691f8` |
| Collection | V2, chunking config `0c2f87411627`, 164 chunks, artifact SHA-256 `f9a8f12a8d1c61aec3580f7f95dd7fd44a26860ff751d85d5a778693f59c55a8` |
| Candidate top-k | 10 |
| Score threshold | none |
| Page deduplication | none at the retrieval layer |
| Fallback | none (no dense, hybrid or V1 fallback) |
| Matching and ties | only chunks containing a query token; descending raw score, then collection order; a query with no token matching any chunk under the BM25 tokenizer (for example empty or punctuation-only) returns no candidates, which is not proof the corpus cannot answer |
| Runtime | no PyTorch, Sentence Transformers, embedding model or embedding matrix |

The complete policy (`RetrievalPolicy.to_dict()`), hashed into the policy ID
`f20913f1c1ec` (SHA-256 of canonical JSON, first 12 hex characters):

```json
{
  "bm25_config_id": "c21bdd0691f8",
  "candidate_top_k": 10,
  "collection": "v2",
  "collection_config_id": "0c2f87411627",
  "fallback": null,
  "matching": "a chunk is returned only if it contains a query token; empty queries return nothing",
  "method": "bm25",
  "page_deduplication": false,
  "schema_version": 1,
  "score_threshold": null,
  "tie_breaking": "descending raw score, then collection order (document, page, chunk index)"
}
```

The policy ID identifies the decision only. It is recorded separately from
the BM25 scoring ID, the collection configuration ID, the collection artifact
SHA-256 (pinned and checked on load) and the evaluation experiment IDs
(`3fd7b205dc81`, `f6982b8eb438`).

### Benchmark k versus candidate k

- **k = 5** stays the frozen primary operating point of the retrieval
  benchmark. Nothing about the protocol or its budgets changes.
- **k = 10** is the number of ranked candidates handed to generation
  (roadmap Stage 4). It was chosen because BM25's page-level evidence
  coverage rises from k = 5 to k = 10 (table above). It is not the final LLM
  context. Context selection and token budgeting belong to the generation
  stage, and k = 10 has not been validated for answer quality.

## Rationale

- At k = 5, BM25 and hybrid hit the same 18 of 19 answerable questions, and
  their mean required-fact coverage is equal.
- Hybrid had higher observed complete coverage (9 versus 8 of 14) and
  MRR@10. Those intervals include or touch zero, so the difference is
  inconclusive. That is not evidence that the two are equivalent either.
- BM25 needs no transformer inference, model files or embedding artifacts on
  the serving path, and has the lowest measured warm latency (p50 1.4 ms
  against 17.9 ms for hybrid) with no model cold start.
- Going from k = 5 to k = 10 raises BM25's observed coverage (fact coverage
  0.829 → 0.947, complete coverage 0.571 → 0.857). Generation can then select
  context from a deeper candidate list.

## Alternatives not selected

**Dense retrieval.** Not selected.
- Lower observed Recall@5 (0.737) and fact coverage@5 (0.658).
- Needs PyTorch, Sentence Transformers, the pinned model and the embedding
  matrix at runtime; its cold start was 15–41 s in Commit 4.2 runs.
- In error analysis it missed specific line items that BM25 found (stock
  repurchases, employee count, Automotive revenue), favouring page-1
  summaries. Fiscal-period mixing (returning the other quarter's release) was
  observed for **both** dense and BM25, so it is not a reason to prefer BM25.

**Hybrid RRF.** Not selected as the initial default.
- No observed gain over BM25 on Recall@5 or macro fact coverage@5.
- Higher observed complete coverage@5 and MRR@10, acknowledged above, without
  an established advantage.
- Carries the full dense runtime cost and a further level of fusion logic.
- Exclusive-candidate suppression was observed: in dev-015, BM25's rank-1
  evidence chunk was fused to position 7 behind shared candidates.
- Did not improve evidence from both companies in two-company questions.

**Deferred, not implemented:** weighted RRF, reranking, metadata-aware
retrieval, query decomposition, evidence diversification, other embedding
models and vector databases. Any of these must be pre-registered and evaluated
on the development split as a new system.

Dense and hybrid retrieval, their CLIs, tests and benchmark modes remain
available unchanged as evaluated alternatives.

## Consequences and limitations

- The development set is small (19 answerable questions); all comparisons are
  exploratory and the held-out split is still unopened.
- Coverage is page-level: a candidate on an evidence page may not state the
  required fact.
- End-to-end answer quality, citation accuracy and abstention are unmeasured.
- Known BM25 weaknesses remain:
  - dev-016 had no evidence in BM25's top 5;
  - in three of the four two-company questions, BM25's top 5 had evidence
    from at most one company;
  - both temporal questions miss one of two facts at k = 5;
  - fiscal-period mixing (the other quarter's release of the same company)
    was observed for BM25 as well as dense.
- Answerability: on the current development sample, no single observed
  top-score threshold perfectly separates answerable from unanswerable
  questions for BM25 (or for dense or hybrid). No threshold is applied, so
  abstention must be decided downstream.
- The standalone `python -m founderlens.retrieval.bm25` CLI keeps its
  historical V1 default and top-5 default; the policy applies to the
  generation pipeline only.

## Integration contract for generation (roadmap Stage 4)

```python
from founderlens.config import load_settings
from founderlens.retrieval.policy import load_policy_retriever

retriever = load_policy_retriever(load_settings())  # once per process; verifies V2
candidates = retriever.search(question)  # <= 10 RetrievalResult, ranked
retriever.identities()  # policy and artifact IDs for logging
```

- Each `RetrievalResult` carries rank, raw BM25 score and the original
  `ChunkRecord` (text, document, page, offsets, company, fiscal period,
  source URL). Scores are ranking scores, not probabilities.
- An empty result tuple means no chunk contains any query token under the
  existing BM25 lexical tokenizer (no eligible lexical match). It does not
  prove that the corpus cannot answer the question.
- A missing or changed V2 artifact, a different BM25 library or tokenizer, or
  an altered policy raises `RetrievalError`; nothing falls back.
- `python -m founderlens.retrieval.policy` verifies the policy against the
  local artifacts without running a query.
