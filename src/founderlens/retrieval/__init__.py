"""Retrieval over the chunk collection (Stage 3).

Commit 3.1 implements the BM25 lexical baseline (``bm25``) with its shared
lexical tokenizer (``tokenize``) and the result/collection types (``schema``)
that dense retrieval will reuse. Commit 3.2 adds the token-length audit and
embedding artifact pipeline (``embeddings``; the model loads lazily, never on
import). Commit 3.3 adds exact dense (cosine) retrieval over the validated
embedding artifact (``dense``). Commit 4.1 adds reciprocal rank fusion of
the two over the same V2 collection (``hybrid``). Commit 4.3 freezes the
retrieval path for generation, BM25 on V2 with ten candidates (``policy``).
"""
