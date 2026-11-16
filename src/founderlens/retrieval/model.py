"""The pinned embedding model's identity, shared by chunking V2 and embeddings.

Plain constants only: importing this module loads no model and makes no
network call.
"""

from __future__ import annotations

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Hugging Face commit of the model repository (resolved 2026-11-16); never "main".
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_LICENSE = "apache-2.0"
EMBEDDING_DIMENSION = 384
# Effective input limit in tokens, [CLS] and [SEP] included (verified on load).
MAX_SEQ_LENGTH = 256
TOKENIZER_CLASS = "BertTokenizer"  # with transformers 5.22.1 (verified on load)
