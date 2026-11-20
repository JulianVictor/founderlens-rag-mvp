"""LLM generation (Stage 4: Grounded RAG Generation).

Commit 5.1 adds the provider layer: a typed, versioned generation
configuration (``config``), request/result/usage types and the provider
protocol (``schema``), sanitized error types (``errors``) and an OpenAI
Responses API adapter (``openai_provider``). Commit 5.2 adds evidence context,
token budgeting and the grounded prompt (``context``, ``token_budget``,
``prompts``). Commit 5.3 adds the structured answer schema and invariants
(``answer``) and the generation service (``service``). Citation resolution is
Commit 5.4.

Importing this package reads no credentials, creates no client and makes no
network call. Retrieval never imports it.
"""
