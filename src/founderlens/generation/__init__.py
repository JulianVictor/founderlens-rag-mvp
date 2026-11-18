"""LLM generation (Stage 4: Grounded RAG Generation).

Commit 5.1 adds the provider layer only: a typed, versioned generation
configuration (``config``), request/result/usage types and the provider
protocol (``schema``), sanitized error types (``errors``) and an OpenAI
Responses API adapter (``openai_provider``). Retrieval, evidence context,
grounded prompts, citations and abstention are later commits (5.2 to 5.4).

Importing this package reads no credentials, creates no client and makes no
network call. Retrieval never imports it.
"""
