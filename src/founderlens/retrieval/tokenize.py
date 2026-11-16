"""Lexical tokenization shared by indexed chunks and queries (BM25).

Rule (tokenizer version 1):

1. ``str.casefold()`` the text (Unicode-aware; e.g. ``"NVIDIA"`` and
   ``"Nvidia"`` both become ``"nvidia"``; ``"ß"`` becomes ``"ss"``).
2. Emit every match of ``[^\\W_]+(?:[.,-][^\\W_]+)*``: a run of Unicode
   letters or digits, optionally joined to further runs by a single ``.``,
   ``,`` or ``-`` with no surrounding space.

Everything else (spaces, ``$``, ``%``, parentheses, apostrophes, slashes,
dashes other than ``-``, underscores, standalone punctuation) separates tokens
and is dropped. Examples:

==================  ==========================
Text                Tokens
==================  ==========================
``NVIDIA``          ``nvidia``
``Blackwell``       ``blackwell``
``FY2025``          ``fy2025``
``Q4``              ``q4``
``$39,331``         ``39,331``
``37.5%``           ``37.5``
``non-GAAP``        ``non-gaap``
``Google Cloud``    ``google``, ``cloud``
``(146)``           ``146``
``NVIDIA's``        ``nvidia``, ``s``
==================  ==========================

Tokens are lexical keys only. Dropping ``$``, ``%`` or the parentheses of a
negative value loses meaning that the original chunk text, which is returned
unchanged, still carries. Numbers are not converted: ``39,331`` and ``39331``
are different tokens. There is no stemming, lemmatization, stop-word removal
or synonym expansion. This is not the ingestion word count
(``founderlens.ingestion.chunk.count_words``).
"""

from __future__ import annotations

import re

TOKENIZER_NAME = "founderlens-lexical"
TOKENIZER_VERSION = 1

_TOKEN_RE = re.compile(r"[^\W_]+(?:[.,\-][^\W_]+)*")


def tokenize(text: str) -> tuple[str, ...]:
    """Lexical tokens of ``text`` in source order (duplicates kept)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    return tuple(_TOKEN_RE.findall(text.casefold()))
