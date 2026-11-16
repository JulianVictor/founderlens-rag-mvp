"""Lexical tokenizer shared by BM25 documents and queries."""

from __future__ import annotations

import pytest

from founderlens.ingestion.chunk import count_words
from founderlens.retrieval.tokenize import tokenize

RSQUO = chr(0x2019)  # typographic apostrophe, as in the corpus


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("NVIDIA", ("nvidia",)),
        ("Blackwell", ("blackwell",)),
        ("FY2025", ("fy2025",)),
        ("Q4", ("q4",)),
        ("$39,331", ("39,331",)),
        ("37.5%", ("37.5",)),
        ("non-GAAP", ("non-gaap",)),
        ("Google Cloud", ("google", "cloud")),
        ("(146)", ("146",)),
        ("NVIDIA" + RSQUO + "s 10-K for the U.S.", ("nvidia", "s", "10-k", "for", "the", "u.s")),
        ("Revenue was $39,331 million.", ("revenue", "was", "39,331", "million")),
        ("$ 1,234.56 (12.5)%  —  3rd", ("1,234.56", "12.5", "3rd")),
    ],
)
def test_representative_financial_terms(text: str, tokens: tuple[str, ...]) -> None:
    assert tokenize(text) == tokens


def test_case_is_folded_consistently_for_queries_and_documents() -> None:
    assert tokenize("NVIDIA Blackwell") == tokenize("nvidia BLACKWELL") == ("nvidia", "blackwell")
    assert tokenize("STRASSE") == tokenize("Straße")  # casefold, not lower()


def test_unicode_letters_and_digits_are_tokens() -> None:
    assert tokenize("Café 日本 ٢٠٢٤ €5") == (
        "café",
        "日本",
        "٢٠٢٤",
        "5",
    )


@pytest.mark.parametrize("text", ["", "   \n\t", "$ % ( ) - — ... , .", "___"])
def test_empty_and_punctuation_only_input(text: str) -> None:
    assert tokenize(text) == ()


def test_tokens_differ_from_ingestion_words_and_leave_text_alone() -> None:
    text = "Revenue  $ 35,082   (1,234)   37.5 %"
    assert count_words(text) == 6  # ingestion words include "$" and "%"
    assert tokenize(text) == ("revenue", "35,082", "1,234", "37.5")
    assert text == "Revenue  $ 35,082   (1,234)   37.5 %"  # input unchanged


def test_numbers_are_not_converted() -> None:
    assert tokenize("39,331") != tokenize("39331")


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        tokenize(123)  # type: ignore[arg-type]  # invalid type on purpose
