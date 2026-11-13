
"""Metrics for the Syntax Agentic RAG evaluation."""
from __future__ import annotations

import re
import unicodedata


_GREEK = str.maketrans({
    "α": " alpha ",
    "β": " beta ",
    "γ": " gamma ",
    "δ": " delta ",
    "κ": " kappa ",
    "λ": " lambda ",
    "μ": " mu ",
    "ω": " omega ",
})
_WORD_SUFFIXES = ("ations", "ation", "ated", "ing", "ed", "es", "s")


def _normalized_tokens(text: str) -> tuple[str, ...]:
    text = text.casefold().translate(_GREEK)
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )
    return tuple(re.findall(r"[a-z0-9]+", text))


def _stem(token: str) -> str:
    for suffix in _WORD_SUFFIXES:
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]
    return token


def required_term_present(answer: str, term: str, alternatives: list[str] | None = None) -> bool:
    """Match normalized phrases, common inflections, Greek symbols, and explicit benchmark aliases."""
    answer_tokens = tuple(_stem(token) for token in _normalized_tokens(answer))
    for candidate in [term, *(alternatives or [])]:
        tokens = tuple(_stem(token) for token in _normalized_tokens(candidate))
        if not tokens:
            continue
        width = len(tokens)
        if any(answer_tokens[start:start + width] == tokens for start in range(len(answer_tokens) - width + 1)):
            return True
    return False


def has_abstention_justification(answer: str) -> bool:
    """Detect an explicit corpus limitation without depending on retrieved/cited documents."""
    text = answer.casefold()
    scope = re.search(r"\b(corpus|document|source|provided information|available information)\b", text)
    absence = re.search(
        r"\b(?:does not|doesn't|do not|don't) (?:contain|include|provide|identify|establish|state|mention|list|give)\b"
        r"|\bnot (?:available|provided|found|included|in the corpus)\b"
        r"|\bcannot (?:determine|establish|answer)\b"
        r"|\babsent from (?:the )?(?:corpus|documents?|sources?)\b",
        text,
    )
    return bool(scope and absence)


def source_recall(expected: list[str], actual: list[str]) -> float:
    expected_set = set(expected)
    actual_set = set(actual)

    if not expected_set:
        return 1.0

    return len(expected_set & actual_set) / len(expected_set)


def source_precision(expected: list[str], actual: list[str]) -> float:
    expected_set = set(expected)
    actual_set = set(actual)

    if not expected_set:
        return 1.0 if not actual_set else 0.0
    if not actual_set:
        return 0.0

    return len(expected_set & actual_set) / len(actual_set)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
