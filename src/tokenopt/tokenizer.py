"""Deterministic token counting and cost estimation.

No model tokenizer is bundled (no network, no downloads). Token counts use a
stable, documented approximation: text is split into word and punctuation
tokens; a word token contributes ``max(1, ceil(len / 4))`` sub-tokens, which
tracks the ~4-characters-per-token behaviour of common BPE tokenizers closely
enough for budgeting, compression ratios and cost accounting. The important
property is determinism: the same text always yields the same count, so
budgets, savings and A/B comparisons are reproducible.
"""
from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Default price list (USD per 1,000 tokens). Override per tenant/model.
DEFAULT_PRICING = {
    "input_per_1k": 0.002,
    "output_per_1k": 0.006,
}


def word_tokens(text: str) -> list[str]:
    """Split text into word/punctuation tokens."""
    return _TOKEN_RE.findall(text or "")


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate for ``text``.

    >>> estimate_tokens("hello world")
    2
    >>> estimate_tokens("") == 0
    True
    """
    total = 0
    for tok in word_tokens(text):
        if tok.isalnum():
            total += max(1, math.ceil(len(tok) / 4))
        else:
            total += 1
    return total


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to at most ``max_tokens`` estimated tokens.

    Truncation happens at token boundaries and never splits a word token.
    """
    if max_tokens <= 0:
        return ""
    kept: list[str] = []
    used = 0
    for tok in word_tokens(text):
        cost = max(1, math.ceil(len(tok) / 4)) if tok.isalnum() else 1
        if used + cost > max_tokens:
            break
        kept.append(tok)
        used += cost
    # Re-join with single spaces; punctuation spacing is cosmetic here.
    out = " ".join(kept)
    # Remove the space inserted before punctuation for readability.
    out = re.sub(r"\s+([,.;:!?%)\]])", r"\1", out)
    out = re.sub(r"([(\[{])\s+", r"\1", out)
    return out


def cost_usd(
    input_tokens: int,
    output_tokens: int = 0,
    *,
    pricing: dict[str, float] | None = None,
) -> float:
    """USD cost for a request at the given price list."""
    p = pricing or DEFAULT_PRICING
    return (
        input_tokens * p.get("input_per_1k", 0.0) / 1_000
        + output_tokens * p.get("output_per_1k", 0.0) / 1_000
    )


def savings_report(before_tokens: int, after_tokens: int) -> dict[str, float]:
    """Token/cost savings summary for a before/after pair."""
    if before_tokens <= 0:
        return {"tokens_saved": 0, "savings_ratio": 0.0, "cost_saved_usd": 0.0}
    saved = max(0, before_tokens - after_tokens)
    return {
        "tokens_saved": saved,
        "savings_ratio": saved / before_tokens,
        "cost_saved_usd": cost_usd(saved),
    }
