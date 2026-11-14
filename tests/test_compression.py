"""Tests: extractive prompt compression."""
import pytest

from tokenopt.compression import compress_prompt, split_sentences
from tokenopt.demo_data import verbose_support_prompt
from tokenopt.tokenizer import estimate_tokens


PROMPT = (
    "Summarize the refund policy for a customer who bought a wireless headset. "
    "The customer wants to know the return window and whether a refund goes to "
    "the original payment method. "
    "The policy states a 30-day return window. "
    "Refunds go to the original payment method within 5 business days. "
    "No restocking fee applies to unopened items. "
    "Please keep the answer concise and factual."
)


def test_split_sentences():
    assert len(split_sentences(PROMPT)) == 6


def test_compression_reduces_tokens():
    result = compress_prompt(PROMPT, 0.6, query="refund policy return window")
    assert result.compressed_tokens < result.original_tokens
    assert result.savings_ratio > 0


def test_compression_never_invents_text():
    result = compress_prompt(PROMPT, 0.6, query="refund policy")
    for sentence in split_sentences(result.compressed_text):
        assert sentence in PROMPT


def test_query_relevant_sentences_survive():
    result = compress_prompt(PROMPT, 0.5, query="restocking fee unopened items")
    assert "restocking" in result.compressed_text.lower()


def test_compression_preserves_sentence_order():
    result = compress_prompt(PROMPT, 0.6, query="refund policy")
    kept = split_sentences(result.compressed_text)
    positions = [PROMPT.index(s) for s in kept]
    assert positions == sorted(positions)


def test_empty_prompt_round_trips():
    result = compress_prompt("", 0.6)
    assert result.compressed_text == ""
    assert result.original_tokens == 0


def test_invalid_target_ratio_rejected():
    with pytest.raises(ValueError):
        compress_prompt(PROMPT, 1.0)
    with pytest.raises(ValueError):
        compress_prompt(PROMPT, 0.0)


def test_min_keep_sentences_guard_rejects_over_compression():
    # One-sentence prompt with a tiny target would leave nothing meaningful.
    result = compress_prompt("Short.", 0.1, min_keep_sentences=2)
    assert result.rejected
    assert result.compressed_text == "Short."


def test_demo_prompt_compresses_measurably():
    result = compress_prompt(verbose_support_prompt(), 0.6, query="refund policy")
    assert result.savings_ratio > 0.15
    assert "30-day" in result.compressed_text
    assert "payment" in result.compressed_text.lower()
