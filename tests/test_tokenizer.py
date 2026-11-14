"""Tests: deterministic token counting and cost estimation."""
import pytest

from tokenopt.tokenizer import (
    cost_usd, estimate_tokens, savings_report, truncate_to_tokens, word_tokens,
)


def test_empty_text_is_zero_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0


def test_deterministic_counts():
    text = "The quick brown fox jumps over the lazy dog."
    assert estimate_tokens(text) == estimate_tokens(text)
    assert estimate_tokens(text) > 0


def test_longer_text_costs_more_tokens():
    short = "Refund policy."
    long = "Refund policy. " * 20
    assert estimate_tokens(long) > estimate_tokens(short)


def test_long_words_cost_more_than_short_words():
    assert estimate_tokens("antidisestablishmentarianism") > estimate_tokens("cat")


def test_word_tokens_splits_punctuation():
    assert word_tokens("hello, world!") == ["hello", ",", "world", "!"]


def test_truncate_respects_budget():
    text = " ".join(["word"] * 50)
    truncated = truncate_to_tokens(text, 10)
    assert estimate_tokens(truncated) <= 10
    assert truncated.startswith("word")


def test_truncate_to_zero_is_empty():
    assert truncate_to_tokens("hello world", 0) == ""


def test_cost_scales_with_tokens():
    assert cost_usd(2000) == pytest.approx(2 * cost_usd(1000))
    assert cost_usd(0) == 0.0


def test_cost_uses_custom_pricing():
    pricing = {"input_per_1k": 0.01, "output_per_1k": 0.03}
    assert cost_usd(1000, 1000, pricing=pricing) == pytest.approx(0.04)


def test_savings_report():
    report = savings_report(100, 60)
    assert report["tokens_saved"] == 40
    assert report["savings_ratio"] == pytest.approx(0.4)
    assert report["cost_saved_usd"] > 0


def test_savings_report_never_negative():
    report = savings_report(50, 80)
    assert report["tokens_saved"] == 0
    assert report["savings_ratio"] == 0.0
