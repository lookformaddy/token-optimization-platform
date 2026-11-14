"""Tests: deterministic evaluation and the quality/cost trade-off."""
import pytest

from tokenopt.evaluation import (
    EvaluationResult, GoldenCase, evaluate_policy, keyword_coverage, rouge_l_f1,
)


def test_keyword_coverage_full_and_empty():
    assert keyword_coverage("refund 30-day payment", {"refund", "payment"}) == 1.0
    assert keyword_coverage("nothing relevant", {"refund", "payment"}) == 0.0
    assert keyword_coverage("anything", set()) == 1.0


def test_rouge_l_identical_is_one():
    assert rouge_l_f1("the refund policy", "the refund policy") == pytest.approx(1.0)


def test_rouge_l_disjoint_is_zero():
    assert rouge_l_f1("alpha beta", "gamma delta") == 0.0


def test_rouge_l_partial_between_zero_and_one():
    score = rouge_l_f1("the quick brown fox", "the quick red fox")
    assert 0.0 < score < 1.0


def test_evaluate_policy_returns_metrics(golden):
    result = evaluate_policy(
        "balanced", {"target_ratio": 0.6, "dedup_threshold": 0.85}, golden)
    assert result.cases == len(golden)
    assert 0.0 <= result.mean_quality_retention <= 1.0
    assert result.mean_tokens_saved_ratio > 0
    assert result.mean_cost_saved_usd > 0
    assert result.mean_latency_ms > 0
    assert result.worst_quality_retention <= result.mean_quality_retention


def test_aggressive_policy_saves_more_than_balanced(golden):
    balanced = evaluate_policy(
        "balanced", {"target_ratio": 0.6, "dedup_threshold": 0.85}, golden)
    aggressive = evaluate_policy(
        "aggressive", {"target_ratio": 0.4, "dedup_threshold": 0.80}, golden)
    assert aggressive.mean_tokens_saved_ratio >= balanced.mean_tokens_saved_ratio


def test_evaluate_empty_golden_set_rejected():
    with pytest.raises(ValueError):
        evaluate_policy("x", {}, [])


def test_passes_gate_logic():
    result = EvaluationResult(
        policy_name="x", cases=4, mean_quality_retention=0.9,
        mean_tokens_saved_ratio=0.3, worst_quality_retention=0.85)
    assert result.passes(min_quality=0.85, min_savings=0.15)
    assert not result.passes(min_quality=0.95, min_savings=0.15)
    assert not result.passes(min_quality=0.85, min_savings=0.40)
