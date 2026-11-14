"""Tests: end-to-end optimizer pipeline."""
import pytest

from tokenopt.budgets import BudgetExceeded, TenantBudget
from tokenopt.demo_data import golden_set, verbose_support_prompt
from tokenopt.optimizer import OptimizeRequest
from tests.conftest import block


def test_optimize_reduces_tokens_and_reports_savings(optimizer):
    result = optimizer.optimize(OptimizeRequest(
        tenant="acme", prompt=verbose_support_prompt(), query="refund policy"))
    assert result.optimized_tokens < result.original_tokens
    assert result.tokens_saved > 0
    assert result.savings_ratio > 0.15
    assert result.cost_saved_usd > 0
    assert result.cache_hit is False
    assert result.compressed is True


def test_optimize_dedups_duplicate_context(optimizer):
    context = [
        block("c1", "Returns are accepted within 30 days of delivery.", 0.95),
        block("c2", "Returns are accepted within 30 days of delivery.", 0.90),
        block("c3", "Refunds go to the original payment method.", 0.92),
    ]
    result = optimizer.optimize(OptimizeRequest(
        tenant="acme", prompt="Summarize the refund policy.", context=context))
    assert result.dropped_block_ids == ["c2"]


def test_optimize_skips_blocks_over_budget(optimizer):
    context = [
        block("big", " ".join(["filler"] * 200), 0.5),
        block("small", "Refunds go to the original payment method.", 0.99),
    ]
    result = optimizer.optimize(OptimizeRequest(
        tenant="acme", prompt="Summarize the refund policy.",
        context=context, max_tokens=60))
    assert "big" in result.skipped_block_ids


def test_optimize_records_metering_and_audit(optimizer):
    optimizer.optimize(OptimizeRequest(tenant="acme", prompt="Hello there."))
    agg = optimizer.ledger.aggregate("acme")
    assert agg.requests == 1
    assert agg.tokens_in > 0
    ok, _ = optimizer.audit.verify()
    assert ok is True
    assert optimizer.audit.entries[-1].event == "optimize"


def test_semantic_cache_second_identical_prompt_hits(optimizer):
    prompt = "What is the refund policy for headsets?"
    first = optimizer.optimize(OptimizeRequest(tenant="acme", prompt=prompt))
    assert first.cache_hit is False
    optimizer.semantic_cache.store(prompt, "30-day returns, no restocking fee.")
    second = optimizer.optimize(OptimizeRequest(tenant="acme", prompt=prompt))
    assert second.cache_hit is True
    assert second.cached_answer == "30-day returns, no restocking fee."
    assert second.optimized_tokens == 0


def test_tiny_tenant_budget_fails_closed(optimizer):
    optimizer.enforcer.budgets["tiny"] = TenantBudget(
        tenant="tiny", max_tokens_per_request=1)
    with pytest.raises(BudgetExceeded):
        optimizer.optimize(OptimizeRequest(
            tenant="tiny", prompt=verbose_support_prompt()))


def test_active_policy_is_used(optimizer, registry):
    registry.submit_candidate("balanced", 1)
    registry.promote("balanced", 1, golden_set())
    result = optimizer.optimize(OptimizeRequest(
        tenant="acme", prompt=verbose_support_prompt()))
    assert result.policy_key == "balanced@v1"
