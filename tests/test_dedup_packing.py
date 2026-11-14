"""Tests: semantic dedup and budget-aware packing."""
import pytest

from tokenopt.budgets import BudgetExceeded, TenantBudget, TokenBudget
from tokenopt.dedup import deduplicate_blocks
from tokenopt.packing import pack_context
from tests.conftest import block


def test_exact_duplicate_dropped():
    blocks = [
        block("a", "Returns are accepted within 30 days of delivery.", 0.95),
        block("b", "Returns are accepted within 30 days of delivery.", 0.90),
    ]
    result = deduplicate_blocks(blocks)
    assert [b.id for b in result.kept] == ["a"]
    assert [b.id for b in result.dropped] == ["b"]
    assert "a" in result.dropped_reasons["b"]


def test_distinct_blocks_all_kept():
    blocks = [
        block("a", "Returns are accepted within 30 days of delivery.", 0.95),
        block("b", "Refunds go to the original payment method.", 0.90),
    ]
    result = deduplicate_blocks(blocks)
    assert len(result.kept) == 2
    assert result.dropped == []


def test_required_blocks_never_deduped():
    blocks = [
        block("a", "Same text here.", 0.95),
        block("b", "Same text here.", 0.90, required=True),
    ]
    result = deduplicate_blocks(blocks)
    assert [b.id for b in result.kept] == ["a", "b"]


def test_dedup_tokens_saved_counts_dropped():
    blocks = [
        block("a", "Returns are accepted within 30 days of delivery.", 0.95),
        block("b", "Returns are accepted within 30 days of delivery.", 0.90),
    ]
    result = deduplicate_blocks(blocks)
    assert result.tokens_saved == blocks[1].tokens


def test_pack_fits_within_budget():
    blocks = [block("a", "Short text one.", 0.9), block("b", "Short text two.", 0.8)]
    result = pack_context(blocks, TokenBudget(max_tokens=100))
    assert len(result.packed) == 2
    assert result.total_tokens <= 100


def test_pack_prefers_high_value_density():
    # Same token count, different scores: the higher score wins the last slot.
    blocks = [
        block("cheap", "Alpha beta gamma delta epsilon.", 0.1),
        block("rich", "Zeta eta theta iota kappa.", 0.99),
    ]
    budget = TokenBudget(max_tokens=blocks[0].tokens + 1)
    result = pack_context(blocks, budget)
    assert [b.id for b in result.packed] == ["rich"]
    assert [b.id for b in result.skipped] == ["cheap"]


def test_pack_respects_reserve_tokens():
    blocks = [block("a", "Short text one.", 0.9)]
    budget = TokenBudget(max_tokens=blocks[0].tokens + 5, reserve_tokens=5)
    result = pack_context(blocks, budget)
    assert len(result.packed) == 1
    tight = TokenBudget(max_tokens=blocks[0].tokens, reserve_tokens=5)
    assert pack_context(blocks, tight).packed == []


def test_required_block_over_budget_fails_closed():
    blocks = [block("must", "This is a long required block that will not fit.", 1.0, required=True)]
    with pytest.raises(BudgetExceeded):
        pack_context(blocks, TokenBudget(max_tokens=3))


def test_tenant_budget_defaults():
    budget = TenantBudget(tenant="acme")
    assert budget.max_tokens_per_request == 8_000
    assert budget.max_cost_usd_per_day == 50.0
