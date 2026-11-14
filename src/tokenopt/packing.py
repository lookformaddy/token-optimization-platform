"""Token-budget-aware context packing.

Greedy value-density knapsack: required blocks first (fail-closed if one
does not fit), then optional blocks ordered by ``score / tokens`` until the
budget is exhausted. See budgets.py for the fail-closed enforcement side.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tokenopt.budgets import BudgetExceeded, TokenBudget
from tokenopt.dedup import ContextBlock


@dataclass
class PackResult:
    packed: list[ContextBlock] = field(default_factory=list)
    skipped: list[ContextBlock] = field(default_factory=list)
    total_tokens: int = 0

    @property
    def utilization(self) -> float:
        return 0.0  # caller divides by the budget when it knows it


def pack_context(
    blocks: list[ContextBlock],
    budget: TokenBudget,
) -> PackResult:
    """Pack blocks into ``budget`` by value density (score per token).

    Required blocks are packed first in input order; if any required block
    does not fit, ``BudgetExceeded`` is raised. Optional blocks are admitted
    greedily by ``score / tokens`` until the budget is exhausted. A block
    with zero tokens is always admitted.
    """
    available = budget.available_tokens
    packed: list[ContextBlock] = []
    skipped: list[ContextBlock] = []
    used = 0

    required = [b for b in blocks if b.required]
    optional = [b for b in blocks if not b.required]
    for block in required:
        if used + block.tokens > available:
            raise BudgetExceeded(
                f"required block {block.id} ({block.tokens} tokens) exceeds "
                f"available budget {available}",
                budget="per-request",
                used=used + block.tokens,
                limit=available,
            )
        packed.append(block)
        used += block.tokens

    def density(block: ContextBlock) -> float:
        return block.score / max(1, block.tokens)

    for block in sorted(optional, key=density, reverse=True):
        if used + block.tokens <= available:
            packed.append(block)
            used += block.tokens
        else:
            skipped.append(block)
    return PackResult(packed=packed, skipped=skipped, total_tokens=used)
