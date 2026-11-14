"""Per-tenant budgets and fail-closed enforcement.

See packing.py for the per-request context packing that consumes
``TokenBudget``. This module owns the enforcement side: per-request and
per-tenant (daily token / cost) ceilings, raising ``BudgetExceeded`` instead
of silently over-running.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised when a request would exceed a token or cost budget.

    Fail-closed: callers must not fall back to an unbudgeted path.
    """

    def __init__(self, message: str, *, budget: str = "", used: int = 0, limit: int = 0):
        super().__init__(message)
        self.budget = budget
        self.used = used
        self.limit = limit


@dataclass
class TokenBudget:
    """Per-request token budget for a single assembled prompt."""

    max_tokens: int
    reserve_tokens: int = 0  # tokens held back for the model's reply

    @property
    def available_tokens(self) -> int:
        return max(0, self.max_tokens - self.reserve_tokens)


@dataclass
class TenantBudget:
    """Daily ceilings for one tenant (fail-closed when exceeded)."""

    tenant: str
    max_tokens_per_request: int = 8_000
    max_tokens_per_day: int = 1_000_000
    max_cost_usd_per_day: float = 50.0


@dataclass
class BudgetEnforcer:
    """Enforces per-request and per-tenant budgets against live usage.

    ``usage`` maps tenant -> {"tokens": int, "cost_usd": float} for today.
    The enforcer is deliberately state-light: production deployments back it
    with the metering ledger (see metering.py).
    """

    budgets: dict[str, TenantBudget] = field(default_factory=dict)
    usage: dict[str, dict[str, float]] = field(default_factory=dict)

    def budget_for(self, tenant: str) -> TenantBudget:
        return self.budgets.get(tenant) or TenantBudget(tenant=tenant)

    def check_request(
        self,
        tenant: str,
        request_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        """Raise ``BudgetExceeded`` if the request would breach any ceiling."""
        budget = self.budget_for(tenant)
        if request_tokens > budget.max_tokens_per_request:
            raise BudgetExceeded(
                f"tenant {tenant}: request of {request_tokens} tokens exceeds "
                f"per-request limit {budget.max_tokens_per_request}",
                budget="per-request",
                used=request_tokens,
                limit=budget.max_tokens_per_request,
            )
        used = self.usage.get(tenant, {"tokens": 0.0, "cost_usd": 0.0})
        if used["tokens"] + request_tokens > budget.max_tokens_per_day:
            raise BudgetExceeded(
                f"tenant {tenant}: daily token budget exceeded "
                f"({used['tokens'] + request_tokens:.0f} > {budget.max_tokens_per_day})",
                budget="daily-tokens",
                used=int(used["tokens"] + request_tokens),
                limit=budget.max_tokens_per_day,
            )
        if used["cost_usd"] + estimated_cost_usd > budget.max_cost_usd_per_day:
            raise BudgetExceeded(
                f"tenant {tenant}: daily cost budget exceeded "
                f"(${used['cost_usd'] + estimated_cost_usd:.2f} > "
                f"${budget.max_cost_usd_per_day:.2f})",
                budget="daily-cost",
                used=int(used["cost_usd"] + estimated_cost_usd),
                limit=int(budget.max_cost_usd_per_day),
            )

    def record(self, tenant: str, tokens: int, cost_usd: float) -> None:
        """Book actual usage after a request completes."""
        entry = self.usage.setdefault(tenant, {"tokens": 0.0, "cost_usd": 0.0})
        entry["tokens"] += tokens
        entry["cost_usd"] += cost_usd
