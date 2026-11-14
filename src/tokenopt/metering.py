"""Per-tenant metering: usage ledger and aggregates.

Every optimized request books one ledger entry (tokens in/out, tokens saved,
cost before/after, cache hits). Aggregates power per-tenant billing, budget
enforcement and the savings dashboard. The ledger is append-only in memory;
production deployments persist it (see docs/architecture.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class UsageRecord:
    tenant: str
    request_id: str
    day: str = field(default_factory=lambda: date.today().isoformat())
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_saved: int = 0
    cost_before_usd: float = 0.0
    cost_after_usd: float = 0.0
    cache_hit: bool = False
    policy_key: str = ""


@dataclass
class TenantAggregate:
    tenant: str
    requests: int = 0
    tokens_in: int = 0
    tokens_saved: int = 0
    cost_before_usd: float = 0.0
    cost_after_usd: float = 0.0
    cache_hits: int = 0

    @property
    def savings_ratio(self) -> float:
        total = self.tokens_in + self.tokens_saved
        return self.tokens_saved / total if total else 0.0

    @property
    def cost_saved_usd(self) -> float:
        return self.cost_before_usd - self.cost_after_usd


@dataclass
class MeteringLedger:
    records: list[UsageRecord] = field(default_factory=list)

    def record(self, entry: UsageRecord) -> UsageRecord:
        self.records.append(entry)
        return entry

    def aggregate(self, tenant: str, day: str | None = None) -> TenantAggregate:
        agg = TenantAggregate(tenant=tenant)
        for r in self.records:
            if r.tenant != tenant:
                continue
            if day is not None and r.day != day:
                continue
            agg.requests += 1
            agg.tokens_in += r.tokens_in
            agg.tokens_saved += r.tokens_saved
            agg.cost_before_usd += r.cost_before_usd
            agg.cost_after_usd += r.cost_after_usd
            agg.cache_hits += 1 if r.cache_hit else 0
        return agg

    def tenants(self) -> list[str]:
        return sorted({r.tenant for r in self.records})
