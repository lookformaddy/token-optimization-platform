"""Tests: per-tenant metering ledger."""
import pytest

from tokenopt.metering import MeteringLedger, UsageRecord


def _record(tenant, tokens_in=100, saved=40, before=0.0002, after=0.00012, hit=False):
    return UsageRecord(
        tenant=tenant, request_id="r1", tokens_in=tokens_in, tokens_saved=saved,
        cost_before_usd=before, cost_after_usd=after, cache_hit=hit)


def test_empty_ledger_aggregates_zero():
    ledger = MeteringLedger()
    agg = ledger.aggregate("acme")
    assert agg.requests == 0
    assert agg.tokens_saved == 0
    assert agg.savings_ratio == 0.0


def test_aggregate_sums_per_tenant():
    ledger = MeteringLedger()
    ledger.record(_record("acme"))
    ledger.record(_record("acme", tokens_in=200, saved=60))
    ledger.record(_record("other", tokens_in=50, saved=10))
    agg = ledger.aggregate("acme")
    assert agg.requests == 2
    assert agg.tokens_in == 300
    assert agg.tokens_saved == 100
    assert agg.cost_saved_usd == pytest.approx(0.00016)


def test_savings_ratio():
    ledger = MeteringLedger()
    ledger.record(_record("acme", tokens_in=100, saved=40))
    agg = ledger.aggregate("acme")
    # 40 saved of 140 original = ~0.286
    assert agg.savings_ratio == pytest.approx(40 / 140)


def test_cache_hits_counted():
    ledger = MeteringLedger()
    ledger.record(_record("acme", hit=True))
    ledger.record(_record("acme", hit=False))
    assert ledger.aggregate("acme").cache_hits == 1


def test_tenant_list():
    ledger = MeteringLedger()
    ledger.record(_record("beta"))
    ledger.record(_record("acme"))
    assert ledger.tenants() == ["acme", "beta"]


def test_day_filter():
    ledger = MeteringLedger()
    ledger.record(UsageRecord(tenant="acme", request_id="r1", day="2026-11-14",
                              tokens_in=100, tokens_saved=10))
    ledger.record(UsageRecord(tenant="acme", request_id="r2", day="2026-11-13",
                              tokens_in=100, tokens_saved=90))
    assert ledger.aggregate("acme", day="2026-11-14").tokens_saved == 10
    assert ledger.aggregate("acme").tokens_saved == 100
