"""Tests: fail-closed budget enforcement."""
import pytest

from tokenopt.budgets import BudgetEnforcer, BudgetExceeded, TenantBudget


def test_request_under_all_limits_passes():
    enforcer = BudgetEnforcer(budgets={"acme": TenantBudget(tenant="acme")})
    enforcer.check_request("acme", 100, 0.01)  # no raise


def test_per_request_limit_enforced():
    enforcer = BudgetEnforcer(
        budgets={"acme": TenantBudget(tenant="acme", max_tokens_per_request=100)})
    with pytest.raises(BudgetExceeded) as exc:
        enforcer.check_request("acme", 101, 0.01)
    assert exc.value.budget == "per-request"


def test_daily_token_limit_enforced():
    enforcer = BudgetEnforcer(
        budgets={"acme": TenantBudget(tenant="acme", max_tokens_per_day=1000)})
    enforcer.record("acme", 900, 0.01)
    with pytest.raises(BudgetExceeded) as exc:
        enforcer.check_request("acme", 200, 0.01)
    assert exc.value.budget == "daily-tokens"


def test_daily_cost_limit_enforced():
    enforcer = BudgetEnforcer(
        budgets={"acme": TenantBudget(tenant="acme", max_cost_usd_per_day=1.0)})
    enforcer.record("acme", 100, 0.95)
    with pytest.raises(BudgetExceeded) as exc:
        enforcer.check_request("acme", 100, 0.10)
    assert exc.value.budget == "daily-cost"


def test_unknown_tenant_gets_defaults():
    enforcer = BudgetEnforcer()
    enforcer.check_request("brand-new", 100, 0.01)  # defaults are generous
    assert enforcer.budget_for("brand-new").tenant == "brand-new"


def test_record_accumulates_usage():
    enforcer = BudgetEnforcer()
    enforcer.record("acme", 100, 0.5)
    enforcer.record("acme", 50, 0.25)
    assert enforcer.usage["acme"]["tokens"] == 150
    assert enforcer.usage["acme"]["cost_usd"] == pytest.approx(0.75)
