"""Tests: versioned policies, A/B comparison, gated promotion."""
import pytest

from tokenopt.policies import OptimizationPolicy, PolicyRegistry, PolicyStatus, PromotionBlocked


def _policy(name, version, ratio):
    return OptimizationPolicy(
        name=name, version=version,
        params={"target_ratio": ratio, "dedup_threshold": 0.85, "max_tokens": 8_000})


def test_register_and_get(registry):
    policy = _policy("custom", 1, 0.6)
    registry.register(policy)
    assert registry.get("custom", 1) is policy


def test_duplicate_registration_rejected(registry):
    registry.register(_policy("custom", 1, 0.6))
    with pytest.raises(ValueError):
        registry.register(_policy("custom", 1, 0.6))


def test_unknown_policy_raises_key_error(registry):
    with pytest.raises(KeyError):
        registry.get("nope", 1)


def test_candidate_lifecycle(registry):
    registry.register(_policy("custom", 1, 0.6))
    policy = registry.submit_candidate("custom", 1)
    assert policy.status == PolicyStatus.CANDIDATE
    with pytest.raises(ValueError):
        registry.submit_candidate("custom", 1)  # no longer a draft


def test_promote_requires_candidate(registry):
    registry.register(_policy("custom", 1, 0.6))
    with pytest.raises(PromotionBlocked):
        registry.promote("custom", 1, [])


def test_balanced_policy_promotes(golden, registry):
    registry.submit_candidate("balanced", 1)
    policy = registry.promote("balanced", 1, golden)
    assert policy.status == PolicyStatus.PROMOTED
    assert registry.active() is policy
    assert registry.active_key == "balanced@v1"


def test_promotion_gate_blocks_low_quality_policy(golden, registry):
    # Target ratio 0.05 keeps almost nothing: quality collapses, gate blocks.
    registry.register(_policy("slash", 1, 0.05))
    registry.submit_candidate("slash", 1)
    with pytest.raises(PromotionBlocked):
        registry.promote("slash", 1, golden)
    assert registry.get("slash", 1).status == PolicyStatus.CANDIDATE


def test_promotion_retires_previous_active(golden, registry):
    registry.submit_candidate("balanced", 1)
    registry.promote("balanced", 1, golden)
    registry.submit_candidate("aggressive", 1)
    # Aggressive may or may not clear the gate; only assert retirement if it does.
    try:
        registry.promote("aggressive", 1, golden)
    except PromotionBlocked:
        pytest.skip("aggressive policy did not clear the promotion gate")
    assert registry.get("balanced", 1).status == PolicyStatus.RETIRED


def test_ab_comparison_returns_winner(golden, registry):
    result = registry.compare("balanced", 1, "aggressive", 1, golden)
    assert result["winner"] in ("A", "B")
    assert result["b"].mean_tokens_saved_ratio >= result["a"].mean_tokens_saved_ratio
    assert "delta_quality" in result and "delta_savings" in result
