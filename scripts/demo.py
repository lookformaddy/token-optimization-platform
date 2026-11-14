"""End-to-end demo: optimize prompts, compare policies, show the savings.

Runs entirely offline on synthetic data. Prints measurable results:
tokens saved %, cost saved, quality retention on the synthetic golden set,
cache behaviour, per-tenant metering and audit-chain verification.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tokenopt.config import Settings
from tokenopt.demo_data import default_policies, golden_set, verbose_support_prompt
from tokenopt.dedup import ContextBlock
from tokenopt.optimizer import OptimizeRequest, TokenOptimizer
from tokenopt.policies import OptimizationPolicy, PolicyRegistry


def main() -> int:
    settings = Settings.from_env()
    registry = PolicyRegistry()
    for spec in default_policies():
        registry.register(OptimizationPolicy(
            name=spec["name"], version=spec["version"],
            params=spec["params"], description=spec["description"]))
    optimizer = TokenOptimizer(settings=settings, policies=registry)
    golden = golden_set()

    print("=== Token Optimization Platform — demo ===\n")

    # 1. Single prompt optimization with duplicate context.
    context = [
        ContextBlock(id="c1", text=(
            "Returns are accepted within 30 days of delivery for most items."),
            source="policy-doc", score=0.95),
        ContextBlock(id="c2", text=(
            "Returns are accepted within 30 days of delivery for most items."),
            source="policy-doc-mirror", score=0.90),
        ContextBlock(id="c3", text=(
            "Refunds are issued to the original payment method within 5 "
            "business days. No restocking fee applies to unopened items."),
            source="policy-doc", score=0.92),
    ]
    result = optimizer.optimize(OptimizeRequest(
        tenant="acme-support", prompt=verbose_support_prompt(),
        context=context, query="refund policy return window"))
    print("1) Optimize a verbose support prompt (duplicate context included)")
    print(f"   original tokens : {result.original_tokens}")
    print(f"   optimized tokens: {result.optimized_tokens}")
    print(f"   tokens saved    : {result.tokens_saved} ({result.savings_ratio:.1%})")
    print(f"   cost saved      : ${result.cost_saved_usd:.6f} per request")
    print(f"   dropped blocks  : {result.dropped_block_ids}")
    print(f"   compressed      : {result.compressed}")
    print(f"   optimized prompt: {result.optimized_prompt[:160]}...\n")

    # 2. Semantic cache: same question twice, second call is free.
    question = "What is the refund policy for headsets?"
    optimizer.optimize(OptimizeRequest(tenant="acme-support", prompt=question))
    optimizer.semantic_cache.store(question, "30-day returns, no restocking fee.")
    cached = optimizer.optimize(OptimizeRequest(tenant="acme-support", prompt=question))
    print("2) Semantic cache")
    print(f"   second identical question cache hit: {cached.cache_hit}")
    print(f"   tokens saved by cache hit         : {cached.tokens_saved}")
    print(f"   semantic cache hit rate           : "
          f"{optimizer.semantic_cache.stats.hit_rate:.0%}\n")

    # 3. Policy A/B on the golden set + gated promotion.
    print("3) Policy A/B on the synthetic golden set "
          f"({len(golden)} cases)")
    comparison = registry.compare("balanced", 1, "aggressive", 1, golden)
    for label, res in (("A balanced ", comparison["a"]),
                       ("B aggressive", comparison["b"])):
        print(f"   {label}: quality {res.mean_quality_retention:.2f} | "
              f"tokens saved {res.mean_tokens_saved_ratio:.1%} | "
              f"cost saved ${res.mean_cost_saved_usd:.6f}/req | "
              f"modelled latency {res.mean_latency_ms:.1f} ms | "
              f"worst-case quality {res.worst_quality_retention:.2f}")
    print(f"   winner: policy {comparison['winner']} "
          f"(Δsavings {comparison['delta_savings']:+.1%}, "
          f"Δquality {comparison['delta_quality']:+.2f})")
    registry.submit_candidate("balanced", 1)
    promoted = registry.promote(
        "balanced", 1, golden,
        min_quality=settings.min_quality_for_promotion,
        min_savings=settings.min_savings_for_promotion)
    print(f"   promoted {promoted.key()} to active "
          f"(gate: quality ≥ {settings.min_quality_for_promotion}, "
          f"savings ≥ {settings.min_savings_for_promotion:.0%})\n")

    # 4. Per-tenant metering and audit chain.
    agg = optimizer.ledger.aggregate("acme-support")
    ok, _ = optimizer.audit.verify()
    print("4) Metering & audit")
    print(f"   tenant acme-support: {agg.requests} requests, "
          f"{agg.tokens_saved} tokens saved ({agg.savings_ratio:.1%}), "
          f"${agg.cost_saved_usd:.6f} saved, {agg.cache_hits} cache hits")
    print(f"   audit chain valid: {ok} ({len(optimizer.audit.entries)} entries)")
    print("\nDemo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
