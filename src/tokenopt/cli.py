"""Command-line interface: optimize a prompt or evaluate a policy."""
from __future__ import annotations

import argparse
import json
import sys

from tokenopt.config import Settings
from tokenopt.demo_data import default_policies, golden_set
from tokenopt.optimizer import OptimizeRequest, TokenOptimizer
from tokenopt.policies import OptimizationPolicy, PolicyRegistry


def build_optimizer() -> TokenOptimizer:
    registry = PolicyRegistry()
    for spec in default_policies():
        registry.register(OptimizationPolicy(
            name=spec["name"], version=spec["version"],
            params=spec["params"], description=spec["description"]))
    return TokenOptimizer(settings=Settings.from_env(), policies=registry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tokenopt", description="Token Optimization Platform")
    sub = parser.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize", help="Optimize a prompt from stdin or --prompt")
    opt.add_argument("--tenant", default="default")
    opt.add_argument("--prompt", default="")
    opt.add_argument("--target-ratio", type=float, default=0.6)

    ev = sub.add_parser("evaluate", help="Evaluate all registered policies on the golden set")
    ev.add_argument("--min-quality", type=float, default=0.85)
    ev.add_argument("--min-savings", type=float, default=0.15)

    args = parser.parse_args(argv)
    optimizer = build_optimizer()

    if args.command == "optimize":
        prompt = args.prompt or sys.stdin.read()
        if not prompt.strip():
            print("error: empty prompt", file=sys.stderr)
            return 2
        result = optimizer.optimize(OptimizeRequest(
            tenant=args.tenant, prompt=prompt,
            max_tokens=optimizer.settings.default_max_tokens))
        print(json.dumps({
            "original_tokens": result.original_tokens,
            "optimized_tokens": result.optimized_tokens,
            "tokens_saved": result.tokens_saved,
            "savings_ratio": round(result.savings_ratio, 4),
            "cost_saved_usd": round(result.cost_saved_usd, 6),
            "optimized_prompt": result.optimized_prompt,
        }, indent=2))
        return 0

    if args.command == "evaluate":
        rows = []
        for spec in default_policies():
            result = optimizer.policies.evaluate(spec["name"], spec["version"], golden_set())
            rows.append({
                "policy": result.policy_name,
                "quality": round(result.mean_quality_retention, 3),
                "savings": round(result.mean_tokens_saved_ratio, 3),
                "cost_saved_usd": round(result.mean_cost_saved_usd, 6),
                "passes": result.passes(min_quality=args.min_quality,
                                        min_savings=args.min_savings),
            })
        print(json.dumps(rows, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
