"""FastAPI service layer (optional — requires fastapi + uvicorn).

Importing this module without FastAPI installed raises ImportError; the
core optimizer never depends on it. Tests skip the API suite when FastAPI
is absent.
"""
from __future__ import annotations

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only with fastapi
    raise ImportError(
        "FastAPI is required for the service layer: pip install -r requirements.txt"
    ) from exc

from tokenopt.audit import AuditLog
from tokenopt.budgets import BudgetEnforcer, BudgetExceeded, TenantBudget
from tokenopt.config import Settings
from tokenopt.demo_data import default_policies, golden_set
from tokenopt.dedup import ContextBlock
from tokenopt.metering import MeteringLedger
from tokenopt.optimizer import OptimizeRequest, TokenOptimizer
from tokenopt.policies import OptimizationPolicy, PolicyRegistry, PromotionBlocked


class ContextBlockIn(BaseModel):
    id: str
    text: str
    source: str = ""
    score: float = 0.0
    required: bool = False


class OptimizeIn(BaseModel):
    tenant: str = "default"
    prompt: str
    context: list[ContextBlockIn] = Field(default_factory=list)
    query: str = ""
    policy_key: str = ""
    max_tokens: int = 0


class PolicyIn(BaseModel):
    name: str
    version: int = 1
    params: dict = Field(default_factory=dict)
    description: str = ""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    registry = PolicyRegistry()
    for spec in default_policies():
        registry.register(OptimizationPolicy(
            name=spec["name"], version=spec["version"],
            params=spec["params"], description=spec["description"]))
    optimizer = TokenOptimizer(
        settings=settings, policies=registry,
        enforcer=BudgetEnforcer(), ledger=MeteringLedger(), audit=AuditLog(),
    )
    app = FastAPI(title="Token Optimization Platform", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "active_policy": registry.active_key or None}

    @app.post("/optimize")
    def optimize(payload: OptimizeIn) -> dict:
        try:
            result = optimizer.optimize(OptimizeRequest(
                tenant=payload.tenant, prompt=payload.prompt,
                context=[ContextBlock(**b.model_dump()) for b in payload.context],
                query=payload.query, policy_key=payload.policy_key,
                max_tokens=payload.max_tokens,
            ))
        except BudgetExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return {
            "request_id": result.request_id,
            "optimized_prompt": result.optimized_prompt,
            "original_tokens": result.original_tokens,
            "optimized_tokens": result.optimized_tokens,
            "tokens_saved": result.tokens_saved,
            "savings_ratio": result.savings_ratio,
            "cost_saved_usd": result.cost_saved_usd,
            "cache_hit": result.cache_hit,
            "cached_answer": result.cached_answer,
            "policy": result.policy_key,
            "dropped_blocks": result.dropped_block_ids,
            "skipped_blocks": result.skipped_block_ids,
        }

    @app.post("/policies", status_code=201)
    def register_policy(payload: PolicyIn) -> dict:
        policy = OptimizationPolicy(
            name=payload.name, version=payload.version,
            params=payload.params, description=payload.description)
        registry.register(policy)
        return {"key": policy.key(), "status": policy.status.value}

    @app.post("/policies/{name}/{version}/candidate")
    def submit_candidate(name: str, version: int) -> dict:
        policy = registry.submit_candidate(name, version)
        return {"key": policy.key(), "status": policy.status.value}

    @app.post("/policies/{name}/{version}/promote")
    def promote(name: str, version: int) -> dict:
        try:
            policy = registry.promote(
                name, version, golden_set(),
                min_quality=settings.min_quality_for_promotion,
                min_savings=settings.min_savings_for_promotion,
            )
        except PromotionBlocked as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"key": policy.key(), "status": policy.status.value,
                "active": registry.active_key}

    @app.get("/policies/{name}/{version}/evaluate")
    def evaluate(name: str, version: int) -> dict:
        result = registry.evaluate(name, version, golden_set())
        return {
            "policy": result.policy_name, "cases": result.cases,
            "mean_quality_retention": result.mean_quality_retention,
            "mean_rouge_l": result.mean_rouge_l,
            "mean_tokens_saved_ratio": result.mean_tokens_saved_ratio,
            "mean_cost_saved_usd": result.mean_cost_saved_usd,
            "mean_latency_ms": result.mean_latency_ms,
            "worst_quality_retention": result.worst_quality_retention,
        }

    @app.get("/meter/{tenant}")
    def meter(tenant: str) -> dict:
        agg = optimizer.ledger.aggregate(tenant)
        return {
            "tenant": tenant, "requests": agg.requests,
            "tokens_in": agg.tokens_in, "tokens_saved": agg.tokens_saved,
            "savings_ratio": agg.savings_ratio,
            "cost_saved_usd": agg.cost_saved_usd,
            "cache_hits": agg.cache_hits,
        }

    @app.get("/audit/verify")
    def audit_verify() -> dict:
        ok, bad_seq = optimizer.audit.verify()
        return {"chain_valid": ok, "first_bad_seq": bad_seq,
                "entries": len(optimizer.audit.entries)}

    return app


app = create_app()
