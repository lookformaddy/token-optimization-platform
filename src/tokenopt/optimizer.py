"""Request orchestrator: cache -> dedup -> pack -> compress -> meter -> audit.

Pipeline order is deliberate:
1. Semantic cache first — a near-duplicate prompt skips the model entirely.
2. Dedup retrieved context, then pack it under the per-request budget.
3. Compress the assembled prompt toward the policy's target ratio.
4. Enforce the tenant budget (fail-closed) before returning anything.
5. Meter the result and append a hash-chained audit entry.

Every step is optional except budget enforcement: a request that cannot be
packed or compressed within budget is rejected, never silently degraded.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from tokenopt.audit import AuditLog
from tokenopt.budgets import BudgetEnforcer, BudgetExceeded, TenantBudget, TokenBudget
from tokenopt.packing import pack_context
from tokenopt.cache import PrefixCache, SemanticCache
from tokenopt.compression import compress_prompt
from tokenopt.config import Settings
from tokenopt.dedup import ContextBlock, deduplicate_blocks
from tokenopt.metering import MeteringLedger, UsageRecord
from tokenopt.policies import OptimizationPolicy, PolicyRegistry
from tokenopt.tokenizer import cost_usd, estimate_tokens


@dataclass
class OptimizeRequest:
    tenant: str
    prompt: str
    context: list[ContextBlock] = field(default_factory=list)
    query: str = ""
    policy_key: str = ""  # "" = use the registry's active policy
    max_tokens: int = 0  # 0 = policy/default budget


@dataclass
class OptimizationResult:
    request_id: str
    tenant: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    savings_ratio: float
    cost_before_usd: float
    cost_after_usd: float
    cache_hit: bool
    cached_answer: str | None = None
    policy_key: str = ""
    dropped_block_ids: list[str] = field(default_factory=list)
    skipped_block_ids: list[str] = field(default_factory=list)
    compressed: bool = False

    @property
    def cost_saved_usd(self) -> float:
        return self.cost_before_usd - self.cost_after_usd


class TokenOptimizer:
    """Stateless-per-request optimizer wired to shared caches/ledger/audit."""

    def __init__(
        self,
        settings: Settings | None = None,
        policies: PolicyRegistry | None = None,
        enforcer: BudgetEnforcer | None = None,
        ledger: MeteringLedger | None = None,
        audit: AuditLog | None = None,
        prefix_cache: PrefixCache | None = None,
        semantic_cache: SemanticCache | None = None,
    ):
        self.settings = settings or Settings.from_env()
        self.policies = policies or PolicyRegistry()
        self.enforcer = enforcer or BudgetEnforcer()
        self.ledger = ledger or MeteringLedger()
        self.audit = audit or AuditLog()
        self.prefix_cache = prefix_cache or PrefixCache(
            cached_cost_ratio=self.settings.cached_cost_ratio)
        self.semantic_cache = semantic_cache or SemanticCache(
            similarity_threshold=self.settings.semantic_cache_threshold)

    def _policy_for(self, key: str) -> OptimizationPolicy | None:
        if key:
            name, _, version = key.rpartition("@v")
            try:
                return self.policies.get(name, int(version))
            except (KeyError, ValueError):
                return None
        return self.policies.active()

    def optimize(self, request: OptimizeRequest) -> OptimizationResult:
        request_id = uuid.uuid4().hex[:12]
        policy = self._policy_for(request.policy_key)
        params = dict(policy.params) if policy else {}
        target_ratio = float(params.get("target_ratio", self.settings.default_target_ratio))
        dedup_threshold = float(params.get("dedup_threshold", self.settings.dedup_threshold))
        max_tokens = request.max_tokens or int(
            params.get("max_tokens", self.settings.default_max_tokens))
        pricing = self.settings.pricing

        # 1. Semantic cache: a near-duplicate prompt reuses a stored answer.
        cached = self.semantic_cache.lookup(request.prompt)
        if cached is not None:
            original_tokens = estimate_tokens(request.prompt)
            self.audit.append("cache_hit", request.tenant, {
                "request_id": request_id, "kind": "semantic",
                "tokens_saved": original_tokens,
            })
            self.ledger.record(UsageRecord(
                tenant=request.tenant, request_id=request_id,
                tokens_in=original_tokens, tokens_saved=original_tokens,
                cost_before_usd=cost_usd(original_tokens, pricing=pricing),
                cost_after_usd=0.0, cache_hit=True,
                policy_key=policy.key() if policy else "",
            ))
            return OptimizationResult(
                request_id=request_id, tenant=request.tenant,
                optimized_prompt=request.prompt, original_tokens=original_tokens,
                optimized_tokens=0, tokens_saved=original_tokens,
                savings_ratio=1.0,
                cost_before_usd=cost_usd(original_tokens, pricing=pricing),
                cost_after_usd=0.0, cache_hit=True, cached_answer=cached,
                policy_key=policy.key() if policy else "",
            )

        # 2. Dedup context, then pack under the per-request budget.
        deduped = deduplicate_blocks(request.context, similarity_threshold=dedup_threshold)
        packed = pack_context(
            deduped.kept, TokenBudget(max_tokens=max_tokens))
        context_text = "\n".join(b.text for b in packed.packed)
        assembled = (
            f"{context_text}\n\n{request.prompt}" if context_text else request.prompt
        )
        original_tokens = estimate_tokens(assembled)

        # 3. Prefix cache accounting on the shared prompt head.
        _hit, prefix_saved = self.prefix_cache.lookup(assembled, original_tokens)
        self.prefix_cache.store(assembled, context_text)

        # 4. Compress toward the policy target.
        compressed = compress_prompt(
            assembled, target_ratio,
            query=request.query or request.prompt,
            similarity_threshold=dedup_threshold,
        )
        optimized_text = compressed.compressed_text
        optimized_tokens = compressed.compressed_tokens
        cost_before = cost_usd(original_tokens, pricing=pricing)
        cost_after = cost_usd(optimized_tokens, pricing=pricing)

        # 5. Fail-closed budget enforcement (per-request + per-tenant).
        budget = self.enforcer.budget_for(request.tenant)
        if isinstance(budget, TenantBudget) and budget.max_tokens_per_request < max_tokens:
            # Tenant ceiling is stricter than the request budget: re-check.
            if optimized_tokens > budget.max_tokens_per_request:
                raise BudgetExceeded(
                    f"tenant {request.tenant}: optimized prompt of "
                    f"{optimized_tokens} tokens exceeds per-request limit "
                    f"{budget.max_tokens_per_request}",
                    budget="per-request",
                    used=optimized_tokens,
                    limit=budget.max_tokens_per_request,
                )
        self.enforcer.check_request(request.tenant, optimized_tokens, cost_after)
        self.enforcer.record(request.tenant, optimized_tokens, cost_after)

        tokens_saved = original_tokens - optimized_tokens + prefix_saved
        result = OptimizationResult(
            request_id=request_id, tenant=request.tenant,
            optimized_prompt=optimized_text,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_saved=max(0, tokens_saved),
            savings_ratio=(max(0, tokens_saved) / original_tokens) if original_tokens else 0.0,
            cost_before_usd=cost_before,
            cost_after_usd=cost_after,
            cache_hit=False,
            policy_key=policy.key() if policy else "",
            dropped_block_ids=[b.id for b in deduped.dropped],
            skipped_block_ids=[b.id for b in packed.skipped],
            compressed=not compressed.rejected,
        )
        self.ledger.record(UsageRecord(
            tenant=request.tenant, request_id=request_id,
            tokens_in=original_tokens, tokens_out=0,
            tokens_saved=result.tokens_saved,
            cost_before_usd=cost_before, cost_after_usd=cost_after,
            cache_hit=False, policy_key=result.policy_key,
        ))
        self.audit.append("optimize", request.tenant, {
            "request_id": request_id,
            "policy": result.policy_key,
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": result.tokens_saved,
            "dropped_blocks": result.dropped_block_ids,
            "skipped_blocks": result.skipped_block_ids,
        })
        return result
