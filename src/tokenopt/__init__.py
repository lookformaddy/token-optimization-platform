"""Token Optimization Platform — enterprise LLM token cost control.

Pure-stdlib core: prompt compression, semantic dedup, budget-aware packing,
caching-aware reuse, policy-gated optimization, per-tenant metering and a
hash-chained audit log. The FastAPI service layer is optional.
"""

from tokenopt.audit import AuditLog
from tokenopt.budgets import BudgetExceeded, TenantBudget, TokenBudget
from tokenopt.cache import PrefixCache, SemanticCache
from tokenopt.compression import CompressionResult, compress_prompt
from tokenopt.dedup import deduplicate_blocks
from tokenopt.evaluation import EvaluationResult, evaluate_policy
from tokenopt.metering import MeteringLedger
from tokenopt.optimizer import OptimizationResult, TokenOptimizer
from tokenopt.packing import pack_context
from tokenopt.policies import OptimizationPolicy, PolicyRegistry
from tokenopt.tokenizer import cost_usd, estimate_tokens

__version__ = "1.0.0"

__all__ = [
    "AuditLog",
    "BudgetExceeded",
    "CompressionResult",
    "EvaluationResult",
    "MeteringLedger",
    "OptimizationPolicy",
    "OptimizationResult",
    "PolicyRegistry",
    "PrefixCache",
    "SemanticCache",
    "TenantBudget",
    "TokenBudget",
    "TokenOptimizer",
    "compress_prompt",
    "cost_usd",
    "deduplicate_blocks",
    "estimate_tokens",
    "evaluate_policy",
    "pack_context",
]
