"""Versioned optimization policies with A/B comparison and gated promotion.

A policy is a versioned bundle of optimizer parameters (compression ratio,
dedup threshold, token budget, cache settings). Policies move through a
lifecycle — draft -> candidate -> promoted — and promotion is gated: a
candidate may only be promoted when its golden-set evaluation clears both a
quality floor and a savings floor. Promotion is the only way a policy
becomes the active default, so a bad policy can never silently reach
production traffic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tokenopt.evaluation import EvaluationResult, GoldenCase, evaluate_policy


class PolicyStatus(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    RETIRED = "retired"


@dataclass
class OptimizationPolicy:
    name: str
    version: int = 1
    params: dict = field(default_factory=dict)
    status: PolicyStatus = PolicyStatus.DRAFT
    description: str = ""

    def key(self) -> str:
        return f"{self.name}@v{self.version}"


class PromotionBlocked(Exception):
    """Raised when a candidate fails the promotion gate."""


@dataclass
class PolicyRegistry:
    """In-memory registry of versioned policies."""

    policies: dict[str, OptimizationPolicy] = field(default_factory=dict)
    active_key: str = ""

    def register(self, policy: OptimizationPolicy) -> OptimizationPolicy:
        key = policy.key()
        if key in self.policies:
            raise ValueError(f"policy {key} already registered")
        self.policies[key] = policy
        return policy

    def get(self, name: str, version: int) -> OptimizationPolicy:
        key = f"{name}@v{version}"
        if key not in self.policies:
            raise KeyError(f"unknown policy {key}")
        return self.policies[key]

    def submit_candidate(self, name: str, version: int) -> OptimizationPolicy:
        policy = self.get(name, version)
        if policy.status != PolicyStatus.DRAFT:
            raise ValueError(f"policy {policy.key()} is {policy.status}, not draft")
        policy.status = PolicyStatus.CANDIDATE
        return policy

    def evaluate(
        self, name: str, version: int, golden_set: list[GoldenCase]
    ) -> EvaluationResult:
        policy = self.get(name, version)
        return evaluate_policy(policy.key(), policy.params, golden_set)

    def compare(
        self,
        name_a: str,
        version_a: int,
        name_b: str,
        version_b: int,
        golden_set: list[GoldenCase],
    ) -> dict:
        """A/B comparison of two policies over the same golden set."""
        result_a = self.evaluate(name_a, version_a, golden_set)
        result_b = self.evaluate(name_b, version_b, golden_set)
        delta_quality = result_b.mean_quality_retention - result_a.mean_quality_retention
        delta_savings = result_b.mean_tokens_saved_ratio - result_a.mean_tokens_saved_ratio
        # B wins only if it saves more without giving up quality beyond 2 points.
        winner = "B" if delta_savings > 0 and delta_quality >= -0.02 else "A"
        return {
            "a": result_a,
            "b": result_b,
            "winner": winner,
            "delta_quality": delta_quality,
            "delta_savings": delta_savings,
            "delta_cost_usd": result_b.mean_cost_saved_usd - result_a.mean_cost_saved_usd,
        }

    def promote(
        self,
        name: str,
        version: int,
        golden_set: list[GoldenCase],
        *,
        min_quality: float = 0.85,
        min_savings: float = 0.15,
    ) -> OptimizationPolicy:
        """Promote a candidate to active, gated on its evaluation.

        The previously active policy is retired. Promotion fails closed:
        a candidate that misses either gate stays a candidate.
        """
        policy = self.get(name, version)
        if policy.status != PolicyStatus.CANDIDATE:
            raise PromotionBlocked(
                f"policy {policy.key()} is {policy.status}; only candidates promote"
            )
        result = self.evaluate(name, version, golden_set)
        if not result.passes(min_quality=min_quality, min_savings=min_savings):
            raise PromotionBlocked(
                f"policy {policy.key()} failed promotion gate: "
                f"quality {result.mean_quality_retention:.2f} (min {min_quality}), "
                f"savings {result.mean_tokens_saved_ratio:.2%} (min {min_savings:.0%}), "
                f"worst-case quality {result.worst_quality_retention:.2f}"
            )
        if self.active_key and self.active_key in self.policies:
            self.policies[self.active_key].status = PolicyStatus.RETIRED
        policy.status = PolicyStatus.PROMOTED
        self.active_key = policy.key()
        return policy

    def active(self) -> OptimizationPolicy | None:
        return self.policies.get(self.active_key) if self.active_key else None
