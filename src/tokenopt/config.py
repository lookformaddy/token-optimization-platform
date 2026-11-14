"""Configuration from environment variables (no secrets needed).

All values have safe defaults; the optimizer runs entirely offline with
synthetic data, so no API keys exist in this service.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    default_max_tokens: int = 8_000
    default_target_ratio: float = 0.6
    dedup_threshold: float = 0.85
    semantic_cache_threshold: float = 0.92
    cached_cost_ratio: float = 0.1
    min_quality_for_promotion: float = 0.85
    min_savings_for_promotion: float = 0.15
    max_tokens_per_request: int = 8_000
    max_tokens_per_day: int = 1_000_000
    max_cost_usd_per_day: float = 50.0
    input_cost_per_1k: float = 0.002
    output_cost_per_1k: float = 0.006
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            default_max_tokens=_int("TOP_DEFAULT_MAX_TOKENS", 8_000),
            default_target_ratio=_float("TOP_DEFAULT_TARGET_RATIO", 0.6),
            dedup_threshold=_float("TOP_DEDUP_THRESHOLD", 0.85),
            semantic_cache_threshold=_float("TOP_SEMANTIC_CACHE_THRESHOLD", 0.92),
            cached_cost_ratio=_float("TOP_CACHED_COST_RATIO", 0.1),
            min_quality_for_promotion=_float("TOP_MIN_QUALITY_PROMOTION", 0.85),
            min_savings_for_promotion=_float("TOP_MIN_SAVINGS_PROMOTION", 0.15),
            max_tokens_per_request=_int("TOP_MAX_TOKENS_PER_REQUEST", 8_000),
            max_tokens_per_day=_int("TOP_MAX_TOKENS_PER_DAY", 1_000_000),
            max_cost_usd_per_day=_float("TOP_MAX_COST_USD_PER_DAY", 50.0),
            input_cost_per_1k=_float("TOP_INPUT_COST_PER_1K", 0.002),
            output_cost_per_1k=_float("TOP_OUTPUT_COST_PER_1K", 0.006),
            host=os.environ.get("TOP_HOST", "0.0.0.0"),
            port=_int("TOP_PORT", 8000),
        )

    @property
    def pricing(self) -> dict[str, float]:
        return {
            "input_per_1k": self.input_cost_per_1k,
            "output_per_1k": self.output_cost_per_1k,
        }
