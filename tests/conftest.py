import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from tokenopt.config import Settings
from tokenopt.demo_data import default_policies, golden_set
from tokenopt.dedup import ContextBlock
from tokenopt.optimizer import TokenOptimizer
from tokenopt.policies import OptimizationPolicy, PolicyRegistry


@pytest.fixture()
def settings():
    return Settings(
        default_max_tokens=8_000,
        max_tokens_per_request=8_000,
        max_tokens_per_day=1_000_000,
        max_cost_usd_per_day=50.0,
    )


@pytest.fixture()
def registry():
    reg = PolicyRegistry()
    for spec in default_policies():
        reg.register(OptimizationPolicy(
            name=spec["name"], version=spec["version"],
            params=spec["params"], description=spec["description"]))
    return reg


@pytest.fixture()
def optimizer(settings, registry):
    return TokenOptimizer(settings=settings, policies=registry)


@pytest.fixture()
def golden():
    return golden_set()


def block(block_id, text, score=0.9, required=False, source="test"):
    return ContextBlock(id=block_id, text=text, source=source,
                        score=score, required=required)
