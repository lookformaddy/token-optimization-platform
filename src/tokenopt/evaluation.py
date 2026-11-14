"""Deterministic evaluation: cost / latency / quality trade-offs.

Quality is measured without a model, using two deterministic scorers:
* keyword coverage — fraction of the golden answer's content words that
  survive in the optimized prompt (proxy for "the model still has the facts
  it needs");
* ROUGE-L-style F1 over word sequences (longest common subsequence), which
  penalizes dropped ordering and content.

Both are reproducible, so policy A/B comparisons are apples-to-apples.
Latency is modelled, not measured: compression and packing add a small,
deterministic per-token overhead, while cache hits and smaller prompts reduce
model time. The model is documented in docs/architecture.md and exists to
make the trade-off visible, not to claim measured latency.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tokenopt.compression import STOPWORDS, compress_prompt
from tokenopt.dedup import ContextBlock, deduplicate_blocks
from tokenopt.packing import pack_context
from tokenopt.budgets import TokenBudget
from tokenopt.tokenizer import cost_usd, estimate_tokens

_WORD_RE = re.compile(r"[a-z0-9]+")

# Modelled latency: 0.4 ms per input token, plus fixed optimizer overhead.
MS_PER_INPUT_TOKEN = 0.4
OPTIMIZER_OVERHEAD_MS = 8.0


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in STOPWORDS}


def keyword_coverage(prompt: str, golden_keywords: set[str]) -> float:
    """Fraction of golden keywords still present in ``prompt``.

    Keywords are tokenized the same way as the prompt, so hyphenated or
    dotted facts (``30-day``, ``4.99``) match when all of their word tokens
    survive.
    """
    if not golden_keywords:
        return 1.0
    words = _content_words(prompt)
    hits = 0
    for keyword in golden_keywords:
        parts = [p for p in _WORD_RE.findall(keyword.lower()) if p not in STOPWORDS]
        if parts and all(p in words for p in parts):
            hits += 1
    return hits / len(golden_keywords)


def _lcs_length(a: list[str], b: list[str]) -> int:
    # Classic DP over word sequences; golden prompts are short, so O(n*m) is fine.
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """ROUGE-L F1 over word sequences (deterministic, model-free)."""
    cand = _WORD_RE.findall((candidate or "").lower())
    ref = _WORD_RE.findall((reference or "").lower())
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class GoldenCase:
    """One synthetic golden-set case: prompt, context, and expected facts."""

    id: str
    prompt: str
    context: list[ContextBlock] = field(default_factory=list)
    expected_keywords: set[str] = field(default_factory=set)


@dataclass
class EvaluationResult:
    policy_name: str
    cases: int = 0
    mean_quality_retention: float = 0.0  # mean of keyword coverage
    mean_rouge_l: float = 0.0
    mean_tokens_saved_ratio: float = 0.0
    mean_cost_saved_usd: float = 0.0
    mean_latency_ms: float = 0.0
    worst_quality_retention: float = 1.0

    def passes(self, *, min_quality: float, min_savings: float) -> bool:
        return (
            self.mean_quality_retention >= min_quality
            and self.mean_tokens_saved_ratio >= min_savings
            and self.worst_quality_retention >= min_quality - 0.10
        )


def _optimize_case(case: GoldenCase, params: dict) -> tuple[str, int, int]:
    """Apply a policy's parameters to one case; return (prompt, before, after)."""
    text = case.prompt
    if case.context:
        deduped = deduplicate_blocks(
            case.context,
            similarity_threshold=params.get("dedup_threshold", 0.85),
        )
        packed = pack_context(
            deduped.kept,
            TokenBudget(max_tokens=params.get("max_tokens", 8_000)),
        )
        context_text = "\n".join(b.text for b in packed.packed)
        text = f"{context_text}\n\n{case.prompt}" if context_text else case.prompt
    before = estimate_tokens(text)
    result = compress_prompt(
        text,
        target_ratio=params.get("target_ratio", 0.6),
        query=case.prompt,
        similarity_threshold=params.get("dedup_threshold", 0.85),
    )
    return result.compressed_text, before, result.compressed_tokens


def evaluate_policy(
    policy_name: str,
    params: dict,
    golden_set: list[GoldenCase],
) -> EvaluationResult:
    """Run a policy over the golden set and score cost/latency/quality."""
    if not golden_set:
        raise ValueError("golden_set must not be empty")
    qualities, rouges, savings, costs, latencies = [], [], [], [], []
    for case in golden_set:
        optimized, before, after = _optimize_case(case, params)
        qualities.append(keyword_coverage(optimized, case.expected_keywords))
        rouges.append(rouge_l_f1(optimized, case.prompt))
        saved = max(0, before - after)
        savings.append(saved / before if before else 0.0)
        costs.append(cost_usd(saved))
        latencies.append(after * MS_PER_INPUT_TOKEN + OPTIMIZER_OVERHEAD_MS)
    n = len(golden_set)
    return EvaluationResult(
        policy_name=policy_name,
        cases=n,
        mean_quality_retention=sum(qualities) / n,
        mean_rouge_l=sum(rouges) / n,
        mean_tokens_saved_ratio=sum(savings) / n,
        mean_cost_saved_usd=sum(costs) / n,
        mean_latency_ms=sum(latencies) / n,
        worst_quality_retention=min(qualities),
    )
