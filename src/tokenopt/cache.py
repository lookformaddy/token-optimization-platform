"""Caching-aware reuse: prefix cache and semantic cache with hit accounting.

Two layers:
* ``PrefixCache`` — exact match on the normalized prompt prefix (the part of
  the prompt that is identical across requests, e.g. a system prompt plus
  shared context). A hit means those tokens are served from cache and cost
  a fraction of a fresh call (configurable ``cached_cost_ratio``).
* ``SemanticCache`` — a near-duplicate prompt (hashed-embedding similarity
  above threshold) reuses a stored answer, skipping the model call entirely.

Both caches are in-memory with a fixed capacity and LRU eviction; both
record hits, misses and tokens saved so savings are measurable, not assumed.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

from tokenopt.embeddings import embed, cosine
from tokenopt.tokenizer import estimate_tokens


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class PrefixCache:
    """Exact-prefix cache keyed by the normalized shared prompt prefix."""

    def __init__(self, capacity: int = 1_000, cached_cost_ratio: float = 0.1):
        self.capacity = capacity
        self.cached_cost_ratio = cached_cost_ratio  # cached tokens cost 10%
        self._entries: OrderedDict[str, str] = OrderedDict()
        self.stats = CacheStats()

    def lookup(self, prompt: str, prefix_tokens: int) -> tuple[bool, int]:
        """Return (hit, tokens_saved) for ``prompt``'s shared prefix."""
        key = _normalize(prompt)
        if key in self._entries:
            self.stats.hits += 1
            saved = int(prefix_tokens * (1 - self.cached_cost_ratio))
            self.stats.tokens_saved += saved
            self._entries.move_to_end(key)
            return True, saved
        self.stats.misses += 1
        return False, 0

    def store(self, prompt: str, prefix: str) -> None:
        key = _normalize(prompt)
        self._entries[key] = prefix
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)


@dataclass
class _SemanticEntry:
    prompt: str
    answer: str
    vector: list[float]


class SemanticCache:
    """Near-duplicate prompt cache: reuse a stored answer when similar enough."""

    def __init__(self, capacity: int = 1_000, similarity_threshold: float = 0.92):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self._entries: OrderedDict[str, _SemanticEntry] = OrderedDict()
        self.stats = CacheStats()

    def lookup(self, prompt: str) -> str | None:
        """Return the cached answer for a near-duplicate prompt, else None."""
        vec = embed(prompt)
        best_key: str | None = None
        best_sim = 0.0
        for key, entry in self._entries.items():
            sim = cosine(vec, entry.vector)
            if sim > best_sim:
                best_sim, best_key = sim, key
        if best_key is not None and best_sim >= self.similarity_threshold:
            entry = self._entries[best_key]
            self.stats.hits += 1
            self.stats.tokens_saved += estimate_tokens(prompt)
            self._entries.move_to_end(best_key)
            return entry.answer
        self.stats.misses += 1
        return None

    def store(self, prompt: str, answer: str) -> None:
        key = _normalize(prompt)
        self._entries[key] = _SemanticEntry(prompt, answer, embed(prompt))
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
