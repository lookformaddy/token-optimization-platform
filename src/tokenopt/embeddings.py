"""Deterministic hashed embeddings for semantic decisions.

No model downloads: each word is hashed with SHA-256 into a fixed-size
vector (feature hashing). Similarity is cosine similarity on L2-normalized
vectors. Two texts sharing many words score high; unrelated texts score near
zero. Deterministic across runs and machines, which keeps dedup, semantic
caching and compression decisions reproducible and testable.
"""
from __future__ import annotations

import hashlib
import math
import re

DIM = 256
_WORD_RE = re.compile(r"[a-z0-9]+")


def _word_index(word: str) -> int:
    digest = hashlib.sha256(word.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % DIM


def embed(text: str, dim: int = DIM) -> list[float]:
    """Return the L2-normalized hashed embedding of ``text``."""
    vec = [0.0] * dim
    for word in _WORD_RE.findall((text or "").lower()):
        vec[_word_index(word) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [0, 1] for non-negative hashed embeddings."""
    if len(a) != len(b):
        raise ValueError("embedding dimensions differ")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def similarity(a: str, b: str) -> float:
    """Cosine similarity between the hashed embeddings of two texts."""
    return cosine(embed(a), embed(b))
