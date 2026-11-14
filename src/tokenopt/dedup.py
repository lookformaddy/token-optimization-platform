"""Semantic dedup of context blocks.

Retrieved context often contains near-duplicate passages (same document
fetched through two connectors, overlapping chunks). Dedup keeps the first
occurrence of each semantic cluster — first wins, because retrieval order
already encodes relevance — and reports what was dropped and why, so the
decision is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tokenopt.embeddings import embed, cosine
from tokenopt.tokenizer import estimate_tokens


@dataclass
class ContextBlock:
    """One retrievable chunk of context."""

    id: str
    text: str
    source: str = ""
    score: float = 0.0  # retrieval relevance score; higher = more relevant
    required: bool = False  # required blocks are never deduped away

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class DedupResult:
    kept: list[ContextBlock] = field(default_factory=list)
    dropped: list[ContextBlock] = field(default_factory=list)
    dropped_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        return sum(b.tokens for b in self.dropped)


def deduplicate_blocks(
    blocks: list[ContextBlock],
    *,
    similarity_threshold: float = 0.85,
) -> DedupResult:
    """Drop near-duplicate blocks, keeping the first of each cluster.

    Blocks are processed in the order given (assumed relevance-ordered).
    A block is dropped when its hashed-embedding similarity to an already
    kept block meets or exceeds ``similarity_threshold``.
    """
    kept: list[ContextBlock] = []
    kept_vecs: list[list[float]] = []
    dropped: list[ContextBlock] = []
    reasons: dict[str, str] = {}
    for block in blocks:
        if block.required:
            kept.append(block)
            kept_vecs.append(embed(block.text))
            continue
        vec = embed(block.text)
        match = next(
            (k for k, kv in zip(kept, kept_vecs) if cosine(vec, kv) >= similarity_threshold),
            None,
        )
        if match is not None:
            dropped.append(block)
            reasons[block.id] = (
                f"near-duplicate of {match.id} "
                f"(similarity >= {similarity_threshold})"
            )
        else:
            kept.append(block)
            kept_vecs.append(vec)
    return DedupResult(kept=kept, dropped=dropped, dropped_reasons=reasons)
