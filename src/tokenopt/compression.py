"""Extractive prompt compression with redundancy removal.

Strategy, in order:
1. Split the prompt into sentences (kept in original order).
2. Score each sentence: overlap with the query terms, keyword salience
   (rare, content-bearing words), and a mild early-position bias.
3. Greedily keep the highest-scoring sentences until the token target is
   met, then drop near-duplicates (hashed-embedding similarity above the
   threshold) among the kept set.
4. Reassemble in original order so the compressed prompt stays readable.

Compression never invents text — it only deletes sentences — so nothing
hallucinated can leak into the prompt. A compression that would keep fewer
than ``min_keep_sentences`` sentences is rejected and the original prompt is
returned with a flag, so quality cannot silently collapse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tokenopt.embeddings import embed, cosine
from tokenopt.tokenizer import estimate_tokens

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Low-information words: dropping them from scoring keeps the signal on
# content-bearing terms. They are never removed from kept sentences.
STOPWORDS = frozenset(
    """a an the and or but if then else when while of to in on at for with by
    from as is are was were be been being it its this that these those there
    here we you he she they them his her their our your my i me do does did
    have has had can could should would will just very really also more most
    other some any no not only""".split()
)


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, preserving order and punctuation."""
    parts = [s.strip() for s in _SENTENCE_RE.split((text or "").strip())]
    return [p for p in parts if p]


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS}


def _score_sentence(sentence: str, query_words: set[str], position: int, total: int) -> float:
    words = _content_words(sentence)
    if not words:
        return 0.0
    query_overlap = len(words & query_words) / max(1, len(query_words)) if query_words else 0.0
    # Longer content-bearing sentences carry more information, up to a cap.
    length_term = min(1.0, len(words) / 12.0)
    # Mild early-position bias: instructions and definitions usually lead.
    position_term = 1.0 - 0.5 * (position / max(1, total))
    return 0.55 * query_overlap + 0.30 * length_term + 0.15 * position_term


@dataclass
class CompressionResult:
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    dropped_sentences: list[str] = field(default_factory=list)
    rejected: bool = False  # True when compression was refused (quality guard)

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def savings_ratio(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return self.tokens_saved / self.original_tokens


def compress_prompt(
    text: str,
    target_ratio: float = 0.6,
    *,
    query: str = "",
    similarity_threshold: float = 0.85,
    min_keep_sentences: int = 1,
) -> CompressionResult:
    """Compress ``text`` to roughly ``target_ratio`` of its token count.

    ``target_ratio`` is the fraction of tokens to keep (0.6 = keep 60%).
    Sentences are removed, never rewritten. Returns the original text with
    ``rejected=True`` when the target would leave too little content.
    """
    original_tokens = estimate_tokens(text)
    sentences = split_sentences(text)
    if not sentences or original_tokens == 0:
        return CompressionResult(text, text, 0, 0)
    if not 0 < target_ratio < 1:
        raise ValueError("target_ratio must be in (0, 1)")

    target_tokens = max(1, int(original_tokens * target_ratio))
    query_words = _content_words(query)
    scored = [
        (_score_sentence(s, query_words, i, len(sentences)), i, s)
        for i, s in enumerate(sentences)
    ]
    # Greedy by score until the target is met, then restore original order.
    kept: list[tuple[int, str]] = []
    kept_tokens = 0
    for _score, idx, sentence in sorted(scored, reverse=True):
        if kept_tokens >= target_tokens and kept:
            break
        kept.append((idx, sentence))
        kept_tokens += estimate_tokens(sentence)
    kept.sort(key=lambda item: item[0])

    if len(kept) < min_keep_sentences:
        return CompressionResult(text, text, original_tokens, original_tokens,
                                 rejected=True)

    # Redundancy removal: drop a kept sentence if it is near-identical to an
    # earlier kept sentence (embedding similarity above the threshold).
    deduped: list[tuple[int, str]] = []
    dropped: list[str] = []
    for idx, sentence in kept:
        vec = embed(sentence)
        if any(cosine(vec, embed(other)) >= similarity_threshold for _, other in deduped):
            dropped.append(sentence)
            continue
        deduped.append((idx, sentence))

    if len(deduped) < min_keep_sentences:
        return CompressionResult(text, text, original_tokens, original_tokens,
                                 rejected=True)

    kept_idx = {idx for idx, _ in deduped}
    dropped.extend(s for i, s in enumerate(sentences) if i not in kept_idx and s not in dropped)
    compressed = " ".join(s for _, s in deduped)
    return CompressionResult(
        original_text=text,
        compressed_text=compressed,
        original_tokens=original_tokens,
        compressed_tokens=estimate_tokens(compressed),
        dropped_sentences=dropped,
    )
