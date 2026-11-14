"""Tests: prefix and semantic caches with hit accounting."""
from tokenopt.cache import PrefixCache, SemanticCache


def test_prefix_cache_miss_then_hit():
    cache = PrefixCache()
    hit, saved = cache.lookup("shared prefix prompt", 100)
    assert hit is False and saved == 0
    cache.store("shared prefix prompt", "shared prefix")
    hit, saved = cache.lookup("shared prefix prompt", 100)
    assert hit is True
    assert saved == 90  # 100 tokens * (1 - 0.1 cached cost ratio)
    assert cache.stats.hit_rate == 0.5


def test_prefix_cache_normalizes_whitespace_and_case():
    cache = PrefixCache()
    cache.store("Shared   Prefix Prompt", "shared prefix")
    hit, _ = cache.lookup("shared prefix prompt", 10)
    assert hit is True


def test_prefix_cache_evicts_oldest():
    cache = PrefixCache(capacity=2)
    cache.store("one", "1")
    cache.store("two", "2")
    cache.store("three", "3")
    hit, _ = cache.lookup("one", 10)
    assert hit is False  # evicted


def test_semantic_cache_hit_on_near_duplicate():
    cache = SemanticCache()
    cache.store("What is the refund policy for headsets?", "30-day returns.")
    answer = cache.lookup("What is the refund policy for wireless headsets?")
    assert answer == "30-day returns."
    assert cache.stats.hits == 1


def test_semantic_cache_miss_on_unrelated_prompt():
    cache = SemanticCache()
    cache.store("What is the refund policy for headsets?", "30-day returns.")
    assert cache.lookup("How do I reset my router password?") is None
    assert cache.stats.misses == 1


def test_semantic_cache_tokens_saved_accounting():
    cache = SemanticCache()
    prompt = "What is the refund policy for headsets?"
    cache.store(prompt, "30-day returns.")
    cache.lookup(prompt)
    from tokenopt.tokenizer import estimate_tokens
    assert cache.stats.tokens_saved == estimate_tokens(prompt)


def test_semantic_cache_threshold_is_enforced():
    strict = SemanticCache(similarity_threshold=0.999)
    strict.store("What is the refund policy for headsets?", "30-day returns.")
    assert strict.lookup("What is the refund policy for wireless headsets?") is None
