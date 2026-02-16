from plmol.cache import LRUCache


def test_lru_cache_eviction():
    cache = LRUCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)
    # "b" should be evicted because "a" was recently used.
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
