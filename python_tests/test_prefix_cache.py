"""prefix_cache 的 find_prefix / store_prefix 单元测试。"""
from scripts.Deploy import prefix_cache


def setup_function():
    prefix_cache.clear()


def test_longest_prefix_match():
    prefix_cache.store_prefix([1, 2, 3, 4], "kv1")
    prefix_cache.store_prefix([1, 2, 5], "kv2")
    length, kv = prefix_cache.find_prefix([1, 2, 3, 4, 6])
    assert (length, kv) == (4, "kv1")


def test_cache_miss_returns_zero():
    length, kv = prefix_cache.find_prefix([9, 9, 9])
    assert (length, kv) == (0, None)


def test_shared_prefix_uses_longest():
    prefix_cache.store_prefix([1, 2, 3], "short")
    prefix_cache.store_prefix([1, 2, 3, 4, 5], "long")
    length, kv = prefix_cache.find_prefix([1, 2, 3, 4, 5, 6])
    assert (length, kv) == (5, "long")


def test_lru_eviction():
    for i in range(prefix_cache.MAX_PREFIX_CACHE + 2):
        prefix_cache.store_prefix([i], f"kv{i}")
    length, kv = prefix_cache.find_prefix([0])   # 最旧的被淘汰
    assert (length, kv) == (0, None)
    length, kv = prefix_cache.find_prefix([prefix_cache.MAX_PREFIX_CACHE + 1])
    assert length == 1


def test_hit_moves_to_front():
    for i in range(prefix_cache.MAX_PREFIX_CACHE):
        prefix_cache.store_prefix([i], f"kv{i}")
    assert prefix_cache.find_prefix([0])[1] == "kv0"   # 命中最旧的 0 → 提到最前
    prefix_cache.store_prefix([100], "new")            # 再存 → 淘汰此时最旧的 1
    assert prefix_cache.find_prefix([1]) == (0, None)
    assert prefix_cache.find_prefix([0])[1] == "kv0"
