"""跨请求 prefix cache：按 token 前缀复用已算好的 K/V（对应 serve_openai_api --enable_kv）。

原理：缓存「已进入模型的完整 token 流 + 对应 past_key_values」。
新请求 tokenize 出完整 prompt 后，找最长匹配前缀，只把前缀之后的增量喂给模型，
历史 K/V 直接复用；前缀不匹配时回退全量重算（结果永远正确）。
"""
from threading import Lock

MAX_PREFIX_CACHE = 8       # 缓存条目上限（大致对应并发会话数）
_lock = Lock()
_cache = []                # [(token_ids: list, past_key_values)]，最近使用在前


def find_prefix(full_ids):
    """在缓存中找最长匹配前缀，返回 (前缀token数, past_key_values)；无命中 → (0, None)。"""
    with _lock:
        best_len, best_kv, best_entry = 0, None, None
        for entry in _cache:
            cached_ids, cached_kv = entry
            n = len(cached_ids)
            if n > best_len and full_ids[:n] == cached_ids:
                best_len, best_kv, best_entry = n, cached_kv, entry
        if best_entry is not None:
            _cache.remove(best_entry)
            _cache.insert(0, best_entry)   # LRU：命中后提到最前
        return best_len, best_kv


def store_prefix(full_ids, past_key_values):
    """保存 (完整token流, K/V)；超过上限淘汰最久未用的。"""
    with _lock:
        _cache.insert(0, (full_ids, past_key_values))
        if len(_cache) > MAX_PREFIX_CACHE:
            _cache.pop()
