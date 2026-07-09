import os
import json
import hashlib
import re
from typing import Any, Optional

import redis


def get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "127.0.0.1"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True,
    )

# normalize question by stripping, lowercasing, and collapsing whitespace
def normalize_question(question: str) -> str:
    normalize = question.strip().lower()
    normalized = re.sub(r"\s+", " ", normalize)# 将多个空格替换为一个
    return normalized
# hash the question for exact cache key
def hash_question(question: str) -> str:
    normalized = normalize_question(question)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def exact_cache(question: str) -> str:
    return f"rag:exact:{hash_question(question)}"

def get_exact_cache(question: str) -> Optional[dict[str, Any]]:
    client = get_redis_client()
    cache_key = exact_cache(question)
    cached_data = client.get(cache_key)
    if not cached_data:
        return None
    return json.loads(cached_data)
# set the exact cache with a TTL (time to live) in seconds
def set_exact_cache(question: str, payload: dict[str, Any], ttl: int) -> None:
    client = get_redis_client()
    cache_key = exact_cache(question)
    client.setex(cache_key, ttl, json.dumps(payload, ensure_ascii=False))