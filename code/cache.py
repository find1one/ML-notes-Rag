"""Best-effort Redis exact cache utilities."""

import hashlib
import json
import logging
import re
from typing import Any, Optional

import redis

from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def get_redis_client(redis_url: Optional[str] = None) -> redis.Redis:
    return redis.Redis.from_url(redis_url or DEFAULT_CONFIG.redis_url, decode_responses=True)


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def hash_question(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def exact_cache_key(question: str) -> str:
    return "rag:exact:" + hash_question(question)


def get_exact_cache(question: str, client: Optional[redis.Redis] = None) -> Optional[dict]:
    try:
        cached = (client or get_redis_client()).get(exact_cache_key(question))
        return json.loads(cached) if cached else None
    except (redis.RedisError, ValueError, TypeError) as exc:
        logger.warning("Exact cache read failed; continuing without cache: %s", exc)
        return None


def set_exact_cache(
    question: str,
    payload: dict[str, Any],
    ttl_seconds: Optional[int] = None,
    client: Optional[redis.Redis] = None,
) -> bool:
    try:
        (client or get_redis_client()).setex(
            exact_cache_key(question),
            ttl_seconds or DEFAULT_CONFIG.cache_ttl_seconds,
            json.dumps(payload, ensure_ascii=False),
        )
        return True
    except (redis.RedisError, TypeError, ValueError) as exc:
        logger.warning("Exact cache write failed; continuing without cache: %s", exc)
        return False
