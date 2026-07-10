import redis

from cache import exact_cache_key, get_exact_cache, normalize_question, set_exact_cache


class BrokenRedis:
    def get(self, key):
        raise redis.ConnectionError("unavailable")

    def setex(self, *args):
        raise redis.ConnectionError("unavailable")


class FakeRedis:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, ttl, value):
        self.values[key] = value


def test_normalize_and_key_are_stable():
    assert normalize_question("  线性  回归 ") == "线性 回归"
    assert exact_cache_key("线性回归") == exact_cache_key("  线性回归  ")


def test_cache_round_trip_and_failure_degrades():
    client = FakeRedis()
    assert set_exact_cache("问题", {"answer": "回答"}, 60, client)
    assert get_exact_cache("问题", client) == {"answer": "回答"}
    assert get_exact_cache("问题", BrokenRedis()) is None
    assert not set_exact_cache("问题", {"answer": "回答"}, 60, BrokenRedis())
