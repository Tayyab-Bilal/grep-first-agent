import asyncio
import random

import fakeredis

from grep_first_agent.backend import KINDS, FakeWorkspaceAPI
from grep_first_agent.cache import InMemoryBackend, ProjectionCache, RedisBackend


async def test_stampede_one_fetch_for_20_concurrent_calls():
    api = FakeWorkspaceAPI(delay=0.05)  # slow fetch: all 20 callers arrive while it is in flight
    cache = ProjectionCache(api)
    results = await asyncio.gather(*(cache.get("token-alice") for _ in range(20)))
    assert cache.fetches == 1
    assert sum(api.calls.values()) == len(KINDS) + 1  # one call per kind plus the hierarchy, not 20x that
    assert all(r == results[0] for r in results)


class Extremes(random.Random):
    def __init__(self, pick):
        super().__init__()
        self.pick = pick

    def uniform(self, a, b):
        return self.pick(a, b)


class SpyBackend(InMemoryBackend):
    def __init__(self):
        super().__init__()
        self.ttls = []

    async def set(self, key, value, ttl):
        self.ttls.append(ttl)
        await super().set(key, value, ttl)


async def test_ttl_jitter_within_bounds():
    ttls = []
    for pick in (min, max):
        spy = SpyBackend()
        await ProjectionCache(FakeWorkspaceAPI(), spy, rng=Extremes(pick)).get("token-alice")
        ttls += spy.ttls
    assert ttls == [270.0, 330.0]
    spy = SpyBackend()  # real rng stays inside the band
    for t in ("token-alice", "token-bob"):
        await ProjectionCache(FakeWorkspaceAPI(), spy).get(t)
    assert all(270 <= t <= 330 for t in spy.ttls)


async def test_entry_expires_after_ttl():
    now = [0.0]
    api = FakeWorkspaceAPI()
    cache = ProjectionCache(api, InMemoryBackend(lambda: now[0]), jitter=0)
    await cache.get("token-alice")
    now[0] = 299
    await cache.get("token-alice")
    assert cache.fetches == 1
    now[0] = 301
    await cache.get("token-alice")
    assert cache.fetches == 2


async def test_error_result_not_cached():
    api = FakeWorkspaceAPI(fail={"email"})
    cache = ProjectionCache(api)
    data = await cache.get("token-alice")
    assert data["task"] and data["email"] == []  # partial result is returned...
    await cache.get("token-alice")
    assert cache.fetches == 2  # ...but never stored
    api.fail.clear()
    assert (await cache.get("token-alice"))["email"]
    await cache.get("token-alice")
    assert cache.fetches == 3  # healthy result is cached


async def test_users_isolated():
    cache = ProjectionCache(FakeWorkspaceAPI())
    a, b = await cache.get("token-alice"), await cache.get("token-bob")
    assert {t["id"] for t in a["task"]}.isdisjoint({t["id"] for t in b["task"]})
    assert cache.fetches == 2
    assert await cache.get("token-bob") == b


async def test_redis_backend_roundtrip():
    backend = RedisBackend(fakeredis.FakeAsyncRedis())
    assert await backend.get("k") is None
    await backend.set("k", '{"a": 1}', 300.4)
    assert await backend.get("k") == '{"a": 1}'
    cache = ProjectionCache(FakeWorkspaceAPI(), backend)
    first = await cache.get("token-alice")
    assert await cache.get("token-alice") == first and cache.fetches == 1
