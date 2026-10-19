"""Per-user projection cache: one parallel fetch of everything the user can see, grepped in memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import Callable
from typing import Any, Protocol

from .backend import KINDS, WorkspaceAPI

# Kind -> records, plus "hierarchy": the project ids this user may see (a list of strings).
Projection = dict[str, list[Any]]


class Backend(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: float) -> None: ...


class InMemoryBackend:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._data: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None or entry[0] <= self._clock():
            return None
        return entry[1]

    async def set(self, key: str, value: str, ttl: float) -> None:
        self._data[key] = (self._clock() + ttl, value)


class RedisBackend:
    """Takes any `redis.asyncio`-style client, so tests can hand it fakeredis."""

    def __init__(self, client: Any) -> None:
        self._r = client

    async def get(self, key: str) -> str | None:
        v = await self._r.get(key)
        return v.decode() if isinstance(v, bytes) else v

    async def set(self, key: str, value: str, ttl: float) -> None:
        await self._r.set(key, value, ex=max(int(ttl), 1))


class ProjectionCache:
    def __init__(
        self,
        api: WorkspaceAPI,
        backend: Backend | None = None,
        ttl: float = 300.0,
        jitter: float = 30.0,
        rng: random.Random | None = None,
    ) -> None:
        self.api = api
        self.backend = backend or InMemoryBackend()
        self.ttl, self.jitter = ttl, jitter
        self.rng = rng or random.Random()
        self.fetches = 0
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, token: str) -> Projection:
        # Key on a hash: the token itself must never land in Redis keys or logs.
        key = "projection:" + hashlib.sha256(token.encode()).hexdigest()[:16]
        if (hit := await self._read(key)) is not None:
            return hit
        async with self._locks.setdefault(key, asyncio.Lock()):
            if (hit := await self._read(key)) is not None:  # double-check: a waiter finds it filled
                return hit
            data, complete = await self._fetch_all(token)
            if complete:
                # Jitter spreads expiries so many users don't refetch on the same tick.
                ttl = self.ttl + self.rng.uniform(-self.jitter, self.jitter)
                await self.backend.set(key, json.dumps(data), ttl)
            return data

    async def _read(self, key: str) -> Projection | None:
        raw = await self.backend.get(key)
        return json.loads(raw) if raw is not None else None

    async def _fetch_all(self, token: str) -> tuple[Projection, bool]:
        self.fetches += 1
        calls = [getattr(self.api, f"list_{k}s")(token) for k in KINDS]
        calls.append(self.api.allowed_project_ids(token))
        results = await asyncio.gather(*calls, return_exceptions=True)
        data = {k: r for k, r in zip((*KINDS, "hierarchy"), results) if not isinstance(r, BaseException)}
        # Rule 6: a partial or empty projection cached for 5 minutes is a 5-minute outage.
        complete = len(data) == len(KINDS) + 1 and any(data.values())
        # A missing hierarchy becomes [] and so hides every scoped record: fail closed.
        return {k: data.get(k, []) for k in (*KINDS, "hierarchy")}, complete
