"""Redis-backed state store (optional dependency)."""

from __future__ import annotations

from typing import Any

try:
    import redis.asyncio as aioredis

    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


class RedisStateStore:
    """State store backed by Redis. Requires the `redis` optional dependency."""

    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "kagent:") -> None:
        if not _HAS_REDIS:
            raise ImportError(
                "redis package is required for RedisStateStore. "
                "Install it with: pip install kagent[redis]"
            )
        self._client: Any = aioredis.from_url(url, decode_responses=True)
        self._prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any:
        return await self._client.get(self._key(key))

    async def set(self, key: str, value: Any) -> None:
        await self._client.set(self._key(key), value)

    async def update(self, key: str, value: Any) -> None:
        await self._client.set(self._key(key), value)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._key(key))
