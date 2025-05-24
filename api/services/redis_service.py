import redis.asyncio as redis
import os
from typing import Optional
import json

_redis_client: Optional[redis.Redis] = None
_memory_store = {}  # Fallback untuk development


async def get_redis_client() -> redis.Redis:
    """Dependency untuk Redis client dengan fallback"""
    global _redis_client

    if _redis_client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            _redis_client = redis.from_url(redis_url, decode_responses=True)
            await _redis_client.ping()  # Test connection
        except Exception as e:
            print(f"⚠️ Redis fallback mode: {e}")
            _redis_client = MockRedisClient()  # Fallback

    return _redis_client


class MockRedisClient:
    """In-memory Redis mock untuk development"""

    def __init__(self):
        self._store = {}

    async def setex(self, key: str, time: int, value: str):
        self._store[key] = value
        return True

    async def get(self, key: str):
        return self._store.get(key)

    async def ping(self):
        return True

    async def close(self):
        pass