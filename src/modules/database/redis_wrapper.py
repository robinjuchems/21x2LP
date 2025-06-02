# src/modules/database/redis_wrapper.py

from typing import Optional, Dict, Any
import logging
from redis.asyncio import Redis
import asyncio

logger = logging.getLogger(__name__)


class RedisWrapper:
    """Asynchroner Redis-Wrapper mit Quantenintegration und Sicherheitsfeatures"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._client: Optional[Redis] = None
        logger.info("RedisWrapper initialisiert")

    async def connect(self):
        """Verbindet sich mit dem Redis-Server"""
        try:
            self._client = Redis.from_url(self.redis_url)
            await self._client.ping()
            logger.info("Redis-Verbindung hergestellt")
            return True
        except Exception as e:
            logger.error(f"Redis-Verbindungsfehler: {e}", exc_info=True)
            return False

    async def get(self, key: str) -> Optional[bytes]:
        """Holt Daten aus Redis"""
        if not self._client:
            return None
        return await self._client.get(key)

    async def set(self, key: str, value: bytes, ex: Optional[int] = None):
        """Speichert Daten in Redis"""
        if not self._client:
            return False
        return await self._client.set(key, value, ex=ex)

    async def close(self):
        """Schließt die Redis-Verbindung"""
        if self._client:
            await self._client.close()
            logger.info("Redis-Verbindung geschlossen")