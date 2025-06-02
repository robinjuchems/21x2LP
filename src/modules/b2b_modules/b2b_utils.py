# src/modules/b2b_modules/b2b_utils.py

from typing import Dict, Any, Optional
import logging
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class B2BUtils:
    """Business-to-Business (B2B) Module mit Redis-basiertem Caching und Datenmanagement"""

    def __init__(self, redis: Redis):
        self.redis = redis
        logger.info("B2B Utils initialisiert")

    async def run(self):
        """Startet den B2B-Modul-Loop"""
        logger.info("B2B-Modul aktiviert")
        await self._initialize_cache()

    async def _initialize_cache(self):
        """Initialisiert den Redis-Cache für B2B-Operationen"""
        await self.redis.set("b2b:status", "active")
        logger.debug("B2B-Cache initialisiert")

    async def get_client_data(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Holt B2B-Kundeninformationen aus dem Cache"""
        data = await self.redis.get(f"b2b:client:{client_id}")
        if data:
            return json.loads(data)
        return None

    async def cache_client_data(self, client_id: str, data: Dict[str, Any]):
        """Speichert B2B-Kundeninformationen im Cache"""
        await self.redis.set(f"b2b:client:{client_id}", json.dumps(data), ex=3600)
        logger.debug(f"B2B-Daten für {client_id} zwischengespeichert")