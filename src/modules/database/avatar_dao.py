# src/modules/database/avatar_dao.py

from typing import Dict, Any, Optional
import logging
from pydantic import BaseModel
from redis.asyncio import Redis as AsyncRedis

logger = logging.getLogger(__name__)


class AvatarParams(BaseModel):
    name: str
    prompt: str
    style: str = "standard"
    use_sd: bool = False
    model: str


class AvatarResult(BaseModel):
    url: str
    generation_time: float


class AvatarDAO:
    """Data Access Object für Avatar-Persistenz mit Redis"""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client = AsyncRedis.from_url(redis_url)
        logger.info("AvatarDAO initialisiert")

    async def get_avatar(self, name: str) -> Optional[Dict]:
        """Lädt Avatar aus Redis-Cache"""
        raw = await self.client.get(f"avatar:{name}")
        if not raw:
            return None
        return json.loads(raw)

    async def save_avatar(self, name: str, data: Dict):
        """Speichert Avatar in Redis-Cache"""
        payload = json.dumps(data)
        await self.client.setex(f"avatar:{name}", 3600, payload)
        logger.info(f"Avatar {name} gespeichert")

    async def delete_avatar(self, name: str):
        """Löscht Avatar aus Cache"""
        await self.client.delete(f"avatar:{name}")
        logger.info(f"Avatar {name} gelöscht")