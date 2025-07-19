# src/modules/config/config.py

from pydantic_settings import BaseSettings
from pydantic import Field, RedisDsn, HttpUrl
from typing import List

class Config(BaseSettings):
    """Zentrale Konfigurationsklasse"""
    redis_url: RedisDsn = Field("redis://localhost:6379", env="REDIS_URL")
    db_url: str = Field("sqlite:///hypercore.db", env="DB_URL")
    quantum_shards: int = Field(2048, env="QUANTUM_SHARDS")
    qec_level: str = Field("topological", env="QEC_LEVEL")
    quantum_cores: int = Field(1024, env="QUANTUM_CORES")
    sgx_enabled: bool = Field(True, env="SGX_ENABLED")
    hsm_nodes: List[str] = Field(default_factory=list, env="HSM_NODES")

# ✅ Instanz der Config-Klasse erstellen
config = Config()