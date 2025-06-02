# src/modules/utils/elastic_log.py

from typing import Dict, Any, Optional
import logging
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, TransportError
from datetime import datetime

logger = logging.getLogger(__name__)


async def elastic_log(level: str, message: str, details: Optional[Dict[str, Any]] = None):
    """Loggt Ereignisse direkt in Elasticsearch"""
    try:
        # Beispiel-Index-Namen basierend auf dem Level
        index_name = f"hypercore-logs-{level.lower()}"

        document = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            "details": details or {}
        }

        # Lokale Test-Instanz (ersetzen mit echter Konfiguration)
        es_client = AsyncElasticsearch(hosts=["https://elastic.instance.com "])
        await es_client.index(index=index_name, document=document)
        logger.debug(f"{level} in Elasticsearch geloggt")
        return True

    except (ConnectionError, TransportError) as e:
        logger.error(f"Elasticsearch-Verbindungsfehler: {e}", exc_info=True)
        return False