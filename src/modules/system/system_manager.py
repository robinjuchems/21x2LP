# src/modules/system/system_manager.py

from typing import Dict, Any, Optional
import logging
import asyncio
from datetime import timedelta
from contextvars import ContextVar

logger = logging.getLogger(__name__)


class SystemManager:
    """Zentraler Manager für Systemressourcen, Sicherheit und Metriken"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._shutdown = asyncio.Event()
        logger.info("SystemManager initialisiert")

    async def graceful_terminate(self):
        """Führt eine geordnete Beendigung durch"""
        logger.info("Starte geordnete Beendigung")
        self._shutdown.set()
        # Hier könnten weitere Beendigungslogiken hinzugefügt werden

    async def monitor_resources(self):
        """Überwacht CPU, RAM und GPU-Nutzung"""
        while not self._shutdown.is_set():
            logger.debug(f"CPU-Nutzung: {psutil.cpu_percent()}%")
            logger.debug(f"RAM-Nutzung: {psutil.virtual_memory().percent}%")
            await asyncio.sleep(5)