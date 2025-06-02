# src/modules/utils/configure_logging.py

import logging
import sys
from typing import Optional, Dict, Any


def configure_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Konfiguriert das Logging-System mit Sicherheits- und Metrikintegration"""
    # Erstelle einen Hauptlogger
    logger = logging.getLogger("hypercore")
    logger.setLevel(logging.DEBUG)

    # Konsole-Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Datei-Handler
    file_handler = logging.FileHandler("hypercore.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Wenn Konfiguration vorhanden ist, anpassen
    if config and hasattr(config, "security") and config.security.SGX_ENABLED:
        logger.info("Sicherheitslogging aktiviert")
        # Hier könnten zusätzliche Sicherheitslogging-Handler hinzugefügt werden

    return logger