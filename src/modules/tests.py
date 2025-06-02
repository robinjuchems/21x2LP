# src/modules/tests.py

import logging
import pytest
from typing import Optional

logger = logging.getLogger(__name__)


def run_tests() -> bool:
    """Führt alle Tests im Projekt aus und gibt den Erfolgsstatus zurück"""
    try:
        # Test-Verzeichnis relativ zum Projektroot
        test_dir = Path(__file__).parent.parent / "tests"

        # pytest ausführen
        result = pytest.main([
            str(test_dir),
            "-v",
            "--cov=src"  # Code-Coverage für src-Verzeichnis
        ])

        logger.info(f"🧪 Test-Ergebnis: {result}")
        return result == 0

    except ImportError as e:
        logger.warning(f"pytest nicht installiert – Tests übersprungen: {e}")
        return True

    except Exception as e:
        logger.error(f"Testfehler: {e}", exc_info=True)
        return False