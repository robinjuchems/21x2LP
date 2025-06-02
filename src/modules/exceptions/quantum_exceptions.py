# src/modules/exceptions/quantum_exceptions.py

from typing import Optional
import logging

logger = logging.getLogger(__name__)

class QuantumSecurityBreach(Exception):
    """Wird ausgelöst bei Sicherheitsverletzungen im Quantensystem."""
    def __init__(self, message: str = "Quantum Security Breach detected"):
        self.message = message
        logger.critical(f"Sicherheitsverletzung: {message}")
        super().__init__(self.message)

class QuantumCircuitTooComplex(Exception):
    """Wird ausgelöst, wenn der Quantenschaltkreis zu komplex ist."""
    def __init__(self, message: str = "Circuit complexity exceeds allowed limits"):
        self.message = message
        logger.warning(f"Schaltkreis zu komplex: {message}")
        super().__init__(self.message)

class ForbiddenGateOperation(Exception):
    """Wird ausgelöst bei verbotenen Gatteroperationen."""
    def __init__(self, message: str = "Forbidden gate operation attempted"):
        self.message = message
        logger.error(f"Verbotene Operation: {message}")
        super().__init__(self.message)

class QuantumDecoherenceError(Exception):
    """Wird ausgelöst bei Quantendekohärenz."""
    def __init__(self, message: str = "Quantum decoherence occurred"):
        self.message = message
        logger.error(f"Dekohärenz erkannt: {message}")
        super().__init__(self.message)

class QuantumResourceError(Exception):
    """Wird ausgelöst bei Ressourcenproblemen im Quantensystem."""
    def __init__(self, message: str = "Quantum resources unavailable"):
        self.message = message
        logger.error(f"Ressourcenfehler: {message}")
        super().__init__(self.message)

class CircuitVersionMismatch(Exception):
    """Wird ausgelöst bei falschen Schaltkreisversionen."""
    def __init__(self, message: str = "Circuit version mismatch detected"):
        self.message = message
        logger.warning(f"Versionen unkompatibel: {message}")
        super().__init__(self.message)