# src/modules/exceptions/__init__.py

from .quantum_exceptions import (
    QuantumSecurityBreach,
    QuantumCircuitTooComplex,
    ForbiddenGateOperation,
    QuantumDecoherenceError,
    QuantumResourceError,
    CircuitVersionMismatch
)