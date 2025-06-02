# src/modules/quantum/rng.py

import numpy as np
from qiskit import QuantumCircuit, transpile  # ✅ Qiskit 1.x-kompatible Importe
from qiskit_aer import Aer  # ✅ Neue Position des Aer-Simulators

def quantum_random(size: int = 1) -> list:
    """Generiert quantenbasierte Zufallszahlen mit Qiskit 1.x."""
    # Quantenschaltung für Zufallszahlengenerierung
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    # Lokale Simulation mit Aer aus qiskit-aer
    simulator = Aer.get_backend('qasm_simulator')
    result = simulator.run(qc, shots=size).result()
    counts = result.get_counts(qc)

    # Ergebnisse in Liste konvertieren
    return [int(bit) for bit in counts]