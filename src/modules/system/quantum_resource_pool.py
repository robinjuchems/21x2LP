async def _health_check(self, provider: str):
    backend = self.get_backend(provider)
    try:
        test_circuit = QuantumCircuit(2)
        test_circuit.h(0)
        test_circuit.cx(0, 1)
        test_circuit.measure_all()

        result = await backend.run(test_circuit, shots=10)
        fidelity = result.results[0].data.counts.get('0x0', 0) / 1000
        RESOURCE_POOL_HEALTH.labels(provider=provider).set(1.0 if fidelity > 0.95 else 0.5)
        return True
    except Exception as e:
        logger.error(f"Health Check für {provider} fehlgeschlagen: {e}")
        RESOURCE_POOL_HEALTH.labels(provider=provider).set(0.0)
        return False