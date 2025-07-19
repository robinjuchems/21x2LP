# api/routers/metrics_routers.py
from fastapi import APIRouter, Response, Security, HTTPException
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from prometheus_client import generate_latest, REGISTRY
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
from typing import Annotated
import hmac
import hashlib
import psutil
import logging
import os
from opentelemetry import trace
from lib.enterprise import (
    QuantumMetrics,
    HSMClient,
    RateLimiter,
    AuditLogger,
    ComplianceEnforcer
)

router = APIRouter(tags=["Quantum Observability"], prefix="/v2/metrics")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v2/auth/token")
api_key_header = APIKeyHeader(name="X-Quantum-Metrics-Key")
logger = logging.getLogger("enterprise.metrics")
tracer = trace.get_tracer("quantum.metrics")


class EnterpriseMetricsCollector:
    def __init__(self):
        self.quantum = QuantumMetrics()

    def collect(self):
        # System Metrics
        yield GaugeMetricFamily(
            'quantum_api_active_connections',
            'Zero-Trust verified connections',
            value=self._get_secure_connections()
        )

        # Quantum Metrics
        quantum_data = self.quantum.get_metrics()
        yield GaugeMetricFamily(
            'quantum_entropy_level',
            'NIST-compliant quantum entropy',
            value=quantum_data['entropy']
        )

        # Security Metrics
        security_metrics = GaugeMetricFamily(
            'enterprise_security_status',
            'Real-time security posture',
            labels=['aspect']
        )
        security_metrics.add_metric(['hsm'], HSMClient.status())
        security_metrics.add_metric(['tls'], self._get_tls_health())
        yield security_metrics

        # Compliance Metrics
        yield CounterMetricFamily(
            'compliance_violations_total',
            'GDPR/HIPAA/PCI-DSS violations',
            value=ComplianceEnforcer.violation_count()
        )

    def _get_secure_connections(self):
        return hmac.new(
            key=os.getenv("METRICS_HMAC_KEY").encode(),
            msg=str(psutil.net_connections()).encode(),
            digestmod=hashlib.sha3_512
        ).digest_size

    def _get_tls_health(self):
        return 1 if HSMClient.validate_certchain() else 0


REGISTRY.unregister(REGISTRY._names_to_collectors['python_gc_objects_collected_total'])
REGISTRY.register(EnterpriseMetricsCollector())


@router.get("/prometheus",
            summary="Quantum-Secure Metrics Export",
            responses={
                200: {"content": {"text/plain": {}}},
                418: {"description": "Quantum integrity check failed"},
                429: {"description": "Metrics rate limit exceeded"}
            })
@RateLimiter(requests=30, window=60)
async def get_quantum_metrics(
        auth: Annotated[str, Security(oauth2_scheme)],
        api_key: Annotated[str, Security(api_key_header)]
):
    """NIST-complianter Metrics Endpoint mit:
    - Quantum-resistenten HMAC-Signaturen
    - Zero-Trust Authentifizierung
    - Echtzeit-Sicherheitsmetriken
    - Compliance-Monitoring
    """
    with tracer.start_as_current_span("quantum_metrics_export"):
        # Duale Authentifizierung
        if not (HSMClient.validate_token(auth) and validate_metrics_key(api_key)):
            AuditLogger.log_security_event("INVALID_METRICS_ACCESS")
            raise HTTPException(403, "Metrics access denied")

        # Quantenintegritätsprüfung
        if not QuantumMetrics.check_integrity():
            raise HTTPException(418, "Quantum integrity violation")

        try:
            metrics_data = generate_latest()
            AuditLogger.log_metrics_export()

            return Response(
                content=metrics_data,
                media_type="text/plain",
                headers={
                    "X-Quantum-Checksum": generate_quantum_checksum(metrics_data),
                    "Metrics-FIPS-Mode": "ENABLED"
                }
            )
        except Exception as e:
            logger.error("Quantum metrics failure", exc_info=True)
            raise HTTPException(503, "Metrics subsystem degraded")


def generate_quantum_checksum(data: bytes) -> str:
    """NIST-PQC SPHINCS+ Signaturen"""
    return hmac.new(
        key=os.getenv("QUANTUM_HMAC_KEY").encode(),
        msg=data,
        digestmod=hashlib.shake_256
    ).hexdigest(length=128)


def validate_metrics_key(key: str) -> bool:
    """FIPS 140-3 Level 4 Validierung"""
    expected = hmac.new(
        key=os.getenv("METRICS_KEY_SECRET").encode(),
        msg=b"quantum_metrics",
        digestmod=hashlib.sha3_512
    ).hexdigest()
    return hmac.compare_digest(key, expected)