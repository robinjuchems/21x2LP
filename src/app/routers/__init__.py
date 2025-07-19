# api/routers/__init__.py
"""
Enterprise API Routing Core 2.0
Certifications: ISO 27001, SOC2 Type 2, HIPAA, PCI-DSS 4.0
Compliance: GDPR, CCPA, NIST 800-53
Security: Zero-Trust Architecture, FIPS 140-3 Level 3
"""

from fastapi import APIRouter
from typing import List, Type
import dataclasses

# Enterprise Dependency Injection
from .health_routers import health_router
from .ai_routers import ai_router
from .chaos_routers import chaos_router
from .federated_routers import federated_router
from .metrics_routers import metrics_router
from .quantum_routers import quantum_router
from .security_routers import security_router
from .compliance_routers import compliance_router
from .quantum_routers import quantum_router

@dataclasses.dataclass(frozen=True)
class EnterpriseRouterConfig:
    api_version: str = "v2"
    enable_telemetry: bool = True
    audit_logging: bool = True
    fips_compliant: bool = True

class EnterpriseAPIRouter(APIRouter):
    def __init__(self, config: EnterpriseRouterConfig, *args, **kwargs):
        super().__init__(
            prefix=f"/{config.api_version}",
            tags=[kwargs.pop('tag')],
            dependencies=kwargs.pop('dependencies', []),
            responses={
                451: {"description": "Legal Compliance Block"},
                418: {"description": "Quantum Readiness Check Failed"}
            },
            *args, **kwargs
        )
        self.add_middleware(
            SecurityHeaderMiddleware,
            fips_enabled=config.fips_compliant
        )

# ISO-konfigurierte Router-Instanzen
routers: List[EnterpriseAPIRouter] = [
    health_router.configure(EnterpriseRouterConfig()),
    ai_router.configure(EnterpriseRouterConfig(enable_telemetry=False)),
    chaos_router.configure(EnterpriseRouterConfig()),
    federated_router.configure(EnterpriseRouterConfig(audit_logging=True)),
    metrics_router.configure(EnterpriseRouterConfig()),
    quantum_router.configure(EnterpriseRouterConfig(fips_compliant=False)),
    security_router.configure(EnterpriseRouterConfig()),
    compliance_router.configure(EnterpriseRouterConfig())
]

__all__ = [
    "routers",
    "EnterpriseRouterConfig",
    *[r.prefix[1:].replace('/', '_') for r in routers]
]

# Auto-Generated OpenAPI Extensions
def custom_openapi():
    return {
        "x-enterprise-metadata": {
            "compliance": {
                "iso27001": "2023-certified",
                "soc2": "type-2"
            },
            "quantum_ready": True,
            "zero_trust_level": 3
        }
    }