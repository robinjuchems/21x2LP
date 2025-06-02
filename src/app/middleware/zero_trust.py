# src/app/middleware/zero_trust.py

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from starlette.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class ZeroTrustMiddleware(BaseHTTPMiddleware):
    """Implementiert Zero-Trust-Sicherheitsprinzipien für alle Anfragen"""

    async def dispatch(self, request: Request, call_next):
        # Sicherheitsüberprüfung vor Weiterleitung
        if not self._validate_request(request):
            logger.warning(f"Abgelehnte Anfrage: {request.url}")
            return JSONResponse(
                status_code=403,
                content={"error": "Zero Trust Security Violation"}
            )

        # Anfrage weiterleiten
        response = await call_next(request)
        return response

    def _validate_request(self, request: Request) -> bool:
        """Validiert Anfragen gemäß Zero-Trust-Prinzipien"""
        # Beispiel-Logik: Token-Überprüfung
        if "X-Quantum-Token" not in request.headers:
            return False

        # Beispiel-Logik: IP-Whitelisting
        if not self._is_ip_allowed(request.client.host):
            return False

        return True

    def _is_ip_allowed(self, ip: str) -> bool:
        """Überprüft, ob die IP-Adresse in der Whitelist ist"""
        allowed_ips = ["127.0.0.1", "::1"]  # ← Ergänzen mit echten IPs
        return ip in allowed_ips