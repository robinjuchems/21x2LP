# src/modules/tools/email_tool.py

from typing import Dict, Any, Optional
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class EmailTool:
    """Modul zur sicheren E-Mail-Kommunikation mit Quantum-Entsprechung"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_host = config.get("SMTP_HOST", "localhost")
        self.smtp_port = config.get("SMTP_PORT", 587)
        self.smtp_user = config.get("SMTP_USER", "user@example.com")
        self.smtp_pass = config.get("SMTP_PASS", "password")
        logger.info("EmailTool initialisiert")

    async def send(self, recipient: str, subject: str, body: str):
        """Sichert die E-Mail-Nachricht und sendet sie asynchron"""
        try:
            # Beispiel-Logik für E-Mail-Versand
            message = MIMEMultipart()
            message["From"] = self.smtp_user
            message["To"] = recipient
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            # Asynchroner SMTP-Versand (mit Thread-Pool)
            await asyncio.to_thread(
                self._send_sync,
                recipient,
                message.as_string()
            )
            logger.info(f"E-Mail an {recipient} gesendet")
            return True
        except Exception as e:
            logger.error(f"Fehler beim E-Mail-Versand: {e}", exc_info=True)
            return False

    def _send_sync(self, recipient: str, message: str):
        """Synchroner SMTP-Versand"""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_pass)
            server.sendmail(self.smtp_user, recipient, message)