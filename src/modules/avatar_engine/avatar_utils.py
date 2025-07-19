# avatar_engine/utils.py
"""
Enterprise-Level Avatar-Utility mit Quantensicherheit, Self-Healing und Prometheus-Metriken

Enthält:
- Typsichere Avatar-Validierung
- Quantum-Token-basierte Sicherheitsprüfung
- Prometheus-Metriken für Avatar-Operationen
- Vault-Integration für Geheimnisse
- Self-Healing bei Quanten-Token-Problemen
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Union, Optional, TypeVar, Generic
from pydantic import BaseModel, Field, validator
from prometheus_client import Histogram, Counter, Gauge
from core_platform.quantum_security import QuantumSecurity, InvalidToken, SecurityViolation
from modules.utils import validate_agent_config, log_agent_activity
from modules.monitoring import SystemMonitor
from core_platform.self_healing import SelfHealingController
from plugin_system.sdk import PluginManager
from modules.config import LLMConfig
from core_platform.redis_wrapper import RedisWrapper
from core_platform.quantum_security import VaultClient

# Typdefinition für Generics
T = TypeVar("T", bound="BaseAvatar")

# Prometheus-Metriken
AVATAR_LATENCY = Histogram(
    "avatar_operation_latency_seconds",
    "Avatar-Operation Latenz",
    ["operation"]
)
AVATAR_FAILURE_COUNTER = Counter(
    "avatar_operations_errors_total",
    "Anzahl der Avatar-Fehler",
    ["operation", "error_type"]
)
AVATAR_SUCCESS_COUNTER = Counter(
    "avatar_operations_success_total",
    "Erfolgreiche Avatar-Operationen",
    ["operation"]
)


class AvatarValidationConfig(BaseModel):
    """
    Konfiguration für Avatar-Validierung

    Beispiel:
        config = AvatarValidationConfig(required_fields=["name", "prompt"], min_shards=1, max_shards=64)
    """
    required_fields: List[str] = Field(["name", "prompt"], description="Erforderliche Felder")
    min_shards: int = Field(1, ge=1)
    max_shards: int = Field(64, ge=1)
    enable_healing: bool = True
    healing_threshold: float = Field(0.8, ge=0.1, le=1.0)
    retry_attempts: int = Field(3, ge=0)
    vault_path: Optional[str] = "avatar/validator"
    enable_tracing: bool = True
    enable_metrics: bool = True


class AvatarValidator:
    """
    Enterprise-Level Avatar-Validierung mit Quanten-Token-Sicherheit

    Features:
    - Quantum-Token-basierte Validierung
    - Prometheus-Metriken für Validierungszeit
    - Self-Healing bei ungültigen Avataren
    - Vault-Integration für Geheimnisse
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton für einheitliche Validierung"""
        if cls._instance is None:
            cls._instance = super(AvatarValidator, cls).__new__(cls)
        return cls._instance

    def __init__(
            self,
            quantum_sec: QuantumSecurity,
            redis: RedisWrapper,
            vault: 'VaultClient',
            logger: Optional[logging.Logger] = None,
            config: Optional[AvatarValidationConfig] = None
    ):
        """
        Initialisiere Avatar-Validator mit Enterprise-Features

        Args:
            quantum_sec: Instanz von QuantumSecurity
            redis: RedisWrapper für Caching
            vault: VaultClient für Geheimnisse
            logger: Optional: Logger-Instanz
            config: Validierungs-Konfiguration

        Raises:
            SecurityViolation: Bei ungültiger Konfiguration
        """
        if AvatarValidator._initialized:
            return

        # Basisvariablen
        self.quantum_sec = quantum_sec
        self.redis = redis
        self.vault = vault
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config = config or AvatarValidationConfig()
        self.tracer = trace.get_tracer(self.__class__.__name__)
        self.healing_controller = SelfHealingController()

        # Initialisierung
        self._validate_config()
        self._load_vault_secrets()

        self.logger.info(f"🧠 Avatar-Validator mit {len(self.config.required_fields)} Pflichtfeldern gestartet")
        AvatarValidator._initialized = True

    def _validate_config(self):
        """Prüfe, ob Validierungs-Konfiguration sicherheitsgeprüft ist"""
        if self.config.min_shards > self.config.max_shards:
            raise SecurityViolation("Ungültige Shard-Konfiguration")

    def _load_vault_secrets(self):
        """Lade Geheimnisse aus Vault"""
        try:
            if self.config.vault_path:
                secrets = self.vault.read_secret(self.config.vault_path)
                self.logger.debug(f"🔐 Geheimnisse geladen aus {self.config.vault_path}")
        except Exception as e:
            self.logger.warning(f"🔒 Vault-Fehler: {e}")
            AVATAR_FAILURE_COUNTER.labels(
                operation="vault",
                error_type="vault_error"
            ).inc()

    def _check_quantum_health(self) -> bool:
        """Prüfe Quanten-Token-Zustand"""
        try:
            if not self.quantum_sec.check_qkd_health():
                self.logger.warning("⚠️ QKD-Schlüssel abgelaufen")
                return False
            return True

        except Exception as e:
            self.logger.error(f"❌ Integritätsprüfung fehlgeschlagen: {e}")
            return False


def validate_avatar_config(config: Dict[str, Any]) -> bool:
    """
    Prüfe Avatar-Konfiguration mit Quanten-Token

    Args:
        config: Avatar-Konfiguration

    Returns:
        bool - Erfolgsstatus

    Raises:
        SecurityViolation: Bei ungültiger Konfiguration
    """
    correlation_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Prüfe, ob alle erforderlichen Felder vorhanden sind
        if not all(field in config for field in config.get("required_fields", ["name", "prompt"])):
            raise SecurityViolation("Fehlende erforderliche Felder")

        # Prüfe Quanten-Token
        if not config.get("token") or not self.quantum_sec.verify(config["token"]):
            raise InvalidToken("Ungültiger Quantum-Token")

        duration = time.time() - start_time
        AVATAR_LATENCY.labels(operation="config_validation").observe(duration)
        AVATAR_SUCCESS_COUNTER.labels(operation="config_validation").inc()
        return True

    except (InvalidToken, SecurityViolation) as se:
        AVATAR_FAILURE_COUNTER.labels(error_type="security").inc()
        logger.warning(f"🔒 Sicherheitsfehler: {se}")
        return False

    except Exception as e:
        AVATAR_FAILURE_COUNTER.labels(error_type="general").inc()
        logger.error(f"💥 Allgemeiner Fehler: {e}")
        raise


def log_avatar_activity(avatar_id: str, action: str, quantum_sec: QuantumSecurity):
    """
    Logge Avatar-Aktivitäten mit Quanten-Token

    Args:
        avatar_id: ID des Avatars
        action: Ausgeführte Aktion
        quantum_sec: QuantumSecurity-Instanz

    Raises:
        SecurityViolation: Bei ungültigem Token
        RuntimeError: Bei allgemeinen Fehlern
    """
    correlation_id = str(uuid.uuid4())

    with tracer.start_as_current_span("avatar_logging") as span:
        span.set_attribute("avatar.id", avatar_id)
        span.set_attribute("correlation.id", correlation_id)

        try:
            # Prüfe Quantum-Token
            if not quantum_sec.verify(action):
                raise InvalidToken("Ungültiger Quantum-Token")

            # Logging
            logger.info(f"🧠 Avatar {avatar_id} führte {action} aus")

        except InvalidToken as it:
            logger.warning(f"🔒 Ungültiger Token: {it}")
            AVATAR_FAILURE_COUNTER.labels(error_type="token").inc()
            raise

        except Exception as e:
            logger.error(f"❌ Loggingfehler: {e}", exc_info=True)
            AVATAR_FAILURE_COUNTER.labels(error_type="general").inc()
            raise


def get_vault_secret(vault: 'VaultClient', path: str, quantum_sec: QuantumSecurity) -> Dict[str, Any]:
    """
    Hole Geheimnis aus Vault mit Quanten-Token-Validierung

    Args:
        vault: VaultClient für Geheimnisabruf
        path: Pfad zum Geheimnis
        quantum_sec: QuantumSecurity-Instanz

    Returns:
        Dict[str, Any] - Geheimnisdaten

    Raises:
        SecurityViolation: Bei ungültigem Token
        RuntimeError: Bei allgemeinen Fehlern
    """
    correlation_id = str(uuid.uuid4())

    with tracer.start_as_current_span("vault_secret_fetch") as span:
        span.set_attribute("vault.path", path)
        span.set_attribute("correlation.id", correlation_id)

        try:
            # Hole Geheimnis mit Quanten-Token-Validierung
            secret = vault.read_secret(path)
            if not quantum_sec.verify(secret.get("token", "")):
                raise InvalidToken("Ungültiger Vault-Token")

            return secret

        except (InvalidToken, SecurityViolation) as se:
            logger.warning(f"🔒 Sicherheitsfehler: {se}")
            AVATAR_FAILURE_COUNTER.labels(
                operation="vault",
                error_type="vault_token"
            ).inc()
            raise

        except Exception as e:
            logger.error(f"❌ Vault-Abfrage fehlgeschlagen: {e}")
            AVATAR_FAILURE_COUNTER.labels(
                operation="vault",
                error_type="vault_general"
            ).inc()
            raise


def get_redis_connection(redis: RedisWrapper, quantum_sec: QuantumSecurity, logger: logging.Logger) -> RedisWrapper:
    """
    Hole gesicherte Redis-Verbindung mit Quanten-Token-Validierung

    Args:
        redis: RedisWrapper für Caching
        quantum_sec: QuantumSecurity-Instanz
        logger: Logger-Instanz

    Returns:
        RedisWrapper - Gesicherte Redis-Verbindung
    """
    correlation_id = str(uuid.uuid4())

    with tracer.start_as_current_span("redis_connection") as span:
        span.set_attribute("correlation.id", correlation_id)

        try:
            # Prüfe Redis-Zustand
            status = await redis.ping()
            if status != "PONG":
                raise SecurityViolation("Redis nicht erreichbar")

            # Prüfe Quanten-Token
            if not quantum_sec.check_qkd_health():
                raise SecurityViolation("QKD-Schlüssel abgelaufen")

            return redis

        except SecurityViolation as sv:
            logger.warning(f"🔒 Sicherheitsfehler: {sv}")
            AVATAR_FAILURE_COUNTER.labels(
                operation="redis",
                error_type="security"
            ).inc()
            raise

        except Exception as e:
            logger.error(f"❌ Redis-Verbindungsfehler: {e}")
            AVATAR_FAILURE_COUNTER.labels(
                operation="redis",
                error_type="general"
            ).inc()
            raise


def dynamic_error_handler(func: Callable) -> Callable:
    """Dekorator für dynamische Fehlerbehandlung"""

    async def wrapper(*args, **kwargs):
        correlation_id = str(uuid.uuid4())
        start_time = time.time()

        with tracer.start_as_current_span("error_handling") as span:
            span.set_attribute("correlation.id", correlation_id)

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                ERROR_LATENCY.observe(duration)
                return result

            except (InvalidToken, SecurityViolation) as se:
                logger.warning(f"🔒 Sicherheitsfehler: {se}")
                ERROR_COUNTER.labels(error_type="security").inc()
                raise

            except RealityAnomaly as ra:
                logger.error(f"🌀 Realitätsanomalie: {ra}")
                ERROR_COUNTER.labels(error_type="reality_anomaly").inc()
                raise

            except Exception as e:
                logger.error(f"💥 Allgemeiner Fehler: {e}", exc_info=True)
                ERROR_COUNTER.labels(error_type="general").inc()
                raise RuntimeError(f"Operation fehlgeschlagen: {e}") from e

    return wrapper


def update_error_rate(avatar_id: str, error_count: int, total_executions: int):
    """Aktualisiere Fehlerquote des Avatars"""
    try:
        error_rate = error_count / total_executions
        ERROR_RATE_GAUGE.labels(avatar_id=avatar_id).set(error_rate)

        if error_rate > HEALING_THRESHOLD:
            logger.warning(f"⚠️ Fehlerquote >{HEALING_THRESHOLD * 100:.0f}% für {avatar_id}")
            asyncio.create_task(healing_controller.heal_avatar(avatar_id))

    except ZeroDivisionError as zde:
        logger.warning(f"❌ Nullteiler bei Fehlerquote: {zde}")
        ERROR_COUNTER.labels(error_type="zero_division").inc()

    except Exception as e:
        logger.warning(f"❌ Metrikaktualisierung fehlgeschlagen: {e}")
        ERROR_COUNTER.labels(error_type="metric_update").inc()


def quantum_config_validator(config: Dict[str, Any]) -> bool:
    """Validiere Quanten-Token in der Konfiguration"""
    if not config.get("token") or not quantum_sec.verify(config["token"]):
        raise SecurityViolation("Ungültiger Quantum-Token")
    return True


def dynamic_shard_selection(task: Dict[str, Any], shard_count: int) -> int:
    """Dynamische Shard-Auswahl mit Quanten-Token"""
    try:
        # Hash-basierte Auswahl mit Quantum-Token
        if not quantum_sec.verify(task.get("token", "")):
            raise InvalidToken("Ungültiger Quantum-Token")

        return hash(task["user_id"]) % shard_count

    except InvalidToken as it:
        logger.warning(f"🔒 Ungültiger Token: {it}")
        AVATAR_FAILURE_COUNTER.labels(error_type="token").inc()
        return -1

    except Exception as e:
        logger.error(f"❌ Shard-Auswahl fehlgeschlagen: {e}")
        AVATAR_FAILURE_COUNTER.labels(error_type="general").inc()
        return -1


def generate_quantum_avatar_id() -> str:
    """Erstelle Quanten-gesicherte Avatar-ID"""
    try:
        token = quantum_sec.generate_token({
            "type": "avatar_id",
            "timestamp": datetime.utcnow().isoformat(),
            "random": str(uuid.uuid4())
        })
        logger.info(f"✅ Quanten-ID generiert für Avatar")
        return token
    except Exception as e:
        logger.error(f"❌ Fehler bei ID-Generierung: {e}")
        AVATAR_FAILURE_COUNTER.labels(error_type="quantum_id").inc()
        raise SecurityViolation(f"Quanten-ID-Generierung fehlgeschlagen: {e}")


def validate_avatar_prompt(prompt: str, quantum_sec: QuantumSecurity) -> bool:
    """
    Prüfe Prompt mit Quanten-Token

    Args:
        prompt: Prompt für den Avatar
        quantum_sec: QuantumSecurity-Instanz

    Returns:
        bool - Erfolgsstatus
    """
    correlation_id = str(uuid.uuid4())

    with tracer.start_as_current_span("prompt_validation") as span:
        span.set_attribute("correlation.id", correlation_id)
        span.set_attribute("avatar.prompt", prompt[:20])

        try:
            # Prüfe Prompt-Länge
            if len(prompt) < 20:
                raise SecurityViolation("Prompt zu kurz")

            # Prüfe Quantum-Token in Prompt
            if not quantum_sec.verify(prompt):
                raise InvalidToken("Ungültiger Prompt-Token")

            # Prüfe auf unerlaubte Begriffe
            if any(bad in prompt.lower() for bad in ["hack", "exploit", "bypass", "admin", "root"]):
                raise SecurityViolation("Unerlaubter Begriff im Prompt")

            AVATAR_SUCCESS_COUNTER.labels(operation="prompt_validation").inc()
            return True

        except (InvalidToken, SecurityViolation) as se:
            logger.warning(f"🔒 Sicherheitsfehler: {se}")
            AVATAR_FAILURE_COUNTER.labels(error_type="security").inc()
            return False

        except Exception as e:
            logger.error(f"💥 Allgemeiner Fehler: {e}")
            AVATAR_FAILURE_COUNTER.labels(error_type="general").inc()
            raise


def log_avatar_activity_with_qkd(avatar_id: str, action: str, quantum_sec: QuantumSecurity):
    """
    Logge Avatar-Aktivitäten mit Quanten-Token-Validierung

    Args:
        avatar_id: ID des Avatars
        action: Ausgeführte Aktion
        quantum_sec: QuantumSecurity-Instanz

    Raises:
        SecurityViolation: Bei ungültigem Token
        RuntimeError: Bei allgemeinen Fehlern
    """
    correlation_id = str(uuid.uuid4())

    with tracer.start_as_current_span("qkd_avatar_logging") as span:
        span.set_attribute("avatar.id", avatar_id)
        span.set_attribute("correlation.id", correlation_id)

        try:
            # Prüfe Quantum-Token
            if not quantum_sec.verify(action):
                raise InvalidToken("Ungültiger Quantum-Token")

            # Logging
            logger.info(f"🧠 Avatar {avatar_id} führte {action} aus")

        except InvalidToken as it:
            logger.warning(f"🔒 Sicherheitsfehler: {it}")
            AVATAR_FAILURE_COUNTER.labels(
                operation="logging",
                error_type="token"
            ).inc()
            raise

        except Exception as e:
            logger.error(f"❌ Loggingfehler: {e}", exc_info=True)
            AVATAR_FAILURE_COUNTER.labels(
                operation="logging",
                error_type="general"
            ).inc()
            raise


def validate_avatar_prompt(prompt: str, quantum_sec: QuantumSecurity) -> bool:
    """
    Prüfe Prompt mit Quanten-Token

    Args:
        prompt: Prompt für den Avatar
        quantum_sec: QuantumSecurity-Instanz

    Returns:
        bool - Erfolgsstatus
    """
    correlation_id = str(uuid.uuid4())

    with tracer.start_as_current_span("prompt_validation") as span:
        span.set_attribute("correlation.id", correlation_id)
        span.set_attribute("avatar.prompt", prompt[:20])

        try:
            # Prüfe Prompt-Länge
            if len(prompt) < 20:
                raise SecurityViolation("Prompt zu kurz")

            # Prüfe Quantum-Token
            if not quantum_sec.verify(prompt):
                raise InvalidToken("Ungültiger Quantum-Token")

            # Prüfe auf unerlaubte Begriffe
            if any(bad in prompt.lower() for bad in ["hack", "exploit", "admin", "root", "password"]):
                raise SecurityViolation("Unerlaubter Begriff im Prompt")

            AVATAR_SUCCESS_COUNTER.labels(operation="prompt_validation").inc()
            return True

        except (InvalidToken, SecurityViolation) as se:
            logger.warning(f"🔒 Sicherheitsfehler: {se}")
            AVATAR_FAILURE_COUNTER.labels(error_type="security").inc()
            return False

        except Exception as e:
            logger.error(f"❌ Prompt-Validierung fehlgeschlagen: {e}")
            AVATAR_FAILURE_COUNTER.labels(error_type="general").inc()
            return False


class QuantumWorkloadBalancer:
    """
    Enterprise-Level Quanten-Shard-Balancer

    Features:
    - Quantum-Token-gesicherte Verteilung
    - Prometheus-Metriken für Lastverteilung
    - Self-Healing bei Quanten-Token-Problemen
    - Vault-Integration für Balancing-Geheimnisse
    """

    def __init__(
            self,
            shard_count: int,
            quantum_sec: QuantumSecurity,
            redis: RedisWrapper,
            vault: 'VaultClient'
    ):
        self.shard_count = shard_count
        self.quantum_sec = quantum_sec
        self.redis = redis
        self.vault = vault
        self.logger = logging.getLogger(__class__.__name__)
        self._shard_states = [1] * shard_count
        self._last_shard = 0
        self._token_cache = {}

    def distribute_task(self, task: Dict[str, Any]) -> int:
        """Verteile Aufgaben auf Quanten-Shards"""
        try:
            # Prüfe Token
            if not self.quantum_sec.verify(task.get("token", "")):
                raise InvalidToken("Ungültiger Quantum-Token")

            # Wähle Shard
            shard_id = hash(task["user_id"]) % self.shard_count
            if self._shard_states[shard_id] == 0:
                shard_id = self._fallback_shard(task)

            # Metriken aktualisieren
            SHARD_LATENCY.labels(shard_id=shard_id).observe(time.time() - task["timestamp"])

            return shard_id

        except (InvalidToken, SecurityViolation) as se:
            logger.warning(f"🔒 Sicherheitsfehler: {se}")
            AVATAR_FAILURE_COUNTER.labels(error_type="security").inc()
            raise

        except Exception as e:
            logger.error(f"❌ Shard-Verteilung fehlgeschlagen: {e}")
            AVATAR_FAILURE_COUNTER.labels(error_type="general").inc()
            raise

    def _fallback_shard(self, task: Dict[str, Any]) -> int:
        """Falls Hauptshard offline, wähle Fallback"""
        for i in range(self.shard_count):
            if self._shard_states[i] == 1:
                return i
        raise SecurityViolation("Kein verfügbares Shard")

    def update_shard_state(self, shard_id: int, state: int):
        """Aktualisiere Quanten-Shard-Zustand"""
        if 0 <= shard_id < self.shard_count:
            self._shard_states[shard_id] = state
            SHARD_STATE_GAUGE.labels(shard_id=shard_id).set(state)
            self.logger.info(f"🔄 Shard {shard_id} Zustand aktualisiert: {state}")

    def get_shard_states(self) -> Dict[int, float]:
        """Hole Zustand aller Quanten-Shards"""
        return {
            i: SHARD_STATE_GAUGE.labels(shard_id=i)._value.get()
            for i in range(self.shard_count)
        }

    def __repr__(self):
        return f"<{self.__class__.__name__} shards={self.shard_count} states={self._shard_states}>"


class QuantumTokenManager:
    """
    Enterprise-Level Token-Management mit Quantum-Sicherheit

    Features:
    - Token-Rotation mit QKD
    - Prometheus-Metriken für Token-Validierung
    - Self-Healing bei Token-Problemen
    """

    def __init__(self, quantum_sec: QuantumSecurity):
        self.quantum_sec = quantum_sec
        self.logger = logging.getLogger(__class__.__name__)
        self._token_cache = {}

    def generate_token(self, data: Dict[str, Any]) -> str:
        """Erstelle Quantum-Token"""
        try:
            # Füge Zeitstempel hinzu
            data["timestamp"] = datetime.utcnow().isoformat()
            token = self.quantum_sec.generate_token(data)
            self._token_cache[token] = data
            self._token_rotations += 1
            return token

        except Exception as e:
            logger.error(f"❌ Token-Generierung fehlgeschlagen: {e}")
            AVATAR_FAILURE_COUNTER.labels(
                operation="token",
                error_type="token_generation"
            ).inc()
            raise

    def verify_token(self, token: str) -> bool:
        """Prüfe Quantum-Token"""
        try:
            if token in self._token_cache:
                if self._token_cache[token]["timestamp"] < (datetime.utcnow() - timedelta(hours=1)).isoformat():
                    self.logger.warning("🔄 Token abgelaufen")
                    self.rotate_token(token)
                    return False
                return True
            return False

        except Exception as e:
            logger.error(f"❌ Token-Validierung fehlgeschlagen: {e}")
            AVATAR_FAILURE_COUNTER.labels(
                operation="token",
                error_type="token_validation"
            ).inc()
            return False

    def rotate_token(self, token: str):
        """Rotiere Quantum-Token"""
        try:
            if token in self._token_cache:
                new_token = self.quantum_sec.generate_token({
                    "data": self._token_cache[token]["data"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "rotation": True
                })
                self._token_cache[new_token] = self._token_cache.pop(token)
                self.logger.info(f"🔄 Token rotiert: {token[:8]}... → {new_token[:8]}...")

        except Exception as e:
            self.logger.error(f"❌ Token-Rotation fehlgeschlagen: {e}")
            AVATAR_FAILURE_COUNTER.labels(
                operation="token",
                error_type="token_rotation"
            ).inc()
            raise

    def get_token(self, token: str) -> Dict[str, Any]:
        """Hole Token-Daten mit Quantum-Token-Validierung"""
        try:
            if not self.verify_token(token):
                raise InvalidToken("Ungültiger Quantum-Token")
            return self._token_cache[token]

        except InvalidToken as it:
            logger.warning(f"🔒 Sicherheitsfehler: {it}")
            AVATAR_FAILURE_COUNTER.labels(error_type="token").inc()
            raise

        except Exception as e:
            logger.error(f"❌ Token-Abfrage fehlgeschlagen: {e}")
            AVATAR_FAILURE_COUNTER.labels(error_type="general").inc()
            raise

    def __repr__(self):
        return f"<{self.__class__.__name__} tokens={len(self._token_cache)}>"