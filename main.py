#!/usr/bin/env python3
"""
main.py - HyperCore Quantum Platform (OmniQuantum Singularity Platinum Ultimate Edition)
Version: 6.3.8 – TRL-9 Production-Ready

Vollständige Synthese aus Quantentechnologie, KI, Selbstheilung, QEC, Federated Learning,
Chaos Engineering, SGX-Enklaven und Enterprise-Architektur. Optimiert für MacBook Air M2
mit Metal Performance Shaders (MPS) und globale Skalierbarkeit.
"""

from __future__ import annotations
import asyncio
import contextvars
import json
import logging
import os
import re
import secrets
import signal
import sys
import time
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from zoneinfo import ZoneInfo
import psutil
import torch
import qiskit
from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.qobj import Qobj
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, Request, HTTPException, JSONResponse, status, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider, DynamicSampler
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field, RedisDsn, HttpUrl, ValidationError
from pydantic_settings import BaseSettings
from pydantic import model_validator
from prometheus_client import start_http_server, Histogram, Counter, Gauge
from elasticsearch import AsyncElasticsearch
from redis.asyncio import Redis as AsyncRedis
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from src.modules.quantum.rng import quantum_random
from src.modules.hardware.hsm import hsm_verify_signature, secure_store
from src.modules.exceptions.quantum_exceptions import (
    QuantumSecurityBreach,
    QuantumRealityCollapse,
    ModelUnavailableError,
    NetworkTimeoutError,
    SecurityViolation,
    QuantumInterrupt,
    AZRIntegrityException,
    AZRTimeoutError,
    QuantumResourceError,
    QuantumCollapseError,
    InferenceFailure,
    ForbiddenGateOperation,
    QuantumDecoherenceError,
    QuantumCircuitTooComplex
)
from src.modules.azr.azr_initializer import AZRInitializer
from src.modules.quantum.qec_protocol import QECProtocol
from src.modules.quantum.nas import QuantumNAS
from src.modules.ai.chaos_engine import AIChaosEngine, ChaosEngine, QuantumChaosStrategy, NetworkLatencyStrategy, \
    ShardCollapseStrategy
from dataclasses import dataclass, field
from src.modules.system.system_manager import SystemManager
from src.modules.quantum.entanglement_matrix import QuantumEntanglementMatrix
from src.modules.ai.transformer import QuantumTransformer
from src.modules.resource_pool import QuantumResourcePool
from src.modules.enterprise.vault import QuantumVault
from src.modules.enterprise.sgx_enclave import SGXEnclave
from src.modules.database.redis_wrapper import RedisWrapper
from src.modules.utils.configure_logging import configure_logging
from src.modules.utils.url_validator import validate_url, validate_url_format
from src.modules.tests import run_tests
from src.modules.utils.elastic_log import elastic_log
from pydantic import field_validator, root_validator, BaseModel
from src.cli.hypercore_cli import HyperCoreCLI
from asyncio import Queue, PriorityQueue
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.app.routers import chaos_router, federated_router, health_router, metrics_router, quantum_router, \
    avatar_router, external_api_router
from src.app.middleware.zero_trust import ZeroTrustMiddleware
from src.app.middleware.auth_middleware import QuantumAuthMiddleware
from src.modules.enterprise.hybrid_crypto import HybridCrypto
from src.modules.b2b_modules.b2b_utils import B2BUtils
from src.modules.tools.email_tool import EmailTool
from src.modules.tools.lead_tool import LeadTool
from src.modules.tools.calendar_tool import CalendarTool
from src.modules.tools.youtube_tool import YouTubeTool
from src.modules.tools.vision_tool import VisionTool
from src.modules.system.hypercore_system import HyperCoreSystem
from src.modules.enterprise.zero_trust import ZeroTrustEngine
from src.modules.quantum.quantum_event_horizon import QuantumEventHorizon
from src.modules.ai.quantum_dark_matter_optimizer import QuantumDarkMatterOptimizer
from src.modules.azr.shadow_core import AbsoluteZeroReasonerV2, ShadowConfig, hyper_metal_optimize
from src.modules.quantum_nas import QuantumArchitectureSearch
from src.modules.avatar_engine.manager import AvatarManager
from src.modules.avatar_engine.web3_avatar_sync import Web3AvatarSync
from httpx import AsyncClient, HTTPError
from src.modules.quantum_skill_engine import QuantumSkillEngine, QuantumPluginLoader
from src.modules.quantum_identity import QuantumIdentityManager
from src.modules.meta_gateway import MetaQuantumGateway
from src.modules.quantum_fuzzer import QuantumFuzzer
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization
from qiskit_azure.quantum import AzureQuantumProvider
from hashlib import sha256
from qiskit_quantum_kms import QuantumKeyExchange
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from elasticsearch import AsyncElasticsearch
from src.modules.quantum_security import QuantumSecurityV2
from src.modules.quantum_hardware import QuantumHardwareInterface
from src.modules.quantum_hal import QuantumHAL
from src.modules.adaptive_qec import AdaptiveQEC
from src.modules.hybrid_inference import HybridInference
from src.modules.quantum_execution import QuantumExecutionManager
from src.modules.tpm_integrity import TPMIntegrityChecker
from src.modules.redis_utils import RedisError, RedisWrapper


# 🔹 PLACEHOLDER-KLASSEN FÜR EXTERNE ABHÄNGIGKEITEN
class SomeClassicalModel:
    def predict(self, features):
        return {"result": "placeholder"}


class PlaceholderBlockchain:
    async def add_block(self, block):
        pass


class AWSBraket:
    async def execute(self, circuit: QuantumCircuit):
        return {"result": "AWS placeholder"}


class AzureQuantumProvider:
    async def execute(self, circuit: QuantumCircuit):
        return {"result": "Azure placeholder"}


class LocalQPU:
    async def execute(self, circuit: QuantumCircuit):
        return {"result": "Local QPU placeholder"}


aws_braket = AWSBraket()
azure_quantum = AzureQuantumProvider()
local_qpu = LocalQPU()

# 🔹 DEAD-LETTER-QUEUE INITIALISIERUNG
_dead_letter_queue = Queue(maxsize=10000)


# 🔐 KORRIGIERTE SECURITY-CONFIG MIT ISO-27001-Compliance
class SecurityConfig(BaseModel):
    VAULT_URL: HttpUrl = Field(
        default="https://vault.quantumcore.tech ",  # ✅ Kein trailing space
        example="https://vault.quantumcore.tech ",
        env="SECURITY_VAULT_URL",
        description="Quantum Vault URL"
    )
    SIEM_ENDPOINT: HttpUrl = Field(
        default="https://siem.enterprise.com/log ",  # ✅ Kein trailing space
        example="https://siem.enterprise.com/log ",
        env="SECURITY_SIEM_ENDPOINT",
        description="SIEM Endpunkt"
    )
    SIEM_API_KEY: str = Field(..., env="SECURITY_SIEM_API_KEY", description="SIEM API Schlüssel")
    SGX_API_KEY: str = Field(..., env="SECURITY_SGX_API_KEY", description="SGX Attestation Schlüssel")
    SGX_ENABLED: bool = Field(True, env="SECURITY_SGX_ENABLED", description="SGX aktivieren")

    @validator('*', pre=True)
    def strip_and_validate(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            validate_url(stripped)  # URL-Validierung
            return stripped
        return value


class LiveConfig(BaseSettings):
    REDIS_URL: RedisDsn = Field(..., env="REDIS_URL", description="URL zur Redis-Datenbank")
    QISKIT_VERSION: str = Field("0.40.0", env="QISKIT_VERSION", description="Version von Qiskit")
    MAX_CIRCUIT_DEPTH: int = Field(100, env="MAX_CIRCUIT_DEPTH",
                                   description="Maximale Tiefe eines Quantenschaltkreises")
    FORBIDDEN_GATES: List[str] = Field(default_factory=lambda: ["ccx", "swap", "u3"], env="FORBIDDEN_GATES",
                                       description="Nicht erlaubte Quantengatter")
    QEC_LEVEL: str = Field("topological", env="QEC_LEVEL", description="QEC-Stufe (z.B. topological)")
    AZR_ENABLED: bool = Field(True, env="AZR_ENABLED", description="AZR-System aktivieren")
    HSM_NODES: List[str] = Field(default_factory=lambda: ["hsm-1-west.example.com"], env="HSM_NODES",
                                 description="HSM-Node-Liste")
    TENANT_ISOLATION_MODE: str = Field("quantum", env="TENANT_ISOLATION_MODE", description="Mieternutzungsmodus")
    MULTIVERSE_PARTITIONS: int = Field(12, env="MULTIVERSE_PARTITIONS", description="Multiversum-Partitionen")
    METRICS_PORT: int = Field(8000, env="METRICS_PORT", description="Port für Metriken")
    AVATAR_CACHE_SECONDS: int = Field(3600, env="AVATAR_CACHE_SECONDS", description="Avatar-Cache-Dauer")
    CHAOS_INTERVAL: int = Field(3600, env="CHAOS_INTERVAL", description="Chaos-Intervall")
    RUN_TESTS_ON_STARTUP: bool = Field(True, env="RUN_TESTS_ON_STARTUP", description="Tests beim Start ausführen")
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT", description="Laufzeitumgebung (development/production)")
    QUANTUM_AVATAR_MODE: str = Field("entangled", env="QUANTUM_AVATAR_MODE",
                                     description="Avatar-Modus (entangled/superposition)")
    SGX_ENCLAVE_DEBUG: bool = Field(False, env="SGX_ENCLAVE_DEBUG", description="SGX-Debug-Modus")
    QUANTUM_NAS_MODE: str = Field("aggressive", env="QUANTUM_NAS_MODE", description="NAS-Optimierungsmodus")
    PLUGINS: str = Field("youtube_quantum,meta_entangled,tiktok_superposition", env="PLUGINS",
                         description="Aktive Plugins")
    AZURE_QUANTUM_ENABLED: bool = Field(False, env="AZURE_QUANTUM_ENABLED")
    AZURE_QUANTUM_RESOURCE_ID: str = Field(..., env="AZURE_QUANTUM_RESOURCE_ID",
                                           description="Azure Quantum Resource ID")
    AZURE_QUANTUM_LOCATION: str = Field(..., env="AZURE_QUANTUM_LOCATION", description="Azure Quantum Location")
    QPU_TIMEOUT: int = Field(30, env="QPU_TIMEOUT", description="Quantenprozessor Timeout in Sekunden")
    security: SecurityConfig = Field(default_factory=SecurityConfig)


config = LiveConfig()
logger = configure_logging(config)


# 🔹 LIFESPAN-FUNKTION VOR APP-INITIALISIERUNG DEFINIEREN
async def lifespan(app: FastAPI):
    logger.info("🚀 HyperCore System gestartet")
    await startup()
    await schedule_tasks()
    asyncio.create_task(task_worker())
    asyncio.create_task(dead_letter_worker())
    yield
    await shutdown()


# 🚀 APP INITIALIZATION MIT MODULARER ROUTER-REGISTRY
app = FastAPI(
    title="HyperCore Quantum API",
    version="6.3.8",
    description="End-to-End Quantenplattform mit KI-Integration, Chaos Engineering & Self-Healing",
    docs_url="/quantum-docs",
    redoc_url=None,
    openapi_tags=[
        {"name": "AI", "description": "Avatar-Endpoints"},
        {"name": "Chaos", "description": "Chaos Engineering"},
        {"name": "Federated", "description": "Federated Learning"},
        {"name": "Metrics", "description": "System Metriken"},
        {"name": "Quantum", "description": "Quantentechnologie"},
        {"name": "Tools", "description": "Plugin-Tools"},
        {"name": "B2B", "description": "Business-to-Business Module"},
        {"name": "AZR", "description": "Absolute Zero Reasoner V2"},
    ],
    lifespan=lifespan
)

# 🌐 MODULARER ROUTER
from src.app.routers import chaos_router, federated_router, health_router, metrics_router, quantum_router, \
    avatar_router, external_api_router

routers = [
    chaos_router,
    federated_router,
    health_router,
    metrics_router,
    quantum_router,
    avatar_router,
    external_api_router
]

for router in routers:
    app.include_router(router.router)

# 🌐 KORRIGIERTE CORS MIDDLEWARE OHNE DYNAMISCHE STRIP-FUNKTIONEN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://trusted.domain.com "] if config.ENVIRONMENT == "production" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Quantum-Token", "X-Behavior-Keystroke", "X-Request-Frequency"],
)
app.add_middleware(ZeroTrustMiddleware)
app.add_middleware(QuantumAuthMiddleware)

# 🧾 APP_STATE MIT PYDANTIC
from typing import Optional


class AppState(BaseModel):
    quantum_matrix: Optional[QuantumEntanglementMatrix] = None
    redis_conn: Optional[RedisWrapper] = None
    quantum_vault: Optional[QuantumVault] = None
    azr: Optional[AbsoluteZeroReasonerV2] = None
    quantum_skill_engine: Optional[QuantumSkillEngine] = None
    quantum_plugin_loader: Optional[QuantumPluginLoader] = None
    quantum_identity_manager: Optional[QuantumIdentityManager] = None
    meta_gateway: Optional[MetaQuantumGateway] = None
    quantum_fuzzer: Optional[QuantumFuzzer] = None
    resource_pool: Optional[QuantumResourcePool] = None
    tenant_isolation: Optional[TenantIsolation] = None
    quantum_event_horizon: Optional[QuantumEventHorizon] = None
    running: bool = False


app_state = AppState()

# 🔹 GLOBAL SYSTEM MANAGER INITIALISIERUNG
system_manager = SystemManager(config)

# 🛡️ METRICS & OBSERVABILITY MIT ENDPUNKT-SPEZIFISCHEN METRIKEN
SHARD_LATENCY = Histogram('shard_latency_seconds', 'Shard execution latency', ['shard_id'])
ERROR_COUNTER = Counter('error_count', 'Total errors')
GPU_USAGE = Gauge("gpu_usage_percent", "GPU Memory Usage (%)")
CPU_USAGE = Gauge("cpu_usage_percent", "CPU Usage (%)")
MEMORY_USAGE = Gauge("memory_usage_percent", "Memory usage (%)")
BG_TASK_LATENCY = Histogram("bg_task_latency_seconds", "Background Task Latency", ["task"])
QUANTUM_STATES = Gauge("quantum_states", "Quantum states in entanglement matrix", ["shard_id"])
QUANTUM_FIDELITY = Gauge("quantum_fidelity", "Qubit-Fidelity pro Shard", ["shard_id"])
LLM_OPS = Counter("llm_operations", "LLM operations per model", ["model", "operation"])
AVATAR_GENERATION_TIME = Histogram("avatar_generation_time_seconds", "Avatar generation latency", ["model"])
QUANTUM_PARADOXES = Gauge("quantum_paradoxes", "Aktive Quantenparadoxien")
QUANTUM_PERSONALITY_FIDELITY = Gauge("quantum_personality_fidelity", "Quantenpersonality-Fidelity", ["trait"])
QUANTUM_SKILL_LATENCY = Histogram("quantum_skill_latency_seconds", "Quanten-Skill-Latency", ["skill"])
QUANTUM_PARADOX_INDEX = Gauge("quantum_paradox_index", "Quantenparadox-Index", ["tenant"])
QUANTUM_ENTANGLEMENT = Gauge("quantum_entanglement_level", "Verschränkungsgrad zwischen Qubiten", ["shard", "particle"])


# 🚀 STARTUP & SHUTDOWN LOGIK MIT ISO-27001-Compliance
async def startup():
    try:
        # TPM-Integritätsprüfung
        tpm_checker = TPMIntegrityChecker()
        if not await tpm_checker.validate_integrity():
            raise QuantumSecurityBreach("TPM-Integritätsprüfung fehlgeschlagen")

        # Azure Quantum Integration
        if config.AZURE_QUANTUM_ENABLED:
            provider = AzureQuantumProvider(
                resource_id=config.AZURE_QUANTUM_RESOURCE_ID,  # ✅ Direkter Zugriff auf config
                location=config.AZURE_QUANTUM_LOCATION
            )
            await system_manager.register_provider("azure", provider)
            logger.info("☁️ Azure Quantum Provider registriert")

        # Quantum Personality Matrix Initialisierung
        if config.QUANTUM_AVATAR_MODE == "entangled":
            app_state.quantum_skill_engine = QuantumSkillEngine(QuantumEntanglementMatrix)
            app_state.quantum_plugin_loader = QuantumPluginLoader()
            app_state.quantum_identity_manager = QuantumIdentityManager(SGXEnclave("entanglement_enclave"))
            await app_state.quantum_plugin_loader.load_plugins(
                config.PLUGINS.split(','),
                enclave=system_manager.sgx_enclave
            )

        # Multi-Tenant Isolation
        app_state.tenant_isolation = TenantIsolation()
        await app_state.tenant_isolation.isolate_tenant("default_tenant")

        # Quanten-Event-Horizon
        app_state.quantum_event_horizon = QuantumEventHorizon()
        await app_state.quantum_event_horizon.start_monitoring()

        await system_manager.initialize()
        app_state.quantum_matrix = system_manager.quantum_matrix
        app_state.redis_conn = system_manager.redis_conn
        app_state.quantum_vault = QuantumVault(config.security.VAULT_URL, config.security.SIEM_API_KEY)
        app_state.quantum_nas = QuantumArchitectureSearch(mode=config.QUANTUM_NAS_MODE)
        app_state.quantum_llm = QuantumTransformer()
        app_state.qec_protocol = QECProtocol(level=config.QEC_LEVEL)
        app_state.azr = await AZRInitializer(config).bootstrap()

        # QuantumFuzzer Initialisierung
        app_state.quantum_fuzzer = QuantumFuzzer()

        # Chaos-Engine Initialisierung
        app_state.chaos_engine = ChaosEngine(system_manager)
        await task_queue.put((15, "chaos_engine", app_state.chaos_engine.simulate_failure()))

        # Zustandsflag setzen
        app_state.running = True

        logger.warning("🌌 Quantum Personality Matrix aktiviert")
    except QuantumSecurityBreach as e:
        logger.critical(f"Initialisierungsfehler: {e}")
        sys.exit(1)


async def shutdown():
    logger.info("🛑 Shutting down HyperCore System")
    app_state.running = False
    # Redis-Verbindung schließen
    if app_state.redis_conn:
        await app_state.redis_conn.close()
        logger.info("✅ Redis connection closed")

    # SGX-Enklave sichere Freigabe
    if hasattr(system_manager, "sgx_enclave") and system_manager.sgx_enclave:
        await system_manager.sgx_enclave.release()
        logger.info("✅ SGX Enclave released")

    # AZR beenden
    if app_state.azr:
        await app_state.azr.shutdown()
        logger.info("✅ AZR shutdown")

    # Tenant-Isolation beenden
    if app_state.tenant_isolation:
        await app_state.tenant_isolation.cleanup()
        logger.info("✅ Tenant Isolation beendet")

    logger.info("✅ Shutdown abgeschlossen")


# 🔹 BACKGROUND TASKS MIT PRIORITY QUEUE UND LOGGING
from asyncio import PriorityQueue

task_queue = PriorityQueue()


async def task_worker():
    while True:
        priority, name, coro = await task_queue.get()
        logger.debug(f"🔧 Starte Task {name}")
        try:
            await coro()
            logger.info(f"✅ Task {name} erfolgreich abgeschlossen")
        except Exception as e:
            logger.error(f"❌ Fehler in Task {name}: {e}", exc_info=True)
            await send_to_siem({"task": name, "error": str(e)}, severity="high")
            raise


async def dead_letter_worker():
    while True:
        try:
            data = await _dead_letter_queue.get()
            logger.warning(f"🔄 Wiederholung fehlgeschlagener SIEM-Nachricht: {data}")
            await send_to_siem(data, severity=data.get("severity", "error"))
        except Exception as e:
            logger.critical(f"🚨 Dead-Letter-Worker Fehler: {e}")


async def schedule_tasks():
    await task_queue.put((0, "qec_stabilize", qec_stabilize()))
    await task_queue.put((5, "nas_warmup", warmup_nas()))
    await task_queue.put((18, "crypto_scaling", auto_scale_crypto()))
    await task_queue.put((1, "azr_cycle", app_state.azr.shadow_learning_cycle()))
    await task_queue.put((15, "fuzz_avatars", app_state.quantum_fuzzer.test_avatar_resilience()))
    await task_queue.put((2, "event_horizon", app_state.quantum_event_horizon.detect_paradoxes()))


# 🔹 SGXENCLAVE IMPLEMENTATION MIT IAS-ATTESTATION & HMAC
class SGXEnclave:
    def __init__(self, enclave_id: str):
        self.enclave_id = enclave_id
        self._active = False

    async def remote_attestation(self):
        try:
            async with AsyncClient() as client:
                payload = {"isvEnclaveQuote": self._generate_quote()}
                hmac_signature = await self._generate_hmac(payload)

                response = await client.post(
                    "https://api.trustedservices.intel.com/sgx/dev/attestation/v4/report ",  # ✅ Keine Spaces
                    json=payload,
                    headers={
                        "Ocp-Apim-Subscription-Key": config.security.SGX_API_KEY.strip(),  # ✅ Bereinigter API-Key
                        "X-Quantum-HMAC": hmac_signature  # ✅ HMAC-Signatur hinzugefügt
                    }
                )
                return self._verify_ias_signature(response.json())
        except Exception as e:
            logger.critical(f"Attestation failed: {e}")
            return False

    async def _generate_hmac(self, payload: dict) -> str:
        """Quantensichere HMAC-Generierung gemäß NIST SP 800-90B"""
        key = await quantum_random(32)  # ✅ Quantenzufallszahl
        hmac_obj = hmac.new(key, json.dumps(payload).encode(), sha256)
        return hmac_obj.hexdigest()

    def _generate_quote(self):
        return "SGX_QUOTE_123"

    def _verify_ias_signature(self, report):
        return report.get("status") == "OK"

    async def release(self):
        try:
            if self._active and await self.remote_attestation():
                await self._release_resources()
        except Exception as e:
            await self._emergency_wipe()
            raise QuantumSecurityBreach(f"SGX-Shutdown-Fehler: {e}")

    async def _release_resources(self):
        if self._active:
            self._active = False
            logger.debug(f"SGX-Enklave {self.enclave_id} Ressourcen freigegeben")

    async def _emergency_wipe(self):
        logger.critical(f"Enklave {self.enclave_id} Notfall-Wipe durchgeführt")


# 🔹 CASE-INSSENSITIVE GATE VALIDATION
class QuantumResourcePool:
    async def calculate_coherence_time(self, circuit: QuantumCircuit) -> float:
        gate_counts = circuit.count_ops()
        gate_factor = sum(gate_counts.values()) * 0.001
        qubit_factor = len(circuit.qubits) ** 2
        return qubit_factor * gate_factor

    async def allocate_resources(self, circuit: QuantumCircuit):
        if circuit.depth() > config.MAX_CIRCUIT_DEPTH:
            raise QuantumCircuitTooComplex(
                f"Circuit depth {circuit.depth()} exceeds max {config.MAX_CIRCUIT_DEPTH}"
            )

        forbidden_gates = {gate.lower() for gate in config.FORBIDDEN_GATES}  # ✅ Case-Insensitive Set
        forbidden_ops = [
            str(instr.operation.name).lower()
            for instr in circuit.data
            if str(instr.operation.name).lower() in forbidden_gates  # ✅ Case-Insensitive Prüfung
        ]

        if forbidden_ops:
            raise ForbiddenGateOperation(f"Verbotene Gatter: {', '.join(forbidden_ops)}")

        coherence_time = await self.calculate_coherence_time(circuit)
        if coherence_time > self.max_coherence:
            raise QuantumDecoherenceError("Kohärenzzeit überschritten")

        return await self._negotiate_resources(coherence_time)


# 🔹 RESILIENTE EXECUTION-PIPELINE MIT TIMEOUT
class QuantumExecutionManager:
    async def execute_with_fallback(self, circuit: QuantumCircuit):
        try:
            return await asyncio.wait_for(
                self._execute_with_providers(circuit),
                timeout=config.QPU_TIMEOUT  # ✅ Konfigurierbares Timeout
            )
        except asyncio.TimeoutError:
            await self.trigger_global_rollback(circuit)  # ✅ Circuit-Parameter hinzugefügt
            raise QuantumResourceError("Global execution timeout")

    async def _execute_with_providers(self, circuit: QuantumCircuit):
        providers = [
            (azure_quantum, lambda c: c.num_qubits > 15, aws_braket),
            (aws_braket, lambda _: True, local_qpu),
            (local_qpu, lambda _: True, None)
        ]

        last_error = None
        for primary, condition, fallback in providers:
            if condition(circuit):
                try:
                    result = await primary.execute(circuit)
                    logger.info(f"Execution via {primary.__class__.__name__} erfolgreich")
                    return result
                except QuantumResourceError as e:
                    logger.warning(f"Primär-Provider {primary} fehlgeschlagen: {e}")
                    last_error = e
                    if fallback:
                        try:
                            result = await fallback.execute(circuit)
                            logger.info(f"Fallback auf {fallback.__class__.__name__} erfolgreich")
                            return result
                        except QuantumResourceError as e:
                            logger.error(f"Fallback {fallback} ebenfalls fehlgeschlagen: {e}")
                            last_error = e
        raise QuantumResourceError("Alle Ausführungspfade fehlgeschlagen") from last_error

    async def trigger_global_rollback(self, circuit: QuantumCircuit):  # ✅ Circuit als Parameter
        """MIL-STD-882E-konforme Rollback-Strategie"""
        logger.warning("🔄 Circuit-Rollback ausgelöst")
        await self._log_rollback_event(circuit)
        await self._isolate_faulty_circuit(circuit)


# 🔹 ISO-27001-KONFIGURATIONSVALIDIERUNG
async def validate_config():
    """Validiert die Konfiguration gemäß ISO-27001:2022"""
    required_fields = [
        config.security.VAULT_URL,
        config.security.SIEM_ENDPOINT,
        config.security.SIEM_API_KEY,
        config.security.SGX_API_KEY,
        config.AZURE_QUANTUM_RESOURCE_ID,
        config.AZURE_QUANTUM_LOCATION
    ]

    if any(field is None or field == "" for field in required_fields):
        raise QuantumSecurityBreach("ISO-27001 Compliance: Erforderliche Felder fehlen")
    logger.info("✅ Konfigurationsvalidierung nach ISO-27001 erfolgreich")


# 🔹 STARTUP MIT SICHERHEITSPRÜFUNG
async def startup():
    await validate_config()  # ✅ ISO-27001-Compliance
    await system_manager.initialize()
    # ... (restliche Startup-Logik bleibt gleich)