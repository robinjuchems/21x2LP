# src/modules/enterprise/hybrid_crypto.py

from typing import Optional, Dict, Any
import logging
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric.padding import OAEP
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)


class HybridCrypto:
    """Hybrid-Kryptografie-Modul mit symmetrischer und asymmetrischer Verschlüsselung"""

    def __init__(self):
        self._generate_keys()
        logger.info("HybridCrypto initialisiert")

    def _generate_keys(self):
        """Generiert RSA-Schlüsselpaar für asymmetrische Verschlüsselung"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        self.public_key = self.private_key.public_key()
        logger.debug("RSA-Schlüsselpaar generiert")

    def encrypt(self, data: bytes) -> Dict[str, bytes]:
        """Verschlüsselt Daten mit Hybridkryptografie"""
        # Symmetrische AES-GCM-Verschlüsselung
        aes_key = os.urandom(32)
        aesgcm = AESGCM(aes_key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, data, None)

        # RSA-Verschlüsselung des AES-Schlüssels
        encrypted_aes_key = self.public_key.encrypt(
            aes_key,
            OAEP(mgf=hashes.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )

        return {
            "ciphertext": ciphertext,
            "nonce": nonce,
            "encrypted_aes_key": encrypted_aes_key
        }

    def decrypt(self, encrypted_data: Dict[str, bytes]) -> bytes:
        """Entschlüsselt Daten mit Hybridkryptografie"""
        aesgcm = AESGCM(self._decrypt_aes_key(encrypted_data["encrypted_aes_key"]))
        return aesgcm.decrypt(encrypted_data["nonce"], encrypted_data["ciphertext"], None)

    def _decrypt_aes_key(self, encrypted_aes_key: bytes) -> bytes:
        """Entschlüsselt den AES-Schlüssel mit dem privaten RSA-Schlüssel"""
        return self.private_key.decrypt(
            encrypted_aes_key,
            OAEP(mgf=hashes.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )