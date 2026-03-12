# utils/encryption.py
import hashlib
import os
from typing import Tuple
from cryptography.fernet import Fernet


class PasswordManager:
    @staticmethod
    def hash_password(password: str) -> Tuple[str, str]:
        """Hash a password with a random salt."""
        salt = os.urandom(32).hex()
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        )
        return key.hex(), salt

    @staticmethod
    def verify_password(password: str, stored_hash: str, salt: str) -> bool:
        """Verify a password against its stored hash."""
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations=100_000,
        )
        return key.hex() == stored_hash


class DataEncryptor:
    """For encrypting sensitive data like API keys or model outputs"""

    def __init__(self, key_file: str = "secret.key"):
        self.key_file = key_file
        self.key = self._load_or_create_key()
        self.fernet = Fernet(self.key)

    def _load_or_create_key(self) -> bytes:
        """Load existing key or create a new one"""
        if os.path.exists(self.key_file):
            with open(self.key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, "wb") as f:
                f.write(key)
            return key

    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except:
            return ""