import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any


# ─── Table Definitions ────────────────────────────────────────────────────────

CREATE_USER_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);
"""

CREATE_SESSION_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_token TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

CREATE_ASL_TRANSLATION_TABLE = """
CREATE TABLE IF NOT EXISTS asl_translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    input_type TEXT NOT NULL,        -- 'video', 'webcam', 'image'
    input_source TEXT,               -- file path or 'live'
    translated_text TEXT NOT NULL,
    confidence_score REAL,           -- model confidence (0.0 - 1.0)
    model_used TEXT,
    duration_seconds REAL,           -- how long the input was
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

CREATE_TRANSLATION_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS translation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    translation_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    session_token TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (translation_id) REFERENCES asl_translations(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

CREATE_USER_PREFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE NOT NULL,
    preferred_model TEXT DEFAULT 'default',
    show_confidence BOOLEAN DEFAULT 1,
    theme TEXT DEFAULT 'light',
    language TEXT DEFAULT 'en',
    auto_save_history BOOLEAN DEFAULT 1,
    camera_index INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

CREATE_MEETING_SESSION_TABLE = """
CREATE TABLE IF NOT EXISTS meeting_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT UNIQUE NOT NULL,
    signer_user_id INTEGER,
    speaker_user_id INTEGER,
    speaker_joined INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (signer_user_id) REFERENCES users(id),
    FOREIGN KEY (speaker_user_id) REFERENCES users(id)
);
"""


# ─── Data Classes ─────────────────────────────────────────────────────────────

class User:
    def __init__(
        self,
        id: int,
        username: str,
        email: str,
        created_at: str,
        last_login: Optional[str] = None,
    ):
        self.id = id
        self.username = username
        self.email = email
        self.created_at = created_at
        self.last_login = last_login

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at,
            "last_login": self.last_login,
        }


class ASLTranslation:
    def __init__(
        self,
        id: int,
        user_id: int,
        input_type: str,
        translated_text: str,
        confidence_score: Optional[float] = None,
        model_used: Optional[str] = None,
        input_source: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        created_at: Optional[str] = None,
    ):
        self.id = id
        self.user_id = user_id
        self.input_type = input_type
        self.translated_text = translated_text
        self.confidence_score = confidence_score
        self.model_used = model_used
        self.input_source = input_source
        self.duration_seconds = duration_seconds
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "input_type": self.input_type,
            "translated_text": self.translated_text,
            "confidence_score": self.confidence_score,
            "model_used": self.model_used,
            "input_source": self.input_source,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
        }