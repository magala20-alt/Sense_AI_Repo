# database/db_manager.py
import json
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from Backend.database.models import (
    CREATE_ASL_TRANSLATION_TABLE,
    CREATE_SESSION_TABLE,
    CREATE_TRANSLATION_HISTORY_TABLE,
    CREATE_USER_PREFERENCES_TABLE,
    CREATE_USER_TABLE,
    CREATE_MEETING_SESSION_TABLE,
    ASLTranslation,
    User,
)
from utils.encryption import PasswordManager


class DBManager:
    def __init__(self, db_path: str = "data/sense_ai.db"):
        self.db_path = str(Path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ─── Connection ───────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(CREATE_USER_TABLE)
            conn.execute(CREATE_SESSION_TABLE)
            conn.execute(CREATE_ASL_TRANSLATION_TABLE)
            conn.execute(CREATE_TRANSLATION_HISTORY_TABLE)
            conn.execute(CREATE_USER_PREFERENCES_TABLE)
            conn.execute(CREATE_MEETING_SESSION_TABLE)
            conn.commit()

    # ─── User Methods ─────────────────────────────────────────────────────────

    def create_user(self, username: str, email: str, password: str) -> Optional[User]:
        try:
            password_hash, salt = PasswordManager.hash_password(password)
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO users (username, email, password_hash, salt)
                    VALUES (?, ?, ?, ?)
                    """,
                    (username, email, password_hash, salt),
                )
                user_id = cur.lastrowid
                conn.commit()

            # Create default preferences for new user
            self._create_default_preferences(user_id)

            return User(
                id=user_id,
                username=username,
                email=email,
                created_at=datetime.now().isoformat(),
            )
        except sqlite3.IntegrityError:
            return None  # Username or email already exists

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, email, password_hash, salt, created_at, last_login
                FROM users WHERE username = ? AND is_active = 1
                """,
                (username,),
            ).fetchone()

            if row and PasswordManager.verify_password(
                password, row["password_hash"], row["salt"]
            ):
                # Update last login
                conn.execute(
                    "UPDATE users SET last_login = ? WHERE id = ?",
                    (datetime.now().isoformat(), row["id"]),
                )
                conn.commit()
                return User(
                    id=row["id"],
                    username=row["username"],
                    email=row["email"],
                    created_at=row["created_at"],
                    last_login=row["last_login"],
                )
        return None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, email, created_at, last_login FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row:
                return User(
                    id=row["id"],
                    username=row["username"],
                    email=row["email"],
                    created_at=row["created_at"],
                    last_login=row["last_login"],
                )
        return None

    # ─── Session Token Methods ────────────────────────────────────────────────

    def create_session_token(self, user_id: int, expires_hours: int = 24) -> str:
        token = secrets.token_hex(32)
        expires_at = (datetime.now() + timedelta(hours=expires_hours)).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (user_id, session_token, expires_at) VALUES (?, ?, ?)",
                (user_id, token, expires_at),
            )
            conn.commit()
        return token

    def validate_session_token(self, token: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT s.user_id, s.expires_at FROM sessions s
                WHERE s.session_token = ?
                """,
                (token,),
            ).fetchone()

            if row:
                if datetime.fromisoformat(row["expires_at"]) > datetime.now():
                    return self.get_user_by_id(row["user_id"])
        return None

    def invalidate_session_token(self, token: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_token = ?", (token,))
            conn.commit()

    # ─── ASL Translation Methods ──────────────────────────────────────────────

    def save_translation(
        self,
        user_id: int,
        input_type: str,
        translated_text: str,
        confidence_score: Optional[float] = None,
        model_used: Optional[str] = None,
        input_source: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        session_token: Optional[str] = None,
    ) -> Optional[ASLTranslation]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO asl_translations 
                (user_id, input_type, input_source, translated_text,
                 confidence_score, model_used, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    input_type,
                    input_source,
                    translated_text,
                    confidence_score,
                    model_used,
                    duration_seconds,
                ),
            )
            translation_id = cur.lastrowid

            # Log to history if auto_save is on
            prefs = self.get_user_preferences(user_id)
            if prefs.get("auto_save_history", True):
                conn.execute(
                    """
                    INSERT INTO translation_history (translation_id, user_id, session_token)
                    VALUES (?, ?, ?)
                    """,
                    (translation_id, user_id, session_token),
                )
            conn.commit()

        return ASLTranslation(
            id=translation_id,
            user_id=user_id,
            input_type=input_type,
            translated_text=translated_text,
            confidence_score=confidence_score,
            model_used=model_used,
            input_source=input_source,
            duration_seconds=duration_seconds,
            created_at=datetime.now().isoformat(),
        )

    def get_translation_history(
        self, user_id: int, limit: int = 50
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.id, t.input_type, t.input_source, t.translated_text,
                       t.confidence_score, t.model_used, t.duration_seconds, t.created_at
                FROM asl_translations t
                WHERE t.user_id = ?
                ORDER BY t.created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_translation(self, translation_id: int, user_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM asl_translations WHERE id = ? AND user_id = ?",
                (translation_id, user_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def clear_translation_history(self, user_id: int):
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM asl_translations WHERE user_id = ?", (user_id,)
            )
            conn.execute(
                "DELETE FROM translation_history WHERE user_id = ?", (user_id,)
            )
            conn.commit()

    # ─── User Preferences ─────────────────────────────────────────────────────

    def _create_default_preferences(self, user_id: int):
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO user_preferences (user_id) VALUES (?)",
                (user_id,),
            )
            conn.commit()

    def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
            ).fetchone()
            if row:
                return dict(row)
        return {
            "preferred_model": "default",
            "show_confidence": True,
            "theme": "light",
            "language": "en",
            "auto_save_history": True,
            "camera_index": 0,
        }

    def update_user_preferences(self, user_id: int, **kwargs):
        allowed = {
            "preferred_model", "show_confidence", "theme",
            "language", "auto_save_history", "camera_index"
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return

        fields = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [user_id]

        with self._connect() as conn:
            conn.execute(
                f"UPDATE user_preferences SET {fields} WHERE user_id = ?", values
            )
            conn.commit()

    # ─── Meeting Session Methods ──────────────────────────────────────────────

    def _normalize_meeting_id(self, meeting_id: str) -> str:
        return (meeting_id or "").strip().upper()

    def create_meeting_session(
        self,
        meeting_id: str,
        signer_user_id: int | None = None,
        ttl_minutes: int = 120,
    ) -> dict:
        meeting_id = self._normalize_meeting_id(meeting_id)
        now = datetime.utcnow()
        expires = now + timedelta(minutes=ttl_minutes)

        created_at = now.strftime("%Y-%m-%d %H:%M:%S")
        expires_at = expires.strftime("%Y-%m-%d %H:%M:%S")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO meeting_sessions
                (meeting_id, signer_user_id, speaker_joined, status, created_at, expires_at)
                VALUES (?, ?, 0, 'active', ?, ?)
                ON CONFLICT(meeting_id) DO UPDATE SET
                    signer_user_id = excluded.signer_user_id,
                    speaker_user_id = NULL,
                    speaker_joined = 0,
                    status = 'active',
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (meeting_id, signer_user_id, created_at, expires_at),
            )
            conn.commit()

        return {"meeting_id": meeting_id, "speaker_joined": False, "status": "active"}

    def join_meeting_session_as_speaker(
        self,
        meeting_id: str,
        speaker_user_id: int | None = None,
    ) -> bool:
        meeting_id = self._normalize_meeting_id(meeting_id)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE meeting_sessions
                SET speaker_joined = 1,
                    speaker_user_id = ?,
                    status = 'active'
                WHERE meeting_id = ?
                  AND status = 'active'
                  AND expires_at > ?
                """,
                (speaker_user_id, meeting_id, now),
            )
            conn.commit()
            return cur.rowcount > 0

    def is_speaker_joined(self, meeting_id: str) -> bool:
        meeting_id = self._normalize_meeting_id(meeting_id)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT speaker_joined
                FROM meeting_sessions
                WHERE meeting_id = ?
                  AND status = 'active'
                  AND expires_at > ?
                """,
                (meeting_id, now),
            ).fetchone()

        return bool(row and row["speaker_joined"] == 1)