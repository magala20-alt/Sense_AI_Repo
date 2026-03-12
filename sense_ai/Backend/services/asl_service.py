from typing import Any, Callable, Dict, List, Optional

from database.db_manager import DBManager


class ASLService:
    """
    Core service layer for ASL Translation.
    Called directly by the Tkinter UI — no HTTP, no API.
    """

    def __init__(self, db: DBManager, translator: Optional[Callable] = None):
        self.db = db
        # Plug in your real ASL model here, falls back to dummy
        self.translator = translator or self._dummy_translator

    def _dummy_translator(self, source: Any, input_type: str) -> Dict[str, Any]:
        return {
            "translated_text": "[ASL Translation Placeholder]",
            "confidence_score": 0.0,
            "model_used": "dummy",
        }

    # ─── Auth ──────────────────────────────────────────────────────────────────

    def register(self, username: str, email: str, password: str) -> Dict[str, Any]:
        user = self.db.create_user(username, email, password)
        if user:
            return {"ok": True, "user": user.to_dict()}
        return {"ok": False, "error": "Username or email already exists."}

    def login(self, username: str, password: str) -> Dict[str, Any]:
        user = self.db.authenticate_user(username, password)
        if user:
            token = self.db.create_session_token(user.id)
            return {"ok": True, "user": user.to_dict(), "token": token}
        return {"ok": False, "error": "Invalid username or password."}

    def logout(self, token: str):
        self.db.invalidate_session_token(token)

    def get_current_user(self, token: str) -> Optional[Dict[str, Any]]:
        user = self.db.validate_session_token(token)
        return user.to_dict() if user else None

    # ─── Translation ───────────────────────────────────────────────────────────

    def translate(
        self,
        user_id: int,
        source: Any,
        input_type: str = "webcam",
        input_source: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        session_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Translate ASL input and save result.
        input_type: 'webcam' | 'video' | 'image'
        source: frame, file path, or stream data
        """
        try:
            result = self.translator(source, input_type)
            translation = self.db.save_translation(
                user_id=user_id,
                input_type=input_type,
                translated_text=result["translated_text"],
                confidence_score=result.get("confidence_score"),
                model_used=result.get("model_used"),
                input_source=input_source,
                duration_seconds=duration_seconds,
                session_token=session_token,
            )
            return {
                "ok": True,
                "translation": translation.to_dict(),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_history(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.get_translation_history(user_id, limit)

    def delete_entry(self, user_id: int, translation_id: int) -> bool:
        return self.db.delete_translation(translation_id, user_id)

    def clear_history(self, user_id: int):
        self.db.clear_translation_history(user_id)

    # ─── Preferences ───────────────────────────────────────────────────────────

    def get_preferences(self, user_id: int) -> Dict[str, Any]:
        return self.db.get_user_preferences(user_id)

    def update_preferences(self, user_id: int, **kwargs):
        self.db.update_user_preferences(user_id, **kwargs)

    # ─── Meeting Sessions ──────────────────────────────────────────────────────

    def create_meeting_session(self, meeting_id: str, signer_user_id: int | None = None) -> dict:
        return self.db.create_meeting_session(meeting_id=meeting_id, signer_user_id=signer_user_id)

    def join_meeting_session_as_speaker(self, meeting_id: str, speaker_user_id: int | None = None) -> bool:
        return self.db.join_meeting_session_as_speaker(meeting_id=meeting_id, speaker_user_id=speaker_user_id)

    def is_speaker_joined(self, meeting_id: str) -> bool:
        return self.db.is_speaker_joined(meeting_id=meeting_id)