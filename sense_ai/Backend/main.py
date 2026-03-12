import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from .core.config import APP_CONFIG
from .database.db_manager import DBManager
from .services.asl_service import ASLService


def create_app() -> ASLService:
    """
    Bootstraps the backend.
    Returns a ready-to-use ASLService instance.
    Called ONLY by Frontend/main.py — never run this file directly.
    """
    db = DBManager(db_path=APP_CONFIG["db_path"])
    service = ASLService(db=db)
    return service


# Example usage from Tkinter UI:
# from Backend.main import create_app
# service = create_app()
# result = service.login("john", "password123")
# translation = service.translate(user_id=1, source=frame, input_type="webcam")