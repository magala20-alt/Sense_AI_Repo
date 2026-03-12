import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = str(DATA_DIR / "sense_ai.db")

APP_CONFIG = {
    "app_name": "Sense AI - ASL Translator",
    "version": "1.0.0",
    "db_path": DB_PATH,
    "default_camera_index": 0,
    "default_model": "default",
    "session_expiry_hours": 24,
}