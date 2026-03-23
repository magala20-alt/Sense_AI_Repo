"""
Backend WebSocket Server - Handles real-time ASL translation and model inference
Serves signer and speaker views with streaming frame processing
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path

# Add backend to path
BACKEND_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_PATH))

from Backend.services.asl_service import ASLService
from database.db_manager import DBManager
from services.model_loader import get_model_manager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global state
connected_clients = set()  # Don't use deprecated WebSocketServerProtocol type hint
db = None
asl_service = None
model_manager = None


class WebSocketServer:
    """ASL Translation WebSocket Server"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.db = None
        self.asl_service = None
        self.model_manager = None

    async def initialize(self):
        """Initialize database and services"""
        try:
            # Initialize database
            db_path = BACKEND_PATH / "database" / "asl_database.db"
            self.db = DBManager(str(db_path))

            # Create translator function
            def translate_func(source: Any, input_type: str) -> Dict[str, Any]:
                """Translate using models"""
                try:
                    if input_type == "text":
                        # Text to gloss (speaker view)
                        result = self.model_manager.translate_text_to_gloss(str(source))
                        return {
                            "translated_text": result.get("gloss", source),
                            "confidence_score": result.get("confidence", 0.0),
                            "model_used": "asl_translator"
                        }
                    elif input_type == "webcam" or input_type == "frame":
                        # Frame to text (signer view)
                        if isinstance(source, dict):
                            # Already processed frame data
                            hand_data = source.get("hand_output", {})
                            facial_data = source.get("facial_output", {})
                            hand_reason = hand_data.get("reason", "")
                            # Simple heuristic: detect if hands are present
                            if hand_data.get("detected"):
                                return {
                                    "translated_text": f"[Sign detected] Expression: {facial_data.get('expression', 'neutral')}",
                                    "confidence_score": hand_data.get("confidence", 0.0),
                                    "model_used": "hand_recognition"
                                }
                            if hand_reason:
                                return {
                                    "translated_text": f"[Signer model unavailable] {hand_reason}",
                                    "confidence_score": 0.0,
                                    "model_used": "hand_recognition"
                                }
                        return {
                            "translated_text": "[No sign detected]",
                            "confidence_score": 0.0,
                            "model_used": "hand_recognition"
                        }
                    return {
                        "translated_text": f"[Unknown input type: {input_type}]",
                        "confidence_score": 0.0,
                        "model_used": None
                    }
                except Exception as e:
                    return {
                        "translated_text": "[Translation error]",
                        "confidence_score": 0.0,
                        "error": str(e)
                    }

            # Initialize ASL service
            self.asl_service = ASLService(self.db, translator=translate_func)

            # Load models
            self.model_manager = get_model_manager()

            logger.info("Server initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False

    async def handle_client(self, websocket, path: Optional[str] = None):
        """Handle WebSocket client connection"""
        connected_clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            connected_clients.discard(websocket)

    async def process_message(self, websocket, message: str):
        """Process incoming message"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            elif message_type == "auth":
                await self._handle_auth(websocket, data)

            elif message_type == "translate_text":
                await self._handle_text_translation(websocket, data)

            elif message_type == "translate_frame":
                await self._handle_frame_translation(websocket, data)

            elif message_type == "translate_multimodal":
                await self._handle_multimodal_translation(websocket, data)

            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"Unknown message type: {message_type}"
                }))
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "error": "Invalid JSON"
            }))
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e)
            }))

    async def _handle_auth(self, websocket, data: Dict):
        """Handle authentication"""
        action = data.get("action")

        if action == "register":
            result = self.asl_service.register(
                username=data.get("username"),
                email=data.get("email"),
                password=data.get("password")
            )
        elif action == "login":
            result = self.asl_service.login(
                username=data.get("username"),
                password=data.get("password")
            )
        else:
            result = {"ok": False, "error": "Unknown auth action"}

        await websocket.send(json.dumps({
            "type": "auth_response",
            **result
        }))

    async def _handle_text_translation(self, websocket, data: Dict):
        """Handle speaker view: text → ASL gloss"""
        try:
            user_id = data.get("user_id", 1)
            session_token = data.get("session_token")
            text = data.get("text", "")

            # Validate token if provided
            if session_token:
                user = self.asl_service.get_current_user(session_token)
                if user:
                    user_id = user["id"]
                else:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Invalid session token"
                    }))
                    return

            # Translate text
            result = self.asl_service.translate(
                user_id=user_id,
                source=text,
                input_type="text",
                input_source="speaker_view",
                session_token=session_token
            )

            await websocket.send(json.dumps({
                "type": "translation_result",
                "ok": result.get("ok"),
                "translation": result.get("translation"),
                "error": result.get("error")
            }))
        except Exception as e:
            logger.error(f"Text translation error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e)
            }))

    async def _handle_frame_translation(self, websocket, data: Dict):
        """Handle signer view: frame data → text"""
        try:
            user_id = data.get("user_id", 1)
            session_token = data.get("session_token")

            # Validate token if provided
            if session_token:
                user = self.asl_service.get_current_user(session_token)
                if user:
                    user_id = user["id"]

            # Use frame metadata for translation
            frame_data = {
                "hand_output": data.get("hand_data", {}),
                "facial_output": data.get("facial_data", {})
            }

            result = self.asl_service.translate(
                user_id=user_id,
                source=frame_data,
                input_type="webcam",
                input_source="signer_view",
                session_token=session_token
            )

            await websocket.send(json.dumps({
                "type": "translation_result",
                "ok": result.get("ok"),
                "translation": result.get("translation"),
                "error": result.get("error")
            }))
        except Exception as e:
            logger.error(f"Frame translation error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e)
            }))

    async def _handle_multimodal_translation(self, websocket, data: Dict):
        """Handle multimodal translation: combine hand, facial, and grammar"""
        try:
            user_id = data.get("user_id", 1)
            session_token = data.get("session_token")

            result = self.asl_service.translate_multimodal(
                user_id=user_id,
                face_output=data.get("facial_data", {}),
                hand_output=data.get("hand_data", {}),
                grammar_state=data.get("grammar_data", {}),
                input_type="webcam",
                input_source="signer_view",
                session_token=session_token
            )

            await websocket.send(json.dumps({
                "type": "translation_result",
                "ok": result.get("ok"),
                "translation": result.get("translation"),
                "error": result.get("error")
            }))
        except Exception as e:
            logger.error(f"Multimodal translation error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e)
            }))

    async def start(self):
        """Start WebSocket server"""
        await self.initialize()

        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        server = await websockets.serve(self.handle_client, self.host, self.port)
        logger.info(f"Server running on ws://{self.host}:{self.port}")

        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Server shutting down")
        finally:
            server.close()
            await server.wait_closed()


async def main():
    """Main entry point"""
    server = WebSocketServer(host="0.0.0.0", port=8765)
    await server.start()


if __name__ == "__main__":
    asyncio.run(main())
