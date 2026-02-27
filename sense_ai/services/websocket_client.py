# services/websocket_client.py — WebSocket client for ASL backend

import asyncio
import websockets
import json
import threading
import time


class WebSocketClient:
    """
    Connects to Python ASL backend at ws://localhost:8765
    Runs in a daemon background thread.
    
    Args:
      url: str — websocket URL from config.py
      on_message: callable(translation, grammar_type, tier_scores)
    
    Incoming JSON format:
      {
        "translation": "Are you coming?",
        "grammar_type": "YES_NO_QUESTION",
        "tier_scores": { "physical": 88, "grammar": 75, "semantic": 69 }
      }
    
    Methods:
      start()  — begin connection in background thread
      stop()   — disconnect cleanly
    
    Auto-reconnects every 3 seconds if disconnected.
    
    Demo/offline mode: if connection fails after 5s,
    emit mock data every 3 seconds.
    """

    def __init__(self, url: str, on_message=None):
        self.url = url
        self.on_message = on_message or (lambda *args, **kwargs: None)
        self.running = False
        self.connected = False
        self.thread = None
        self.is_demo_mode = False

    def start(self):
        """Start WebSocket client in background thread."""
        if self.thread and self.thread.is_alive():
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop WebSocket client."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _run(self):
        """Main loop running asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            loop.close()

    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages."""
        retry_count = 0
        max_retries = 2  # Try to connect twice (5 seconds total)
        
        while self.running:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=10) as websocket:
                    self.connected = True
                    self.is_demo_mode = False
                    retry_count = 0
                    print(f"Connected to {self.url}")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            translation = data.get("translation", "")
                            grammar_type = data.get("grammar_type", "")
                            tier_scores = data.get("tier_scores", {"physical": 0, "grammar": 0, "semantic": 0})
                            self.on_message(translation, grammar_type, tier_scores)
                        except json.JSONDecodeError:
                            print(f"Invalid JSON: {message}")
            
            except (ConnectionRefusedError, OSError) as e:
                self.connected = False
                retry_count += 1
                
                if retry_count >= max_retries and not self.is_demo_mode:
                    print(f"Failed to connect to {self.url}, entering demo mode")
                    self.is_demo_mode = True
                    await self._demo_mode()
                else:
                    if not self.is_demo_mode:
                        print(f"Connection failed, retrying in 3s... ({retry_count}/{max_retries})")
                        await asyncio.sleep(3)
            
            except Exception as e:
                print(f"Unexpected error: {e}")
                self.connected = False
                await asyncio.sleep(3)

    async def _demo_mode(self):
        """Emit mock data every 3 seconds in demo mode."""
        demo_messages = [
            {
                "translation": "Are you coming?",
                "grammar_type": "YES_NO_QUESTION",
                "tier_scores": {"physical": 88, "grammar": 75, "semantic": 69}
            },
            {
                "translation": "What time is it?",
                "grammar_type": "WH_QUESTION",
                "tier_scores": {"physical": 92, "grammar": 80, "semantic": 78}
            },
            {
                "translation": "I am happy.",
                "grammar_type": "STATEMENT",
                "tier_scores": {"physical": 85, "grammar": 72, "semantic": 80}
            },
            {
                "translation": "Don't go.",
                "grammar_type": "NEGATION",
                "tier_scores": {"physical": 79, "grammar": 68, "semantic": 75}
            },
        ]
        
        msg_index = 0
        while self.running and self.is_demo_mode:
            try:
                msg = demo_messages[msg_index % len(demo_messages)]
                self.on_message(
                    msg["translation"],
                    msg["grammar_type"],
                    msg["tier_scores"]
                )
                msg_index += 1
                await asyncio.sleep(3)
            except Exception as e:
                print(f"Demo mode error: {e}")
                await asyncio.sleep(3)


# Convenience function
def create_websocket_client(url: str, on_message=None):
    """Create and start a WebSocket client."""
    client = WebSocketClient(url, on_message)
    client.start()
    return client
