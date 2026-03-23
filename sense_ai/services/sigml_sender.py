# services/sigml_sender.py — send SiGML commands to CWASA avatar

import socket
import threading
import time
from typing import List, Optional
import logging


class SiGMLSender:
    """
    Sends SiGML XML commands to CWASA SiGML Player over TCP socket.
    
    Args:
      host: str — CWASA player host (default 127.0.0.1)
      port: int — CWASA player port (default 4569)
    
    Methods:
      send_sigml_text(sigml_xml: str) -> bool
      send_sigml_file(filepath: str) -> bool
      sign_gloss(gloss: str) -> bool
      speak_to_avatar(text: str, on_done=None)  — convert text to glosses and sign
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4569):
        self.host = host
        self.port = port
        self.connected = False
        self.socket = None
        self.last_error = ""
        self.logger = logging.getLogger(__name__)

    def _connect(self) -> bool:
        """Connect to CWASA player."""
        try:
            # Always recreate socket for a clean connection attempt.
            if self.socket is not None:
                try:
                    self.socket.close()
                except Exception:
                    pass
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(2)
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.last_error = ""
            return True
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"SIGML connection error ({self.host}:{self.port}): {e}")
            self.connected = False
            self._disconnect()
            return False

    def _disconnect(self):
        """Disconnect from CWASA player."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False

    def send_sigml_text(self, sigml_xml: str) -> bool:
        """Send SiGML XML string to CWASA player."""
        if not self._connect():
            self.logger.error(f"Could not connect to CWASA player at {self.host}:{self.port}")
            return False
        
        try:
            self.socket.sendall(sigml_xml.encode('utf-8'))
            self._disconnect()
            return True
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Error sending SIGML: {e}")
            self._disconnect()
            return False

    def test_connection(self) -> bool:
        """Lightweight connection check for UI diagnostics."""
        ok = self._connect()
        self._disconnect()
        return ok

    def send_sigml_file(self, filepath: str) -> bool:
        """Load .sigml file and send to CWASA player."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                sigml_xml = f.read()
            return self.send_sigml_text(sigml_xml)
        except Exception as e:
            print(f"Error loading SIGML file: {e}")
            return False

    def sign_gloss(self, gloss: str) -> bool:
        """Sign a single gloss. e.g. sign_gloss('HELLO')"""
        # Build minimal SiGML command
        sigml_xml = f'<sigml>{gloss}</sigml>'
        return self.send_sigml_text(sigml_xml)

    def sign_sentence(self, glosses: List[str], delay: float = 1.2):
        """Sign a list of glosses with delay between each (runs in thread)."""
        def _sign():
            for gloss in glosses:
                self.sign_gloss(gloss)
                time.sleep(delay)
        
        thread = threading.Thread(target=_sign, daemon=True)
        thread.start()

    def speak_to_avatar(self, text: str, on_done=None):
        """
        Convert English text → gloss list → send to CWASA.
        Runs in background thread so tkinter UI stays responsive.
        on_done: optional callback when signing finishes.
        
        Demo implementation: splits text on spaces and uses uppercase as glosses.
        """
        def _speak():
            try:
                # Simple demo: split text into words and use as glosses
                words = text.upper().split()
                
                # Filter to alphanumeric glosses
                glosses = [w for w in words if any(c.isalnum() for c in w)]
                
                if glosses:
                    self.sign_sentence(glosses, delay=0.8)
                    time.sleep(len(glosses) * 0.8)
                
                if on_done:
                    on_done()
            except Exception as e:
                print(f"Error in speak_to_avatar: {e}")
                if on_done:
                    on_done()
        
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()


# Global instance
_sigml_sender = None


def get_sigml_sender(host: str = "127.0.0.1", port: int = 4569) -> SiGMLSender:
    """Get or create the global SiGML sender instance."""
    global _sigml_sender
    if _sigml_sender is None:
        _sigml_sender = SiGMLSender(host, port)
    return _sigml_sender
