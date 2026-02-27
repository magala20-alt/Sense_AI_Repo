# screens/signer_screen.py

import tkinter as tk
import cv2
from PIL import Image, ImageTk
import threading
from components.window_bar import WindowBar
from components.status_bar import StatusBar
from components.sidebar import Sidebar
from components.grammar_tag import GrammarTag
from components.detection_chips import DetectionChips
from components.conv_history import ConversationHistory
from services.websocket_client import create_websocket_client
from config import BACKEND_WS_URL, CAMERA_INDEX
from theme import COLORS, FONTS


class SignerScreen(tk.Frame):
    """Main screen for Deaf signers — camera capture + translation + history."""

    SCREEN_NAME = "signer"

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate

        # Top container: sidebar + main
        top_container = tk.Frame(self, bg=COLORS["cream"])
        top_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Sidebar
        def on_logout():
            self.stop_camera()
            self.stop_websocket()
            self.navigate("welcome")

        self.sidebar = Sidebar(top_container, role="signer", on_logout=on_logout)

        # Main content area
        self.main_frame = tk.Frame(top_container, bg=COLORS["cream"])
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Window bar
        WindowBar(self.main_frame, title="Sense.AI Signer", highlight="Sense")

        # Scrollable content
        canvas = tk.Canvas(self.main_frame, bg=COLORS["cream"], highlightthickness=0)
        scrollbar = tk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg=COLORS["cream"])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Camera Section ---
        self.camera_label = tk.Label(self.scrollable_frame, bg=COLORS["navy"], width=45, height=9)
        self.camera_label.pack(padx=10, pady=10)

        # Overlay canvas for detection dots
        self.camera_frame = tk.Frame(self.scrollable_frame, width=430, height=160, bg=COLORS["cream"])
        self.camera_frame.pack(padx=10, pady=0)

        self.overlay_canvas = tk.Canvas(self.camera_frame, bg=COLORS["navy"], highlightthickness=0, width=430, height=160)
        self.overlay_canvas.pack()

        # Listening indicator
        indicator_frame = tk.Frame(self.scrollable_frame, bg=COLORS["navy_dark"], height=28)
        indicator_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        indicator_frame.pack_propagate(False)

        # Pulsing dot
        self.status_canvas = tk.Canvas(indicator_frame, bg=COLORS["navy_dark"], highlightthickness=0, width=20, height=28)
        self.status_canvas.pack(side=tk.LEFT, padx=6)
        self.status_canvas.create_oval(6, 11, 14, 19, fill=COLORS["success"], outline=COLORS["success"])
        self.pulse_on = True

        # Status text
        tk.Label(indicator_frame, text="Listening… · Facial detection active", font=FONTS["small"], bg=COLORS["navy_dark"], fg=COLORS["teal_lt"]).pack(side=tk.LEFT, padx=6)

        # Detection chips
        self.detection_chips = DetectionChips(self.scrollable_frame)

        # --- Translation Output ---
        trans_frame = tk.Frame(self.scrollable_frame, bg=COLORS["white"], relief=tk.SOLID, bd=1, highlightbackground=COLORS["teal"], highlightthickness=1)
        trans_frame.pack(fill=tk.X, padx=10, pady=10)

        # Header
        trans_header = tk.Frame(trans_frame, bg=COLORS["teal_bg"], pady=4)
        trans_header.pack(fill=tk.X)

        tk.Label(trans_header, text="TEXT TRANSLATION", font=FONTS["label"], fg=COLORS["teal"], bg=COLORS["teal_bg"]).pack(side=tk.LEFT, padx=6)
        self.grammar_tag = GrammarTag(trans_header, "")
        self.grammar_tag.pack(side=tk.RIGHT, padx=6)

        # Translation text
        self.translation_label = tk.Label(
            trans_frame, text="",
            font=("Helvetica", 15, "bold"),
            fg=COLORS["navy"],
            bg=COLORS["white"],
            wraplength=410,
            justify=tk.CENTER,
            pady=12
        )
        self.translation_label.pack(fill=tk.X, padx=6, pady=6)

        # --- Conversation History ---
        self.conv_history = ConversationHistory(self.scrollable_frame)

        # Status bar
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Listening")

        # Camera and WebSocket
        self.cap = None
        self.camera_running = False
        self.websocket_client = None
        self.pulse_id = None

    def on_show(self):
        """Initialize when screen becomes visible."""
        self.state.clear_history()
        self.conv_history.clear_history()
        self.start_camera()
        self.start_websocket()
        self.status_bar.set_live(True)
        self._pulse_status()

    def start_camera(self):
        """Start camera capture in background thread."""
        if self.camera_running:
            return
        
        self.camera_running = True
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        thread = threading.Thread(target=self._camera_loop, daemon=True)
        thread.start()

    def _camera_loop(self):
        """Continuously read and display camera frames."""
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Resize for display
            frame = cv2.resize(frame, (430, 160))
            
            # Convert BGR → RGB → PIL → PhotoImage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update UI (thread-safe)
            self.after(0, lambda p=photo: self._update_camera_display(p))
            
            # ~30fps
            threading.Event().wait(0.033)

    def _update_camera_display(self, photo):
        """Update camera label with new frame."""
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo  # Keep a reference

    def start_websocket(self):
        """Connect to ASL backend via WebSocket."""
        if self.websocket_client:
            return
        
        self.websocket_client = create_websocket_client(BACKEND_WS_URL, self._on_websocket_message)

    def _on_websocket_message(self, translation: str, grammar_type: str, tier_scores: dict):
        """Handle incoming WebSocket message."""
        self.after(0, lambda: self._handle_translation(translation, grammar_type, tier_scores))

    def _handle_translation(self, translation: str, grammar_type: str, tier_scores: dict):
        """Update UI with translation data (called from main thread)."""
        self.state.current_translation = translation
        self.state.current_grammar = grammar_type
        self.state.tier_scores = tier_scores
        
        # Update translation label
        self.translation_label.config(text=translation)
        
        # Update grammar tag
        self.grammar_tag.update_grammar(grammar_type)
        
        # Update detection chips
        self.detection_chips.update_chips(
            tier_scores.get("physical", 0),
            tier_scores.get("grammar", 0),
            tier_scores.get("semantic", 0)
        )
        
        # Add to history
        if translation:
            self.state.add_message("You", translation, grammar_type)
            self.conv_history.add_entry("You", translation, grammar_type)

    def stop_camera(self):
        """Stop camera capture."""
        self.camera_running = False
        if self.cap:
            self.cap.release()

    def stop_websocket(self):
        """Stop WebSocket client."""
        if self.websocket_client:
            self.websocket_client.stop()

    def _pulse_status(self):
        """Pulse the status dot."""
        self.pulse_on = not self.pulse_on
        color = COLORS["success"] if self.pulse_on else COLORS["navy_dark"]
        self.status_canvas.delete("all")
        self.status_canvas.create_oval(6, 11, 14, 19, fill=color, outline=color)
        self.pulse_id = self.after(800, self._pulse_status)

    def __del__(self):
        """Cleanup when screen is destroyed."""
        self.stop_camera()
        self.stop_websocket()
        if self.pulse_id:
            self.after_cancel(self.pulse_id)
