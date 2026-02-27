# screens/speaker_screen.py

import tkinter as tk
import threading
from components.window_bar import WindowBar
from components.status_bar import StatusBar
from components.sidebar import Sidebar
from components.face_avatar import FaceAvatar
from components.conv_history import ConversationHistory
from services.sigml_sender import get_sigml_sender
from theme import COLORS, FONTS


class SpeakerScreen(tk.Frame):
    """Main screen for hearing speakers — text input + avatar signing + history."""

    SCREEN_NAME = "speaker"

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate

        # Top container: sidebar + main
        top_container = tk.Frame(self, bg=COLORS["cream"])
        top_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Sidebar
        def on_logout():
            self.navigate("welcome")

        self.sidebar = Sidebar(top_container, role="speaker", on_logout=on_logout)

        # Main content area
        self.main_frame = tk.Frame(top_container, bg=COLORS["cream"])
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Window bar
        WindowBar(self.main_frame, title="Sense.AI Speaker", highlight="Sense")

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

        # --- Text Input Section ---
        input_frame = tk.Frame(self.scrollable_frame, bg=COLORS["white"], relief=tk.SOLID, bd=1, highlightbackground=COLORS["gold"], highlightthickness=1)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Header
        input_header = tk.Frame(input_frame, bg=COLORS["teal_bg"], pady=4)
        input_header.pack(fill=tk.X)
        tk.Label(input_header, text="✏️ TEXT INPUT", font=FONTS["label"], fg=COLORS["gold"], bg=COLORS["teal_bg"]).pack(side=tk.LEFT, padx=6)

        # Input row
        input_row = tk.Frame(input_frame, bg=COLORS["white"])
        input_row.pack(fill=tk.X, padx=6, pady=6)

        self.text_entry = tk.Entry(input_row, font=FONTS["body"], bg=COLORS["white"], relief=tk.SOLID, bd=1)
        self.text_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.text_entry.bind("<Return>", lambda e: self._on_sign_button())

        sign_btn = tk.Button(
            input_row,
            text="→ Sign",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["teal"],
            fg=COLORS["white"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._on_sign_button,
            padx=12
        )
        sign_btn.pack(side=tk.LEFT)

        # Avatar status
        self.avatar_status_label = tk.Label(
            input_frame, text="🤟 Avatar: Ready",
            font=FONTS["small"], fg=COLORS["success"],
            bg=COLORS["white"]
        )
        self.avatar_status_label.pack(anchor="w", padx=6, pady=(0, 6))

        # --- Speak Button ---
        speak_btn = tk.Button(
            self.scrollable_frame,
            text="🎤\nTap to Speak\nVoice → ASL translation",
            font=("Helvetica", 13, "bold"),
            bg=COLORS["gold"],
            fg=COLORS["navy"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._on_speak_button,
            justify=tk.CENTER,
            pady=14
        )
        speak_btn.pack(fill=tk.X, padx=10, pady=6)

        # --- Avatar Display Panel ---
        avatar_panel = tk.Frame(self.scrollable_frame, bg=COLORS["navy"], relief=tk.SOLID, bd=1)
        avatar_panel.pack(padx=10, pady=10, fill=tk.X)

        avatar_inner = tk.Frame(avatar_panel, bg=COLORS["navy"])
        avatar_inner.pack(padx=12, pady=12)

        # Avatar canvas (centered)
        avatar_container = tk.Frame(avatar_inner, bg=COLORS["navy"])
        avatar_container.pack(expand=True)

        self.face_avatar = FaceAvatar(avatar_container, size=80)
        self.face_avatar.pack()

        tk.Label(avatar_inner, text="AVATAR (ASL Signing)",
                font=FONTS["tiny"], fg=COLORS["teal"],
                bg=COLORS["navy"]).pack(pady=(8, 4))

        self.avatar_phrase_label = tk.Label(
            avatar_inner, text="Ready to sign",
            font=FONTS["body"], fg=COLORS["white"],
            bg=COLORS["navy"], wraplength=140, justify=tk.CENTER
        )
        self.avatar_phrase_label.pack(pady=4)

        # --- Conversation History ---
        self.conv_history = ConversationHistory(self.scrollable_frame)

        # Status bar
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Ready")

        # SIGML sender
        self.sigml_sender = get_sigml_sender()

    def on_show(self):
        """Initialize when screen becomes visible."""
        self.state.clear_history()
        self.conv_history.clear_history()
        self.status_bar.set_live(False)
        self.status_bar.set_text("Status: Ready")
        self.text_entry.focus()

    def _on_sign_button(self):
        """Handle Sign button press."""
        text = self.text_entry.get().strip()
        if not text:
            return
        
        # Update status
        self.avatar_status_label.config(text="🤟 Avatar: Signing...", fg=COLORS["gold"])
        self.avatar_phrase_label.config(text=text)
        
        # Add to history
        self.state.add_message("You", text)
        self.conv_history.add_entry("You", text)

        def on_done():
            self.after(0, lambda: self.avatar_status_label.config(text="🤟 Avatar: Ready", fg=COLORS["success"]))
        
        # Send to avatar (runs in background thread)
        self.sigml_sender.speak_to_avatar(text, on_done)
        
        # Clear input
        self.text_entry.delete(0, tk.END)

    def _on_speak_button(self):
        """Handle Speak button press."""
        self.avatar_status_label.config(text="🎤 Listening...", fg=COLORS["gold"])
        
        # Demo: add mock spoken text
        def mock_speech():
            import time
            time.sleep(2)
            mock_text = "Hello, how are you?"
            self.after(0, lambda: self._handle_speech_result(mock_text))
        
        threading.Thread(target=mock_speech, daemon=True).start()

    def _handle_speech_result(self, text: str):
        """Process speech recognition result."""
        self.text_entry.delete(0, tk.END)
        self.text_entry.insert(0, text)
        self.avatar_status_label.config(text="🤟 Avatar: Ready", fg=COLORS["success"])
        self.text_entry.focus()
