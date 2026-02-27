# screens/session_screen.py

import tkinter as tk
from components.window_bar import WindowBar
from components.status_bar import StatusBar
from theme import COLORS, FONTS


class SessionScreen(tk.Frame):
    """Session ID entry screen before launching Signer or Speaker mode."""

    SCREEN_NAME = "session"

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate

        # Window bar
        WindowBar(self, title="Sense.AI · Session")

        # Body — centered dark panel
        body = tk.Frame(self, bg=COLORS["cream"])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Centering frame
        center_frame = tk.Frame(body, bg=COLORS["cream"])
        center_frame.pack(expand=True, fill=tk.BOTH)

        # Dark panel (navy background)
        panel = tk.Frame(center_frame, bg=COLORS["navy_dark"], relief=tk.SOLID, bd=1, highlightbackground=COLORS["border"], highlightthickness=1)
        panel.pack(expand=True, ipady=20, ipadx=20)

        # Panel content
        tk.Label(panel, text="JOIN A SESSION", font=FONTS["label"],
                bg=COLORS["navy_dark"], fg=COLORS["teal_lt"]).pack(pady=8)
        tk.Label(panel, text="Enter Conversation ID", font=("Helvetica", 16, "bold"),
                bg=COLORS["navy_dark"], fg=COLORS["white"]).pack(pady=8)

        # Session ID entry
        self.session_entry = tk.Entry(panel, font=FONTS["body"], bg=COLORS["navy"], fg=COLORS["white"], relief=tk.SOLID, bd=1, width=32, justify=tk.CENTER)
        self.session_entry.pack(fill=tk.X, pady=12, ipady=6)
        self.session_entry.insert(0, "Session-123456")  # Demo value

        # Start button
        tk.Button(
            panel,
            text="Start Conversation →",
            font=FONTS["body_bold"],
            bg=COLORS["gold"],
            fg=COLORS["navy"],
            relief=tk.FLAT,
            height=2,
            cursor="hand2",
            command=self._on_start
        ).pack(fill=tk.X, pady=12)

        # Hint
        tk.Label(panel, text="Don't have an ID? Ask your partner.",
                font=FONTS["small"], bg=COLORS["navy_dark"], fg=COLORS["muted"]).pack(pady=4)

        # Bottom buttons (side by side)
        button_frame = tk.Frame(body, bg=COLORS["cream"])
        button_frame.pack(fill=tk.X, pady=12)

        tk.Button(
            button_frame,
            text="🤟\nSigner View",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["teal"],
            fg=COLORS["white"],
            relief=tk.FLAT,
            height=3,
            cursor="hand2",
            command=lambda: self._set_role_and_navigate("signer")
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Button(
            button_frame,
            text="🗣️\nSpeaker View",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["gold"],
            fg=COLORS["navy"],
            relief=tk.FLAT,
            height=3,
            cursor="hand2",
            command=lambda: self._set_role_and_navigate("speaker")
        ).pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        # Status bar
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Waiting…")

    def _on_start(self):
        """Start conversation with the entered session ID."""
        session_id = self.session_entry.get().strip()
        if session_id:
            self.state.session_id = session_id
            # Determine which role to navigate to based on state
            if self.state.user_role == "signer":
                self.navigate("signer")
            elif self.state.user_role == "speaker":
                self.navigate("speaker")
            else:
                # If no role set yet, default to signer
                self.state.user_role = "signer"
                self.navigate("signer")

    def _set_role_and_navigate(self, role: str):
        """Set role and navigate to the appropriate screen."""
        self.state.user_role = role
        session_id = self.session_entry.get().strip()
        if session_id:
            self.state.session_id = session_id
        self.navigate(role)

    def on_show(self):
        """Called when screen is shown."""
        self.status_bar.set_live(False)
        self.status_bar.set_text("Status: Waiting…")
