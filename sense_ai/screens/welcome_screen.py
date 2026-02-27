# screens/welcome_screen.py

import tkinter as tk
from components.window_bar import WindowBar
from components.status_bar import StatusBar
from theme import COLORS, FONTS, APP_WIDTH, APP_HEIGHT


class WelcomeScreen(tk.Frame):
    """Welcome screen with role selection (Signer vs Speaker)."""

    SCREEN_NAME = "welcome"

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate

        # Window bar
        WindowBar(self, title="Sense.AI Login", highlight="Sense")

        # Body
        body = tk.Frame(self, bg=COLORS["cream"])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Emoji logo
        tk.Label(body, text="🤟", font=("Helvetica", 32), bg=COLORS["cream"], fg=COLORS["text"]).pack(pady=8)

        # Title
        tk.Label(body, text="Welcome to",
                font=("Helvetica", 18, "bold"), bg=COLORS["cream"], fg=COLORS["navy"]).pack()
        tk.Label(body, text="Sense.AI",
                font=("Helvetica", 22, "bold"), bg=COLORS["cream"], fg=COLORS["teal"]).pack()

        # Subtitle
        tk.Label(body, text="Your ASL communication bridge",
                font=FONTS["small"], bg=COLORS["cream"], fg=COLORS["muted"]).pack(pady=4)

        # Spacer
        tk.Frame(body, bg=COLORS["cream"], height=10).pack()

        # Login button
        tk.Button(
            body,
            text="Login",
            font=FONTS["body_bold"],
            bg=COLORS["teal"],
            fg=COLORS["white"],
            relief=tk.FLAT,
            height=2,
            cursor="hand2",
            command=lambda: self.navigate("login")
        ).pack(fill=tk.X, pady=4)

        # Create Account button
        tk.Button(
            body,
            text="Create Account",
            font=FONTS["body_bold"],
            bg=COLORS["white"],
            fg=COLORS["teal"],
            relief=tk.SOLID,
            bd=2,
            height=2,
            cursor="hand2",
            command=lambda: self.navigate("signup")
        ).pack(fill=tk.X, pady=4)

        # Divider with text
        divider_frame = tk.Frame(body, bg=COLORS["cream"])
        divider_frame.pack(fill=tk.X, pady=12)
        tk.Label(divider_frame, text="Continue As", font=FONTS["small"], bg=COLORS["cream"], fg=COLORS["muted"]).pack()

        # Role card — Signer
        signer_card = tk.Frame(body, bg=COLORS["teal_bg"], relief=tk.SOLID, bd=1, highlightbackground=COLORS["border"], highlightthickness=1)
        signer_card.pack(fill=tk.X, pady=4)
        signer_card_inner = tk.Frame(signer_card, bg=COLORS["teal_bg"])
        signer_card_inner.pack(fill=tk.X, padx=6, pady=6)

        # Icon area
        icon_frame = tk.Frame(signer_card_inner, bg=COLORS["teal"], width=40, height=40)
        icon_frame.pack(side=tk.LEFT, padx=(0, 12))
        icon_frame.pack_propagate(False)
        tk.Label(icon_frame, text="🤟", font=("Helvetica", 20), bg=COLORS["teal"], fg=COLORS["white"]).pack(expand=True)

        # Text area
        tk.Label(signer_card_inner, text="I use ASL (Signer)", font=FONTS["body"], bg=COLORS["teal_bg"], fg=COLORS["navy"]).pack(side=tk.LEFT, anchor="w", expand=True)

        # Bind click event
        def on_signer_click(event=None):
            self.state.user_role = "signer"
            self.navigate("session")

        signer_card.bind("<Button-1>", on_signer_click)
        for widget in signer_card.winfo_children():
            widget.bind("<Button-1>", on_signer_click)

        # Role card — Speaker
        speaker_card = tk.Frame(body, bg=COLORS["teal_bg"], relief=tk.SOLID, bd=1, highlightbackground=COLORS["border"], highlightthickness=1)
        speaker_card.pack(fill=tk.X, pady=4)
        speaker_card_inner = tk.Frame(speaker_card, bg=COLORS["teal_bg"])
        speaker_card_inner.pack(fill=tk.X, padx=6, pady=6)

        # Icon area
        icon_frame2 = tk.Frame(speaker_card_inner, bg=COLORS["gold"], width=40, height=40)
        icon_frame2.pack(side=tk.LEFT, padx=(0, 12))
        icon_frame2.pack_propagate(False)
        tk.Label(icon_frame2, text="🗣️", font=("Helvetica", 20), bg=COLORS["gold"], fg=COLORS["white"]).pack(expand=True)

        # Text area
        tk.Label(speaker_card_inner, text="I Speak / Hear", font=FONTS["body"], bg=COLORS["teal_bg"], fg=COLORS["navy"]).pack(side=tk.LEFT, anchor="w", expand=True)

        # Bind click event
        def on_speaker_click(event=None):
            self.state.user_role = "speaker"
            self.navigate("session")

        speaker_card.bind("<Button-1>", on_speaker_click)
        for widget in speaker_card.winfo_children():
            widget.bind("<Button-1>", on_speaker_click)

        # Status bar
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Ready")
        self.status_bar.set_live(True)

    def on_show(self):
        """Called when screen is shown."""
        self.status_bar.set_live(True)
