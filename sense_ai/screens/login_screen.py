# screens/login_screen.py

import re
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk
from components.status_bar import StatusBar
from theme import COLORS, FONTS


class LoginScreen(tk.Frame):
    """Login screen with validation and connected navigation."""

    SCREEN_NAME = "login"
    EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate
        self._layout_mode = None

        self.bind("<Configure>", self._on_resize)

        self.container = tk.Frame(self, bg=COLORS["cream"])
        self.container.pack(fill=tk.BOTH, expand=True)

        self.header = tk.Frame(self.container, bg=COLORS["navy"], height=160)
        self.header.pack_propagate(False)
        self.header_bg_label = tk.Label(self.header, bg=COLORS["navy"], bd=0)
        self.header_bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.header_title = tk.Label(
            self.header,
            text="Sense.AI",
            font=("Georgia", 36, "bold"),
            bg=COLORS["navy"],
            fg=COLORS["white"],
        )
        self.header_title.place(x=28, y=36, anchor="nw")
        self.header_subtitle = tk.Label(
            self.header,
            text="Translate (Signer) · Speak (Speaker)",
            font=("Helvetica", 14),
            bg=COLORS["navy"],
            fg=COLORS["teal"],
        )
        self.header_subtitle.place(x=28, y=80, anchor="nw")
        self._header_source_image = None
        self._header_photo = None
        self._load_header_image()
        self.header.bind("<Configure>", self._update_header_image)

        self.card = tk.Frame(
            self.container,
            bg=COLORS["white"],
            bd=0,
            relief=tk.FLAT,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        tk.Label(
            self.card,
            text="Welcome Back",
            font=("Helvetica", 22, "bold"),
            bg=COLORS["white"],
            fg=COLORS["black"],
        ).pack(anchor="w", padx=32, pady=(20, 2))

        tk.Label(
            self.card,
            text="Sign in to continue",
            font=("Helvetica", 14),
            bg=COLORS["white"],
            fg=COLORS["text_lt"],
        ).pack(anchor="w", padx=32, pady=(0, 8))
        
        tk.Label(self.card, text="EMAIL", font=FONTS["label"], bg=COLORS["white"], fg=COLORS["muted"]).pack(anchor="w", padx=32, pady=(24, 4))
        self.email_entry = tk.Entry(self.card, font=FONTS["body"], bg=COLORS["cream"], relief=tk.FLAT, bd=0, highlightbackground=COLORS["border"], highlightthickness=1)
        self.email_entry.pack(fill=tk.X, padx=32, pady=(0, 12), ipady=8)

        tk.Label(self.card, text="PASSWORD", font=FONTS["label"], bg=COLORS["white"], fg=COLORS["muted"]).pack(anchor="w", padx=32, pady=(0, 4))
        self.password_entry = tk.Entry(self.card, font=FONTS["body"], bg=COLORS["cream"], relief=tk.FLAT, bd=0, highlightbackground=COLORS["border"], highlightthickness=1, show="•")
        self.password_entry.pack(fill=tk.X, padx=32, pady=(0, 8), ipady=8)

        self.feedback_label = tk.Label(
            self.card,
            text="",
            font=FONTS["small"],
            bg=COLORS["white"],
            fg=COLORS["danger"],
            justify=tk.LEFT,
            wraplength=320,
        )
        self.feedback_label.pack(anchor="w", padx=32, pady=(0, 8))

        forgot_frame = tk.Frame(self.card, bg=COLORS["white"])
        forgot_frame.pack(fill=tk.X, padx=32, pady=(0, 8))
        tk.Label(forgot_frame, text="Forgot password?", font=FONTS["small"], bg=COLORS["white"], fg=COLORS["teal"], cursor="hand2").pack(side=tk.RIGHT)

        tk.Button(
            self.card,
            text="Sign In",
            font=FONTS["body_bold"],
            bg=COLORS["navy"],
            fg=COLORS["white"],
            relief=tk.FLAT,
            height=2,
            cursor="hand2",
            command=self._on_login,
            borderwidth=0,
            highlightbackground=COLORS["navy"],
            highlightthickness=0,
        ).pack(fill=tk.X, padx=32, pady=(0, 12), ipady=8)

        divider_frame = tk.Frame(self.card, bg=COLORS["white"])
        divider_frame.pack(fill=tk.X, padx=32, pady=(0, 8))
        divider_left = tk.Frame(divider_frame, bg=COLORS["border"], height=1)
        divider_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), pady=8)
        tk.Label(divider_frame, text="or", font=FONTS["small"], bg=COLORS["white"], fg=COLORS["muted"]).pack(side=tk.LEFT)
        divider_right = tk.Frame(divider_frame, bg=COLORS["border"], height=1)
        divider_right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), pady=8)

        tk.Button(
            self.card,
            text="Create an account",
            font=FONTS["body_bold"],
            bg=COLORS["white"],
            fg=COLORS["teal"],
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda: self.navigate("signup"),
            borderwidth=1,
            highlightbackground=COLORS["teal"],
            highlightthickness=1,
        ).pack(fill=tk.X, padx=32, pady=(0, 8), ipady=8)

        tk.Label(
            self.card,
            text="By signing in you agree to our Terms of Use",
            font=FONTS["tiny"],
            bg=COLORS["white"],
            fg=COLORS["muted"],
        ).pack(pady=(0, 8))

        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Demo login available · demo@sense.ai / Sense1234")

        self.after_idle(self._apply_layout)

    def _apply_layout(self):
        width = max(self.winfo_width(), self.winfo_reqwidth())
        mode = "desktop" if width >= 800 else "mobile"
        if mode == self._layout_mode:
            if mode == "desktop":
                self._update_desktop_split()
            return

        self._layout_mode = mode
        for widget in self.container.winfo_children():
            widget.pack_forget()

        if mode == "desktop":
            self.header.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, pady=32)
            self.card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 32), pady=32)
            self._update_desktop_split()
        else:
            self.header.pack(side=tk.TOP, fill=tk.X)
            self.card.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

    def _update_desktop_split(self):
        container_width = max(self.container.winfo_width(), self.container.winfo_reqwidth(), 1)
        self.header.configure(width=int(container_width * (2 / 3)))

    def _load_header_image(self):
        asset_path = Path(__file__).resolve().parent.parent / "assets" / "darkNavy_bg.webp"
        try:
            self._header_source_image = Image.open(asset_path)
            self._update_header_image()
        except Exception:
            self._header_source_image = None

    def _update_header_image(self, _event=None):
        if self._header_source_image is None:
            return

        width = max(self.header.winfo_width(), 1)
        height = max(self.header.winfo_height(), 1)
        resample = getattr(Image, "Resampling", Image).LANCZOS
        resized = self._header_source_image.resize((width, height), resample)
        self._header_photo = ImageTk.PhotoImage(resized)
        self.header_bg_label.config(image=self._header_photo)

    def _on_resize(self, _event):
        self._apply_layout()

    def _set_feedback(self, message: str, is_error: bool = True):
        self.feedback_label.config(text=message, fg=COLORS["danger"] if is_error else COLORS["teal"])
        self.status_bar.set_text(f"Status: {message}")

    def _set_error(self, message: str):
        self._set_feedback(message, is_error=True)

    def _validate_credentials(self, email: str, password: str):
        if not email or not password:
            return False, "Enter both email and password."
        if not self.EMAIL_PATTERN.match(email):
            return False, "Enter a valid email address."
        return True, ""

    def _on_login(self):
        email = self.email_entry.get().strip()
        password = self.password_entry.get().strip()

        ok, msg = self._validate_credentials(email, password)
        if not ok:
            self._set_error(msg)
            return

        # Use AppState wrapper (handles backend and username/email compatibility)
        success, message = self.state.authenticate_user(email, password)
        if success:
            self._set_feedback(message, is_error=False)
            self.navigate("session")
        else:
            self._set_error(message)

    def on_show(self):
        self._apply_layout()
        self.feedback_label.config(text="")
        self.email_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.status_bar.set_live(False)
        self.status_bar.set_text("Status: Enter your account credentials")
