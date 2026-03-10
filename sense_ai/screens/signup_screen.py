# screens/signup_screen.py

import tkinter as tk
import re

from components.status_bar import StatusBar
from theme import COLORS, FONTS


class SignUpScreen(tk.Frame):
    """Sign Up screen with email, password, and confirm password fields."""

    SCREEN_NAME = "signup"
    EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate
        self._layout_mode = None

        # Responsive layout logic
        self.bind('<Configure>', self._on_resize)

        # Main container
        self.container = tk.Frame(self, bg=COLORS["cream"])
        self.container.pack(fill=tk.BOTH, expand=True)

        # Card
        self.card = tk.Frame(self.container, bg=COLORS["white"], bd=0, relief=tk.FLAT, highlightbackground=COLORS["border"], highlightthickness=1)

        # Emoji
        tk.Label(self.card, text="✨", font=("Helvetica", 32), bg=COLORS["white"]).pack(pady=(18, 6))

        # Heading
        tk.Label(self.card, text="Create account", font=FONTS["heading"], bg=COLORS["white"], fg=COLORS["navy"]).pack(pady=(0, 8))

        # Full Name field
        tk.Label(self.card, text="FULL NAME", font=FONTS["label"], bg=COLORS["white"], fg=COLORS["muted"]).pack(anchor="w", padx=32, pady=(12, 4))
        self.name_entry = tk.Entry(self.card, font=FONTS["body"], bg=COLORS["cream"], relief=tk.FLAT, bd=0, highlightbackground=COLORS["border"], highlightthickness=1)
        self.name_entry.pack(fill=tk.X, padx=32, pady=(0, 12), ipady=8)

        # Email field
        tk.Label(self.card, text="EMAIL", font=FONTS["label"], bg=COLORS["white"], fg=COLORS["muted"]).pack(anchor="w", padx=32, pady=(0, 4))
        self.email_entry = tk.Entry(self.card, font=FONTS["body"], bg=COLORS["cream"], relief=tk.FLAT, bd=0, highlightbackground=COLORS["border"], highlightthickness=1)
        self.email_entry.pack(fill=tk.X, padx=32, pady=(0, 12), ipady=8)

        # Password field
        tk.Label(self.card, text="PASSWORD", font=FONTS["label"], bg=COLORS["white"], fg=COLORS["muted"]).pack(anchor="w", padx=32, pady=(0, 4))
        self.password_entry = tk.Entry(self.card, font=FONTS["body"], bg=COLORS["cream"], relief=tk.FLAT, bd=0, highlightbackground=COLORS["border"], highlightthickness=1, show="•")
        self.password_entry.pack(fill=tk.X, padx=32, pady=(0, 12), ipady=8)

        # Confirm Password field
        tk.Label(self.card, text="CONFIRM PASSWORD", font=FONTS["label"], bg=COLORS["white"], fg=COLORS["muted"]).pack(anchor="w", padx=32, pady=(0, 4))
        self.confirm_entry = tk.Entry(self.card, font=FONTS["body"], bg=COLORS["cream"], relief=tk.FLAT, bd=0, highlightbackground=COLORS["border"], highlightthickness=1, show="•")
        self.confirm_entry.pack(fill=tk.X, padx=32, pady=(0, 16), ipady=8)

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

        # Create Account button
        tk.Button(
            self.card,
            text="Get Started →",
            font=FONTS["body_bold"],
            bg=COLORS["gold"],
            fg=COLORS["navy"],
            relief=tk.FLAT,
            height=2,
            cursor="hand2",
            command=self._on_create_account,
            borderwidth=0,
            highlightbackground=COLORS["gold"],
            highlightthickness=0
        ).pack(fill=tk.X, padx=32, pady=(0, 12), ipady=8)

        # Login link
        login_frame = tk.Frame(self.card, bg=COLORS["white"])
        login_frame.pack(fill=tk.X, pady=(0, 8))
        tk.Label(login_frame, text="Have an account? ",
            font=FONTS["small"], bg=COLORS["white"], fg=COLORS["muted"]).pack(side=tk.LEFT)
        login_link = tk.Label(login_frame, text="Sign in",
                     font=("Helvetica", 9, "bold", "underline"), bg=COLORS["white"], fg=COLORS["teal"],
                     cursor="hand2")
        login_link.pack(side=tk.LEFT)
        login_link.bind("<Button-1>", lambda e: self.navigate("login"))

        # Status bar
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Ready")

        self.after_idle(self._apply_layout)

    def _apply_layout(self):
        width = max(self.winfo_width(), self.winfo_reqwidth())
        mode = "desktop" if width >= 800 else "mobile"
        if mode == self._layout_mode:
            return

        self._layout_mode = mode
        for widget in self.container.winfo_children():
            widget.pack_forget()

        if mode == "desktop":
            self.card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(160, 32), pady=32)
        else:
            self.card.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

    def _on_resize(self, event):
        self._apply_layout()

    def _set_feedback(self, message: str, is_error: bool = True):
        self.feedback_label.config(fg=COLORS["danger"] if is_error else COLORS["teal"], text=message)
        self.status_bar.set_text(f"Status: {message}")

    def _validate_registration(self, full_name: str, email: str, password: str, confirm: str):
        if not full_name or not email or not password or not confirm:
            return False, "Complete every field before continuing."
        if len(full_name.split()) < 2:
            return False, "Enter your full name."
        if not self.EMAIL_PATTERN.match(email):
            return False, "Enter a valid email address."
        if len(password) < 8:
            return False, "Password must be at least 8 characters long."
        if password != confirm:
            return False, "Passwords do not match."
        return True, ""

    def _on_create_account(self):
        """Handle account creation."""
        full_name = self.name_entry.get().strip()
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        confirm = self.confirm_entry.get()

        is_valid, message = self._validate_registration(full_name, email, password, confirm)
        if not is_valid:
            self._set_feedback(message)
            return

        success, message = self.state.register_user(full_name, email, password)
        if not success:
            self._set_feedback(message)
            return

        self._set_feedback(message, is_error=False)
        self.navigate("session")

    def on_show(self):
        """Called when screen is shown."""
        self._apply_layout()
        self.feedback_label.config(text="")
        self.name_entry.delete(0, tk.END)
        self.email_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.confirm_entry.delete(0, tk.END)
        self.status_bar.set_live(False)
        self.status_bar.set_text("Status: Create your account")

    def back(self):
        """Navigate back to login screen."""
        self.navigate("login")