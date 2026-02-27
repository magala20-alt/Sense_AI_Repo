# screens/signup_screen.py

import tkinter as tk
from components.window_bar import WindowBar
from components.status_bar import StatusBar
from theme import COLORS, FONTS


class SignUpScreen(tk.Frame):
    """Sign Up screen with email, password, and confirm password fields."""

    SCREEN_NAME = "signup"

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate

        # Window bar
        WindowBar(self, title="Sense.AI · Sign Up")

        # Body
        body = tk.Frame(self, bg=COLORS["cream"])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Emoji
        tk.Label(body, text="✨", font=("Helvetica", 28), bg=COLORS["cream"]).pack(pady=6)

        # Heading
        tk.Label(body, text="Sign Up", font=FONTS["heading"], bg=COLORS["cream"], fg=COLORS["navy"]).pack(pady=8)

        # Email field
        tk.Label(body, text="EMAIL", font=FONTS["label"], bg=COLORS["cream"], fg=COLORS["muted"]).pack(anchor="w", pady=(12, 4))
        self.email_entry = tk.Entry(body, font=FONTS["body"], bg=COLORS["white"], relief=tk.SOLID, bd=1, width=40)
        self.email_entry.pack(fill=tk.X, pady=(0, 12), ipady=6)

        # Password field
        tk.Label(body, text="PASSWORD", font=FONTS["label"], bg=COLORS["cream"], fg=COLORS["muted"]).pack(anchor="w", pady=(0, 4))
        self.password_entry = tk.Entry(body, font=FONTS["body"], bg=COLORS["white"], relief=tk.SOLID, bd=1, width=40, show="•")
        self.password_entry.pack(fill=tk.X, pady=(0, 12), ipady=6)

        # Confirm Password field
        tk.Label(body, text="CONFIRM PASSWORD", font=FONTS["label"], bg=COLORS["cream"], fg=COLORS["muted"]).pack(anchor="w", pady=(0, 4))
        self.confirm_entry = tk.Entry(body, font=FONTS["body"], bg=COLORS["white"], relief=tk.SOLID, bd=1, width=40, show="•")
        self.confirm_entry.pack(fill=tk.X, pady=(0, 16), ipady=6)

        # Create Account button
        tk.Button(
            body,
            text="Create Account →",
            font=FONTS["body_bold"],
            bg=COLORS["gold"],
            fg=COLORS["navy"],
            relief=tk.FLAT,
            height=2,
            cursor="hand2",
            command=self._on_create_account
        ).pack(fill=tk.X, pady=8)

        # Spacer
        tk.Frame(body, bg=COLORS["cream"]).pack(expand=True, fill=tk.BOTH)

        # Login link
        login_frame = tk.Frame(body, bg=COLORS["cream"])
        login_frame.pack(fill=tk.X, pady=8)
        tk.Label(login_frame, text="Already have an account? ",
                font=FONTS["small"], bg=COLORS["cream"], fg=COLORS["muted"]).pack(side=tk.LEFT)
        login_link = tk.Label(login_frame, text="Login",
                             font=("Helvetica", 9, "bold", "underline"), bg=COLORS["cream"], fg=COLORS["teal"],
                             cursor="hand2")
        login_link.pack(side=tk.LEFT)
        login_link.bind("<Button-1>", lambda e: self.navigate("login"))

        # Status bar
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Ready")

    def _on_create_account(self):
        """Handle account creation."""
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        confirm = self.confirm_entry.get()
        
        if email and password and password == confirm:
            self.navigate("session")

    def on_show(self):
        """Called when screen is shown."""
        self.email_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.confirm_entry.delete(0, tk.END)
        self.status_bar.set_live(False)
