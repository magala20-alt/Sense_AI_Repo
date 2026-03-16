# screens/signup_screen.py

import tkinter as tk
import re
from pathlib import Path

from PIL import Image, ImageTk
from components.status_bar import StatusBar
from theme import COLORS, FONTS
from app_state import AppState


class SignUpScreen(tk.Frame):
    """Sign Up screen with email, password, and confirm password fields."""

    SCREEN_NAME = "signup"
    EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate
        self._layout_mode = None
        self._header_source_image = None
        self._header_photo = None

        # Responsive layout logic
        self.bind('<Configure>', self._on_resize)

        # Main container
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
        self.header_title.place(x=28, y=40, anchor="nw")
        self.header_subtitle = tk.Label(
            self.header,
            text="Translate (Signer) · Speak (Speaker)",
            font=("Helvetica", 14),
            bg=COLORS["navy"],
            fg=COLORS["teal"],
        )
        self.header_subtitle.place(x=28, y=100, anchor="nw")
        self._load_header_image()
        self.header.bind("<Configure>", self._update_header_image)

        # Card host (scrollable)
        self.card_host = tk.Frame(self.container, bg=COLORS["cream"])

        self.card_canvas = tk.Canvas(
            self.card_host,
            bg=COLORS["cream"],
            highlightthickness=0,
            bd=0,
        )
        self.card_scrollbar = tk.Scrollbar(
            self.card_host,
            orient=tk.VERTICAL,
            command=self.card_canvas.yview,
        )
        self.card_canvas.configure(yscrollcommand=self.card_scrollbar.set)

        self.card_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.card_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Card (inside canvas)
        self.card = tk.Frame(
            self.card_canvas,
            bg=COLORS["white"],
            bd=0,
            relief=tk.FLAT,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        self._card_window = self.card_canvas.create_window((0, 0), window=self.card, anchor="nw")

        self.card.bind("<Configure>", self._on_card_configure)
        self.card_canvas.bind("<Configure>", self._on_canvas_configure)
        self.card_canvas.bind("<Enter>", self._bind_mousewheel)
        self.card_canvas.bind("<Leave>", self._unbind_mousewheel)

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

        # Security Question field
        tk.Label(self.card, text="SECURITY QUESTION", font=FONTS["label"], bg=COLORS["white"], fg=COLORS["muted"]).pack(anchor="w", padx=32, pady=(0, 4))
        self.security_question_entry = tk.Entry(self.card, font=FONTS["body"], bg=COLORS["cream"], relief=tk.FLAT, bd=0, highlightbackground=COLORS["border"], highlightthickness=1)
        self.security_question_entry.pack(fill=tk.X, padx=32, pady=(0, 12), ipady=8)

        # Security Answer field
        tk.Label(self.card, text="SECURITY ANSWER", font=FONTS["label"], bg=COLORS["white"], fg=COLORS["muted"]).pack(anchor="w", padx=32, pady=(0, 4))
        self.security_answer_entry = tk.Entry(self.card, font=FONTS["body"], bg=COLORS["cream"], relief=tk.FLAT, bd=0, highlightbackground=COLORS["border"], highlightthickness=1, show="•")
        self.security_answer_entry.pack(fill=tk.X, padx=32, pady=(0, 12), ipady=8)

        # Show/Hide password fields
        self._signup_show_password = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.card,
            text="Show password fields",
            variable=self._signup_show_password,
            command=self._toggle_signup_passwords,
            bg=COLORS["white"],
            fg=COLORS["muted"],
            activebackground=COLORS["white"],
            activeforeground=COLORS["muted"],
            selectcolor=COLORS["white"],
            font=FONTS["small"],
            relief=tk.FLAT,
            highlightthickness=0,
            bd=0,
        ).pack(anchor="w", padx=32, pady=(0, 8))

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
            height=1,
            cursor="hand2",
            command=self._on_create_account,
            borderwidth=0,
            highlightbackground=COLORS["gold"],
            highlightthickness=0
        ).pack(fill=tk.X, padx=32, pady=(0, 12), ipady=8)

        # Login link (normal flow; do NOT pin at bottom)
        login_frame = tk.Frame(self.card, bg=COLORS["white"])
        login_frame.pack(fill=tk.X, padx=32, pady=(0, 10))

        tk.Label(
            login_frame,
            text="Have an account? ",
            font=FONTS["small"],
            bg=COLORS["white"],
            fg=COLORS["muted"],
        ).pack(side=tk.LEFT)

        login_link = tk.Label(
            login_frame,
            text="Sign in",
            font=("Helvetica", 9, "bold", "underline"),
            bg=COLORS["white"],
            fg=COLORS["teal"],
            cursor="hand2",
        )
        login_link.pack(side=tk.LEFT)
        login_link.bind("<Button-1>", self._go_to_login)

        # Status bar
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Ready")

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
            self.card_host.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 32), pady=32)
            self.card_scrollbar.pack_forget()  # hide scrollbar on wide layout
            self._update_desktop_split()
        else:
            self.header.pack(side=tk.TOP, fill=tk.X)
            self.card_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
            self.card_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # show scrollbar on small layout

        self.after_idle(self._refresh_scrollregion)

    def _update_desktop_split(self):
        """Set a stable desktop split between left header and right form area."""
        total_w = max(self.container.winfo_width(), 1)
        header_w = max(360, int(total_w * 0.45))
        self.header.config(width=header_w)
        self.after_idle(self._refresh_scrollregion)

    def _on_card_configure(self, _event=None):
        self._refresh_scrollregion()

    def _on_canvas_configure(self, event):
        self.card_canvas.itemconfigure(self._card_window, width=event.width)

    def _refresh_scrollregion(self):
        self.card_canvas.configure(scrollregion=self.card_canvas.bbox("all"))

    def _on_mousewheel(self, event):
        # Windows wheel delta
        self.card_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_mousewheel(self, _event=None):
        self.card_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self.card_canvas.unbind_all("<MouseWheel>")

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

    def _on_resize(self, event):
        self._apply_layout()

    def _set_feedback(self, message: str, is_error: bool = True):
        self.feedback_label.config(fg=COLORS["danger"] if is_error else COLORS["teal"], text=message)
        self.status_bar.set_text(f"Status: {message}")

    def _validate_registration(
        self,
        full_name: str,
        email: str,
        password: str,
        confirm: str,
        security_question: str,
        security_answer: str,
    ):
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
        if not security_question.strip():
            return False, "Security question is required."
        if not security_answer.strip():
            return False, "Security answer is required."
        return True, ""

    def _on_create_account(self):
        """Handle account creation."""
        full_name = self.name_entry.get().strip()
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        confirm = self.confirm_entry.get()
        security_question = self.security_question_entry.get().strip()
        security_answer = self.security_answer_entry.get().strip()

        is_valid, message = self._validate_registration(
            full_name,
            email,
            password,
            confirm,
            security_question,
            security_answer,
        )
        if not is_valid:
            self._set_feedback(message)
            return

        success, message = self.state.register_user(
            full_name,
            email,
            password,
            security_question,
            security_answer,
        )
        if not success:
            self._set_feedback(message)
            return

        # Optional DB sync if state exposes method
        if hasattr(self.state, "set_security_qa"):
            self.state.set_security_qa(email, security_question, security_answer)

        self._set_feedback(message, is_error=False)
        self.navigate("session")

    def _toggle_signup_passwords(self):
        mask = "" if self._signup_show_password.get() else "•"
        self.password_entry.config(show=mask)
        self.confirm_entry.config(show=mask)
        self.security_answer_entry.config(show=mask)

    def _clear_form(self):
        self.feedback_label.config(text="")
        self.name_entry.delete(0, tk.END)
        self.email_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.confirm_entry.delete(0, tk.END)
        self.security_question_entry.delete(0, tk.END)
        self.security_answer_entry.delete(0, tk.END)

        self._signup_show_password.set(False)
        self._toggle_signup_passwords()
        self.card_canvas.yview_moveto(0)

    def _clear_auth_state(self):
        if hasattr(self.state, "logout"):
            self.state.logout()
            return

        if hasattr(self.state, "current_user"):
            self.state.current_user = None
        if hasattr(self.state, "session_token"):
            self.state.session_token = None
        if hasattr(self.state, "session_id"):
            self.state.session_id = ""

    def _go_to_login(self, _event=None):
        self._clear_form()
        self._clear_auth_state()
        self.navigate("login")

    def on_show(self):
        """Called when screen is shown."""
        self._apply_layout()
        self._clear_form()
        self.status_bar.set_live(False)
        self.status_bar.set_text("Status: Create your account")

    def back(self):
        """Navigate back to login screen."""
        self._go_to_login()