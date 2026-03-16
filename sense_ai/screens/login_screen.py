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
        self.header_title.place(x=28, y=40, anchor="nw")
        self.header_subtitle = tk.Label(
            self.header,
            text="Translate (Signer) · Speak (Speaker)",
            font=("Helvetica", 14),
            bg=COLORS["navy"],
            fg=COLORS["teal"],
        )
        self.header_subtitle.place(x=28, y=100, anchor="nw")
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
        self.password_entry = tk.Entry(
            self.card,
            font=FONTS["body"],
            bg=COLORS["cream"],
            relief=tk.FLAT,
            bd=0,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
            show="•",
        )
        self.password_entry.pack(fill=tk.X, padx=32, pady=(0, 4), ipady=8)

        self._login_show_password = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.card,
            text="Show password",
            variable=self._login_show_password,
            command=self._toggle_login_password,
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

        forgot_frame = tk.Frame(self.card, bg=COLORS["white"])
        forgot_frame.pack(fill=tk.X, padx=32, pady=(0, 8))

        self.forgot_password_label = tk.Label(
            forgot_frame,
            text="Forgot password?",
            font=FONTS["small"],
            bg=COLORS["white"],
            fg=COLORS["teal"],
            cursor="hand2",
        )
        self.forgot_password_label.pack(side=tk.RIGHT)
        self.forgot_password_label.bind("<Button-1>", self._open_forgot_password_window)

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

    def _toggle_login_password(self):
        self.password_entry.config(show="" if self._login_show_password.get() else "•")

    def _get_security_question(self, email: str):
        if hasattr(self.state, "get_security_question"):
            return self.state.get_security_question(email)

        users = getattr(self.state, "users", {})
        record = users.get(email.lower()) if isinstance(users, dict) else None
        if isinstance(record, dict):
            return record.get("security_question")
        return None

    def _verify_security_answer(self, email: str, answer: str):
        if hasattr(self.state, "verify_security_answer"):
            return self.state.verify_security_answer(email, answer)

        users = getattr(self.state, "users", {})
        record = users.get(email.lower()) if isinstance(users, dict) else None
        if isinstance(record, dict):
            saved = str(record.get("security_answer", "")).strip().casefold()
            return saved and saved == answer.strip().casefold()
        return False

    def _apply_password_update(self, email: str, new_password: str):
        if hasattr(self.state, "update_password"):
            return self.state.update_password(email, new_password)

        users = getattr(self.state, "users", {})
        if isinstance(users, dict) and email.lower() in users and isinstance(users[email.lower()], dict):
            users[email.lower()]["password"] = new_password
            return True, "Password updated. Please sign in."
        return False, "Could not update password."

    def _open_forgot_password_window(self, _event=None):
        win = tk.Toplevel(self)
        win.title("Forgot Password")
        win.geometry("460x420")
        win.resizable(False, False)
        win.transient(self.winfo_toplevel())
        win.grab_set()

        tk.Label(win, text="Email", font=FONTS["label"]).pack(anchor="w", padx=16, pady=(16, 4))
        email_entry = tk.Entry(win, font=FONTS["body"])
        email_entry.insert(0, self.email_entry.get().strip())
        email_entry.pack(fill=tk.X, padx=16, ipady=6)

        tk.Label(win, text="Security question", font=FONTS["label"]).pack(anchor="w", padx=16, pady=(12, 4))
        question_lbl = tk.Label(win, text="Enter email, then click Load Question", font=FONTS["small"], justify=tk.LEFT, wraplength=420)
        question_lbl.pack(anchor="w", padx=16)

        tk.Button(
            win,
            text="Load Question",
            font=FONTS["small"],
            command=lambda: _load_question(),
            cursor="hand2",
        ).pack(anchor="w", padx=16, pady=(8, 0))

        tk.Label(win, text="Answer", font=FONTS["label"]).pack(anchor="w", padx=16, pady=(12, 4))
        answer_entry = tk.Entry(win, font=FONTS["body"])
        answer_entry.pack(fill=tk.X, padx=16, ipady=6)

        tk.Label(win, text="New password", font=FONTS["label"]).pack(anchor="w", padx=16, pady=(12, 4))
        new_pass_entry = tk.Entry(win, font=FONTS["body"], show="•")
        new_pass_entry.pack(fill=tk.X, padx=16, ipady=6)

        tk.Label(win, text="Confirm password", font=FONTS["label"]).pack(anchor="w", padx=16, pady=(12, 4))
        confirm_pass_entry = tk.Entry(win, font=FONTS["body"], show="•")
        confirm_pass_entry.pack(fill=tk.X, padx=16, ipady=6)

        show_reset_pw = tk.BooleanVar(value=False)

        def _toggle_reset_password():
            mask = "" if show_reset_pw.get() else "•"
            new_pass_entry.config(show=mask)
            confirm_pass_entry.config(show=mask)

        tk.Checkbutton(
            win,
            text="Show passwords",
            variable=show_reset_pw,
            command=_toggle_reset_password,
            font=FONTS["small"],
        ).pack(anchor="w", padx=16, pady=(6, 0))

        status_lbl = tk.Label(win, text="", font=FONTS["small"], fg=COLORS["danger"])
        status_lbl.pack(anchor="w", padx=16, pady=(10, 0))

        def _load_question():
            email = email_entry.get().strip().lower()
            if not self.EMAIL_PATTERN.match(email):
                question_lbl.config(text="Enter a valid email.")
                return
            q = self._get_security_question(email)
            if not q:
                question_lbl.config(text="No security question configured for this account.")
                return
            question_lbl.config(text=q)

        def _submit():
            email = email_entry.get().strip().lower()
            answer = answer_entry.get().strip()
            new_pw = new_pass_entry.get().strip()
            confirm_pw = confirm_pass_entry.get().strip()

            if not self.EMAIL_PATTERN.match(email):
                status_lbl.config(text="Enter a valid email.")
                return
            if not self._get_security_question(email):
                status_lbl.config(text="No security question configured for this account.")
                return
            if not self._verify_security_answer(email, answer):
                status_lbl.config(text="Incorrect security answer.")
                return
            if len(new_pw) < 6:
                status_lbl.config(text="Password must be at least 6 characters.")
                return
            if new_pw != confirm_pw:
                status_lbl.config(text="Passwords do not match.")
                return

            ok, msg = self._apply_password_update(email, new_pw)
            if ok:
                self._set_feedback(msg, is_error=False)
                win.destroy()
            else:
                status_lbl.config(text=msg)

        tk.Button(
            win,
            text="Update Password",
            font=FONTS["body_bold"],
            bg=COLORS["navy"],
            fg=COLORS["white"],
            relief=tk.FLAT,
            command=_submit,
            cursor="hand2",
        ).pack(fill=tk.X, padx=16, pady=16, ipady=8)

        self._apply_layout()

    def on_show(self):
        self._apply_layout()
        self.feedback_label.config(text="")
        self.email_entry.delete(0, tk.END)
        self.password_entry.delete(0, tk.END)
        self.status_bar.set_live(False)
        self.status_bar.set_text("Status: Enter your account credentials")
