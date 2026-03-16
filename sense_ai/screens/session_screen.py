# screens/session_screen.py

import tkinter as tk
import secrets
import string

try:
    from components.status_bar import StatusBar
    from theme import COLORS, FONTS
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from components.status_bar import StatusBar
    from theme import COLORS, FONTS


class SessionScreen(tk.Frame):
    """Session ID entry screen before launching Signer or Speaker mode."""

    SCREEN_NAME = "session"

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate
        self._layout_mode = None
        self._join_dialog = None
        self._join_entry = None
        self._selected_role = None
        self._role_cards = {}

        self.bind("<Configure>", self._on_resize)

        # Main container
        self.container = tk.Frame(self, bg=COLORS["cream"])
        self.container.pack(fill=tk.BOTH, expand=True)

        self.scroll_canvas = tk.Canvas(self.container, bg=COLORS["cream"], highlightthickness=0)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.container, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.shell = tk.Frame(self.scroll_canvas, bg=COLORS["cream"])
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.shell, anchor="nw")

        self.shell.bind("<Configure>", self._sync_scroll_region)
        self.scroll_canvas.bind("<Configure>", self._sync_canvas_width)
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.content = tk.Frame(self.shell, bg=COLORS["cream"])
        self.content.pack(fill=tk.BOTH, expand=True)
 # mobile view wrap
        self.mobile_notch_wrap = tk.Frame(self.content, bg=COLORS["cream"], height=54)
        self.mobile_notch = tk.Frame(self.mobile_notch_wrap, bg="#07122d", width=110, height=28)
        self.mobile_notch.pack(pady=(12, 8))
        self.mobile_notch.pack_propagate(False)
        tk.Label(self.mobile_notch, text="", bg="#07122d").pack()

        self.top_spacer = tk.Frame(self.content, bg=COLORS["cream"], height=24)
        self.top_spacer.pack(fill=tk.X)

#back button
        self.backtologin = tk.Label(
            self.content,
            text="← Back to login",
            font=("Helvetica", 15, "bold"),
            fg="#007e9b",
            bg=COLORS["cream"],
            cursor="hand2",
        )
        self.backtologin.pack(anchor="w", padx=40, pady=(0, 6))
        self.backtologin.bind("<Button-1>", lambda _event: self.navigate("login"))

    # Welcome text
        self.kicker = tk.Label(
            self.content,
            text="WELCOME BACK 👋",
            font=("Helvetica", 11, "bold"),
            fg="#007e9b",
            bg=COLORS["cream"],

        )
        self.kicker.pack(anchor="w", padx=40, pady=(8, 6))

        self.title_label = tk.Label(
            self.content,
            text="How are you\ncommunicating today?",
            font=("Georgia", 30, "bold"),
            fg=COLORS["navy"],
            bg=COLORS["cream"],
            justify=tk.LEFT,
        )
        self.title_label.pack(anchor="w", padx=40)

        self.subtitle = tk.Label(
            self.content,
            text="Pick a role to get started — you can switch roles anytime from Settings",
            font=("Helvetica", 12),
            fg="#96a0b2",
            bg=COLORS["cream"],
            justify=tk.LEFT,
        )
        self.subtitle.pack(anchor="w", padx=40, pady=(10, 24))

# Role selection cards
        self.cards_row = tk.Frame(self.content, bg=COLORS["cream"])
        self.cards_row.pack(fill=tk.X, padx=40)

        self.signer_card = self._build_role_card(
            parent=self.cards_row,
            role="signer",
            emoji="🤟",
            title="Signer",
            description="I use sign language — translate my signs to text for others",
            border="#b8dde4",
            icon_bg="#e4f1f3",
            command=lambda: self._set_role_and_navigate("signer"),
        )
        self.speaker_card = self._build_role_card(
            parent=self.cards_row,
            role="speaker",
            emoji="🗣️",
            title="Speaker",
            description="I speak or type — translate my words to sign language",
            border="#f1d8aa",
            icon_bg="#f8efdf",
            command=lambda: self._set_role_and_navigate("speaker"),
        )
        self._role_cards = {
            "signer": self.signer_card,
            "speaker": self.speaker_card,
        }
        self._set_selected_role(getattr(self.state, "user_role", "signer"))

        self.join_card = tk.Frame(
            self.content,
            bg=COLORS["white"],
            highlightbackground="#e2e8ef",
            highlightthickness=1,
            bd=0,
        )

        self.join_icon = tk.Label(
            self.join_card,
            text="🔗",
            font=("Helvetica", 24),
            fg="#b7a8c8",
            bg=COLORS["white"],
            width=2,
        )
        self.join_copy = tk.Frame(self.join_card, bg=COLORS["white"])
        self.join_title = tk.Label(
            self.join_copy,
            text="Join a session",
            font=("Helvetica", 16, "bold"),
            fg=COLORS["navy"],
            bg=COLORS["white"],
        )
        self.join_desc = tk.Label(
            self.join_copy,
            text="Enter an ID to join someone else's conversation",
            font=("Helvetica", 11),
            fg=COLORS["text"],
            bg=COLORS["white"],
            justify=tk.LEFT,
        )
        self.join_button = tk.Button(
            self.join_card,
            text="Enter ID →",
            font=("Helvetica", 11, "bold"),
            bg="#dff0f3",
            fg="#0c7fa0",
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=8,
            cursor="hand2",
            command=self._open_join_dialog,
        )

        # Status bar
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Waiting…")

        # Lay out panels after all widgets exist
        self.after_idle(self._layout_panels)

    def _layout_panels(self):
        w = max(self.winfo_width(), self.winfo_reqwidth())
        next_mode = "desktop" if w >= 800 else "mobile"

        if self._layout_mode == next_mode:
            return

        self._layout_mode = next_mode

        self.mobile_notch_wrap.pack_forget()
        for widget in self.cards_row.winfo_children():
            widget.pack_forget()

        self.join_card.pack_forget()
        for widget in self.join_card.winfo_children():
            widget.pack_forget()
        for widget in self.join_copy.winfo_children():
            widget.pack_forget()

        if next_mode == "desktop":
            self.content.pack_configure(padx=0, pady=0)
            self.top_spacer.pack_configure(fill=tk.X)
            self.top_spacer.configure(height=24)
            self.kicker.pack_configure(anchor="w", padx=180, pady=(24, 8))
            self.title_label.pack_configure(anchor="w", padx=180)
            self.title_label.configure(font=("Georgia", 28, "bold"))
            self.subtitle.pack_configure(anchor="w", padx=180, pady=(10, 28))
            self.cards_row.pack_configure(fill=tk.X, padx=180)

            self._configure_role_card(self.signer_card, wraplength=250, font_size=12, title_size=19, icon_size=24, arrow_size=18, pady=24)
            self._configure_role_card(self.speaker_card, wraplength=250, font_size=12, title_size=19, icon_size=24, arrow_size=18, pady=24)

            self.signer_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))
            self.speaker_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 0))

            self.join_card.pack(fill=tk.X, padx=180, pady=(24, 0))
            self.join_icon.pack(side=tk.LEFT, padx=(24, 16), pady=22)
            self.join_copy.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=20)
            self.join_title.pack(anchor="w")
            self.join_desc.pack(anchor="w", pady=(4, 0))
            self.join_title.configure(font=("Helvetica", 16, "bold"))
            self.join_desc.configure(font=("Helvetica", 11,), wraplength=420)
            self.join_button.configure(text="Enter ID →", font=("Helvetica", 11, "bold"), padx=18, pady=8)
            self.join_button.pack(side=tk.RIGHT, padx=24, pady=20)
        else:
            self.mobile_notch_wrap.pack(fill=tk.X, before=self.top_spacer)
            self.kicker.pack_configure(anchor="w", padx=32, pady=(12, 8))
            self.title_label.pack_configure(anchor="w", padx=32)
            self.title_label.configure(font=("Georgia", 19, "bold"))
            self.subtitle.pack_configure(anchor="w", padx=32, pady=(8, 20))
            self.subtitle.configure(font=("Helvetica", 10))
            self.cards_row.pack_configure(fill=tk.X, padx=32)
            self.top_spacer.configure(height=0)

            self._configure_role_card(self.signer_card, wraplength=210, font_size=10, title_size=17, icon_size=20, arrow_size=15, pady=20)
            self._configure_role_card(self.speaker_card, wraplength=210, font_size=10, title_size=17, icon_size=20, arrow_size=15, pady=20)

            self.signer_card.pack(fill=tk.X, pady=(0, 16))
            self.speaker_card.pack(fill=tk.X)

            self.join_card.pack(fill=tk.X, padx=32, pady=(16, 0))
            self.join_icon.pack(side=tk.LEFT, padx=(18, 12), pady=18)
            self.join_copy.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=16)
            self.join_title.pack(anchor="w")
            self.join_desc.pack(anchor="w", pady=(4, 0))
            self.join_title.configure(font=("Helvetica", 14, "bold"))
            self.join_desc.configure(font=("Helvetica", 10), wraplength=180)
            self.join_button.configure(text="Join", font=("Helvetica", 10, "bold"), padx=14, pady=6)
            self.join_button.pack(side=tk.RIGHT, padx=18, pady=18)

    def _on_resize(self, event):
        self._layout_panels()
        self._update_scrollbar_visibility()

    def _sync_scroll_region(self, _event):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
        self._update_scrollbar_visibility()

    def _sync_canvas_width(self, event):
        self.scroll_canvas.itemconfigure(self.canvas_window, width=event.width)
        self._update_scrollbar_visibility()

    def _update_scrollbar_visibility(self):
        self.update_idletasks()
        content_height = self.shell.winfo_reqheight()
        canvas_height = self.scroll_canvas.winfo_height()

        if content_height > canvas_height:
            if not self.scrollbar.winfo_ismapped():
                self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        else:
            if self.scrollbar.winfo_ismapped():
                self.scrollbar.pack_forget()

    def _on_mousewheel(self, event):
        if self.scrollbar.winfo_ismapped():
            self.scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _build_role_card(self, parent, role, emoji, title, description, border, icon_bg, command):
        card = tk.Frame(
            parent,
            bg=COLORS["white"],
            highlightbackground=border,
            highlightthickness=2,
            bd=0,
            padx=24,
            pady=24,
        )

        top_row = tk.Frame(card, bg=COLORS["white"])
        top_row.pack(fill=tk.X)

        icon = tk.Label(
            top_row,
            text=emoji,
            font=("Helvetica", 24),
            bg=icon_bg,
            fg=COLORS["navy"],
            width=2,
            height=1,
            padx=10,
            pady=10,
        )
        icon.pack(side=tk.LEFT)

        arrow = tk.Label(
            top_row,
            text="→",
            font=("Helvetica", 18, "bold"),
            fg="#d9dee6",
            bg=COLORS["white"],
        )
        arrow.pack(side=tk.RIGHT, anchor="s", pady=(34, 0))

        title_label = tk.Label(
            card,
            text=title,
            font=("Helvetica", 19, "bold"),
            fg=COLORS["navy"],
            bg=COLORS["white"],
        )
        title_label.pack(anchor="w", pady=(18, 8))

        desc_label = tk.Label(
            card,
            text=description,
            font=("Helvetica", 12),
            fg="#8f9ab0",
            bg=COLORS["white"],
            justify=tk.LEFT,
            wraplength=280,
        )
        desc_label.pack(anchor="w")

        button = tk.Button(
            card,
            text="",
            command=command,
            bg=COLORS["white"],
            activebackground=COLORS["white"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        button.place(relx=0, rely=0, relwidth=1, relheight=1)
        button.lower()

        for widget in (card, top_row, icon, arrow, title_label, desc_label):
            widget.bind("<Button-1>", lambda _event, action=command: action())

        card.role = role
        card.default_border = border
        card.selected_border = "#45d9f8" if role == "signer" else "#f7c965"
        card.selected_icon_bg = "#d9f7ff" if role == "signer" else "#fff2db"
        card.default_icon_bg = icon_bg
        card.icon_label = icon
        card.arrow_label = arrow
        card.title_label = title_label
        card.desc_label = desc_label
        card.top_row = top_row

        return card

    def _configure_role_card(self, card, wraplength, font_size, title_size, icon_size, arrow_size, pady):
        card.configure(padx=20, pady=pady)
        card.icon_label.configure(font=("Helvetica", icon_size), padx=10, pady=10)
        card.arrow_label.configure(font=("Helvetica", arrow_size, "bold"))
        card.title_label.configure(font=("Helvetica", title_size, "bold"))
        card.desc_label.configure(font=("Helvetica", font_size), wraplength=wraplength)

    def _set_selected_role(self, role: str):
        if role not in self._role_cards:
            role = "signer"

        self._selected_role = role
        for name, card in self._role_cards.items():
            is_selected = name == role
            card.configure(
                highlightbackground=card.selected_border if is_selected else card.default_border,
                highlightthickness=4 if is_selected else 2,
            )
            card.icon_label.configure(bg=card.selected_icon_bg if is_selected else card.default_icon_bg)
            card.arrow_label.configure(fg=COLORS["teal"] if is_selected else "#d9dee6")
            card.title_label.configure(fg=COLORS["navy"] if is_selected else COLORS["text_lt"])
            card.desc_label.configure(fg=COLORS["text"] if is_selected else "#8f9ab0")

    def _open_join_dialog(self):
        if self._join_dialog and self._join_dialog.winfo_exists():
            self._join_dialog.lift()
            self._join_dialog.focus_force()
            return

        dialog = tk.Toplevel(self)
        dialog.title("Join Session")
        dialog.configure(bg=COLORS["white"])
        dialog.resizable(False, False)
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()

        body = tk.Frame(dialog, bg=COLORS["white"], padx=20, pady=18)
        body.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            body,
            text="Meeting ID",
            font=("Helvetica", 14, "bold"),
            fg=COLORS["text"],
            bg=COLORS["white"],
        ).pack(anchor="w", pady=(0, 10))

        self._join_entry = tk.Entry(
            body,
            font=("Helvetica", 14),
            bg="#c8d9bd",
            fg=COLORS["text"],
            relief=tk.FLAT,
            bd=0,
            width=24,
        )
        self._join_entry.pack(fill=tk.X, ipady=8)
        self._join_entry.insert(0, self.state.session_id or "")
        self._join_entry.focus_set()
        self._join_entry.bind("<Return>", lambda _event: self._confirm_join_dialog())

        actions = tk.Frame(dialog, bg="#d8d8d8")
        actions.pack(fill=tk.X)

        tk.Button(
            actions,
            text="Cancel",
            command=self._close_join_dialog,
            bg="#d8d8d8",
            fg=COLORS["text"],
            relief=tk.FLAT,
            bd=0,
            padx=30,
            pady=12,
            cursor="hand2",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            actions,
            text="Enter",
            command=self._confirm_join_dialog,
            bg="#d8d8d8",
            fg=COLORS["text"],
            relief=tk.FLAT,
            bd=0,
            padx=30,
            pady=12,
            cursor="hand2",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._join_dialog = dialog

    def _close_join_dialog(self):
        if self._join_dialog and self._join_dialog.winfo_exists():
            self._join_dialog.destroy()
        self._join_dialog = None
        self._join_entry = None

    def _confirm_join_dialog(self):
        if not self._join_entry:
            return

        value = self._join_entry.get().strip()
        if not value:
            self.status_bar.set_text("Status: Ask the Signer for the Meeting ID")
            return

        self.state.session_id = value

        # Only mark speaker joined if speaker role is selected.
        if (self._selected_role or "").strip() == "speaker":
            self._register_speaker_join(value)
            self.status_bar.set_text(f"Status: Joined {value} as Speaker")
            self._close_join_dialog()
            self.state.user_role = "speaker"
            self.navigate("speaker")
            return

        # Non-speaker flow: set ID only.
        self.status_bar.set_text(f"Status: Meeting ID set to {value}")
        self._close_join_dialog()

    def _set_role_and_navigate(self, role: str):
        """Set role and navigate to the appropriate screen."""
        previous_role = self._selected_role or getattr(self.state, "user_role", "")

        self._set_selected_role(role)

        # Persist selected role immediately to avoid stale-role behavior.
        self.state.user_role = role

        # If switching from speaker -> signer, clear stale meeting ID so signer creates fresh session.
        if previous_role == "speaker" and role == "signer":
            self.state.session_id = ""

        is_valid, selected_role, meeting_id = self._validate_session_requirements(role)
        if not is_valid:
            return

        self.state.user_role = selected_role
        self.state.session_id = meeting_id

        if selected_role == "speaker":
            self._register_speaker_join(meeting_id)

        self.navigate(selected_role)

    def _generate_meeting_id(self, length: int = 6) -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def _register_signer_session(self, meeting_id: str):
        backend = getattr(self.state, "backend", None)
        if backend and hasattr(backend, "create_meeting_session"):
            try:
                backend.create_meeting_session(meeting_id=meeting_id)
            except Exception:
                pass

        registry = getattr(self.state, "_session_registry", {})
        session = registry.get(meeting_id, {})
        # Signer creates/owns the session start state; always reset speaker flag.
        session["speaker_joined"] = False
        registry[meeting_id] = session
        self.state._session_registry = registry

    def _register_speaker_join(self, meeting_id: str):
        backend = getattr(self.state, "backend", None)
        if backend and hasattr(backend, "join_meeting_session_as_speaker"):
            try:
                backend.join_meeting_session_as_speaker(meeting_id=meeting_id)
                return
            except Exception:
                pass

        registry = getattr(self.state, "_session_registry", {})
        session = registry.get(meeting_id, {})
        session["speaker_joined"] = True
        registry[meeting_id] = session
        self.state._session_registry = registry

    def _speaker_has_joined(self, meeting_id: str) -> bool:
        backend = getattr(self.state, "backend", None)
        if backend and hasattr(backend, "is_speaker_joined"):
            try:
                return bool(backend.is_speaker_joined(meeting_id=meeting_id))
            except Exception:
                pass

        registry = getattr(self.state, "_session_registry", {})
        return bool(registry.get(meeting_id, {}).get("speaker_joined", False))

    def _validate_session_requirements(self, role: str = None):
        selected_role = role or self._selected_role or getattr(self.state, "user_role", "")
        if selected_role not in {"signer", "speaker"}:
            self.status_bar.set_text("Status: Select a role (Signer or Speaker) before continuing")
            return False, None, None

        meeting_id = getattr(self.state, "session_id", "").strip()

        if selected_role == "signer":
            if not meeting_id:
                meeting_id = self._generate_meeting_id()
                self.state.session_id = meeting_id
                self._register_signer_session(meeting_id)
                self.status_bar.set_text(f"Status: Meeting ID {meeting_id} created. Share it with Speaker.")

            if not self._speaker_has_joined(meeting_id):
                self.status_bar.set_text(f"Status: Waiting for Speaker to join {meeting_id}")
                return False, selected_role, meeting_id

            return True, selected_role, meeting_id

        # Speaker path
        if not meeting_id:
            self.status_bar.set_text("Status: Ask the Signer for the Meeting ID")
            self._open_join_dialog()
            return False, selected_role, None

        return True, selected_role, meeting_id

    def _on_start(self):
        """Start conversation with the entered session ID."""
        is_valid, selected_role, meeting_id = self._validate_session_requirements()
        if not is_valid:
            return

        self.state.user_role = selected_role
        self.state.session_id = meeting_id
        self.navigate(selected_role)

    def on_show(self):
        """Called when screen is shown."""
        self.status_bar.set_live(False)
        user = getattr(self.state, "current_user", None)
        name = user["full_name"].split()[0] if user else "there"
        self.kicker.config(text=f"WELCOME BACK {name.upper()} 👋")
        self._set_selected_role(getattr(self.state, "user_role", "signer"))

        meeting_id = getattr(self.state, "session_id", "").strip()
        if meeting_id:
            self.status_bar.set_text(f"Status: Current Meeting ID {meeting_id}")
        else:
            self.status_bar.set_text("Status: Signer creates ID · Speaker enters ID")
        self._layout_panels()

# temp to sstart session screen to see how it looks
# def main():
#     root = tk.Tk()
#     root.title("Sense.AI")
#     root.geometry("900x600")
#     state = type("State", (), {"session_id": "", "user_role": "signer"})()
#     session_screen = SessionScreen(root, state, lambda screen: print(f"Navigate to {screen}"))
#     session_screen.pack(fill=tk.BOTH, expand=True)
#     root.mainloop()


# if __name__ == "__main__":
#     main()