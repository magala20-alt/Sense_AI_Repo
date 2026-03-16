# screens/speaker_screen.py

import threading
import tkinter as tk

try:
    from app_state import AppState
    from components.mobile_nav import MobileNavBar
    from components.status_bar import StatusBar
    from components.sidebar import Sidebar
    from components.face_avatar import FaceAvatar
    from services.sigml_sender import get_sigml_sender
    from theme import COLORS
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from app_state import AppState
    from components.mobile_nav import MobileNavBar
    from components.status_bar import StatusBar
    from components.sidebar import Sidebar
    from components.face_avatar import FaceAvatar
    from services.sigml_sender import get_sigml_sender
    from theme import COLORS


class SpeakerScreen(tk.Frame):
    """Responsive speaker screen matching the supplied mockups."""

    SCREEN_NAME = "speaker"

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate
        self._layout_mode = None

        self.bind("<Configure>", self._on_resize)

        self.desktop_sidebar = Sidebar(
            self,
            active_screen="speaker",
            navigate=self.navigate,
            role="speaker",
            on_logout=self._logout,
            state=self.state,
        )
        self.main_area = tk.Frame(self, bg=COLORS["cream"])
        self.header = tk.Frame(self.main_area, bg=COLORS["cream"], height=70)
        self.header.pack(fill=tk.X)
        self.header.pack_propagate(False)
        tk.Label(self.header, text="Speak", font=("Georgia", 24, "bold"), bg=COLORS["cream"], fg=COLORS["navy"]).pack(side=tk.LEFT, padx=28, pady=16)
        self.session_badge = tk.Label(self.header, text="Session #4821", font=("Helvetica", 11, "bold"), bg="#f7ecd8", fg=COLORS["gold"], padx=14, pady=6)
        self.session_badge.pack(side=tk.RIGHT, padx=28, pady=14)

        self.scroll_canvas = tk.Canvas(self.main_area, bg=COLORS["cream"], highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.main_area, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.content = tk.Frame(self.scroll_canvas, bg=COLORS["cream"])
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda _e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.bind("<Configure>", lambda e: self.scroll_canvas.itemconfigure(self.canvas_window, width=e.width))

        self.mobile_top = tk.Frame(self.content, bg=COLORS["gold"], height=56)
        self.mobile_top.pack_propagate(False)
        tk.Label(self.mobile_top, text="Speak", font=("Georgia", 18, "bold"), bg=COLORS["gold"], fg=COLORS["white"]).pack(side=tk.LEFT, padx=18, pady=14)
        tk.Label(self.mobile_top, text="Session #4821", font=("Helvetica", 10, "bold"), bg="#f5c98a", fg=COLORS["white"], padx=12, pady=6).pack(side=tk.RIGHT, padx=12, pady=10)

        self.hero_row = tk.Frame(self.content, bg=COLORS["cream"])
        self.avatar_card = tk.Frame(self.hero_row, bg=COLORS["navy"], highlightbackground="#21345d", highlightthickness=1, padx=20, pady=20)
        self.face_avatar = FaceAvatar(self.avatar_card, size=300)
        self.face_avatar.pack()
        tk.Label(self.avatar_card, text="SIGNING AVATAR", font=("Helvetica", 11, "bold"), bg=COLORS["navy"], fg=COLORS["teal"]).pack(pady=(10, 8))
        self.avatar_phrase_label = tk.Label(self.avatar_card, text='"Yes, I am coming."', font=("Helvetica", 16, "bold"), bg="#2b385a", fg=COLORS["white"], padx=18, pady=10)
        self.avatar_phrase_label.pack()

        self.controls_column = tk.Frame(self.hero_row, bg=COLORS["cream"])
        self.type_card = tk.Frame(self.controls_column, bg="#e2f1f4", highlightbackground="#d3e4ea", highlightthickness=1)
        header = tk.Frame(self.type_card, bg="#e2f1f4")
        header.pack(fill=tk.X, padx=18, pady=(12, 8))
        tk.Label(header, text="✏️ TYPE TO SIGN", font=("Helvetica", 11, "bold"), bg="#e2f1f4", fg=COLORS["teal"]).pack(anchor="w")
        input_row = tk.Frame(self.type_card, bg="#e2f1f4")
        input_row.pack(fill=tk.X, padx=18, pady=(0, 16))
        self.text_entry = tk.Entry(input_row, font=("Helvetica", 14), bg=COLORS["cream"], fg="#6b7280", relief=tk.FLAT, bd=0)
        self.text_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=12)
        self.text_entry.insert(0, "Type your message...")
        self.text_entry.bind("<Return>", lambda _e: self._on_sign_button())
        self.sign_button = tk.Button(input_row, text="→ Sign", font=("Helvetica", 13, "bold"), bg=COLORS["teal"], fg=COLORS["white"], relief=tk.FLAT, bd=0, padx=20, pady=12, cursor="hand2", command=self._on_sign_button)
        self.sign_button.pack(side=tk.LEFT, padx=(10, 0))

        self.speak_button = tk.Button(self.controls_column, text="🎤\n\nTap to Speak\nVoice → sign language", font=("Helvetica", 18, "bold"), bg=COLORS["gold"], fg=COLORS["white"], relief=tk.FLAT, bd=0, cursor="hand2", justify=tk.CENTER, pady=30, command=self._on_speak_button)

        self.conversation_card = tk.Frame(self.content, bg=COLORS["white"], highlightbackground="#e2e8ef", highlightthickness=1, padx=18, pady=18)
        tk.Label(self.conversation_card, text="CONVERSATION", font=("Helvetica", 12, "bold"), bg=COLORS["white"], fg="#97a2b5").pack(anchor="w", pady=(0, 12))
        self.history_frame = tk.Frame(self.conversation_card, bg=COLORS["white"])
        self.history_frame.pack(fill=tk.BOTH, expand=True)

        self.mobile_nav = MobileNavBar(self, active_screen="speaker", navigate=self.navigate, role="speaker")
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Ready")
        self.sigml_sender = get_sigml_sender()

        self.after_idle(self._apply_layout)

    def _logout(self):
        self.state.logout()
        self.navigate("login")

    def _apply_layout(self):
        width = max(self.winfo_width(), self.winfo_reqwidth())
        mode = "desktop" if width >= 900 else "mobile"
        if mode == self._layout_mode:
            return
        self._layout_mode = mode

        self.desktop_sidebar.pack_forget()
        self.main_area.pack_forget()
        self.scroll_canvas.pack_forget()
        self.scrollbar.pack_forget()
        self.mobile_nav.pack_forget()
        self.mobile_top.pack_forget()
        self.hero_row.pack_forget()
        self.avatar_card.pack_forget()
        self.controls_column.pack_forget()
        self.type_card.pack_forget()
        self.speak_button.pack_forget()
        self.conversation_card.pack_forget()

        if mode == "desktop":
            self.desktop_sidebar.pack(side=tk.LEFT, fill=tk.Y)
            self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.header.pack(fill=tk.X)
            self.mobile_top.pack_forget()
            self.hero_row.pack(fill=tk.X, padx=28, pady=(18, 18))
            self.avatar_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 18))
            self.controls_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.type_card.pack(fill=tk.X, pady=(0, 14))
            self.speak_button.pack(fill=tk.BOTH, expand=True)
            self.conversation_card.pack(fill=tk.BOTH, expand=True, padx=28, pady=(0, 28))
        else:
            self.main_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.header.pack_forget()
            self.mobile_top.pack(fill=tk.X)
            self.hero_row.pack(fill=tk.X, padx=12, pady=(12, 12))
            self.avatar_card.pack(fill=tk.X, pady=(0, 12))
            self.controls_column.pack(fill=tk.X)
            self.type_card.pack(fill=tk.X, pady=(0, 12))
            self.speak_button.pack(fill=tk.X)
            self.conversation_card.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
            self.mobile_nav.pack(side=tk.BOTTOM, fill=tk.X)

    def _on_resize(self, _event):
        self._apply_layout()

    def _render_history(self):
        for child in self.history_frame.winfo_children():
            child.destroy()

        messages = self.state.conv_history[-4:] if self.state.conv_history else []

        if not messages:
            tk.Label(
                self.history_frame,
                text="No conversations yet",
                font=("Helvetica", 12, "italic"),
                bg=COLORS["white"],
                fg="#97a2b5",
                pady=12,
            ).pack(anchor="w")
            return

        for entry in messages:
            author = entry.get("who", "You")
            bubble_bg = COLORS["white"] if author == "You" else "#e2f1f4"
            label_fg = COLORS["teal"] if author == "You" else COLORS["gold"]
            row = tk.Frame(self.history_frame, bg=COLORS["white"])
            row.pack(fill=tk.X, pady=8)
            tk.Label(
                row,
                text=author,
                font=("Helvetica", 10, "bold"),
                bg=COLORS["white"],
                fg=label_fg,
            ).pack(anchor="w")
            tk.Label(
                row,
                text=entry.get("text", ""),
                font=("Helvetica", 12),
                bg=bubble_bg,
                fg=COLORS["navy"],
                wraplength=520,
                justify=tk.LEFT,
                padx=14,
                pady=10,
            ).pack(fill=tk.X, pady=(4, 0))

    def on_show(self):
        self._apply_layout()
        self._render_history()
        self.status_bar.set_live(False)
        self.status_bar.set_text("Status: Ready")
        self.text_entry.focus()

    def _on_sign_button(self):
        text = self.text_entry.get().strip()
        if not text or text == "Type your message...":
            return
        self.avatar_phrase_label.config(text=text)
        self.state.add_message("You", text)
        self._render_history()

        def on_done():
            self.after(0, lambda: self.status_bar.set_text("Status: Ready"))

        self.status_bar.set_text("Status: Avatar signing")
        self.sigml_sender.speak_to_avatar(text, on_done)
        self.text_entry.delete(0, tk.END)

    def _on_speak_button(self):
        self.status_bar.set_text("Status: Listening")
        def mock_speech():
            import time
            time.sleep(2)
            mock_text = "Hello, how are you?"
            self.after(0, lambda: self._handle_speech_result(mock_text))
        threading.Thread(target=mock_speech, daemon=True).start()

    def _handle_speech_result(self, text: str):
        self.text_entry.delete(0, tk.END)
        self.text_entry.insert(0, text)
        self.status_bar.set_text("Status: Ready")
        self.text_entry.focus()

# temp to sstart speaker screen to see how it looks
# def main():
#     root = tk.Tk()
#     root.geometry("400x600")
#     state = AppState()
#     speaker_screen = SpeakerScreen(root, state, lambda x: print(f"Navigate to {x}"))
#     speaker_screen.pack(fill=tk.BOTH, expand=True)
#     root.mainloop()

# if __name__ == "__main__":
#     main()