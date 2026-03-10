import tkinter as tk

from components.mobile_nav import MobileNavBar
from components.sidebar import Sidebar
from components.status_bar import StatusBar
from theme import COLORS


class HistoryScreen(tk.Frame):
    SCREEN_NAME = "history"

    SAMPLE_ITEMS = [
        ("TODAY", "Sarah K.", "2:34 PM · 12 min", '"Are you coming tomorrow?" → "Yes, I\'ll be there at 3 pm."', "YES_NO_Q", "8 turns"),
        ("TODAY", "Dr. Patel", "10:15 AM · 24 min", '"What time is the appointment?" → "Friday at 10 am."', "WH_Q", "15 turns"),
        ("YESTERDAY", "Alex M.", "4:50 PM · 6 min", '"Do you want coffee?" → "No, thank you."', "NEGATION", "4 turns"),
        ("YESTERDAY", "Team Meeting", "9:00 AM · 45 min", '"When is the deadline?" → "End of next week."', "WH_Q", "32 turns"),
    ]

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate
        self.role = self.state.user_role or "signer"
        self._layout_mode = None

        self.bind("<Configure>", self._on_resize)

        self.desktop_sidebar = Sidebar(self, active_screen="history", navigate=self.navigate, role=self.role, on_logout=self._logout)
        self.main_area = tk.Frame(self, bg=COLORS["cream"])
        self.header = tk.Frame(self.main_area, bg=COLORS["cream"], height=70)
        self.header.pack(fill=tk.X)
        self.header.pack_propagate(False)
        self.title_label = tk.Label(self.header, text="History", font=("Georgia", 24, "bold"), bg=COLORS["cream"], fg=COLORS["navy"])
        self.title_label.pack(side=tk.LEFT, padx=28, pady=18)
        self.badge = tk.Label(self.header, text="4 sessions today", font=("Helvetica", 11, "bold"), bg="#dff0f3", fg=COLORS["teal"], padx=14, pady=8)
        self.badge.pack(side=tk.RIGHT, padx=28, pady=16)

        self.scroll_canvas = tk.Canvas(self.main_area, bg=COLORS["cream"], highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.main_area, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.content = tk.Frame(self.scroll_canvas, bg=COLORS["cream"])
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda _e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.bind("<Configure>", lambda e: self.scroll_canvas.itemconfigure(self.canvas_window, width=e.width))

        self.search_frame = tk.Frame(self.content, bg=COLORS["white"], highlightbackground="#d8dfe8", highlightthickness=1)
        self.search_frame.pack(fill=tk.X, padx=28, pady=(16, 18))
        tk.Label(self.search_frame, text="🔍", font=("Helvetica", 14), bg=COLORS["white"], fg="#8e99ac").pack(side=tk.LEFT, padx=(18, 8), pady=12)
        tk.Label(self.search_frame, text="Search past conversations...", font=("Helvetica", 12), bg=COLORS["white"], fg="#97a2b5").pack(side=tk.LEFT, pady=12)

        self.section_frames = {}
        for section in ("TODAY", "YESTERDAY"):
            frame = tk.Frame(self.content, bg=COLORS["cream"])
            frame.pack(fill=tk.X, padx=28, pady=(0, 16))
            tk.Label(frame, text=section, font=("Helvetica", 11, "bold"), bg=COLORS["cream"], fg="#8f9aab").pack(anchor="w", pady=(0, 10))
            cards = tk.Frame(frame, bg=COLORS["cream"])
            cards.pack(fill=tk.X)
            self.section_frames[section] = cards

        for item in self.SAMPLE_ITEMS:
            self._add_card(*item)

        self.mobile_nav = MobileNavBar(self, active_screen="history", navigate=self.navigate, role=self.role)
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: History ready")
        self.after_idle(self._apply_layout)

    def _logout(self):
        self.state.logout()
        self.navigate("login")

    def _refresh_navigation(self):
        role = self.state.user_role or "signer"
        if role == self.role:
            return

        self.role = role
        self.desktop_sidebar.destroy()
        self.mobile_nav.destroy()
        self.desktop_sidebar = Sidebar(self, active_screen="history", navigate=self.navigate, role=self.role, on_logout=self._logout)
        self.mobile_nav = MobileNavBar(self, active_screen="history", navigate=self.navigate, role=self.role)

    def _add_card(self, section, title, meta, summary, tag, turns):
        parent = self.section_frames[section]
        card = tk.Frame(parent, bg=COLORS["white"], highlightbackground="#e3e8ef", highlightthickness=1, padx=18, pady=18)
        icon = "🗣️" if "Dr." in title or title == "Alex M." else "👥" if "Team" in title else "🗣"
        tk.Label(card, text=icon, font=("Helvetica", 18), bg=COLORS["white"], fg="#8f7ab4").pack(side=tk.LEFT, anchor="n", padx=(0, 12))
        content = tk.Frame(card, bg=COLORS["white"])
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        head = tk.Frame(content, bg=COLORS["white"])
        head.pack(fill=tk.X)
        tk.Label(head, text=title, font=("Helvetica", 14, "bold"), bg=COLORS["white"], fg=COLORS["navy"]).pack(side=tk.LEFT)
        tk.Label(head, text=meta, font=("Helvetica", 10), bg=COLORS["white"], fg="#9ea8b6").pack(side=tk.RIGHT)
        tk.Label(content, text=summary, font=("Helvetica", 12), bg=COLORS["white"], fg="#556277", wraplength=360, justify=tk.LEFT).pack(anchor="w", pady=(10, 12))
        footer = tk.Frame(content, bg=COLORS["white"])
        footer.pack(anchor="w")
        tk.Label(footer, text=tag, font=("Helvetica", 9, "bold"), bg="#f0e9ff" if tag == "WH_Q" else "#ffe9ea" if tag == "NEGATION" else "#efeaff", fg="#7c3aed" if tag != "NEGATION" else "#eb5b64", padx=8, pady=3).pack(side=tk.LEFT)
        tk.Label(footer, text=turns, font=("Helvetica", 9, "bold"), bg="#dff0f3" if section == "TODAY" else "#e2f3f8", fg=COLORS["teal"], padx=8, pady=3).pack(side=tk.LEFT, padx=(8, 0))
        card._section = section
        card._meta = meta
        card._title = title
        card._summary = summary
        parent.children[str(len(parent.children))] = card

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

        for cards in self.section_frames.values():
            for child in cards.winfo_children():
                child.pack_forget()

        if mode == "desktop":
            self.desktop_sidebar.pack(side=tk.LEFT, fill=tk.Y)
            self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.title_label.configure(font=("Georgia", 24, "bold"))
            self.badge.pack(side=tk.RIGHT, padx=28, pady=16)
            self.search_frame.pack_configure(padx=28, pady=(16, 18))
            for section in self.section_frames.values():
                row = section.winfo_children()
                for index, child in enumerate(row):
                    child.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14) if index % 2 == 0 else (14, 0), pady=0)
        else:
            self.main_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.mobile_nav.pack(side=tk.BOTTOM, fill=tk.X)
            self.title_label.configure(font=("Georgia", 20, "bold"))
            self.badge.pack_forget()
            self.search_frame.pack_configure(padx=22, pady=(14, 16))
            for section in self.section_frames.values():
                for child in section.winfo_children():
                    child.pack(fill=tk.X, pady=6)

    def _on_resize(self, _event):
        self._apply_layout()

    def on_show(self):
        self._refresh_navigation()
        self._apply_layout()
        self.status_bar.set_text("Status: History ready")
