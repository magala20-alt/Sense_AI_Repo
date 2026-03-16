from logging import root
import tkinter as tk

try:
    from app_state import AppState
    from config import BACKEND_WS_URL, CAMERA_INDEX
    from components.mobile_nav import MobileNavBar
    from components.status_bar import StatusBar
    from components.sidebar import Sidebar
    from services.websocket_client import create_websocket_client
    from theme import COLORS
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        
from app_state import AppState
from components.mobile_nav import MobileNavBar
from components.sidebar import Sidebar
from components.status_bar import StatusBar
from theme import COLORS


class SettingsScreen(tk.Frame):
    SCREEN_NAME = "settings"

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate
        self.role = self.state.user_role or "signer"
        self._layout_mode = None

        self.analysis_var = tk.BooleanVar(value=True)
        self.grammar_var = tk.BooleanVar(value=True)
        self.notifications_var = tk.BooleanVar(value=False)

        self.bind("<Configure>", self._on_resize)

        self.desktop_sidebar = Sidebar(self, active_screen="settings", navigate=self.navigate, role=self.role, on_logout=self._logout)
        self.main_area = tk.Frame(self, bg=COLORS["cream"])
        header = tk.Frame(self.main_area, bg=COLORS["cream"], height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="Settings", font=("Georgia", 24, "bold"), bg=COLORS["cream"], fg=COLORS["navy"]).pack(side=tk.LEFT, padx=28, pady=18)

        self.scroll_canvas = tk.Canvas(self.main_area, bg=COLORS["cream"], highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.main_area, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.content = tk.Frame(self.scroll_canvas, bg=COLORS["cream"])
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda _e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.bind("<Configure>", lambda e: self.scroll_canvas.itemconfigure(self.canvas_window, width=e.width))

        self.profile_card = tk.Frame(self.content, bg=COLORS["white"], highlightbackground="#e3e8ef", highlightthickness=1, padx=28, pady=28)
        tk.Label(self.profile_card, text="🤟", font=("Helvetica", 28), bg=COLORS["white"], fg=COLORS["gold"]).pack(pady=(0, 10))
        tk.Label(self.profile_card, text="Maya Johnson", font=("Helvetica", 18, "bold"), bg=COLORS["white"], fg=COLORS["navy"]).pack()
        tk.Label(self.profile_card, text="Signer", font=("Helvetica", 10, "bold"), bg=COLORS["teal"], fg=COLORS["white"], padx=12, pady=3).pack(pady=10)
        tk.Button(self.profile_card, text="Edit Profile", font=("Helvetica", 11, "bold"), bg=COLORS["white"], fg=COLORS["teal"], relief=tk.FLAT, highlightbackground=COLORS["teal"], highlightthickness=1, padx=18, pady=8).pack()

        self.right_stack = tk.Frame(self.content, bg=COLORS["cream"])
        self.communication_card = self._settings_group(self.right_stack, "COMMUNICATION", [
            ("🔄", "Switch Role", "Currently: Signer", None, lambda: self.navigate("session")),
            ("🤟", "Signing Avatar", "CWASA · Anna", None, None),
            ("🌐", "Language", "ASL — American", None, None),
        ])
        self.detection_card = self._settings_group(self.right_stack, "DETECTION", [
            ("📷", "Camera", "Front-facing", None, None),
            ("✨", "Three-Tier Analysis", "Physical · Grammar · Semantic", self.analysis_var, None),
            ("🎯", "Grammar Badges", "Show YES_NO_Q, WH_Q tags", self.grammar_var, None),
        ])
        self.app_card = self._settings_group(self.right_stack, "APP", [
            ("🔔", "Notifications", "", self.notifications_var, None),
            ("🚪", "Sign Out", "", None, lambda: self.navigate("login")),
        ], danger_last=True)

        self.mobile_nav = MobileNavBar(self, active_screen="settings", navigate=self.navigate, role=self.role)
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Settings ready")
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
        self.desktop_sidebar = Sidebar(self, active_screen="settings", navigate=self.navigate, role=self.role, on_logout=self._logout)
        self.mobile_nav = MobileNavBar(self, active_screen="settings", navigate=self.navigate, role=self.role)

        # Force next _apply_layout call to repack recreated widgets
        self._layout_mode = None


    def _settings_group(self, parent, title, items, danger_last=False):
        card = tk.Frame(parent, bg=COLORS["white"], highlightbackground="#e3e8ef", highlightthickness=1)
        tk.Label(card, text=title, font=("Helvetica", 10, "bold"), bg=COLORS["white"], fg="#8f9aab").pack(anchor="w", padx=18, pady=(14, 8))
        for index, (icon, name, subtitle, variable, command) in enumerate(items):
            row = tk.Frame(card, bg=COLORS["white"], height=62)
            row.pack(fill=tk.X)
            row.pack_propagate(False)
            if index > 0:
                tk.Frame(row, bg="#e8edf2", height=1).pack(fill=tk.X, side=tk.TOP)
            icon_box = tk.Label(row, text=icon, font=("Helvetica", 14), bg="#eef4f5" if command is None else "#f1eef9", fg=COLORS["teal"], width=2)
            icon_box.pack(side=tk.LEFT, padx=(16, 12), pady=12)
            labels = tk.Frame(row, bg=COLORS["white"])
            labels.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10)
            title_fg = COLORS["danger"] if danger_last and index == len(items) - 1 else COLORS["navy"]
            tk.Label(labels, text=name, font=("Helvetica", 13, "bold"), bg=COLORS["white"], fg=title_fg).pack(anchor="w")
            if subtitle:
                tk.Label(labels, text=subtitle, font=("Helvetica", 10), bg=COLORS["white"], fg="#9ba5b5").pack(anchor="w")
            if variable is not None:
                tk.Checkbutton(row, variable=variable, onvalue=True, offvalue=False, bg=COLORS["white"], activebackground=COLORS["white"], selectcolor=COLORS["white"], highlightthickness=0, bd=0).pack(side=tk.RIGHT, padx=18)
            else:
                tk.Label(row, text="›", font=("Helvetica", 16, "bold"), bg=COLORS["white"], fg="#b0b9c6").pack(side=tk.RIGHT, padx=18)
            if command:
                row.bind("<Button-1>", lambda _e, action=command: action())
                for child in row.winfo_children():
                    child.bind("<Button-1>", lambda _e, action=command: action())
        return card

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
        self.profile_card.pack_forget()
        self.right_stack.pack_forget()
        self.communication_card.pack_forget()
        self.detection_card.pack_forget()
        self.app_card.pack_forget()

        if mode == "desktop":
            self.desktop_sidebar.pack(side=tk.LEFT, fill=tk.Y)
            self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.profile_card.pack(side=tk.LEFT, fill=tk.X, padx=(28, 12), pady=24, anchor="n")
            self.right_stack.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 28), pady=24)
            self.communication_card.pack(fill=tk.X, pady=(0, 16))
            self.detection_card.pack(fill=tk.X, pady=(0, 16))
            self.app_card.pack(fill=tk.X)
        else:
            self.mobile_nav.pack(side=tk.LEFT, fill=tk.Y)
            self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.profile_card.pack(fill=tk.X, padx=22, pady=(18, 14))
            self.communication_card.pack(fill=tk.X, padx=22, pady=(0, 14))
            self.detection_card.pack(fill=tk.X, padx=22, pady=(0, 14))
            self.app_card.pack(fill=tk.X, padx=22, pady=(0, 18))

    def _on_resize(self, _event):
        self._apply_layout()

    def on_show(self):
        self._refresh_navigation()
        self._apply_layout()
        self.status_bar.set_text("Status: Settings ready")

# temp to start settings screen to see how it looks
def main():
        root = tk.Tk()
        root.geometry("400x600")
        state = AppState()
        settings_screen = SettingsScreen(root, state, lambda x: print(f"Navigate to {x}"))
        settings_screen.pack(fill=tk.BOTH, expand=True)
        root.mainloop()
        
if __name__ == "__main__":
    main()