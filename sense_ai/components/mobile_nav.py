import tkinter as tk

from theme import COLORS


class MobileNavBar(tk.Frame):
    """Bottom mobile navigation matching the phone mockups."""

    def __init__(self, parent, active_screen, navigate, role="signer"):
        super().__init__(parent, bg=COLORS["white"], height=78, highlightbackground="#dfe4ea", highlightthickness=1)
        self.pack_propagate(False)

        primary_screen = "signer" if role == "signer" else "speaker"
        primary_label = "Sign" if role == "signer" else "Speak"
        primary_icon = "🤟" if role == "signer" else "🗣️"

        items = [
            (primary_screen, primary_icon, primary_label),
            ("history", "💭", "History"),
            ("settings", "⚙️", "Settings"),
        ]

        for screen_name, icon, label in items:
            self._item(screen_name, icon, label, active_screen == screen_name, navigate)

    def _item(self, screen_name, icon, label, active, navigate):
        fg = COLORS["teal"] if active else "#9ca6b6"
        cell = tk.Frame(self, bg=COLORS["white"])
        cell.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(cell, text=icon, font=("Helvetica", 18), bg=COLORS["white"], fg=fg).pack(pady=(10, 2))
        tk.Label(cell, text=label, font=("Helvetica", 9, "bold"), bg=COLORS["white"], fg=fg).pack()

        def handle(_event=None):
            navigate(screen_name)

        cell.bind("<Button-1>", handle)
        for child in cell.winfo_children():
            child.bind("<Button-1>", handle)