# components/sidebar.py

import tkinter as tk
from tkinter import messagebox
from theme import COLORS, FONTS


class Sidebar(tk.Frame):
  """Desktop navigation sidebar matching the wider mockups."""

  def __init__(self, parent, active_screen, navigate, role="signer", on_logout=None):
    super().__init__(parent, bg=COLORS["navy"], width=262)
    self.pack_propagate(False)
    self.navigate = navigate
    self.active_screen = active_screen
    self.role = role
    self.on_logout = on_logout or (lambda: None)

    accent = COLORS["teal"] if role == "signer" else COLORS["gold"]

    header = tk.Frame(self, bg=COLORS["navy"], height=86)
    header.pack(fill=tk.X)
    header.pack_propagate(False)
    tk.Label(header, text="🤟", font=("Helvetica", 22), bg=COLORS["navy"], fg=COLORS["gold_lt"]).pack(side=tk.LEFT, padx=(28, 12), pady=22)
    tk.Label(header, text="Sense.AI", font=("Georgia", 20, "bold"), bg=COLORS["navy"], fg=COLORS["white"]).pack(side=tk.LEFT, pady=24)

    sections = [
      ("MAIN", [
        ("signer", "🤟", "Translate (Signer)"),
        ("speaker", "🗣️", "Speak (Speaker)"),
      ]),
      ("MANAGE", [
        ("history", "💬", "History"),
        ("settings", "⚙️", "Settings"),
      ]),
      ("ACCOUNT", [
        ("session", "🔄", "Switch Role"),
      ]),
    ]

    for title, items in sections:
      tk.Label(self, text=title, font=("Helvetica", 10, "bold"), bg=COLORS["navy"], fg="#5f6c8b").pack(anchor="w", padx=24, pady=(18, 8))
      for screen_name, icon, label in items:
        self._nav_button(screen_name, icon, label, accent)

    tk.Frame(self, bg=COLORS["navy"]).pack(fill=tk.BOTH, expand=True)

    footer = tk.Frame(self, bg=COLORS["navy"], height=72)
    footer.pack(fill=tk.X, side=tk.BOTTOM)
    footer.pack_propagate(False)
    avatar = tk.Label(footer, text="🤟", font=("Helvetica", 18), bg=COLORS["white"], fg=COLORS["gold"], width=2)
    avatar.pack(side=tk.LEFT, padx=(18, 10), pady=16)
    info = tk.Frame(footer, bg=COLORS["navy"])
    info.pack(side=tk.LEFT, pady=14)
    tk.Label(info, text="Maya Johnson", font=("Helvetica", 12, "bold"), bg=COLORS["navy"], fg=COLORS["white"]).pack(anchor="w")
    tk.Label(info, text=role.capitalize(), font=("Helvetica", 10), bg=COLORS["navy"], fg=accent).pack(anchor="w")
    tk.Button(footer, text="⎋", font=("Helvetica", 12), bg=COLORS["navy"], fg="#8b97b2", relief=tk.FLAT, bd=0, cursor="hand2", command=self.on_logout).pack(side=tk.RIGHT, padx=18)

  def _nav_button(self, screen_name, icon, label, accent):
    is_locked = (
      (self.active_screen == "speaker" and screen_name == "signer") or
      (self.active_screen == "signer" and screen_name == "speaker")
    )
    is_active = screen_name == self.active_screen
    bg = "#15516e" if is_active else COLORS["navy"]
    fg = COLORS["white"] if is_active else ("#7f8aa5" if is_locked else "#c3cadb")
    row = tk.Frame(self, bg=bg, height=48)
    row.pack(fill=tk.X, padx=12, pady=4)
    row.pack_propagate(False)
    tk.Label(row, text=icon, font=("Helvetica", 16), bg=bg, fg=accent if is_active else ("#6e7895" if is_locked else "#8b97b2")).pack(side=tk.LEFT, padx=(18, 12))
    tk.Label(row, text=label, font=("Helvetica", 12, "bold"), bg=bg, fg=fg).pack(side=tk.LEFT)

    def handle_click(_event=None):
      if is_locked:
        messagebox.showinfo("Switch Role", "Go to Settings to switch role.")
        return
      self.navigate(screen_name)

    for widget in (row,):
      widget.bind("<Button-1>", handle_click)
    for child in row.winfo_children():
      child.bind("<Button-1>", handle_click)
