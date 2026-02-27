# components/detection_chips.py — tier detection percentage chips

import tkinter as tk
from theme import COLORS, FONTS


class DetectionChips(tk.Frame):
    """
    Row of 3 colored pill labels showing tier detection %.
    Args: parent
    Chips:
      "Physical XX%"  — blue:   fg #2563eb, bg #dde8fb
      "Grammar XX%"   — purple: fg #7c3aed, bg #ece5fb
      "Semantic XX%"  — pink:   fg #db2777, bg #fce5f0
    Each chip: tk.Label, font Helvetica 8 bold, padx 6, pady 2,
               highlightbackground (same as fg), highlightthickness 1
    Method:
      update(physical: int, grammar: int, semantic: int) — update text values
    """

    BLUE = "#2563eb"
    PURPLE = "#7c3aed"
    PINK = "#db2777"

    def __init__(self, parent):
        super().__init__(parent, bg=COLORS["cream"])
        self.pack(fill=tk.X, padx=6, pady=4)

        # Define chip styles
        chips_config = [
            ("Physical", self.BLUE, "#dde8fb"),
            ("Grammar", self.PURPLE, "#ece5fb"),
            ("Semantic", self.PINK, "#fce5f0"),
        ]

        self.chip_labels = {}

        for name, fg_color, bg_color in chips_config:
            label = tk.Label(
                self,
                text=f"{name} 0%",
                font=FONTS["tiny"],
                fg=fg_color,
                bg=bg_color,
                highlightbackground=fg_color,
                highlightthickness=1,
                padx=6,
                pady=2
            )
            label.pack(side=tk.LEFT, padx=3)
            self.chip_labels[name.lower()] = label

    def update_chips(self, physical: int = 0, grammar: int = 0, semantic: int = 0):
        """Update chip percentages."""
        self.chip_labels["physical"].config(text=f"Physical {physical}%")
        self.chip_labels["grammar"].config(text=f"Grammar {grammar}%")
        self.chip_labels["semantic"].config(text=f"Semantic {semantic}%")
