# components/tier_confidence.py — three-tier confidence progress bars

import tkinter as tk
from theme import COLORS, FONTS


class TierConfidence(tk.Frame):
    """
    Three horizontal progress bar rows.
    Each row: colored dot (Canvas oval) | label | progress bar Frame | percentage label
    Colors:
      PHYSICAL: #2563eb
      GRAMMAR:  #7c3aed
      SEMANTIC: #db2777
    Progress bar: outer Frame with border, inner Frame width set as percentage of total
    Method:
      update(physical: int, grammar: int, semantic: int) — animate bar widths
    """

    PHYSICAL = "#2563eb"
    GRAMMAR = "#7c3aed"
    SEMANTIC = "#db2777"

    def __init__(self, parent):
        super().__init__(parent, bg=COLORS["cream"])
        self.pack(fill=tk.X, padx=6, pady=6)

        self.bars = {}

        tiers = [
            ("Physical", self.PHYSICAL, "physical"),
            ("Grammar", self.GRAMMAR, "grammar"),
            ("Semantic", self.SEMANTIC, "semantic"),
        ]

        for label_text, color, tier_name in tiers:
            # Row frame
            row = tk.Frame(self, bg=COLORS["cream"])
            row.pack(fill=tk.X, pady=3)

            # Dot canvas
            dot_canvas = tk.Canvas(row, bg=COLORS["cream"], highlightthickness=0, width=12, height=12)
            dot_canvas.pack(side=tk.LEFT, padx=(0, 6))
            dot_canvas.create_oval(3, 3, 9, 9, fill=color, outline=color)

            # Label
            tk.Label(row, text=label_text, font=FONTS["tiny"], fg=color, bg=COLORS["cream"]).pack(side=tk.LEFT, width=8)

            # Progress bar (outer)
            bar_outer = tk.Frame(row, bg=COLORS["border"], height=8, relief=tk.SOLID, bd=1)
            bar_outer.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
            bar_outer.pack_propagate(False)

            # Progress bar (inner)
            bar_inner = tk.Frame(bar_outer, bg=color, height=6)
            bar_inner.pack(side=tk.LEFT, fill=tk.Y)
            bar_inner_label = tk.Label(row, text="0%", font=FONTS["tiny"], fg=color, bg=COLORS["cream"], width=3)
            bar_inner_label.pack(side=tk.LEFT, padx=6)

            self.bars[tier_name] = {
                "outer": bar_outer,
                "inner": bar_inner,
                "label": bar_inner_label,
                "color": color,
            }

    def update_tiers(self, physical: int = 0, grammar: int = 0, semantic: int = 0):
        """Update tier progress bars with percentage values."""
        for tier_name, value in [("physical", physical), ("grammar", grammar), ("semantic", semantic)]:
            bar_data = self.bars[tier_name]
            # Animate width change (simplified: just set it)
            outer_width = bar_data["outer"].winfo_width()
            if outer_width > 1:
                new_width = int((value / 100) * (outer_width - 2))
                bar_data["inner"].configure(width=max(0, new_width))
            bar_data["label"].config(text=f"{value}%")
