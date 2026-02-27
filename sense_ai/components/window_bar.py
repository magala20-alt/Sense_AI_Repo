# components/window_bar.py — top title bar with colored dots

import tkinter as tk
from theme import COLORS, FONTS


class WindowBar(tk.Frame):
    """
    Top title bar with 3 colored dots on left, title centered.
    Args: parent, title (str), highlight (str) — first word colored gold
    Height: 40px, bg: navy #1a2744
    Dots: 9px circles using tk.Canvas, colors: #e05a5a, #e8a020, #2ecc8a
    Title: tk.Label centered, white text, highlight portion in #f5c84a
    """

    def __init__(self, parent, title="Sense.AI", highlight=None):
        super().__init__(parent, bg=COLORS["navy"], height=40)
        self.pack_propagate(False)
        self.pack(side=tk.TOP, fill=tk.X)

        # Canvas for colored dots
        dot_canvas = tk.Canvas(self, bg=COLORS["navy"], highlightthickness=0, height=40, width=35)
        dot_canvas.pack(side=tk.LEFT, padx=8)

        # Draw 3 colored dots
        dot_colors = [COLORS["danger"], COLORS["gold"], COLORS["success"]]
        for i, color in enumerate(dot_colors):
            x = 8 + i * 12
            y = 20
            dot_canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline=color)

        # Title label (centered)
        title_frame = tk.Frame(self, bg=COLORS["navy"])
        title_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        if highlight and highlight in title:
            # Split title into highlight and rest
            parts = title.split(highlight, 1)
            before = parts[0]
            after = parts[1] if len(parts) > 1 else ""

            title_container = tk.Frame(title_frame, bg=COLORS["navy"])
            title_container.pack(expand=True)

            if before:
                tk.Label(title_container, text=before, font=FONTS["body_bold"], bg=COLORS["navy"], fg=COLORS["white"]).pack(side=tk.LEFT)
            tk.Label(title_container, text=highlight, font=FONTS["body_bold"], bg=COLORS["navy"], fg=COLORS["gold_lt"]).pack(side=tk.LEFT)
            if after:
                tk.Label(title_container, text=after, font=FONTS["body_bold"], bg=COLORS["navy"], fg=COLORS["white"]).pack(side=tk.LEFT)
        else:
            tk.Label(title_frame, text=title, font=FONTS["body_bold"], bg=COLORS["navy"], fg=COLORS["white"]).pack(expand=True)

        # Spacer on right
        tk.Frame(self, bg=COLORS["navy"], width=8).pack(side=tk.RIGHT)
