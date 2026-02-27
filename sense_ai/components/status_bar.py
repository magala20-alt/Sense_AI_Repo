# components/status_bar.py — bottom status strip with pulsing dot

import tkinter as tk
from theme import COLORS, FONTS


class StatusBar(tk.Frame):
    """
    Bottom strip. Height 28px. navy bg.
    Left: 6px circle Canvas — pulses green #2ecc8a using after() every 800ms
    Right: status text label in teal #2aa3b8, Courier 9
    Methods:
      set_text(text: str)   — update status message
      set_live(live: bool)  — start/stop pulsing
    """

    def __init__(self, parent):
        super().__init__(parent, bg=COLORS["navy"], height=28)
        self.pack_propagate(False)
        self.pack(side=tk.BOTTOM, fill=tk.X)

        # State
        self.is_live = False
        self.pulse_on = True
        self.status_text = "Status: Ready"

        # Left: pulsing dot canvas
        self.dot_canvas = tk.Canvas(
            self, bg=COLORS["navy"], highlightthickness=0, 
            width=20, height=28
        )
        self.dot_canvas.pack(side=tk.LEFT, padx=6)

        # Draw static pulsing dot (circle)
        self.dot_canvas.create_oval(6, 11, 14, 19, fill=COLORS["navy"], outline=COLORS["navy"])
        self.dot_id = None

        # Right: status text label
        self.status_label = tk.Label(
            self, text=self.status_text, font=FONTS["status"],
            bg=COLORS["navy"], fg=COLORS["teal_lt"], justify=tk.LEFT
        )
        self.status_label.pack(side=tk.LEFT, padx=6, anchor="w")

        # Spacer
        tk.Frame(self, bg=COLORS["navy"]).pack(side=tk.RIGHT, expand=True, fill=tk.X)

    def set_text(self, text: str):
        """Update status message."""
        self.status_text = text
        self.status_label.config(text=text)

    def set_live(self, live: bool):
        """Start/stop pulsing."""
        self.is_live = live
        if live:
            self._pulse()
        else:
            self.after_cancel(self.pulse_id) if hasattr(self, "pulse_id") else None
            self._draw_dot(False)

    def _pulse(self):
        """Pulse the dot between colors."""
        self.pulse_on = not self.pulse_on
        self._draw_dot(self.pulse_on)
        self.pulse_id = self.after(800, self._pulse)

    def _draw_dot(self, on: bool):
        """Draw the dot."""
        self.dot_canvas.delete("all")
        color = COLORS["success"] if on else COLORS["navy"]
        self.dot_canvas.create_oval(6, 11, 14, 19, fill=color, outline=color)
