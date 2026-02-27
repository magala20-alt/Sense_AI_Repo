# components/conv_history.py — scrollable conversation history

import tkinter as tk
from theme import COLORS, FONTS


class ConversationHistory(tk.Frame):
    """
    Scrollable conversation log.
    Uses: outer Frame → tk.Canvas + tk.Scrollbar → inner tk.Frame
    Header: teal_bg bg, "💬 Conversation History" teal label
    Method:
      add_entry(who: str, text: str, grammar_type: str)
        who="You"     → teal name label
        who="Speaker" → gold name label
        Appends a new row Frame to inner Frame
        Auto-scrolls to bottom after adding
        Alternates row bg between white and teal_bg
    """

    def __init__(self, parent):
        super().__init__(parent, bg=COLORS["cream"])
        self.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Header
        header = tk.Frame(self, bg=COLORS["teal_bg"])
        header.pack(fill=tk.X, pady=(0, 4))
        tk.Label(
            header, text="💬 Conversation History",
            font=FONTS["label"], fg=COLORS["teal"],
            bg=COLORS["teal_bg"]
        ).pack(padx=6, pady=4)

        # Canvas + Scrollbar (scrollable)
        canvas_frame = tk.Frame(self, bg=COLORS["cream"])
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=COLORS["cream"], highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLORS["cream"])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.entry_count = 0

    def add_entry(self, who: str, text: str, grammar_type: str = ""):
        """Add a conversation entry."""
        # Alternate background
        bg_color = COLORS["white"] if self.entry_count % 2 == 0 else COLORS["teal_bg"]
        
        # Entry frame
        entry_frame = tk.Frame(self.scrollable_frame, bg=bg_color, relief=tk.SOLID, bd=0)
        entry_frame.pack(fill=tk.X, padx=0, pady=2)

        # Who label (teal or gold)
        who_color = COLORS["teal"] if who == "You" else COLORS["gold"]
        tk.Label(
            entry_frame, text=f"{who}:",
            font=FONTS["body_bold"], fg=who_color, bg=bg_color
        ).pack(anchor="w", padx=6, pady=(4, 0))

        # Message text
        tk.Label(
            entry_frame, text=text,
            font=FONTS["small"], fg=COLORS["text"],
            bg=bg_color, wraplength=340, justify=tk.LEFT
        ).pack(anchor="w", padx=6, pady=(0, 4))

        # Grammar tag (if present)
        if grammar_type:
            grammar_frame = tk.Frame(entry_frame, bg=bg_color)
            grammar_frame.pack(anchor="e", padx=6, pady=(0, 4))
            tk.Label(
                grammar_frame, text=grammar_type,
                font=FONTS["tiny"], fg=COLORS["text"],
                bg=bg_color, relief=tk.SOLID, bd=1,
                highlightbackground=COLORS["border"],
                highlightthickness=1, padx=3, pady=1
            ).pack()

        self.entry_count += 1
        self.canvas.yview_moveto(1.0)  # Auto-scroll to bottom

    def clear_history(self):
        """Clear all entries."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.entry_count = 0
