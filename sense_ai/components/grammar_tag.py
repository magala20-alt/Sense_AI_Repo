# components/grammar_tag.py — grammar type badge label

import tkinter as tk
from theme import COLORS, FONTS


class GrammarTag(tk.Label):
    """
    Small label with purple border and text.
    Args: parent, grammar_type (str)
    Style: bg white, fg #7c3aed, font Helvetica 8 bold,
           highlightbackground #7c3aed, highlightthickness 1,
           padx 4, pady 1
    Method:
      update(grammar_type: str) — change text
    """

    PURPLE = "#7c3aed"

    def __init__(self, parent, grammar_type=""):
        super().__init__(
            parent,
            text=grammar_type,
            font=FONTS["tiny"],
            bg=COLORS["white"],
            fg=self.PURPLE,
            highlightbackground=self.PURPLE,
            highlightthickness=1,
            padx=4,
            pady=1
        )

    def update_grammar(self, grammar_type: str):
        """Update grammar type text."""
        self.config(text=grammar_type)
