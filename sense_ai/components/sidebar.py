# components/sidebar.py — left icon sidebar

import tkinter as tk
from theme import COLORS, FONTS


class Sidebar(tk.Frame):
    """
    Left column, width=52, full height, navy bg.
    Args: parent, role ("signer"|"speaker"), on_logout (callable)
    - Vertical role label using Canvas.create_text with angle=90
      color: teal for signer, gold for speaker
    - 3 icon buttons (32x32 tk.Button, relief FLAT):
      icons depend on role:
        signer:  ["📷", "⚙️", "📋"]
        speaker: ["⌨️", "🎤", "📋"]
      first button is active (teal/gold bg), rest are dark translucent
    - Bottom: "Out" button (danger red bg, white text, width=40)
    """

    def __init__(self, parent, role="signer", on_logout=None):
        super().__init__(parent, bg=COLORS["navy"], width=52)
        self.pack_propagate(False)
        self.pack(side=tk.LEFT, fill=tk.Y)

        self.role = role
        self.on_logout = on_logout or (lambda: None)

        # Role label canvas (rotated text)
        role_canvas = tk.Canvas(self, bg=COLORS["navy"], highlightthickness=0, width=52, height=80)
        role_canvas.pack(side=tk.TOP, pady=12)

        role_text = "SIGNER" if role == "signer" else "SPEAKER"
        role_color = COLORS["teal"] if role == "signer" else COLORS["gold"]
        role_canvas.create_text(26, 40, text=role_text, font=FONTS["label"], fill=role_color, angle=90)

        # Icon buttons container
        icons_frame = tk.Frame(self, bg=COLORS["navy"])
        icons_frame.pack(side=tk.TOP, pady=8)

        signer_icons = ["📷", "⚙️", "📋"]
        speaker_icons = ["⌨️", "🎤", "📋"]
        icons = signer_icons if role == "signer" else speaker_icons
        active_color = COLORS["teal"] if role == "signer" else COLORS["gold"]

        for i, icon in enumerate(icons):
            btn = tk.Button(
                icons_frame,
                text=icon,
                font=("Helvetica", 16),
                bg=active_color if i == 0 else COLORS["navy_dark"],
                fg=COLORS["white"],
                relief=tk.FLAT,
                width=3,
                height=1,
                cursor="hand2" if i == 0 else "arrow"
            )
            btn.pack(pady=4)

        # Spacer to push logout button to bottom
        tk.Frame(self, bg=COLORS["navy"]).pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # Logout button at bottom
        logout_btn = tk.Button(
            self,
            text="Out",
            font=FONTS["body_bold"],
            bg=COLORS["danger"],
            fg=COLORS["white"],
            relief=tk.FLAT,
            width=4,
            height=2,
            cursor="hand2",
            command=self.on_logout
        )
        logout_btn.pack(side=tk.BOTTOM, pady=8)
