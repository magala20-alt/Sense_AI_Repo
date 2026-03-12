# components/face_avatar.py — face avatar drawn with canvas primitives

import tkinter as tk
from theme import COLORS

# this shall connect to the AI-generated avatar in the future, but for now it's a static drawing using canvas primitives to match the mockups.
class FaceAvatar(tk.Canvas):
    """
    Draws a face avatar using tk.Canvas primitives.
    Args: parent, size=80
    Drawing order (use create_oval, create_arc, create_line):
      1. Hair: dark oval, color #4a3728
      2. Ears: two small ovals, skin #fcd7a8
      3. Face: larger oval, skin #fcd7a8
      4. Eyes: white ovals + dark blue iris ovals #2c4a6e
      5. Eyebrows: create_line or create_arc, color #4a3728
      6. Nose: small arc, color #c8956a
      7. Smile: create_arc (arc style=ARC), color #c87840
      8. Shoulders: create_arc at bottom, color teal #1d7a8a, width=10
    """

    HAIR = "#4a3728"
    SKIN = "#fcd7a8"
    IRIS = "#2c4a6e"
    NOSE = "#c8956a"
    MOUTH = "#c87840"
    SHOULDERS = COLORS["teal"]

    def __init__(self, parent, size=80):
        super().__init__(parent, width=size, height=size, bg=COLORS["cream"], highlightthickness=0)
        self.size = size
        self._draw_face()

    def _draw_face(self):
        """Draw the avatar face."""
        s = self.size
        mid = s / 2

        # Hair (dark oval at top)
        self.create_oval(mid - 28, mid - 32, mid + 28, mid - 2, fill=self.HAIR, outline=self.HAIR)

        # Ears (two small ovals)
        self.create_oval(mid - 34, mid - 8, mid - 26, mid + 8, fill=self.SKIN, outline=self.SKIN)
        self.create_oval(mid + 26, mid - 8, mid + 34, mid + 8, fill=self.SKIN, outline=self.SKIN)

        # Face (main oval)
        self.create_oval(mid - 26, mid - 10, mid + 26, mid + 32, fill=self.SKIN, outline=self.SKIN)

        # Eyes (white ovals + blue iris)
        # Left eye
        self.create_oval(mid - 14, mid - 4, mid - 6, mid + 4, fill="white", outline="white")
        self.create_oval(mid - 12, mid - 2, mid - 8, mid + 2, fill=self.IRIS, outline=self.IRIS)
        # Right eye
        self.create_oval(mid + 6, mid - 4, mid + 14, mid + 4, fill="white", outline="white")
        self.create_oval(mid + 8, mid - 2, mid + 12, mid + 2, fill=self.IRIS, outline=self.IRIS)

        # Eyebrows (dark arcs)
        self.create_arc(mid - 16, mid - 14, mid - 4, mid - 6, start=0, extent=180, outline=self.HAIR, width=2)
        self.create_arc(mid + 4, mid - 14, mid + 16, mid - 6, start=0, extent=180, outline=self.HAIR, width=2)

        # Nose (small arc)
        self.create_arc(mid - 2, mid - 2, mid + 2, mid + 6, start=0, extent=180, outline=self.NOSE, width=1.5)

        # Mouth (smile arc)
        self.create_arc(mid - 10, mid + 8, mid + 10, mid + 18, start=0, extent=180, outline=self.MOUTH, width=2)

        # Shoulders (teal arc at bottom)
        self.create_arc(mid - 28, mid + 20, mid + 28, mid + 60, start=0, extent=180, fill=self.SHOULDERS, outline=self.SHOULDERS, width=10)
