# theme.py — design system for Sense.AI

COLORS = {
    "navy":      "#1a2744",   # window title bar, sidebar, headings
    "navy_dark": "#111b33",   # darkest panel backgrounds
    "teal":      "#1d7a8a",   # PRIMARY action color (buttons, active states)
    "teal_lt":   "#2aa3b8",   # lighter teal for status text
    "teal_bg":   "#e8f6f8",   # soft section backgrounds
    "gold":      "#e8a020",   # Speaker mode accent, secondary buttons
    "gold_lt":   "#f5c84a",   # lighter gold highlights
    "cream":     "#fdf8f2",   # main window backgrounds
    "white":     "#ffffff",   # card/panel surfaces
    "muted":     "#394453",   # label text, placeholders
    "border":    "#c8dde0",   # widget borders, dividers
    "success":   "#2ecc8a",   # live status indicator
    "danger":    "#e05a5a",   # logout button, errors
    "text":      "#1a2744",   # primary text
    "text_lt":   "#23395e",   # secondary text
    "black":     "#000000"
}

FONTS = {
    "heading":    ("Helvetica", 20, "bold"),
    "subheading": ("Helvetica", 18, "bold"),
    "body":       ("Helvetica", 12),
    "body_bold":  ("Helvetica", 12, "bold"),
    "label":      ("Helvetica", 11, "bold"),
    "small":      ("Helvetica", 10),
    "tiny":       ("Helvetica", 9),
    "status":     ("Courier", 8),
}

# Window size
APP_WIDTH  = 480
APP_HEIGHT = 720
