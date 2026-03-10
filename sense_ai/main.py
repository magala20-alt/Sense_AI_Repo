# main.py — Sense.AI desktop application entry point

import tkinter as tk
from app_state import AppState
from theme import COLORS, APP_WIDTH, APP_HEIGHT

# Import all screens
from screens.login_screen import LoginScreen
from screens.signup_screen import SignUpScreen
from screens.session_screen import SessionScreen
from screens.signer_screen import SignerScreen
from screens.speaker_screen import SpeakerScreen
from screens.history_screen import HistoryScreen
from screens.settings_screen import SettingsScreen


class SenseAIApp(tk.Tk):
    """Main application window managing screen navigation."""

    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("Sense.AI")
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.resizable(True, True)
        self.configure(bg=COLORS["cream"])
        
        # Center window on screen
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - APP_WIDTH) // 2
        y = (screen_height - APP_HEIGHT) // 2
        self.geometry(f"+{x}+{y}")
        
        # Global app state
        self.state = AppState()
        
        # Frame stack for screen management
        self.frames = {}
        self._build_screens()
        # Show login screen first
        self.show_screen("login")

    def _build_screens(self):
        """Instantiate all screens and lay them on top of each other."""
        screen_classes = [
            LoginScreen,
            SignUpScreen,
            SessionScreen,
            SignerScreen,
            SpeakerScreen,
            HistoryScreen,
            SettingsScreen,
        ]
        
        for ScreenClass in screen_classes:
            frame = ScreenClass(self, self.state, self.show_screen)
            self.frames[frame.SCREEN_NAME] = frame
            # Place all frames on top of each other
            frame.place(x=0, y=0, relwidth=1, relheight=1)

    def show_screen(self, name: str):
        """Switch to a different screen."""
        if name not in self.frames:
            print(f"Screen '{name}' not found")
            return
        
        frame = self.frames[name]
        frame.lift()  # Bring to front
        
        # Call on_show hook if it exists
        if hasattr(frame, "on_show"):
            frame.on_show()

    def on_closing(self):
        """Handle app closing."""
        # Clean up resources
        for frame in self.frames.values():
            if hasattr(frame, "stop_camera"):
                frame.stop_camera()
            if hasattr(frame, "stop_websocket"):
                frame.stop_websocket()
        self.destroy()


def main():
    """Entry point for the application."""
    app = SenseAIApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
