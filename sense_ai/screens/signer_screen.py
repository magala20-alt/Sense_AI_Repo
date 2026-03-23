import threading
import tkinter as tk

import cv2
from PIL import Image, ImageTk

try:
    from app_state import AppState
    from config import BACKEND_WS_URL, CAMERA_INDEX
    from components.mobile_nav import MobileNavBar
    from components.status_bar import StatusBar
    from components.sidebar import Sidebar
    from components.grammar_tag import GrammarTag
    from services.websocket_client import create_websocket_client
    from services.camera_processor import CameraFrameProcessor, FrameBuffer
    from theme import COLORS
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from app_state import AppState
    from config import BACKEND_WS_URL, CAMERA_INDEX
    from components.mobile_nav import MobileNavBar
    from components.status_bar import StatusBar
    from components.sidebar import Sidebar
    from components.grammar_tag import GrammarTag
    from services.websocket_client import create_websocket_client
    from services.camera_processor import CameraFrameProcessor, FrameBuffer
    from theme import COLORS


class SignerScreen(tk.Frame):
    """Responsive signer screen matching the supplied mockups."""

    SCREEN_NAME = "signer"

    def __init__(self, parent, state, navigate):
        super().__init__(parent, bg=COLORS["cream"])
        self.state = state
        self.navigate = navigate
        self._layout_mode = None
        self.cap = None
        self.camera_running = False
        self.websocket_client = None
        self.pulse_id = None
        self.preview_photo = None
        self.display_loop_id = None
        
        # Camera frame processor
        self.frame_processor = None
        self.frame_buffer = FrameBuffer(max_frames=30)
        self.translation_update_frequency = 10  # Update translation every 10 frames

        self.bind("<Configure>", self._on_resize)

        self.desktop_sidebar = Sidebar(
            self,
            active_screen="signer",
            navigate=self.navigate,
            role="signer",
            on_logout=self._logout,
            state=self.state,
        )
        self.main_area = tk.Frame(self, bg=COLORS["cream"])

        self.header = tk.Frame(self.main_area, bg=COLORS["cream"], height=70)
        self.header.pack(fill=tk.X)
        self.header.pack_propagate(False)
        tk.Label(self.header, text="Translate", font=("Georgia", 24, "bold"), bg=COLORS["cream"], fg=COLORS["navy"]).pack(side=tk.LEFT, padx=28, pady=16)
        self.live_badge = tk.Label(self.header, text="● LIVE", font=("Helvetica", 11, "bold"), bg="#dff0f3", fg=COLORS["teal"], padx=14, pady=6)
        self.live_badge.pack(side=tk.RIGHT, padx=(0, 16), pady=14)
        self.history_button = tk.Button(self.header, text="View History", font=("Helvetica", 11, "bold"), bg=COLORS["teal"], fg=COLORS["white"], relief=tk.FLAT, bd=0, padx=18, pady=8, cursor="hand2", command=lambda: self.navigate("history"))
        self.history_button.pack(side=tk.RIGHT, padx=16, pady=12)

        self.scroll_canvas = tk.Canvas(self.main_area, bg=COLORS["cream"], highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.main_area, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.content = tk.Frame(self.scroll_canvas, bg=COLORS["cream"])
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda _e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        self.scroll_canvas.bind("<Configure>", lambda e: self.scroll_canvas.itemconfigure(self.canvas_window, width=e.width))

        self.mobile_top = tk.Frame(self.content, bg=COLORS["navy"], height=60)
        self.mobile_top.pack_propagate(False)
        tk.Label(self.mobile_top, text="Translate", font=("Georgia", 18, "bold"), bg=COLORS["navy"], fg=COLORS["white"]).pack(side=tk.LEFT, padx=18, pady=14)
        self.mobile_live_badge = tk.Label(self.mobile_top, text="● LIVE", font=("Helvetica", 10, "bold"), bg="#314165", fg=COLORS["white"], padx=12, pady=6)
        self.mobile_live_badge.pack(side=tk.RIGHT, padx=12, pady=12)

        self.main_row = tk.Frame(self.content, bg=COLORS["cream"])
        self.video_card = tk.Frame(self.main_row, bg="#021632", highlightbackground="#091b38", highlightthickness=1)
        self.aside = tk.Frame(self.main_row, bg=COLORS["cream"])

        self.video_stage = tk.Frame(self.video_card, bg="#021632")
        self.video_stage.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)
        self.camera_label = tk.Label(
            self.video_stage,
            bg="#021632",
            fg=COLORS["white"],
            text="Starting camera...",
            font=("Helvetica", 12),
        )
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        self.video_footer = tk.Frame(self.video_card, bg="#021632")
        self.video_footer.pack(fill=tk.X, padx=16, pady=(0, 14))
        self.live_dot = tk.Label(self.video_footer, text="●", font=("Helvetica", 10), bg="#021632", fg=COLORS["success"])
        self.live_dot.pack(side=tk.LEFT)
        tk.Label(self.video_footer, text="Camera active", font=("Helvetica", 11), bg="#021632", fg=COLORS["white"]).pack(side=tk.LEFT, padx=(6, 0))
        self.grammar_overlay = GrammarTag(self.video_footer, "")
        self.grammar_overlay.configure(bg="#021632")
        self.grammar_overlay.pack(side=tk.RIGHT)

        self.translation_card = tk.Frame(self.aside, bg=COLORS["white"], highlightbackground="#e2e8ef", highlightthickness=1, padx=18, pady=16)
        self.translation_title = tk.Label(self.translation_card, text='"Are you coming tomorrow?"', font=("Helvetica", 16, "bold"), bg=COLORS["white"], fg=COLORS["navy"], wraplength=280, justify=tk.LEFT)
        self.translation_title.pack(anchor="w")
        self.translation_meta = tk.Label(self.translation_card, text="Translated", font=("Helvetica", 11), bg=COLORS["white"], fg="#95a0b2")
        self.translation_meta.pack(anchor="w", pady=(8, 0))

        self.conversation_card = tk.Frame(self.aside, bg=COLORS["white"], highlightbackground="#e2e8ef", highlightthickness=1, padx=18, pady=18)
        tk.Label(self.conversation_card, text="CONVERSATION", font=("Helvetica", 12, "bold"), bg=COLORS["white"], fg="#97a2b5").pack(anchor="w", pady=(0, 10))
        self.history_frame = tk.Frame(self.conversation_card, bg=COLORS["white"])
        self.history_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.mobile_nav = MobileNavBar(self, active_screen="signer", navigate=self.navigate, role="signer")
        self.status_bar = StatusBar(self)
        self.status_bar.set_text("Status: Listening")

        self.after_idle(self._apply_layout)

    def _logout(self):
        self.stop_camera()
        self.stop_websocket()
        self.state.logout()
        self.navigate("login")

    def _apply_layout(self):
        width = max(self.winfo_width(), self.winfo_reqwidth())
        mode = "desktop" if width >= 900 else "mobile"
        if mode == self._layout_mode:
            return
        self._layout_mode = mode

        self.desktop_sidebar.pack_forget()
        self.main_area.pack_forget()
        self.scroll_canvas.pack_forget()
        self.scrollbar.pack_forget()
        self.mobile_nav.pack_forget()
        self.mobile_top.pack_forget()
        self.main_row.pack_forget()
        self.video_card.pack_forget()
        self.aside.pack_forget()
        self.translation_card.pack_forget()
        self.conversation_card.pack_forget()

        if mode == "desktop":
            self.desktop_sidebar.pack(side=tk.LEFT, fill=tk.Y)
            self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.header.pack(fill=tk.X)
            self.mobile_top.pack_forget()
            self.main_row.pack(fill=tk.BOTH, expand=True, padx=28, pady=(16, 28))
            self.video_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 18))
            self.aside.pack(side=tk.LEFT, fill=tk.Y)
            self.translation_card.pack(fill=tk.X, pady=(0, 12))
            self.conversation_card.pack(fill=tk.BOTH, expand=True)
        else:
            self.main_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.header.pack_forget()
            self.mobile_top.pack(fill=tk.X, pady=(0, 12))
            self.main_row.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 14))
            self.video_card.pack(fill=tk.X, pady=(0, 12))
            self.aside.pack(fill=tk.BOTH, expand=True)
            self.translation_card.pack(fill=tk.X, pady=(0, 12))
            self.conversation_card.pack(fill=tk.BOTH, expand=True)
            self.mobile_nav.pack(side=tk.BOTTOM, fill=tk.X)

    def _on_resize(self, _event):
        self._apply_layout()

    def _render_history(self):
        for child in self.history_frame.winfo_children():
            child.destroy()

        messages = self.state.conv_history[-4:] if self.state.conv_history else []

        if not messages:
            tk.Label(
                self.history_frame,
                text="No conversations yet",
                font=("Helvetica", 12, "italic"),
                bg=COLORS["white"],
                fg="#97a2b5",
                pady=12,
            ).pack(anchor="w")
            return

        for entry in messages:
            author = entry.get("who", "You")
            bubble_bg = COLORS["white"] if author == "You" else "#e2f1f4"
            label_fg = COLORS["teal"] if author == "You" else COLORS["gold"]
            row = tk.Frame(self.history_frame, bg=COLORS["white"])
            row.pack(fill=tk.X, pady=6)
            head = tk.Frame(row, bg=COLORS["white"])
            head.pack(anchor="w")
            tk.Label(head, text=author, font=("Helvetica", 10, "bold"), bg=COLORS["white"], fg=label_fg).pack(side=tk.LEFT)
            tk.Label(row, text=entry.get("text", ""), font=("Helvetica", 12), bg=bubble_bg, fg=COLORS["navy"], wraplength=260, justify=tk.LEFT, padx=14, pady=10).pack(fill=tk.X, pady=(4, 0))

    def on_show(self):
        self._apply_layout()
        self._render_history()
        self.start_camera()
        self.start_websocket()
        self.status_bar.set_live(True)
        self.status_bar.set_text("Status: Listening")
        self._pulse_status()

    def start_camera(self):
        """Start camera capture using frame processor."""
        if self.camera_running:
            return
        
        try:
            self.frame_processor = CameraFrameProcessor(camera_index=CAMERA_INDEX, fps=30)
            self.frame_processor.on_frame_processed = self._on_frame_processed
            
            if self.frame_processor.start():
                self.camera_running = True
                self.preview_photo = None
                self._update_camera_status("")
                if self.display_loop_id is None:
                    self._camera_display_loop()
            else:
                self._update_camera_status("Failed to start camera")
        except Exception as e:
            self._update_camera_status(f"Camera error: {str(e)}")

    def _on_frame_processed(self, processed_frame):
        """Callback when frame is processed by models"""
        try:
            # Add to buffer
            self.frame_buffer.add_frame(processed_frame)
            
            # Periodically send to server for translation
            if processed_frame.frame_id % self.translation_update_frequency == 0:
                self._send_frame_for_translation(processed_frame)
        except Exception as e:
            import logging
            logging.error(f"Frame processing callback error: {e}")

    def _send_frame_for_translation(self, processed_frame):
        """Send frame to WebSocket server for translation"""
        try:
            if not self.websocket_client:
                return
            
            message = {
                "type": "translate_frame",
                "hand_data": processed_frame.hand_data,
                "facial_data": processed_frame.facial_data,
                "frame_id": processed_frame.frame_id,
                "user_id": self.state.current_user.get("id", 1) if self.state.current_user else 1
            }
            
            self.websocket_client.send(message)
        except Exception as e:
            import logging
            logging.error(f"Error sending frame to server: {e}")

    def _camera_display_loop(self):
        """Update displayed camera frame"""
        if not self.camera_running or not self.winfo_exists():
            self.display_loop_id = None
            return
        
        try:
            latest_frame = self.frame_processor.get_latest_frame()
            
            if latest_frame is not None:
                # Draw keypoints and expression on frame
                display_frame = latest_frame.frame.copy()
                display_frame = CameraFrameProcessor.draw_hand_keypoints(display_frame, latest_frame.hand_data)
                display_frame = CameraFrameProcessor.draw_facial_expression(display_frame, latest_frame.facial_data)
                
                # Convert to PhotoImage
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (640, 360))
                pil_image = Image.fromarray(rgb_frame)
                if self.preview_photo is None:
                    self.preview_photo = ImageTk.PhotoImage(pil_image)
                else:
                    # Reuse existing Tk image buffer to avoid unbounded allocations.
                    self.preview_photo.paste(pil_image)

                self.camera_label.configure(image=self.preview_photo, text="")
                self.camera_label.image = self.preview_photo
            
            # Continue looping at ~30fps
            self.display_loop_id = self.after(40, self._camera_display_loop)
        except Exception as e:
            import logging
            logging.error(f"Display loop error: {e}")
            # If image buffer is exhausted, reset image object and continue with smaller updates.
            self.preview_photo = None
            self.display_loop_id = self.after(120, self._camera_display_loop)

    def _camera_loop(self):
        """Legacy camera loop - kept for compatibility but not used"""
        while self.camera_running:
            ret, frame = self.cap.read() if self.cap else (False, None)
            if not ret:
                threading.Event().wait(0.2)
                continue
            frame = cv2.resize(frame, (640, 360))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            self.after(0, lambda p=photo: self._update_camera_display(p))
            threading.Event().wait(0.033)

    def _update_camera_display(self, photo):
        self.preview_photo = photo
        self.camera_label.configure(image=photo, text="")
        self.camera_label.image = photo

    def _update_camera_status(self, message):
        self.camera_label.configure(image="", text=message)
        self.camera_label.image = None

    def start_websocket(self):
        if self.websocket_client:
            return
        self.websocket_client = create_websocket_client(BACKEND_WS_URL, self._on_websocket_message)

    def _on_websocket_message(self, translation: str, grammar_type: str, tier_scores: dict):
        self.after(0, lambda: self._handle_translation(translation, grammar_type, tier_scores))

    def _handle_translation(self, translation: str, grammar_type: str, tier_scores: dict):
        self.state.current_translation = translation
        self.state.current_grammar = grammar_type
        self.state.tier_scores = tier_scores
        if translation:
            self.translation_title.config(text=f'"{translation}"')
        self.translation_meta.config(text="Translated")
        self.grammar_overlay.update_grammar(grammar_type)
        if translation:
            self.state.add_message("You", translation, grammar_type)
            self._render_history()

    def stop_camera(self):
        self.camera_running = False
        if self.display_loop_id:
            try:
                self.after_cancel(self.display_loop_id)
            except Exception:
                pass
            self.display_loop_id = None
        if self.frame_processor:
            self.frame_processor.stop()
            self.frame_processor = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.frame_buffer.clear()
        self.preview_photo = None
        self.camera_label.configure(image="")
        self.camera_label.image = None

    def stop_websocket(self):
        if self.websocket_client:
            self.websocket_client.stop()
            self.websocket_client = None

    def _pulse_status(self):
        if not self.winfo_exists():
            return
        current = self.live_dot.cget("fg")
        color = COLORS["success"] if current != COLORS["success"] else "#4f5d7b"
        self.live_dot.config(fg=color)
        self.pulse_id = self.after(800, self._pulse_status)

    def __del__(self):
        self.stop_camera()
        self.stop_websocket()
        if self.pulse_id:
            try:
                self.after_cancel(self.pulse_id)
            except Exception:
                pass

# temp to see view of screen
# def main():
#     root = tk.Tk()
#     root.geometry("400x600")
#     state = AppState()
#     signer_screen = SignerScreen(root, state, lambda x: print(f"Navigate to {x}"))
#     signer_screen.pack(fill=tk.BOTH, expand=True)
#     signer_screen.on_show()
#     root.mainloop()

# if __name__ == "__main__":
#     main()