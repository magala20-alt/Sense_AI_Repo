"""
Real-time Camera Frame Processor - Processes video frames for hand/facial recognition
and coordinates with ASL translation
"""

import cv2
import numpy as np
import threading
import queue
import os
from typing import Dict, Any, Optional, Callable
import logging
from dataclasses import dataclass
from datetime import datetime

from services.model_loader import get_model_manager

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFrame:
    """Container for processed frame data"""
    timestamp: datetime
    frame: np.ndarray
    hand_data: Dict[str, Any]
    facial_data: Dict[str, Any]
    frame_id: int


class CameraFrameProcessor:
    """Real-time camera frame processor with model inference"""

    def __init__(self, camera_index: int = 0, fps: int = 30):
        self.camera_index = camera_index
        self.fps = fps
        self.frame_delay_ms = int(1000 / fps)
        
        self.cap = None
        self.is_running = False
        self.processing_thread = None
        
        # Frame queue for processed data
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Model manager
        self.model_manager = get_model_manager()
        
        # Callback for frame updates
        self.on_frame_processed: Optional[Callable] = None
        
        # Statistics
        self.frame_count = 0
        self.skipped_frames = 0
        self._consecutive_read_failures = 0

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Open camera with backend preference. On Windows, prefer DirectShow over MSMF."""
        backends = [None]
        if os.name == "nt":
            # MSMF is known to intermittently fail on some Windows camera drivers.
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]

        for backend in backends:
            cap = cv2.VideoCapture(self.camera_index, backend) if backend is not None else cv2.VideoCapture(self.camera_index)
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                logger.info(f"Camera opened on index {self.camera_index} using backend {backend}")
                return cap
            if cap:
                cap.release()

        return None

    def _reopen_camera(self) -> bool:
        """Attempt to recover camera after repeated read failures."""
        if self.cap:
            self.cap.release()
            self.cap = None

        self.cap = self._open_camera()
        self._consecutive_read_failures = 0
        if self.cap is None:
            logger.error("Camera re-open failed")
            return False
        logger.info("Camera re-opened successfully")
        return True

    def start(self):
        """Start camera and processing thread"""
        try:
            # Initialize camera
            self.cap = self._open_camera()
            if not self.cap:
                logger.error(f"Failed to open camera at index {self.camera_index}")
                return False

            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()

            logger.info(f"Camera processor started (camera {self.camera_index})")
            return True
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False

    def stop(self):
        """Stop camera and processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        logger.info("Camera processor stopped")

    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        try:
            while self.is_running:
                if self.cap is None:
                    if not self._reopen_camera():
                        threading.Event().wait(0.5)
                        continue

                ret, frame = self.cap.read()
                if not ret:
                    self._consecutive_read_failures += 1
                    if self._consecutive_read_failures in (1, 5, 10):
                        logger.warning(f"Failed to read frame from camera (x{self._consecutive_read_failures})")

                    # Try to recover after repeated failures.
                    if self._consecutive_read_failures >= 15:
                        logger.warning("Too many camera read failures. Re-opening camera...")
                        self._reopen_camera()
                    threading.Event().wait(0.05)
                    continue
                self._consecutive_read_failures = 0

                # Flip frame horizontally (mirror effect)
                frame = cv2.flip(frame, 1)

                # Process frame with models
                try:
                    hand_data = self.model_manager.detect_hand_pose(frame)
                    facial_data = self.model_manager.detect_facial_expression(frame)

                    processed = ProcessedFrame(
                        timestamp=datetime.now(),
                        frame=frame,
                        hand_data=hand_data,
                        facial_data=facial_data,
                        frame_id=self.frame_count
                    )

                    self.frame_count += 1

                    # Put in queue (drop old frames if queue is full)
                    try:
                        self.frame_queue.put_nowait(processed)
                    except queue.Full:
                        self.skipped_frames += 1
                        try:
                            self.frame_queue.get_nowait()  # Remove oldest
                            self.frame_queue.put_nowait(processed)
                        except queue.Empty:
                            pass

                    # Call callback if set
                    if self.on_frame_processed:
                        self.on_frame_processed(processed)

                except Exception as e:
                    logger.error(f"Frame processing error: {e}")

                # Maintain frame rate
                if self.frame_delay_ms > 0:
                    threading.Event().wait(self.frame_delay_ms / 1000.0)

        except Exception as e:
            logger.error(f"Processing loop error: {e}")

    def get_latest_frame(self) -> Optional[ProcessedFrame]:
        """Get the latest processed frame without blocking"""
        try:
            # Empty queue and get last item
            frames = []
            while True:
                try:
                    frames.append(self.frame_queue.get_nowait())
                except queue.Empty:
                    break
            return frames[-1] if frames else None
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "frames_processed": self.frame_count,
            "frames_skipped": self.skipped_frames,
            "fps": self.fps,
            "is_running": self.is_running
        }

    @staticmethod
    def draw_hand_keypoints(frame: np.ndarray, hand_data: Dict[str, Any]) -> np.ndarray:
        """Draw hand keypoints on frame"""
        if not hand_data.get("detected"):
            return frame

        keypoints = hand_data.get("keypoints", [])
        if not keypoints:
            return frame

        # Draw circles at keypoints
        for point in keypoints:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Draw connections between main hand joints
        if len(keypoints) >= 20:
            # Typical hand pose: 21 keypoints
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            ]
            for start, end in connections:
                if start < len(keypoints) and end < len(keypoints):
                    p1 = keypoints[start]
                    p2 = keypoints[end]
                    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

        return frame

    @staticmethod
    def draw_facial_expression(frame: np.ndarray, facial_data: Dict[str, Any]) -> np.ndarray:
        """Draw facial expression label on frame"""
        if not facial_data.get("detected"):
            return frame

        expression = facial_data.get("expression", "unknown")
        confidence = facial_data.get("confidence", 0.0)

        label = f"{expression}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame


class FrameBuffer:
    """Accumulates frames for gesture recognition"""

    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames
        self.frames = []
        self.lock = threading.Lock()

    def add_frame(self, processed_frame: ProcessedFrame):
        """Add frame to buffer"""
        with self.lock:
            self.frames.append(processed_frame)
            if len(self.frames) > self.max_frames:
                self.frames.pop(0)

    def get_frames(self) -> list:
        """Get all frames in buffer"""
        with self.lock:
            return self.frames.copy()

    def clear(self):
        """Clear all frames"""
        with self.lock:
            self.frames = []

    def get_hand_trajectory(self) -> Optional[list]:
        """Extract hand position trajectory from frames"""
        with self.lock:
            if not self.frames:
                return None

            trajectories = []
            for frame_data in self.frames:
                hand_data = frame_data.hand_data
                if hand_data.get("detected") and hand_data.get("keypoints"):
                    # Get palm center (average of all keypoints)
                    keypoints = np.array(hand_data["keypoints"])
                    palm_center = keypoints.mean(axis=0)
                    trajectories.append({
                        "timestamp": frame_data.timestamp,
                        "position": palm_center.tolist(),
                        "confidence": hand_data.get("confidence", 0.0)
                    })

            return trajectories if trajectories else None

    def get_expression_sequence(self) -> Optional[list]:
        """Extract facial expression sequence from frames"""
        with self.lock:
            if not self.frames:
                return None

            sequences = []
            for frame_data in self.frames:
                facial_data = frame_data.facial_data
                if facial_data.get("detected"):
                    sequences.append({
                        "timestamp": frame_data.timestamp,
                        "expression": facial_data.get("expression"),
                        "confidence": facial_data.get("confidence", 0.0)
                    })

            return sequences if sequences else None
