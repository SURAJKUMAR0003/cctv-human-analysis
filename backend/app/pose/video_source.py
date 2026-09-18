"""Video capture.

A background thread keeps `cv2.VideoCapture` drained so the analysis pipeline
always works on the newest frame instead of a stale queue - the usual failure
mode of RTSP feeds. Reconnection is automatic.

Sources: `webcam` (device index), `rtsp` (URL), `file` (path, loops - useful
for offline testing without a camera).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from collections import deque
from typing import Optional

import cv2
import numpy as np

from app.core.config import Settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class VideoSource:
    """Threaded frame grabber with reconnect and FPS measurement."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.kind = settings.video_source
        self.target = self._target()
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._timestamps: deque[float] = deque(maxlen=30)
        self.connected = False
        self.last_error = ""
        self.width = 0
        self.height = 0
        self.generation = 0     # increments on every (re)connect

    # ------------------------------------------------------------------ util
    def _target(self):
        if self.kind == "webcam":
            return self.settings.webcam_index
        if self.kind == "rtsp":
            return self.settings.rtsp_url
        return str(self.settings.video_file_path) if self.settings.video_file else ""

    def describe(self) -> str:
        return f"{self.kind}:{self.target}"

    @property
    def capture_fps(self) -> float:
        with self._lock:
            stamps = list(self._timestamps)
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 0 else 0.0

    # --------------------------------------------------------------- control
    def open(self) -> bool:
        target = self._target()
        if target == "" or target is None:
            self.last_error = f"No target configured for VIDEO_SOURCE={self.kind}"
            logger.error(self.last_error)
            self.connected = False
            return False

        logger.info("Opening video source %s", self.describe())
        if self.kind == "webcam":
            capture = cv2.VideoCapture(int(target), cv2.CAP_ANY)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.capture_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.capture_height)
            capture.set(cv2.CAP_PROP_FPS, self.settings.target_fps)
        else:
            capture = cv2.VideoCapture(str(target), cv2.CAP_ANY)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not capture.isOpened():
            self.last_error = (
                f"Could not open video source {self.describe()}. "
                + ("File not found." if self.kind == "file" and not Path(str(target)).exists()
                   else "Check that the device/URL exists and is not in use by another application.")
            )
            logger.error(self.last_error)
            capture.release()
            self.connected = False
            return False

        ok, frame = capture.read()
        if not ok or frame is None:
            self.last_error = f"Video source {self.describe()} opened but returned no frames"
            logger.error(self.last_error)
            capture.release()
            self.connected = False
            return False

        self._capture = capture
        self.height, self.width = frame.shape[:2]
        self.connected = True
        self.last_error = ""
        self.generation += 1
        with self._lock:
            self._frame = frame
            self._frame_id += 1
            self._timestamps.append(time.time())
        logger.info("Video source connected: %s (%dx%d)", self.describe(), self.width, self.height)
        return True

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="video-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.release()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.connected = False

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        interval = 1.0 / max(self.settings.target_fps, 1e-3)
        while not self._stop.is_set():
            if self._capture is None or not self.connected:
                if not self.open():
                    time.sleep(self.settings.reconnect_delay_seconds)
                    continue

            started = time.time()
            ok, frame = self._capture.read()
            if not ok or frame is None:
                if self.kind == "file":
                    # Loop test footage instead of dying at EOF.
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._capture.read()
                if not ok or frame is None:
                    self.last_error = f"Lost connection to {self.describe()}"
                    logger.warning("%s - reconnecting in %.1fs",
                                   self.last_error, self.settings.reconnect_delay_seconds)
                    self.release()
                    time.sleep(self.settings.reconnect_delay_seconds)
                    continue

            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._timestamps.append(time.time())
                self.height, self.width = frame.shape[:2]

            elapsed = time.time() - started
            if interval > elapsed:
                time.sleep(interval - elapsed)

    # ------------------------------------------------------------------ read
    def read(self) -> tuple[Optional[np.ndarray], int]:
        """Latest frame and its monotonically increasing id."""
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id
