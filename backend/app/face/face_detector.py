"""Face detection.

Primary backend: MediaPipe Face Detector (BlazeFace short-range) through the
MediaPipe Tasks API. It needs the model file `blaze_face_short_range.tflite`
(see `scripts/download_models.py`).

Fallback backend: the Haar cascade that ships inside the `opencv-python`
wheel. It is weaker, but it needs no download, so the pipeline keeps working
(and keeps saying which detector produced a box) on machines that cannot
reach Google's model CDN.

`FACE_DETECTOR=auto` picks MediaPipe when its model is present and falls back
to Haar with a warning. Never raises into the pipeline: a missing face is a
normal, expected result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FaceBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def clip(self, width: int, height: int) -> "FaceBox":
        return FaceBox(
            max(0.0, min(self.x1, width - 1)),
            max(0.0, min(self.y1, height - 1)),
            max(0.0, min(self.x2, width - 1)),
            max(0.0, min(self.y2, height - 1)),
            self.confidence,
        )

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class BaseFaceDetector(ABC):
    """Interface for face detectors."""

    name: str = "abstract-face-detector"

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        """Detect faces in a BGR image, returning boxes in that image's coords."""

    def close(self) -> None:  # pragma: no cover - optional hook
        return None


class MediaPipeFaceDetector(BaseFaceDetector):
    """MediaPipe Tasks BlazeFace detector."""

    name = "mediapipe"

    def __init__(self, model_path: str | Path, min_confidence: float = 0.4) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face model not found: {self.model_path}. "
                "Run `python scripts/download_models.py`."
            )
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            min_detection_confidence=min_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        logger.info("MediaPipe face detector loaded (%s)", self.model_path.name)

    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        boxes: list[FaceBox] = []
        for detection in getattr(result, "detections", []) or []:
            bb = detection.bounding_box
            score = 1.0
            if detection.categories:
                score = float(detection.categories[0].score)
            boxes.append(
                FaceBox(
                    float(bb.origin_x),
                    float(bb.origin_y),
                    float(bb.origin_x + bb.width),
                    float(bb.origin_y + bb.height),
                    score,
                )
            )
        return boxes

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:  # pragma: no cover
            pass


class HaarFaceDetector(BaseFaceDetector):
    """OpenCV Haar cascade detector - bundled with opencv-python, no download."""

    name = "haar"

    def __init__(self, min_confidence: float = 0.4, min_size: int = 24) -> None:
        cascade_file = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_file))
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_file}")
        self._profile = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml")
        )
        self.min_size = min_size
        self.min_confidence = min_confidence
        logger.info("Haar cascade face detector loaded (%s)", cascade_file.name)

    def _run(self, cascade, gray: np.ndarray) -> list[tuple]:
        if cascade is None or cascade.empty():
            return []
        faces, _, weights = cascade.detectMultiScale3(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(self.min_size, self.min_size),
            outputRejectLevels=True,
        )
        out = []
        for (x, y, w, h), weight in zip(faces, weights):
            # `levelWeight` is not a probability; squash it into 0..1 so the
            # field stays comparable across backends. Reported as-is, never
            # presented as a calibrated score.
            score = float(1.0 / (1.0 + np.exp(-float(weight))))
            out.append((x, y, w, h, score))
        return out

    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        if image_bgr is None or image_bgr.size == 0:
            return []
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        found = self._run(self._cascade, gray)
        if not found:
            found = self._run(self._profile, gray)
        boxes = [
            FaceBox(float(x), float(y), float(x + w), float(y + h), score)
            for (x, y, w, h, score) in found
            if score >= self.min_confidence
        ]
        return boxes


class NullFaceDetector(BaseFaceDetector):
    """Explicitly disabled face detection."""

    name = "none"

    def detect(self, image_bgr: np.ndarray) -> list[FaceBox]:
        return []


def create_face_detector(
    backend: str = "auto",
    model_path: str | Path | None = None,
    min_confidence: float = 0.4,
) -> BaseFaceDetector:
    """Factory with graceful degradation. Never raises for `auto`."""
    backend = (backend or "auto").lower()

    if backend == "none":
        logger.info("Face detection disabled by configuration")
        return NullFaceDetector()

    if backend in ("mediapipe", "auto") and model_path is not None:
        try:
            return MediaPipeFaceDetector(model_path, min_confidence)
        except Exception as exc:
            if backend == "mediapipe":
                raise
            logger.warning(
                "MediaPipe face detector unavailable (%s) - falling back to the "
                "bundled OpenCV Haar cascade. Run scripts/download_models.py to "
                "enable MediaPipe.", exc,
            )

    if backend in ("haar", "auto"):
        try:
            return HaarFaceDetector(min_confidence)
        except Exception as exc:
            if backend == "haar":
                raise
            logger.error("Haar face detector failed to load: %s", exc)

    logger.error("No face detector available - face stage disabled")
    return NullFaceDetector()


def pick_best_face(
    boxes: list[FaceBox],
    prefer_center: Optional[tuple[float, float]] = None,
) -> Optional[FaceBox]:
    """Choose one face from candidates: biggest, or nearest to a hint point."""
    if not boxes:
        return None
    if prefer_center is None:
        return max(boxes, key=lambda b: b.area)

    cx, cy = prefer_center

    def score(b: FaceBox) -> float:
        bx = (b.x1 + b.x2) / 2.0
        by = (b.y1 + b.y2) / 2.0
        distance = float(np.hypot(bx - cx, by - cy))
        return b.area / (1.0 + distance)

    return max(boxes, key=score)
