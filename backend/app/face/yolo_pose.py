"""YOLO Pose person detector.

Wraps an Ultralytics pose model (`*-pose.pt` / `*-pose.onnx` / `*-pose.engine`).
Swap in your own trained weights by pointing `YOLO_MODEL` at them - the rest of
the pipeline does not care which checkpoint produced the keypoints, as long as
it uses the COCO-17 layout.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.core.logging_config import get_logger
from app.detection.base import PersonDetection, PersonDetector

logger = get_logger(__name__)

PERSON_CLASS_ID = 0


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string, falling back to CPU when unavailable."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency
        return "cpu"

    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available - falling back to CPU")
            return "cpu"
        if requested == "mps" and not getattr(torch.backends, "mps", None):
            logger.warning("MPS requested but not available - falling back to CPU")
            return "cpu"
        return requested

    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloPoseEstimator(PersonDetector):
    """Person detection + 17-keypoint pose estimation in a single pass."""

    name = "yolo-pose"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        confidence: float = 0.4,
        iou: float = 0.5,
        imgsz: int = 640,
        max_detections: int = 32,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO pose weights not found: {self.model_path}. "
                "Run `python scripts/download_models.py` first."
            )

        from ultralytics import YOLO  # imported lazily: heavy

        self.device = resolve_device(device)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.max_detections = max_detections
        self.last_inference_ms = 0.0

        logger.info("Loading YOLO pose model %s on %s", self.model_path.name, self.device)
        self.model = YOLO(str(self.model_path))
        try:
            self.model.to(self.device)
        except Exception as exc:  # pragma: no cover - exotic device setups
            logger.warning("Could not move model to %s (%s); using CPU", self.device, exc)
            self.device = "cpu"

        self.keypoint_count = 17
        logger.info("YOLO pose model ready (task=%s)", getattr(self.model, "task", "pose"))

    @property
    def provides_keypoints(self) -> bool:
        return True

    def warmup(self) -> None:
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.detect(blank)
        logger.info("YOLO pose warmup complete (%.0f ms)", self.last_inference_ms)

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        started = time.perf_counter()
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            max_det=self.max_detections,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        keypoints_xy = None
        keypoints_conf = None
        if result.keypoints is not None and result.keypoints.xy is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                keypoints_conf = result.keypoints.conf.cpu().numpy()

        detections: list[PersonDetection] = []
        for i in range(len(boxes)):
            if classes[i] != PERSON_CLASS_ID:
                continue
            kp = None
            if keypoints_xy is not None and i < len(keypoints_xy):
                xy = keypoints_xy[i]
                conf = (
                    keypoints_conf[i]
                    if keypoints_conf is not None and i < len(keypoints_conf)
                    else np.ones(len(xy), dtype=np.float32)
                )
                kp = np.concatenate([xy, conf.reshape(-1, 1)], axis=1).astype(np.float32)
            detections.append(
                PersonDetection(
                    box=tuple(float(v) for v in boxes[i]),
                    confidence=float(confs[i]),
                    keypoints=kp,
                    class_id=int(classes[i]),
                )
            )
        return detections
