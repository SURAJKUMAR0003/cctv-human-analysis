"""Model registry and status reporting.

Every model the system can use is declared here together with the exact URL it
comes from and its licence. Nothing is downloaded implicitly at runtime: the
app reports what is missing and `scripts/download_models.py` fetches it after
printing what it is about to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.models.schemas import ModelInfo

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    license: str
    description: str
    required: bool = True
    sha256: Optional[str] = None
    extra_files: tuple[tuple[str, str], ...] = ()   # (filename, url)


# --------------------------------------------------------------------------
# Known, documented model sources.
#
# YOLO pose  : Ultralytics release assets (AGPL-3.0). Any *-pose checkpoint
#              works; add an entry here or drop the file into backend/models/.
# Expression : HSEmotion (Savchenko et al.), EfficientNet-B0 trained on
#              AffectNet, exported to ONNX, Apache-2.0.
# Face       : MediaPipe BlazeFace short-range, Apache-2.0, from Google's
#              official model CDN.
# --------------------------------------------------------------------------
YOLO_RELEASE = "https://github.com/ultralytics/assets/releases/download"

KNOWN_YOLO_WEIGHTS: dict[str, str] = {
    "yolo11n-pose.pt": f"{YOLO_RELEASE}/v8.4.0/yolo11n-pose.pt",
    "yolo11s-pose.pt": f"{YOLO_RELEASE}/v8.4.0/yolo11s-pose.pt",
    "yolo11m-pose.pt": f"{YOLO_RELEASE}/v8.4.0/yolo11m-pose.pt",
    "yolo11l-pose.pt": f"{YOLO_RELEASE}/v8.4.0/yolo11l-pose.pt",
    "yolo11x-pose.pt": f"{YOLO_RELEASE}/v8.4.0/yolo11x-pose.pt",
    "yolov8n-pose.pt": f"{YOLO_RELEASE}/v8.3.0/yolov8n-pose.pt",
    "yolov8s-pose.pt": f"{YOLO_RELEASE}/v8.3.0/yolov8s-pose.pt",
    "yolov8m-pose.pt": f"{YOLO_RELEASE}/v8.3.0/yolov8m-pose.pt",
}

EXPRESSION_URL = (
    "https://raw.githubusercontent.com/av-savchenko/face-emotion-recognition/"
    "main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx"
)

FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def build_specs(settings: Settings) -> list[ModelSpec]:
    yolo_name = Path(settings.yolo_model).name
    specs = [
        ModelSpec(
            key="yolo_pose",
            filename=settings.yolo_model,
            url=KNOWN_YOLO_WEIGHTS.get(yolo_name, ""),
            license="AGPL-3.0 (Ultralytics)",
            description="YOLO Pose - person detection + 17 COCO keypoints",
            required=True,
        ),
        ModelSpec(
            key="expression",
            filename=settings.expression_model,
            url=EXPRESSION_URL,
            license="Apache-2.0 (HSEmotion / av-savchenko)",
            description=(
                "EfficientNet-B0 facial-expression classifier (AffectNet, 8 classes), "
                "ONNX. Outputs visible-expression probabilities only."
            ),
            required=False,
        ),
        ModelSpec(
            key="face",
            filename=settings.face_model,
            url=FACE_URL,
            license="Apache-2.0 (Google MediaPipe)",
            description=(
                "MediaPipe BlazeFace short-range face detector. Optional: without it "
                "the system falls back to the OpenCV Haar cascade bundled with opencv-python."
            ),
            required=False,
        ),
    ]
    return specs


class ModelManager:
    """Reports which model files exist on disk and which are loaded."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.specs = build_specs(settings)
        self._loaded: dict[str, bool] = {}
        self._errors: dict[str, str] = {}

    def path_for(self, spec: ModelSpec) -> Path:
        path = Path(spec.filename).expanduser()
        if path.is_absolute():
            return path
        return (self.settings.model_dir / path).resolve()

    def mark_loaded(self, key: str, loaded: bool, error: str = "") -> None:
        self._loaded[key] = loaded
        if error:
            self._errors[key] = error
        else:
            self._errors.pop(key, None)

    def missing_required(self) -> list[ModelSpec]:
        return [s for s in self.specs if s.required and not self.path_for(s).exists()]

    def report(self) -> list[ModelInfo]:
        infos: list[ModelInfo] = []
        for spec in self.specs:
            path = self.path_for(spec)
            present = path.exists()
            infos.append(
                ModelInfo(
                    key=spec.key,
                    name=path.name,
                    path=str(path),
                    present=present,
                    loaded=self._loaded.get(spec.key, False),
                    size_bytes=path.stat().st_size if present else 0,
                    source_url=spec.url,
                    license=spec.license,
                    description=spec.description,
                    error=self._errors.get(spec.key, ""),
                )
            )
        return infos

    def log_status(self) -> None:
        for info in self.report():
            if info.present:
                logger.info("Model OK       %-11s %s (%.1f MB)",
                            info.key, info.name, info.size_bytes / 1e6)
            else:
                logger.warning("Model MISSING  %-11s %s - expected at %s",
                               info.key, info.name, info.path)
