"""Application configuration.

Every tunable value lives here and is read from the environment (or a .env
file at the repository root / backend folder). Nothing important is
hard-coded elsewhere in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo_root/backend/app/core/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
DEFAULT_MODEL_DIR = BACKEND_ROOT / "models"


class Settings(BaseSettings):
    """Runtime settings for the CCTV human-analysis system."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---------------------------------------------------------------- video
    video_source: Literal["webcam", "rtsp", "file"] = "webcam"
    webcam_index: int = 0
    rtsp_url: str = ""
    video_file: str = ""
    capture_width: int = 1280
    capture_height: int = 720
    target_fps: float = Field(default=15.0, gt=0, le=120)
    reconnect_delay_seconds: float = 3.0

    # --------------------------------------------------------------- models
    model_dir: Path = DEFAULT_MODEL_DIR
    yolo_model: str = "yolo11n-pose.pt"
    expression_model: str = "emotion_enet_b0_8.onnx"
    face_model: str = "blaze_face_short_range.tflite"

    # ------------------------------------------------------------ inference
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    inference_imgsz: int = 640
    keypoint_confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    # ------------------------------------------------------------- tracking
    track_high_threshold: float = 0.5
    track_low_threshold: float = 0.1
    new_track_threshold: float = 0.6
    track_buffer: int = 30
    match_threshold: float = 0.8

    # ----------------------------------------------------------------- face
    face_detector: Literal["auto", "mediapipe", "haar", "none"] = "auto"
    face_detection_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    face_padding_ratio: float = 0.15
    # Only the upper part of a person box is searched for a face.
    face_search_region_ratio: float = Field(default=0.45, gt=0.0, le=1.0)

    # ----------------------------------------------------------- expression
    expression_enabled: bool = True
    expression_every_n_frames: int = 3
    expression_smoothing: float = Field(default=0.6, ge=0.0, le=1.0)

    # ----------------------------------------------------------- behaviour
    behavior_history_seconds: float = 3.0
    activity_low_threshold: float = 0.02   # normalised px/s of hip movement
    activity_high_threshold: float = 0.12

    # ------------------------------------------------------------ pipeline
    process_every_n_frames: int = 1        # frame skipping
    max_people: int = 32
    stale_person_seconds: float = 2.0

    # ------------------------------------------------------------- serving
    host: str = "0.0.0.0"
    port: int = 8000
    ws_fps: float = Field(default=10.0, gt=0, le=60)
    mjpeg_fps: float = Field(default=12.0, gt=0, le=60)
    mjpeg_quality: int = Field(default=75, ge=10, le=100)
    annotate_stream: bool = True
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ------------------------------------------------------------- logging
    log_level: str = "INFO"

    @field_validator("model_dir", mode="before")
    @classmethod
    def _expand_model_dir(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def video_file_path(self) -> Path:
        """Absolute path of VIDEO_FILE.

        Relative paths are resolved against the repository root, not the
        process working directory, so `VIDEO_FILE=datasets/samples/x.mp4`
        works whether uvicorn was started from the repo root or from
        `backend/`.
        """
        path = Path(self.video_file).expanduser()
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    @property
    def yolo_model_path(self) -> Path:
        return self._resolve(self.yolo_model)

    @property
    def expression_model_path(self) -> Path:
        return self._resolve(self.expression_model)

    @property
    def face_model_path(self) -> Path:
        return self._resolve(self.face_model)

    def _resolve(self, name: str) -> Path:
        path = Path(name).expanduser()
        if path.is_absolute():
            return path
        return (self.model_dir / path).resolve()

    def describe_source(self) -> str:
        if self.video_source == "webcam":
            return f"webcam:{self.webcam_index}"
        if self.video_source == "rtsp":
            return self.rtsp_url or "rtsp:<unset>"
        return str(self.video_file_path) if self.video_file else "file:<unset>"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
