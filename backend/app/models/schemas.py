"""Data structures exchanged between the pipeline, the API and the frontend.

These models are the contract for `/api/*` and `/ws/live`. They intentionally
contain only *observable* measurements - no psychological interpretation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# COCO-17 keypoint order used by the YOLO pose models.
KEYPOINT_NAMES: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Pairs of keypoint indices that form the drawn skeleton.
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class Keypoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float
    visible: bool


class PoseData(BaseModel):
    keypoints: list[Keypoint] = Field(default_factory=list)
    mean_confidence: float = 0.0
    visible_keypoints: int = 0

    def get(self, name: str) -> Optional[Keypoint]:
        for kp in self.keypoints:
            if kp.name == name:
                return kp
        return None


class FaceData(BaseModel):
    detected: bool = False
    box: Optional[BoundingBox] = None
    confidence: float = 0.0
    source: str = "none"  # which detector produced it


class ExpressionResult(BaseModel):
    """Raw classifier output. Labels describe the *visible facial expression*
    only; they are not statements about a person's inner or mental state."""

    available: bool = False
    probabilities: dict[str, float] = Field(default_factory=dict)
    top_label: Optional[str] = None
    top_probability: float = 0.0
    model_name: Optional[str] = None
    inference_ms: float = 0.0


class BehaviorFeatures(BaseModel):
    """Purely computational, observable measurements."""

    movement_speed: float = 0.0          # normalised image-widths per second
    movement_amount: float = 0.0         # normalised path length over window
    activity_level: str = "unknown"      # low | moderate | high | unknown
    posture: str = "unknown"             # standing | sitting_or_crouching | lying | unknown
    posture_confidence: float = 0.0
    pose_change: float = 0.0             # mean normalised keypoint displacement
    head_orientation: Optional[str] = None   # left | right | forward | unknown
    head_yaw_ratio: Optional[float] = None
    torso_lean_degrees: Optional[float] = None
    time_visible_seconds: float = 0.0


class PersonAnalysis(BaseModel):
    track_id: int
    box: BoundingBox
    confidence: float
    center: tuple[float, float]
    pose: PoseData
    face: FaceData = Field(default_factory=FaceData)
    expression: ExpressionResult = Field(default_factory=ExpressionResult)
    behavior: BehaviorFeatures = Field(default_factory=BehaviorFeatures)
    first_seen: float = 0.0
    last_seen: float = 0.0


class PipelineStage(BaseModel):
    name: str
    status: str            # ok | degraded | disabled | error
    detail: str = ""
    last_duration_ms: float = 0.0


class FrameAnalysis(BaseModel):
    """One analysed frame - the payload pushed over the WebSocket."""

    type: str = "analysis"
    frame_id: int = 0
    timestamp: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    people_count: int = 0
    people: list[PersonAnalysis] = Field(default_factory=list)
    fps: float = 0.0
    capture_fps: float = 0.0
    processing_ms: float = 0.0
    camera_connected: bool = False
    source: str = ""
    stages: list[PipelineStage] = Field(default_factory=list)


class ModelInfo(BaseModel):
    key: str
    name: str
    path: str
    present: bool
    loaded: bool
    size_bytes: int = 0
    source_url: str = ""
    license: str = ""
    description: str = ""
    error: str = ""


class SourceInfo(BaseModel):
    kind: str
    target: str
    connected: bool
    width: int = 0
    height: int = 0
    capture_fps: float = 0.0
    requested_fps: float = 0.0
    last_error: str = ""


class HealthInfo(BaseModel):
    status: str
    uptime_seconds: float
    pipeline_running: bool
    camera_connected: bool
    fps: float
    people_count: int
    device: str
    version: str
