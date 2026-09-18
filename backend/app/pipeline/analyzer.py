"""The analysis pipeline.

    frame -> YOLO Pose -> ByteTrack -> face detection -> expression -> behaviour

Each stage is an injected object with a narrow interface, so any of them can be
replaced. A stage that fails is reported as `error`/`degraded` in the payload
and the remaining stages keep running; one broken face crop must never take
down the stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.expression.base import ExpressionClassifier
from app.face.face_detector import BaseFaceDetector, FaceBox, pick_best_face
from app.models.schemas import (
    BoundingBox,
    ExpressionResult,
    FaceData,
    FrameAnalysis,
    PersonAnalysis,
    PipelineStage,
)
from app.pipeline.behavior import BehaviorAnalyzer
from app.pose.pose_utils import keypoints_to_pose
from app.pose.yolo_pose import YoloPoseEstimator
from app.tracking.tracker import PersonTracker

logger = get_logger(__name__)


@dataclass
class _ExpressionCache:
    probabilities: dict[str, float] = field(default_factory=dict)
    last_frame: int = -10_000
    inference_ms: float = 0.0
    model_name: str = ""


class HumanAnalysisPipeline:
    """Runs every stage for one frame and returns a `FrameAnalysis`."""

    def __init__(
        self,
        settings: Settings,
        pose_estimator: Optional[YoloPoseEstimator] = None,
        tracker: Optional[PersonTracker] = None,
        face_detector: Optional[BaseFaceDetector] = None,
        expression_classifier: Optional[ExpressionClassifier] = None,
    ) -> None:
        self.settings = settings
        self.pose_estimator = pose_estimator
        self.tracker = tracker
        self.face_detector = face_detector
        self.expression_classifier = expression_classifier

        self.behavior = BehaviorAnalyzer(
            window_seconds=settings.behavior_history_seconds,
            low_threshold=settings.activity_low_threshold,
            high_threshold=settings.activity_high_threshold,
            keypoint_min_confidence=settings.keypoint_confidence_threshold,
        )
        self._expression_cache: dict[int, _ExpressionCache] = {}
        self._stage_errors: dict[str, str] = {}
        self._stage_ms: dict[str, float] = {}
        self._last_analysis: Optional[FrameAnalysis] = None

    # ----------------------------------------------------------------- setup
    def reset_tracks(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()
        self.behavior.reset()
        self._expression_cache.clear()

    @property
    def stages(self) -> list[PipelineStage]:
        def stage(name: str, obj, disabled_detail: str = "not configured") -> PipelineStage:
            error = self._stage_errors.get(name, "")
            duration = round(self._stage_ms.get(name, 0.0), 2)
            if obj is None:
                return PipelineStage(name=name, status="disabled", detail=disabled_detail)
            if error:
                return PipelineStage(name=name, status="error", detail=error,
                                     last_duration_ms=duration)
            return PipelineStage(name=name, status="ok",
                                 detail=getattr(obj, "name", type(obj).__name__),
                                 last_duration_ms=duration)

        stages = [
            stage("pose", self.pose_estimator, "YOLO pose model not loaded"),
            stage("tracking", self.tracker, "tracker not loaded"),
            stage("face", self.face_detector, "face detection disabled"),
            stage("expression", self.expression_classifier,
                  "expression model not loaded"),
            PipelineStage(
                name="behavior",
                status="ok",
                detail="observable features only",
                last_duration_ms=round(self._stage_ms.get("behavior", 0.0), 2),
            ),
        ]
        if self.face_detector is not None and getattr(self.face_detector, "name", "") == "haar":
            for item in stages:
                if item.name == "face":
                    item.status = "degraded"
                    item.detail = "haar fallback (MediaPipe model missing)"
        return stages

    # --------------------------------------------------------------- process
    def process(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
    ) -> FrameAnalysis:
        timestamp = timestamp if timestamp is not None else time.time()
        started = time.perf_counter()
        height, width = frame.shape[:2]

        detections = []
        if self.pose_estimator is not None:
            t0 = time.perf_counter()
            try:
                detections = self.pose_estimator.detect(frame)
                self._stage_errors.pop("pose", None)
            except Exception as exc:
                logger.exception("Pose stage failed: %s", exc)
                self._stage_errors["pose"] = str(exc)
            self._stage_ms["pose"] = (time.perf_counter() - t0) * 1000.0

        tracked = []
        if self.tracker is not None:
            t0 = time.perf_counter()
            try:
                tracked = self.tracker.update(detections, frame)
                self._stage_errors.pop("tracking", None)
            except Exception as exc:
                logger.exception("Tracking stage failed: %s", exc)
                self._stage_errors["tracking"] = str(exc)
            self._stage_ms["tracking"] = (time.perf_counter() - t0) * 1000.0
        else:
            tracked = [
                type("T", (), {
                    "track_id": -1,
                    "box": d.box,
                    "confidence": d.confidence,
                    "keypoints": d.keypoints,
                })()
                for d in detections
            ]

        people: list[PersonAnalysis] = []
        face_ms = 0.0
        expression_ms = 0.0
        behavior_ms = 0.0
        active_ids: set[int] = set()

        for item in tracked[: self.settings.max_people]:
            x1, y1, x2, y2 = item.box
            box = BoundingBox(
                x1=round(float(x1), 1), y1=round(float(y1), 1),
                x2=round(float(x2), 1), y2=round(float(y2), 1),
            )
            pose = keypoints_to_pose(
                item.keypoints, self.settings.keypoint_confidence_threshold
            )
            track_id = int(item.track_id)
            active_ids.add(track_id)

            t0 = time.perf_counter()
            face = self._detect_face(frame, box, pose)
            face_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            expression = self._classify_expression(frame, face, track_id, frame_id)
            expression_ms += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            behavior = self.behavior.update(
                track_id, timestamp, box, pose, width, height
            )
            behavior_ms += (time.perf_counter() - t0) * 1000.0

            people.append(
                PersonAnalysis(
                    track_id=track_id,
                    box=box,
                    confidence=round(float(item.confidence), 4),
                    center=(round(box.center[0], 1), round(box.center[1], 1)),
                    pose=pose,
                    face=face,
                    expression=expression,
                    behavior=behavior,
                    first_seen=round(self.behavior.first_seen(track_id), 3),
                    last_seen=round(timestamp, 3),
                )
            )

        self._stage_ms["face"] = face_ms
        self._stage_ms["expression"] = expression_ms
        self._stage_ms["behavior"] = behavior_ms

        self.behavior.prune(active_ids)
        for track_id in list(self._expression_cache):
            if track_id not in active_ids and len(self._expression_cache) > self.settings.max_people:
                self._expression_cache.pop(track_id, None)

        analysis = FrameAnalysis(
            frame_id=frame_id,
            timestamp=round(timestamp, 3),
            frame_width=width,
            frame_height=height,
            people_count=len(people),
            people=people,
            processing_ms=round((time.perf_counter() - started) * 1000.0, 2),
            source=self.settings.describe_source(),
            stages=self.stages,
        )
        self._last_analysis = analysis
        return analysis

    # ------------------------------------------------------------------ face
    def _detect_face(self, frame: np.ndarray, box: BoundingBox, pose) -> FaceData:
        if self.face_detector is None:
            return FaceData(detected=False, source="disabled")

        height, width = frame.shape[:2]
        region_height = max(
            32.0, box.height * self.settings.face_search_region_ratio
        )
        pad_x = box.width * 0.1
        rx1 = int(max(0, box.x1 - pad_x))
        ry1 = int(max(0, box.y1 - box.height * 0.05))
        rx2 = int(min(width, box.x2 + pad_x))
        ry2 = int(min(height, box.y1 + region_height))
        if rx2 - rx1 < 16 or ry2 - ry1 < 16:
            return FaceData(detected=False, source=self.face_detector.name)

        crop = frame[ry1:ry2, rx1:rx2]
        try:
            boxes = self.face_detector.detect(crop)
            self._stage_errors.pop("face", None)
        except Exception as exc:
            logger.error("Face stage failed: %s", exc)
            self._stage_errors["face"] = str(exc)
            return FaceData(detected=False, source=self.face_detector.name)

        if not boxes:
            return FaceData(detected=False, source=self.face_detector.name)

        # Prefer the candidate nearest the nose keypoint when we have one.
        hint = None
        nose = pose.get("nose") if pose else None
        if nose is not None and nose.visible:
            hint = (nose.x - rx1, nose.y - ry1)
        best = pick_best_face(boxes, hint)
        if best is None:
            return FaceData(detected=False, source=self.face_detector.name)

        absolute = FaceBox(
            best.x1 + rx1, best.y1 + ry1, best.x2 + rx1, best.y2 + ry1, best.confidence
        ).clip(width, height)
        if absolute.area < 64:
            return FaceData(detected=False, source=self.face_detector.name)

        return FaceData(
            detected=True,
            box=BoundingBox(
                x1=round(absolute.x1, 1), y1=round(absolute.y1, 1),
                x2=round(absolute.x2, 1), y2=round(absolute.y2, 1),
            ),
            confidence=round(float(absolute.confidence), 4),
            source=self.face_detector.name,
        )

    # ------------------------------------------------------------ expression
    def _crop_face(self, frame: np.ndarray, face: FaceData) -> Optional[np.ndarray]:
        if not face.detected or face.box is None:
            return None
        height, width = frame.shape[:2]
        pad_w = face.box.width * self.settings.face_padding_ratio
        pad_h = face.box.height * self.settings.face_padding_ratio
        x1 = int(max(0, face.box.x1 - pad_w))
        y1 = int(max(0, face.box.y1 - pad_h))
        x2 = int(min(width, face.box.x2 + pad_w))
        y2 = int(min(height, face.box.y2 + pad_h))
        if x2 - x1 < 12 or y2 - y1 < 12:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None

    def _classify_expression(
        self,
        frame: np.ndarray,
        face: FaceData,
        track_id: int,
        frame_id: int,
    ) -> ExpressionResult:
        if self.expression_classifier is None or not self.settings.expression_enabled:
            return ExpressionResult(available=False)
        if not face.detected:
            # Nothing to classify: report unavailable rather than a stale guess.
            return ExpressionResult(available=False,
                                    model_name=getattr(self.expression_classifier, "name", None))

        cache = self._expression_cache.get(track_id)
        interval = max(1, self.settings.expression_every_n_frames)
        needs_run = cache is None or (frame_id - cache.last_frame) >= interval

        if needs_run:
            crop = self._crop_face(frame, face)
            if crop is None:
                return ExpressionResult(available=False)
            try:
                probabilities = self.expression_classifier.predict(crop)
                self._stage_errors.pop("expression", None)
            except Exception as exc:
                logger.error("Expression stage failed: %s", exc)
                self._stage_errors["expression"] = str(exc)
                return ExpressionResult(available=False)

            if not probabilities:
                return ExpressionResult(available=False)

            inference_ms = float(
                getattr(self.expression_classifier, "last_inference_ms", 0.0)
            )
            if cache is not None and cache.probabilities and self.settings.expression_smoothing > 0:
                alpha = self.settings.expression_smoothing
                probabilities = {
                    label: round(alpha * value + (1 - alpha) * cache.probabilities.get(label, value), 4)
                    for label, value in probabilities.items()
                }
            cache = _ExpressionCache(
                probabilities=probabilities,
                last_frame=frame_id,
                inference_ms=inference_ms,
                model_name=getattr(self.expression_classifier, "name", "expression"),
            )
            self._expression_cache[track_id] = cache

        if cache is None or not cache.probabilities:
            return ExpressionResult(available=False)

        top_label, top_probability = max(cache.probabilities.items(), key=lambda kv: kv[1])
        return ExpressionResult(
            available=True,
            probabilities=cache.probabilities,
            top_label=top_label,
            top_probability=round(float(top_probability), 4),
            model_name=cache.model_name,
            inference_ms=round(cache.inference_ms, 2),
        )
