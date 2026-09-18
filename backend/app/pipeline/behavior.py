"""Observable behaviour features.

Everything computed here is a measurement of what the camera can see:
how far a tracked box moved, how fast, how much the skeleton changed,
which way the head is turned, and a coarse posture label.

Explicitly out of scope: emotions, intentions, honesty, stress, and any
mental-health inference. Speed is speed; it is not "agitation".
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np

from app.models.schemas import BehaviorFeatures, BoundingBox, PoseData
from app.pose.pose_utils import (
    estimate_head_orientation,
    estimate_posture,
    pose_anchor,
    torso_lean_degrees,
)


@dataclass
class _TrackHistory:
    first_seen: float
    last_seen: float
    positions: Deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=180))
    keypoints: Deque[tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=8))


class BehaviorAnalyzer:
    """Rolling per-track statistics over a short time window."""

    def __init__(
        self,
        window_seconds: float = 3.0,
        low_threshold: float = 0.02,
        high_threshold: float = 0.12,
        keypoint_min_confidence: float = 0.4,
    ) -> None:
        self.window_seconds = window_seconds
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.keypoint_min_confidence = keypoint_min_confidence
        self._history: dict[int, _TrackHistory] = {}

    # ------------------------------------------------------------------ util
    def reset(self) -> None:
        self._history.clear()

    def forget(self, track_id: int) -> None:
        self._history.pop(track_id, None)

    def prune(self, active_ids: set[int], max_age_seconds: float = 10.0) -> None:
        now = time.time()
        for track_id in list(self._history):
            history = self._history[track_id]
            if track_id not in active_ids and now - history.last_seen > max_age_seconds:
                del self._history[track_id]

    def first_seen(self, track_id: int) -> float:
        history = self._history.get(track_id)
        return history.first_seen if history else time.time()

    # --------------------------------------------------------------- compute
    def update(
        self,
        track_id: int,
        timestamp: float,
        box: BoundingBox,
        pose: PoseData,
        frame_width: int,
        frame_height: int,
    ) -> BehaviorFeatures:
        scale = float(max(frame_width, 1))
        anchor = pose_anchor(pose, box, self.keypoint_min_confidence)
        norm_anchor = (anchor[0] / scale, anchor[1] / max(frame_height, 1))

        history = self._history.get(track_id)
        if history is None:
            history = _TrackHistory(first_seen=timestamp, last_seen=timestamp)
            self._history[track_id] = history
        history.last_seen = timestamp
        history.positions.append((timestamp, norm_anchor[0], norm_anchor[1]))

        cutoff = timestamp - self.window_seconds
        while len(history.positions) > 2 and history.positions[0][0] < cutoff:
            history.positions.popleft()

        movement_amount = 0.0
        points = list(history.positions)
        for (_, x0, y0), (_, x1, y1) in zip(points, points[1:]):
            movement_amount += float(np.hypot(x1 - x0, y1 - y0))

        elapsed = max(points[-1][0] - points[0][0], 1e-6) if len(points) > 1 else 0.0
        movement_speed = movement_amount / elapsed if elapsed > 1e-3 else 0.0

        pose_change = self._pose_change(history, timestamp, pose, scale, frame_height)

        activity = "unknown"
        if elapsed >= 0.4:
            if movement_speed < self.low_threshold and pose_change < self.low_threshold:
                activity = "low"
            elif movement_speed < self.high_threshold:
                activity = "moderate"
            else:
                activity = "high"

        posture, posture_conf = estimate_posture(
            pose, box.width, box.height, self.keypoint_min_confidence
        )
        head_orientation, head_ratio = estimate_head_orientation(
            pose, self.keypoint_min_confidence
        )

        return BehaviorFeatures(
            movement_speed=round(movement_speed, 4),
            movement_amount=round(movement_amount, 4),
            activity_level=activity,
            posture=posture,
            posture_confidence=round(posture_conf, 3),
            pose_change=round(pose_change, 4),
            head_orientation=head_orientation,
            head_yaw_ratio=head_ratio,
            torso_lean_degrees=torso_lean_degrees(pose, self.keypoint_min_confidence),
            time_visible_seconds=round(timestamp - history.first_seen, 2),
        )

    # -------------------------------------------------------------- internal
    def _pose_change(
        self,
        history: _TrackHistory,
        timestamp: float,
        pose: PoseData,
        scale: float,
        frame_height: int,
    ) -> float:
        current = self._pose_vector(pose, scale, frame_height)
        previous: Optional[np.ndarray] = None
        previous_time = timestamp
        if history.keypoints:
            previous_time, previous = history.keypoints[-1]
        history.keypoints.append((timestamp, current))

        if previous is None or current is None:
            return 0.0
        dt = max(timestamp - previous_time, 1e-6)
        mask = ~np.isnan(current) & ~np.isnan(previous)
        if mask.sum() < 4:
            return 0.0
        delta = np.abs(current[mask] - previous[mask])
        return float(delta.mean() / dt)

    def _pose_vector(self, pose: PoseData, scale: float, frame_height: int) -> np.ndarray:
        values: list[float] = []
        for kp in pose.keypoints:
            if kp.confidence >= self.keypoint_min_confidence:
                values.extend([kp.x / scale, kp.y / max(frame_height, 1)])
            else:
                values.extend([np.nan, np.nan])
        return np.array(values, dtype=np.float32) if values else np.array([], dtype=np.float32)
