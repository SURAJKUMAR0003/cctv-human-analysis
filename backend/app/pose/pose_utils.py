"""Helpers that turn raw keypoint arrays into the API pose structure and into
simple geometric descriptions (posture, head orientation).

Everything here is geometry. None of it is a psychological statement.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from app.models.schemas import KEYPOINT_NAMES, Keypoint, PoseData


def keypoints_to_pose(
    keypoints: Optional[np.ndarray],
    min_confidence: float = 0.4,
) -> PoseData:
    """Convert a (K, 3) array into the wire-format `PoseData`."""
    if keypoints is None or len(keypoints) == 0:
        return PoseData()

    items: list[Keypoint] = []
    confs: list[float] = []
    visible = 0
    for idx, row in enumerate(keypoints):
        name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kp_{idx}"
        x, y = float(row[0]), float(row[1])
        conf = float(row[2]) if len(row) > 2 else 0.0
        is_visible = conf >= min_confidence and (x > 0 or y > 0)
        if is_visible:
            visible += 1
        confs.append(conf)
        items.append(Keypoint(name=name, x=round(x, 2), y=round(y, 2),
                              confidence=round(conf, 4), visible=is_visible))

    return PoseData(
        keypoints=items,
        mean_confidence=round(float(np.mean(confs)) if confs else 0.0, 4),
        visible_keypoints=visible,
    )


def _point(pose: PoseData, name: str, min_conf: float) -> Optional[tuple[float, float]]:
    kp = pose.get(name)
    if kp is None or kp.confidence < min_conf:
        return None
    return (kp.x, kp.y)


def _midpoint(a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]):
    if a and b:
        return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return a or b


def estimate_posture(
    pose: PoseData,
    box_width: float,
    box_height: float,
    min_conf: float = 0.4,
) -> tuple[str, float]:
    """Classify gross body posture from keypoint geometry.

    Returns (label, confidence) where label is one of
    standing / sitting_or_crouching / lying / unknown.
    """
    shoulders = _midpoint(_point(pose, "left_shoulder", min_conf),
                          _point(pose, "right_shoulder", min_conf))
    hips = _midpoint(_point(pose, "left_hip", min_conf),
                     _point(pose, "right_hip", min_conf))
    knees = _midpoint(_point(pose, "left_knee", min_conf),
                      _point(pose, "right_knee", min_conf))
    ankles = _midpoint(_point(pose, "left_ankle", min_conf),
                       _point(pose, "right_ankle", min_conf))

    if shoulders is None or hips is None:
        # Without a visible torso there is nothing to measure. A wide box is
        # just as likely to be a close-up of an upright person as someone
        # lying down, so report "unknown" instead of guessing a label the
        # dashboard would display as fact.
        return "unknown", 0.0

    torso_dx = hips[0] - shoulders[0]
    torso_dy = hips[1] - shoulders[1]
    torso_len = math.hypot(torso_dx, torso_dy)
    if torso_len < 1e-3:
        return "unknown", 0.0

    # Angle of the torso away from vertical (0 = upright, 90 = horizontal).
    torso_angle = abs(math.degrees(math.atan2(abs(torso_dx), abs(torso_dy))))
    if torso_angle > 60.0:
        return "lying", min(1.0, 0.5 + (torso_angle - 60.0) / 60.0)

    lower = knees or ankles
    if lower is None:
        # Upright torso but legs not visible (common on close-range CCTV).
        return "standing", 0.4

    leg_len = abs(lower[1] - hips[1])
    ratio = leg_len / torso_len
    if ankles and knees:
        knee_to_ankle = abs(ankles[1] - knees[1])
        hip_to_knee = abs(knees[1] - hips[1])
        if hip_to_knee > 1e-3:
            ratio = max(ratio, (hip_to_knee + knee_to_ankle) / torso_len)

    if ratio < 0.85:
        return "sitting_or_crouching", min(1.0, 0.45 + (0.85 - ratio))
    return "standing", min(1.0, 0.5 + (ratio - 0.85) * 0.5)


def estimate_head_orientation(
    pose: PoseData,
    min_conf: float = 0.4,
) -> tuple[Optional[str], Optional[float]]:
    """Rough left/right/forward head orientation from eye/ear/nose geometry."""
    nose = _point(pose, "nose", min_conf)
    left_eye = _point(pose, "left_eye", min_conf)
    right_eye = _point(pose, "right_eye", min_conf)
    left_ear = _point(pose, "left_ear", min_conf)
    right_ear = _point(pose, "right_ear", min_conf)

    if nose is None:
        if left_ear and not right_ear:
            return "right", None
        if right_ear and not left_ear:
            return "left", None
        return None, None

    reference = None
    if left_eye and right_eye:
        reference = (left_eye, right_eye)
    elif left_ear and right_ear:
        reference = (left_ear, right_ear)

    if reference is None:
        if left_ear or left_eye:
            return "right", None
        if right_ear or right_eye:
            return "left", None
        return "unknown", None

    (lx, _), (rx, _) = reference
    span = abs(lx - rx)
    if span < 1e-3:
        return "unknown", None
    center = (lx + rx) / 2.0
    # Positive ratio -> nose shifted towards image-left of face centre.
    ratio = (nose[0] - center) / span
    if ratio > 0.35:
        return "left", round(float(ratio), 3)
    if ratio < -0.35:
        return "right", round(float(ratio), 3)
    return "forward", round(float(ratio), 3)


def torso_lean_degrees(pose: PoseData, min_conf: float = 0.4) -> Optional[float]:
    shoulders = _midpoint(_point(pose, "left_shoulder", min_conf),
                          _point(pose, "right_shoulder", min_conf))
    hips = _midpoint(_point(pose, "left_hip", min_conf),
                     _point(pose, "right_hip", min_conf))
    if not shoulders or not hips:
        return None
    dx = hips[0] - shoulders[0]
    dy = hips[1] - shoulders[1]
    if abs(dy) < 1e-6 and abs(dx) < 1e-6:
        return None
    return round(float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))), 2)


def pose_anchor(pose: PoseData, box, min_conf: float = 0.4) -> tuple[float, float]:
    """Stable reference point for movement measurements (hip centre if visible)."""
    hips = _midpoint(_point(pose, "left_hip", min_conf),
                     _point(pose, "right_hip", min_conf))
    if hips:
        return hips
    shoulders = _midpoint(_point(pose, "left_shoulder", min_conf),
                          _point(pose, "right_shoulder", min_conf))
    if shoulders:
        return shoulders
    return box.center
