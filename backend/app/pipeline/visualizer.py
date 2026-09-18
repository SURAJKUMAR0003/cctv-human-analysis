"""Draws the analysis onto frames for the MJPEG stream."""

from __future__ import annotations

import cv2
import numpy as np

from app.models.schemas import SKELETON_EDGES, FrameAnalysis

# Stable per-ID colours (BGR).
_PALETTE = [
    (92, 214, 255), (128, 222, 137), (255, 176, 102), (200, 160, 255),
    (120, 255, 232), (255, 130, 180), (176, 232, 96), (255, 214, 120),
]


def color_for(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[abs(int(track_id)) % len(_PALETTE)]


def annotate(
    frame: np.ndarray,
    analysis: FrameAnalysis,
    draw_pose: bool = True,
    draw_face: bool = True,
    min_keypoint_confidence: float = 0.4,
) -> np.ndarray:
    """Return a copy of `frame` with detections drawn on it."""
    canvas = frame.copy()

    for person in analysis.people:
        color = color_for(person.track_id)
        x1, y1 = int(person.box.x1), int(person.box.y1)
        x2, y2 = int(person.box.x2), int(person.box.y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"ID {person.track_id:02d}  {person.confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 4, max(11, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

        if draw_pose and person.pose.keypoints:
            points = person.pose.keypoints
            for a, b in SKELETON_EDGES:
                if a < len(points) and b < len(points):
                    ka, kb = points[a], points[b]
                    if ka.confidence >= min_keypoint_confidence and kb.confidence >= min_keypoint_confidence:
                        cv2.line(canvas, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)),
                                 color, 2, cv2.LINE_AA)
            for kp in points:
                if kp.confidence >= min_keypoint_confidence:
                    cv2.circle(canvas, (int(kp.x), int(kp.y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (int(kp.x), int(kp.y)), 3, color, 1, cv2.LINE_AA)

        if draw_face and person.face.detected and person.face.box is not None:
            fb = person.face.box
            cv2.rectangle(canvas, (int(fb.x1), int(fb.y1)), (int(fb.x2), int(fb.y2)),
                          (255, 255, 255), 1)
            if person.expression.available and person.expression.top_label:
                text = f"{person.expression.top_label} {person.expression.top_probability * 100:.0f}%"
                cv2.putText(canvas, text, (int(fb.x1), max(11, int(fb.y1) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    header = (
        f"people {analysis.people_count}   "
        f"{analysis.fps:.1f} fps   "
        f"{analysis.processing_ms:.0f} ms"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (18, 20, 24), -1)
    cv2.putText(canvas, header, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 238, 242), 1, cv2.LINE_AA)
    return canvas


def encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
