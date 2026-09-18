"""Multi-person tracking with ByteTrack.

`PersonTracker` takes the detections produced by the pose stage and returns
the same detections enriched with a *persistent* track ID. Pose keypoints are
carried through by index, so a track ID always keeps the skeleton that
belongs to it.

The tracker is deliberately isolated behind `PersonTracker.update()` so it can
be replaced (BoT-SORT, OC-SORT, a re-ID tracker, ...) without touching the
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

from app.core.logging_config import get_logger
from app.detection.base import PersonDetection

logger = get_logger(__name__)


class _DetectionArray:
    """Minimal `Results.boxes`-like adapter that BYTETracker can consume.

    BYTETracker only needs `.conf`, `.xywh`, `.cls` and boolean-mask indexing.
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask) -> "_DetectionArray":
        return _DetectionArray(self.xywh[mask], self.conf[mask], self.cls[mask])


@dataclass
class TrackedPerson:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    keypoints: Optional[np.ndarray]
    detection_index: int


class PersonTracker:
    """Persistent-ID tracker for people."""

    name = "bytetrack"

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        fuse_score: bool = True,
    ) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.args = SimpleNamespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self._tracker_cls = BYTETracker
        self.tracker = BYTETracker(self.args)
        logger.info(
            "ByteTrack initialised (high=%.2f low=%.2f new=%.2f buffer=%d match=%.2f)",
            track_high_thresh, track_low_thresh, new_track_thresh, track_buffer, match_thresh,
        )

    def reset(self) -> None:
        """Drop all tracks (used when the camera reconnects)."""
        self.tracker = self._tracker_cls(self.args)
        logger.info("Tracker state reset")

    def update(
        self,
        detections: list[PersonDetection],
        frame: Optional[np.ndarray] = None,
    ) -> list[TrackedPerson]:
        if not detections:
            # Still step the tracker so lost tracks age out correctly.
            empty = _DetectionArray(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
            self.tracker.update(empty, frame)
            return []

        xywh = np.array([d.xywh for d in detections], dtype=np.float32)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        cls = np.array([d.class_id for d in detections], dtype=np.float32)

        outputs = self.tracker.update(_DetectionArray(xywh, conf, cls), frame)

        tracked: list[TrackedPerson] = []
        if outputs is None or len(outputs) == 0:
            return tracked

        for row in outputs:
            x1, y1, x2, y2, track_id, score = row[:6]
            det_index = int(row[7]) if len(row) > 7 else -1
            keypoints = None
            if 0 <= det_index < len(detections):
                keypoints = detections[det_index].keypoints
            tracked.append(
                TrackedPerson(
                    track_id=int(track_id),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(score),
                    keypoints=keypoints,
                    detection_index=det_index,
                )
            )
        return tracked
