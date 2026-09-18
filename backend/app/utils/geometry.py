"""Small geometry helpers shared by the pipeline stages."""

from __future__ import annotations


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def contains(outer: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = outer
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2
