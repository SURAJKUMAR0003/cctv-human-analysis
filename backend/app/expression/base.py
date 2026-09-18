"""Facial-expression classifier interface.

The pipeline only ever calls `ExpressionClassifier.predict(face_image)` and
gets back label -> probability. Replace the implementation (your own ONNX
export, a TorchScript model, a remote service) without touching anything else.

Scope note: these labels describe the *visible facial expression* in a single
crop. They are not measurements of mood, intent, honesty or any mental-health
condition, and nothing downstream may treat them as such.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ExpressionClassifier(ABC):
    """Base class for facial-expression models."""

    name: str = "abstract-expression-classifier"
    labels: list[str] = []

    @abstractmethod
    def predict(self, face_image: np.ndarray) -> dict[str, float]:
        """Classify one BGR face crop.

        Returns a mapping of label -> probability that sums to ~1.0.
        Returns an empty dict when the crop cannot be classified.
        """

    def warmup(self) -> None:  # pragma: no cover - optional hook
        return None

    def close(self) -> None:  # pragma: no cover - optional hook
        return None


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    total = e.sum()
    if total <= 0:
        return np.full_like(e, 1.0 / len(e))
    return e / total
