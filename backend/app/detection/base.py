from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PersonDetection:
    box: tuple[float, float, float, float]
    confidence: float
    keypoints: Optional[np.ndarray] = None
    class_id: int = 0

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.box
        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        )


class PersonDetector(ABC):
    name: str = "person-detector"

    @property
    @abstractmethod
    def provides_keypoints(self) -> bool:
        ...


    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        ...


    def warmup(self) -> None:
        pass