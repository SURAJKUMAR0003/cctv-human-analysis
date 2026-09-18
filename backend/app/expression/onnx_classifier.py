"""ONNX Runtime facial-expression classifier.

Defaults match the bundled HSEmotion model (`enet_b0_8_best_afew.onnx`,
AffectNet 8 classes, 224x224, ImageNet normalisation). Anything model-specific
can be overridden with a sidecar JSON file next to the weights:

    models/emotion_enet_b0_8.onnx
    models/emotion_enet_b0_8.json

    {
      "labels": ["anger", "contempt", ...],
      "input_size": [224, 224],
      "color_order": "rgb",
      "scale": 0.00392156862745098,
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225],
      "layout": "nchw",
      "apply_softmax": true
    }

That is the whole contract for swapping in your own trained model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from app.core.logging_config import get_logger
from app.expression.base import ExpressionClassifier, softmax

logger = get_logger(__name__)

DEFAULT_LABELS = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

DEFAULT_CONFIG: dict = {
    "labels": DEFAULT_LABELS,
    "input_size": [224, 224],
    "color_order": "rgb",
    "scale": 1.0 / 255.0,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "layout": "nchw",
    "apply_softmax": True,
}


class OnnxExpressionClassifier(ExpressionClassifier):
    """Runs a classification ONNX graph over face crops."""

    name = "onnx-expression"

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        config: dict | None = None,
        intra_op_threads: int = 1,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Expression model not found: {self.model_path}. "
                "Run `python scripts/download_models.py`."
            )

        self.config = dict(DEFAULT_CONFIG)
        sidecar = self.model_path.with_suffix(".json")
        if sidecar.exists():
            try:
                self.config.update(json.loads(sidecar.read_text()))
                logger.info("Loaded expression model config from %s", sidecar.name)
            except Exception as exc:
                logger.warning("Ignoring malformed %s: %s", sidecar.name, exc)
        if config:
            self.config.update(config)

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, intra_op_threads)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = self.session.get_providers()
        self.name = f"onnx:{self.model_path.name}"

        self.labels = [str(label) for label in self.config["labels"]]
        self.input_size = tuple(int(v) for v in self.config["input_size"])
        self.layout = str(self.config["layout"]).lower()
        self.mean = np.array(self.config["mean"], dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(self.config["std"], dtype=np.float32).reshape(1, 1, -1)
        self.scale = float(self.config["scale"])
        self.color_order = str(self.config["color_order"]).lower()
        self.apply_softmax = bool(self.config["apply_softmax"])
        self.last_inference_ms = 0.0

        # Validate the label count against the graph where it is static.
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) == 2 and isinstance(out_shape[1], int):
            if out_shape[1] != len(self.labels):
                raise ValueError(
                    f"Expression model outputs {out_shape[1]} classes but "
                    f"{len(self.labels)} labels are configured. Fix "
                    f"{sidecar.name} before using this model."
                )

        logger.info(
            "Expression model loaded: %s (%d classes, %dx%d, providers=%s)",
            self.model_path.name, len(self.labels),
            self.input_size[0], self.input_size[1], ",".join(self.providers),
        )

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        if self.color_order == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.color_order == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, None]
        tensor = img.astype(np.float32) * self.scale
        if tensor.shape[2] == self.mean.shape[2]:
            tensor = (tensor - self.mean) / self.std
        if self.layout == "nchw":
            tensor = np.transpose(tensor, (2, 0, 1))
        return np.expand_dims(tensor, axis=0).astype(np.float32)

    def predict(self, face_image: np.ndarray) -> dict[str, float]:
        if face_image is None or face_image.size == 0:
            return {}
        if face_image.shape[0] < 8 or face_image.shape[1] < 8:
            return {}
        try:
            started = time.perf_counter()
            tensor = self.preprocess(face_image)
            raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
            self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            logger.error("Expression inference failed: %s", exc)
            return {}

        scores = np.asarray(raw).reshape(-1)
        if self.apply_softmax:
            scores = softmax(scores)
        else:
            total = float(scores.sum())
            if total > 0:
                scores = scores / total
        if len(scores) != len(self.labels):
            logger.error(
                "Expression output size %d does not match %d labels",
                len(scores), len(self.labels),
            )
            return {}
        return {label: round(float(p), 4) for label, p in zip(self.labels, scores)}

    def warmup(self) -> None:
        blank = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        self.predict(blank)
        logger.info("Expression model warmup complete (%.0f ms)", self.last_inference_ms)
