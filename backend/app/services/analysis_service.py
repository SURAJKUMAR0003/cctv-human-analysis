"""The long-running analysis service.

Owns the models, the capture thread and the worker thread; publishes the most
recent `FrameAnalysis` and the most recent annotated JPEG. The API layer only
reads from here, so a slow or disconnected client can never stall inference.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.expression.onnx_classifier import OnnxExpressionClassifier
from app.face.face_detector import create_face_detector
from app.models.schemas import FrameAnalysis, HealthInfo, PersonAnalysis, SourceInfo
from app.pipeline.analyzer import HumanAnalysisPipeline
from app.pipeline.visualizer import annotate, encode_jpeg
from app.pose.yolo_pose import YoloPoseEstimator, resolve_device
from app.services.model_manager import ModelManager
from app.services.video_source import VideoSource
from app.tracking.tracker import PersonTracker

logger = get_logger(__name__)

VERSION = "1.0.0"


class AnalysisService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models = ModelManager(settings)
        self.source = VideoSource(settings)
        self.pipeline: Optional[HumanAnalysisPipeline] = None
        self.device = resolve_device(settings.device)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._analysis: Optional[FrameAnalysis] = None
        self._jpeg: Optional[bytes] = None
        self._jpeg_frame_id = -1
        self._frame_times: deque[float] = deque(maxlen=30)
        self._last_generation = 0
        self._started_at = time.time()
        self._processed_frames = 0
        self.running = False
        self.startup_errors: list[str] = []

    # ---------------------------------------------------------------- models
    def _build_pipeline(self) -> HumanAnalysisPipeline:
        settings = self.settings
        self.models.log_status()

        pose_estimator = None
        try:
            pose_estimator = YoloPoseEstimator(
                model_path=settings.yolo_model_path,
                device=settings.device,
                confidence=settings.confidence_threshold,
                iou=settings.iou_threshold,
                imgsz=settings.inference_imgsz,
                max_detections=settings.max_people,
            )
            pose_estimator.warmup()
            self.models.mark_loaded("yolo_pose", True)
        except Exception as exc:
            message = f"YOLO pose model could not be loaded: {exc}"
            logger.error(message)
            self.startup_errors.append(message)
            self.models.mark_loaded("yolo_pose", False, str(exc))

        tracker = PersonTracker(
            track_high_thresh=settings.track_high_threshold,
            track_low_thresh=settings.track_low_threshold,
            new_track_thresh=settings.new_track_threshold,
            track_buffer=settings.track_buffer,
            match_thresh=settings.match_threshold,
        )

        face_detector = create_face_detector(
            backend=settings.face_detector,
            model_path=settings.face_model_path,
            min_confidence=settings.face_detection_confidence,
        )
        self.models.mark_loaded("face", getattr(face_detector, "name", "") == "mediapipe")

        expression_classifier = None
        if settings.expression_enabled:
            try:
                expression_classifier = OnnxExpressionClassifier(
                    model_path=settings.expression_model_path,
                    device=self.device,
                )
                expression_classifier.warmup()
                self.models.mark_loaded("expression", True)
            except Exception as exc:
                message = f"Expression model unavailable: {exc}"
                logger.warning(message)
                self.startup_errors.append(message)
                self.models.mark_loaded("expression", False, str(exc))

        return HumanAnalysisPipeline(
            settings=settings,
            pose_estimator=pose_estimator,
            tracker=tracker,
            face_detector=face_detector,
            expression_classifier=expression_classifier,
        )

    # --------------------------------------------------------------- control
    def start(self) -> None:
        if self.running:
            return
        logger.info("Starting analysis service (device=%s)", self.device)
        self.pipeline = self._build_pipeline()
        self.source.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="analysis", daemon=True)
        self._thread.start()
        self.running = True

    def stop(self) -> None:
        logger.info("Stopping analysis service")
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.source.stop()
        self.running = False

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        interval = 1.0 / max(self.settings.target_fps, 1e-3)
        last_frame_id = -1
        counter = 0

        while not self._stop.is_set():
            cycle_started = time.perf_counter()
            frame, frame_id = self.source.read()

            if frame is None or frame_id == last_frame_id:
                time.sleep(0.005)
                continue
            last_frame_id = frame_id

            if self.source.generation != self._last_generation:
                if self._last_generation != 0 and self.pipeline is not None:
                    logger.info("Camera reconnected - resetting tracker state")
                    self.pipeline.reset_tracks()
                self._last_generation = self.source.generation

            counter += 1
            if self.settings.process_every_n_frames > 1 and counter % self.settings.process_every_n_frames:
                continue

            try:
                analysis = self.pipeline.process(frame, frame_id=frame_id, timestamp=time.time())
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Pipeline error: %s", exc)
                time.sleep(0.05)
                continue

            now = time.time()
            self._frame_times.append(now)
            analysis.fps = round(self._measured_fps(), 2)
            analysis.capture_fps = round(self.source.capture_fps, 2)
            analysis.camera_connected = self.source.connected
            self._processed_frames += 1

            jpeg: Optional[bytes] = None
            try:
                rendered = (
                    annotate(frame, analysis,
                             min_keypoint_confidence=self.settings.keypoint_confidence_threshold)
                    if self.settings.annotate_stream else frame
                )
                jpeg = encode_jpeg(rendered, self.settings.mjpeg_quality)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Frame encoding failed: %s", exc)

            with self._lock:
                self._analysis = analysis
                if jpeg is not None:
                    self._jpeg = jpeg
                    self._jpeg_frame_id = frame_id

            elapsed = time.perf_counter() - cycle_started
            if interval > elapsed:
                time.sleep(interval - elapsed)

        logger.info("Analysis loop finished after %d frames", self._processed_frames)

    def _measured_fps(self) -> float:
        stamps = list(self._frame_times)
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 0 else 0.0

    # ------------------------------------------------------------------ read
    @property
    def analysis(self) -> Optional[FrameAnalysis]:
        with self._lock:
            return self._analysis

    def jpeg(self) -> tuple[Optional[bytes], int]:
        with self._lock:
            return self._jpeg, self._jpeg_frame_id

    def people(self) -> list[PersonAnalysis]:
        analysis = self.analysis
        if analysis is None:
            return []
        cutoff = time.time() - self.settings.stale_person_seconds
        return [p for p in analysis.people if p.last_seen >= cutoff]

    def health(self) -> HealthInfo:
        analysis = self.analysis
        status = "ok"
        if not self.running:
            status = "stopped"
        elif not self.source.connected:
            status = "degraded"
        elif self.pipeline is None or self.pipeline.pose_estimator is None:
            status = "degraded"
        return HealthInfo(
            status=status,
            uptime_seconds=round(time.time() - self._started_at, 1),
            pipeline_running=self.running,
            camera_connected=self.source.connected,
            fps=round(self._measured_fps(), 2),
            people_count=analysis.people_count if analysis else 0,
            device=self.device,
            version=VERSION,
        )

    def source_info(self) -> SourceInfo:
        return SourceInfo(
            kind=self.source.kind,
            target=str(self.source.target),
            connected=self.source.connected,
            width=self.source.width,
            height=self.source.height,
            capture_fps=round(self.source.capture_fps, 2),
            requested_fps=self.settings.target_fps,
            last_error=self.source.last_error,
        )

    def stage_report(self) -> list:
        return self.pipeline.stages if self.pipeline else []

    def empty_analysis(self) -> FrameAnalysis:
        """Payload used before the first frame has been processed."""
        return FrameAnalysis(
            frame_id=0,
            timestamp=round(time.time(), 3),
            people_count=0,
            people=[],
            fps=0.0,
            camera_connected=self.source.connected,
            source=self.source.describe(),
            stages=self.stage_report(),
        )
