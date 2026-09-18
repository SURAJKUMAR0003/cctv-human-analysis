# Backend

FastAPI service that runs the analysis pipeline and serves the results.

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Configuration comes from `.env` at the repository root (see `.env.example`).
Interactive API docs: <http://localhost:8000/docs>.

## Threading model

Three threads, so a slow consumer can never stall inference:

1. **capture** (`services/video_source.py`) - drains `cv2.VideoCapture` and
   keeps only the newest frame. Reconnects on failure; loops files at EOF.
2. **analysis** (`services/analysis_service.py`) - reads the newest frame, runs
   the pipeline, publishes the `FrameAnalysis` and an annotated JPEG.
3. **event loop** - REST, MJPEG and WebSocket handlers read the published
   result. They never block the pipeline, and clients that fall behind simply
   skip frames.

## Stage contracts

| Stage | Interface | Default implementation |
|---|---|---|
| detection + pose | `PersonDetector.detect(frame)` | `YoloPoseEstimator` |
| tracking | `PersonTracker.update(detections, frame)` | ByteTrack |
| face | `BaseFaceDetector.detect(image)` | MediaPipe, Haar fallback |
| expression | `ExpressionClassifier.predict(face_image)` | `OnnxExpressionClassifier` |
| behaviour | `BehaviorAnalyzer.update(...)` | rolling per-track statistics |

A stage that throws is caught, reported as `error` in `stages[]`, and skipped -
the other stages keep producing output.

## Error handling rules

- No face found is a normal result, not an error: `face.detected = false`.
- No face means no expression: `expression.available = false` with empty
  probabilities. Stale results are never re-reported as current.
- A missing model file disables its stage and is reported on `/api/models`.
  Nothing is estimated to fill the gap.
- Camera loss flips `camera_connected`, triggers reconnect, and resets tracker
  state when the feed returns (so IDs do not silently continue across a gap).

## Tests

```bash
pytest -v                       # everything
pytest tests/test_api.py -v     # boots the app and hits the real endpoints
```

The suite forces `VIDEO_SOURCE=file` against `datasets/samples/test_scene.mp4`
(`tests/conftest.py`), so it never needs a camera. Tests that require a model
file skip rather than fake it.
