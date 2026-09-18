"""WebSocket endpoint `/ws/live`.

Pushes one JSON `FrameAnalysis` per processed frame, rate-limited to
`WS_FPS`. Frames are only sent when the frame id changed, so an idle camera
costs nothing. Slow clients cannot stall the pipeline: the socket reads the
latest published result and skips whatever it missed.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active.add(websocket)
        logger.info("WebSocket connected (%d active)", len(self.active))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active.discard(websocket)
        logger.info("WebSocket disconnected (%d active)", len(self.active))

    @property
    def count(self) -> int:
        return len(self.active)


manager = ConnectionManager()


@router.websocket("/ws/live")
async def live(websocket: WebSocket) -> None:
    service = websocket.app.state.service
    await manager.connect(websocket)

    interval = 1.0 / max(settings.ws_fps, 1.0)
    last_frame_id = -1
    last_heartbeat = 0.0

    try:
        # Tell the client what it is connected to before any frames arrive.
        await websocket.send_json({
            "type": "hello",
            "source": service.source.describe(),
            "device": service.device,
            "ws_fps": settings.ws_fps,
            "stream_url": "/api/stream.mjpg",
            "models": [m.model_dump() for m in service.models.report()],
            "stages": [s.model_dump() for s in service.stage_report()],
            "notice": (
                "Expression labels describe visible facial expressions only. "
                "They are not psychological or medical assessments."
            ),
        })

        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break

            analysis = service.analysis
            now = time.time()

            if analysis is not None and analysis.frame_id != last_frame_id:
                last_frame_id = analysis.frame_id
                last_heartbeat = now
                await websocket.send_json(analysis.model_dump())
            elif now - last_heartbeat > 2.0:
                # Keeps the dashboard's connection indicator honest while the
                # camera is down or reconnecting.
                last_heartbeat = now
                await websocket.send_json({
                    "type": "status",
                    "timestamp": round(now, 3),
                    "camera_connected": service.source.connected,
                    "source": service.source.describe(),
                    "last_error": service.source.last_error,
                    "fps": service.health().fps,
                    "stages": [s.model_dump() for s in service.stage_report()],
                })

            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - transport level
        logger.warning("WebSocket error: %s", exc)
    finally:
        await manager.disconnect(websocket)
