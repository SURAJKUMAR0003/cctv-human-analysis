"""REST endpoints."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import (
    FrameAnalysis,
    HealthInfo,
    ModelInfo,
    PersonAnalysis,
    SourceInfo,
)
from app.services.analysis_service import AnalysisService

logger = get_logger(__name__)
router = APIRouter()


def get_service(request: Request) -> AnalysisService:
    return request.app.state.service


@router.get("/health", response_model=HealthInfo, tags=["system"])
def health(request: Request) -> HealthInfo:
    return get_service(request).health()


@router.get("/api/source", response_model=SourceInfo, tags=["system"])
def source(request: Request) -> SourceInfo:
    return get_service(request).source_info()


@router.get("/api/models", response_model=list[ModelInfo], tags=["system"])
def models(request: Request) -> list[ModelInfo]:
    return get_service(request).models.report()


@router.get("/api/people", response_model=list[PersonAnalysis], tags=["analysis"])
def people(request: Request) -> list[PersonAnalysis]:
    return get_service(request).people()


@router.get("/api/analysis", response_model=FrameAnalysis, tags=["analysis"])
def analysis(request: Request) -> FrameAnalysis:
    service = get_service(request)
    return service.analysis or service.empty_analysis()


@router.get("/api/stream.mjpg", tags=["video"])
def mjpeg_stream(request: Request) -> StreamingResponse:
    """Live MJPEG stream of the annotated frames."""
    service = get_service(request)
    boundary = "frame"

    async def generator():
        last_id = -1
        interval = 1.0 / max(settings.mjpeg_fps, 1.0)
        idle_since = time.time()
        while True:
            if await request.is_disconnected():
                break
            jpeg, frame_id = service.jpeg()
            if jpeg is not None and frame_id != last_id:
                last_id = frame_id
                idle_since = time.time()
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            elif time.time() - idle_since > 15:
                logger.warning("MJPEG stream idle for 15s (camera connected=%s)",
                               service.source.connected)
                idle_since = time.time()
            await asyncio.sleep(interval)

    return StreamingResponse(
        generator(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@router.get("/api/snapshot.jpg", tags=["video"])
def snapshot(request: Request):
    """Single annotated JPEG - handy for debugging without a browser."""
    from fastapi.responses import Response

    jpeg, _ = get_service(request).jpeg()
    if jpeg is None:
        return Response(status_code=503, content=b"no frame available")
    return Response(content=jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})
