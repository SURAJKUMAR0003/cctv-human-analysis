"""FastAPI application entrypoint.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as rest_router
from app.api.websocket import router as ws_router
from app.core.config import settings
from app.core.logging_config import configure_logging, get_logger
from app.services.analysis_service import VERSION, AnalysisService

configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 68)
    logger.info("CCTV Human Analysis System v%s", VERSION)
    logger.info("Source: %s | target fps: %s | device: %s",
                settings.describe_source(), settings.target_fps, settings.device)
    logger.info("=" * 68)

    service = AnalysisService(settings)
    app.state.service = service

    missing = service.models.missing_required()
    if missing:
        for spec in missing:
            logger.error(
                "REQUIRED MODEL MISSING: %s (expected at %s). "
                "Run: python scripts/download_models.py",
                spec.filename, service.models.path_for(spec),
            )

    service.start()
    try:
        yield
    finally:
        service.stop()


app = FastAPI(
    title="CCTV Human Analysis System",
    version=VERSION,
    description=(
        "Real-time person detection, pose estimation, tracking, face detection "
        "and facial-expression classification for CCTV/webcam streams.\n\n"
        "**Scope note:** expression labels and behaviour features describe what is "
        "visible in the frame (movement, posture, facial expression). They are not "
        "psychological or medical assessments and must not be used as such."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest_router)
app.include_router(ws_router)


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "name": "CCTV Human Analysis System",
        "version": VERSION,
        "docs": "/docs",
        "endpoints": [
            "/health", "/api/source", "/api/models", "/api/people",
            "/api/analysis", "/api/stream.mjpg", "/api/snapshot.jpg", "/ws/live",
        ],
    }
