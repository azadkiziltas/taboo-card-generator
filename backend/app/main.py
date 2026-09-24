"""FastAPI application factory.

Run with:  uvicorn app.main:create_app --factory
The internal review UI is served at /ui/ (same origin as the API).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.admin.routes import router as admin_router
from app.api.errors import register_error_handlers
from app.api.v1.cards import router as cards_router
from app.api.v1.health import router as health_router
from app.bootstrap import Container, build_container
from app.config.settings import Settings, get_settings
from app.infrastructure.observability.logging import configure_logging

UI_DIR = Path(__file__).parent / "ui" / "static"


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = container is None
        app.state.container = container or await build_container(settings)
        try:
            yield
        finally:
            if owned:
                await app.state.container.aclose()

    app = FastAPI(
        title="Taboo Card Generation API",
        version="0.1.0",
        description=(
            "Provider-agnostic AI card generation backend. The game client never selects "
            "a provider or model; routing is a server-side concern."
        ),
        lifespan=lifespan,
    )
    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    register_error_handlers(app)
    app.include_router(health_router, prefix="/v1")
    app.include_router(cards_router, prefix="/v1")
    app.include_router(admin_router, prefix="/admin/v1")
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return app
