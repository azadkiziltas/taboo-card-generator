from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import ContainerDep
from app.api.v1.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Public service health")
async def health(container: ContainerDep) -> HealthResponse:
    """Coarse component status only; provider names and details are admin-only."""
    return HealthResponse(**await container.health.public_status())
