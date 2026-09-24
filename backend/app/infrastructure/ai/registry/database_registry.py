"""PostgreSQL-backed model registry with a short read cache."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.ai.capabilities import ModelCapabilities
from app.domain.ai.routing import (
    HealthStatus,
    ModelDescriptor,
    ModelPricing,
    ModelRegistry,
    ModelType,
)
from app.infrastructure.ai.registry.common import apply_changes, next_latency
from app.infrastructure.db.models import AIModelRow
from app.infrastructure.db.session import translate_db_errors

logger = logging.getLogger(__name__)


def row_to_descriptor(row: AIModelRow) -> ModelDescriptor:
    return ModelDescriptor(
        id=row.id,
        provider=row.provider,
        model_name=row.model_name,
        model_type=ModelType(row.model_type),
        enabled=row.enabled,
        is_local=row.is_local,
        priority=row.priority,
        pricing=ModelPricing(
            input_cost_per_million=row.input_cost_per_million,
            output_cost_per_million=row.output_cost_per_million,
            currency=row.currency,
        ),
        capabilities=ModelCapabilities(
            structured_output=row.supports_structured_output,
            json_mode=row.supports_json_mode,
            tool_calling=row.supports_tools,
            vision=row.supports_vision,
            streaming=row.supports_streaming,
            max_context_tokens=row.context_window,
        ),
        languages=tuple(row.languages) if row.languages else None,
        quality_score=row.quality_score,
        avg_latency_ms=row.avg_latency_ms,
        health_status=HealthStatus(row.health_status),
        provider_options=dict(row.provider_options or {}),
    )


def _descriptor_values(model: ModelDescriptor) -> dict[str, Any]:
    return {
        "enabled": model.enabled,
        "is_local": model.is_local,
        "priority": model.priority,
        "input_cost_per_million": model.pricing.input_cost_per_million,
        "output_cost_per_million": model.pricing.output_cost_per_million,
        "currency": model.pricing.currency,
        "supports_structured_output": model.capabilities.structured_output,
        "supports_json_mode": model.capabilities.json_mode,
        "supports_tools": model.capabilities.tool_calling,
        "supports_vision": model.capabilities.vision,
        "supports_streaming": model.capabilities.streaming,
        "context_window": model.capabilities.max_context_tokens,
        "languages": list(model.languages) if model.languages else None,
        "quality_score": model.quality_score,
        "provider_options": dict(model.provider_options),
    }


class DatabaseModelRegistry:
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], *, cache_seconds: float = 15.0
    ) -> None:
        self._sessions = session_factory
        self._cache_seconds = cache_seconds
        self._cache: tuple[float, list[ModelDescriptor]] | None = None

    def invalidate(self) -> None:
        self._cache = None

    @translate_db_errors
    async def _all(self) -> list[ModelDescriptor]:
        now = time.monotonic()
        if self._cache and now - self._cache[0] < self._cache_seconds:
            return self._cache[1]
        async with self._sessions() as session:
            rows = await session.scalars(
                select(AIModelRow).order_by(AIModelRow.priority, AIModelRow.provider)
            )
            models = [row_to_descriptor(row) for row in rows]
        self._cache = (now, models)
        return models

    async def list_models(self, model_type: ModelType | None = None) -> list[ModelDescriptor]:
        return [m for m in await self._all() if model_type in (None, m.model_type)]

    async def get(
        self, provider: str, model_name: str, model_type: ModelType
    ) -> ModelDescriptor | None:
        for model in await self._all():
            if (model.provider, model.model_name, model.model_type) == (
                provider,
                model_name,
                model_type,
            ):
                return model
        return None

    async def get_by_id(self, model_id: uuid.UUID) -> ModelDescriptor | None:
        return next((m for m in await self._all() if m.id == model_id), None)

    @translate_db_errors
    async def upsert(self, model: ModelDescriptor, *, overwrite: bool) -> ModelDescriptor:
        async with self._sessions() as session, session.begin():
            row = await session.scalar(
                select(AIModelRow).where(
                    AIModelRow.provider == model.provider,
                    AIModelRow.model_name == model.model_name,
                    AIModelRow.model_type == model.model_type.value,
                )
            )
            if row is None:
                row = AIModelRow(
                    id=model.id,
                    provider=model.provider,
                    model_name=model.model_name,
                    model_type=model.model_type.value,
                    health_status=HealthStatus.UNKNOWN.value,
                    **_descriptor_values(model),
                )
                session.add(row)
            elif overwrite:
                values = _descriptor_values(model)
                values["quality_score"] = row.quality_score  # benchmark results survive a sync
                for key, value in values.items():
                    setattr(row, key, value)
            await session.flush()
            result = row_to_descriptor(row)
        self.invalidate()
        return result

    @translate_db_errors
    async def update(self, model_id: uuid.UUID, changes: Mapping[str, Any]) -> ModelDescriptor:
        async with self._sessions() as session, session.begin():
            row = await session.get(AIModelRow, model_id)
            if row is None:
                raise KeyError(str(model_id))
            updated = apply_changes(row_to_descriptor(row), changes)
            for key, value in _descriptor_values(updated).items():
                setattr(row, key, value)
            await session.flush()
            result = row_to_descriptor(row)
        self.invalidate()
        return result

    @translate_db_errors
    async def record_health(self, model_id: uuid.UUID, status: HealthStatus) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(AIModelRow)
                .where(AIModelRow.id == model_id)
                .values(health_status=status.value)
            )
        self.invalidate()

    @translate_db_errors
    async def record_latency(self, model_id: uuid.UUID, latency_ms: int) -> None:
        async with self._sessions() as session, session.begin():
            row = await session.get(AIModelRow, model_id)
            if row is not None:
                row.avg_latency_ms = next_latency(row.avg_latency_ms, latency_ms)


async def sync_registry(
    registry: ModelRegistry, models: Iterable[ModelDescriptor], *, overwrite: bool
) -> list[ModelDescriptor]:
    """Seed the registry from config. By default existing entries (admin edits) win."""
    synced = []
    for model in models:
        synced.append(await registry.upsert(model, overwrite=overwrite))
    logger.info("model registry synced: %s", ", ".join(m.key for m in synced) or "-")
    return synced
