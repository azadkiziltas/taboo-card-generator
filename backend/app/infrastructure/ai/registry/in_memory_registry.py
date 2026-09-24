"""Process-local model registry (tests, benchmarks, DB-less tooling)."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from app.domain.ai.routing import HealthStatus, ModelDescriptor, ModelType
from app.infrastructure.ai.registry.common import apply_changes, next_latency


class InMemoryModelRegistry:
    def __init__(self, models: Iterable[ModelDescriptor] = ()) -> None:
        self._models: dict[uuid.UUID, ModelDescriptor] = {m.id: m for m in models}

    async def list_models(self, model_type: ModelType | None = None) -> list[ModelDescriptor]:
        models = [m for m in self._models.values() if model_type in (None, m.model_type)]
        return sorted(models, key=lambda m: (m.priority, m.key))

    async def get(
        self, provider: str, model_name: str, model_type: ModelType
    ) -> ModelDescriptor | None:
        for model in self._models.values():
            if (model.provider, model.model_name, model.model_type) == (
                provider,
                model_name,
                model_type,
            ):
                return model
        return None

    async def get_by_id(self, model_id: uuid.UUID) -> ModelDescriptor | None:
        return self._models.get(model_id)

    async def upsert(self, model: ModelDescriptor, *, overwrite: bool) -> ModelDescriptor:
        existing = await self.get(model.provider, model.model_name, model.model_type)
        if existing is None:
            self._models[model.id] = model
            return model
        if not overwrite:
            return existing
        updated = replace(model, id=existing.id)
        self._models[existing.id] = updated
        return updated

    async def update(self, model_id: uuid.UUID, changes: Mapping[str, Any]) -> ModelDescriptor:
        model = self._models[model_id]
        self._models[model_id] = apply_changes(model, changes)
        return self._models[model_id]

    async def record_health(self, model_id: uuid.UUID, status: HealthStatus) -> None:
        if model_id in self._models:
            self._models[model_id] = replace(self._models[model_id], health_status=status)

    async def record_latency(self, model_id: uuid.UUID, latency_ms: int) -> None:
        if model_id in self._models:
            model = self._models[model_id]
            self._models[model_id] = replace(
                model, avg_latency_ms=next_latency(model.avg_latency_ms, latency_ms)
            )
