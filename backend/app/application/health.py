"""Health reporting. The public view never exposes provider names, URLs or errors."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.application.ai.embedding_router import EmbeddingRouter
from app.application.ai.model_router import ModelRouter
from app.domain.ai.routing import CircuitBreaker

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 5.0
OK = "ok"
UNAVAILABLE = "unavailable"
DISABLED = "disabled"


class HealthService:
    def __init__(
        self,
        *,
        database_check: Callable[[], Awaitable[bool]],
        router: ModelRouter,
        embeddings: EmbeddingRouter,
        circuit_breaker: CircuitBreaker,
        rag_enabled: bool,
        primary_task: str = "card_generation",
        cache_seconds: float = 10.0,
    ) -> None:
        self._database_check = database_check
        self._router = router
        self._embeddings = embeddings
        self._breaker = circuit_breaker
        self._rag_enabled = rag_enabled
        self._task = primary_task
        self._cache_seconds = cache_seconds
        self._cached: tuple[float, dict[str, str]] | None = None

    async def public_status(self) -> dict[str, Any]:
        components = await self._components()
        required = [components["database"], components["generation"]]
        if all(value == OK for value in required):
            status = "ok" if components["embedding"] in (OK, DISABLED) else "degraded"
        elif components["database"] == OK or components["generation"] == OK:
            status = "degraded"
        else:
            status = "unavailable"
        return {"status": status, "components": components}

    async def detailed_status(self) -> dict[str, Any]:
        providers: dict[str, bool] = {}
        for name in self._router.provider_names:
            provider = self._router.provider(name)
            providers[name] = bool(provider and await _probe(provider.health_check))
        candidates = await self._router.candidates(self._task)
        return {
            **(await self.public_status()),
            "generation_providers": providers,
            "embedding_space": self._embeddings.space.key,
            "routing": {self._task: [model.key for model in candidates]},
            "circuit_breaker": self._breaker.snapshot(),
        }

    async def _components(self) -> dict[str, str]:
        now = time.monotonic()
        if self._cached and now - self._cached[0] < self._cache_seconds:
            return self._cached[1]
        database, generation, embedding = await asyncio.gather(
            _probe(self._database_check),
            self._generation_ok(),
            _probe(self._embeddings.health_check) if self._rag_enabled else _disabled(),
        )
        components = {
            "database": OK if database else UNAVAILABLE,
            "generation": OK if generation else UNAVAILABLE,
            "embedding": DISABLED if embedding is None else (OK if embedding else UNAVAILABLE),
        }
        self._cached = (now, components)
        return components

    async def _generation_ok(self) -> bool:
        try:
            candidates = await self._router.candidates(self._task)
        except Exception:
            logger.exception("could not load routing candidates")
            return False
        for name in dict.fromkeys(model.provider for model in candidates):
            provider = self._router.provider(name)
            if provider and await _probe(provider.health_check):
                return True
        return False


async def _disabled() -> None:
    return None


async def _probe(check: Callable[[], Awaitable[bool]]) -> bool:
    try:
        return bool(await asyncio.wait_for(check(), timeout=_PROBE_TIMEOUT_SECONDS))
    except Exception:
        return False
