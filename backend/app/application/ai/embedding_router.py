"""Embedding router.

Unlike text generation there is no cross-model fallback here: vectors from different
models live in different spaces and must never be mixed in one index. The router always
embeds into the single *active* embedding space from configuration.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence

from app.domain.ai.cost import estimate_api_cost
from app.domain.ai.embeddings import EmbeddingProvider, EmbeddingResult, EmbeddingSpace
from app.domain.ai.exceptions import (
    EmbeddingError,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)
from app.domain.ai.routing import CircuitBreaker, ModelDescriptor, ModelRegistry, ModelType
from app.domain.ai.telemetry import AIRequestContext, AIRequestRecord, AIUsageRecorder

logger = logging.getLogger(__name__)


class EmbeddingRouter:
    def __init__(
        self,
        *,
        providers: Mapping[str, EmbeddingProvider],
        space: EmbeddingSpace,
        registry: ModelRegistry,
        usage_recorder: AIUsageRecorder,
        circuit_breaker: CircuitBreaker,
        max_retries: int = 1,
    ) -> None:
        self._providers = dict(providers)
        self._max_retries = max_retries
        self._space = space
        self._registry = registry
        self._usage = usage_recorder
        self._breaker = circuit_breaker

    @property
    def space(self) -> EmbeddingSpace:
        return self._space

    @property
    def configured(self) -> bool:
        return bool(self._space.model) and self._space.provider in self._providers

    def _circuit_key(self) -> str:
        return f"{self._space.provider}:{self._space.model}"

    async def health_check(self) -> bool:
        provider = self._providers.get(self._space.provider)
        return bool(provider and self._space.model and await provider.health_check())

    async def embed(
        self,
        texts: Sequence[str],
        *,
        task: str = "embedding",
        context: AIRequestContext | None = None,
    ) -> EmbeddingResult:
        if not texts:
            raise EmbeddingError("nothing to embed")
        provider = self._providers.get(self._space.provider)
        if provider is None or not self._space.model:
            raise EmbeddingError("embedding provider is not configured")
        key = self._circuit_key()
        if not self._breaker.allow(key):
            raise EmbeddingError("embedding provider temporarily disabled")

        descriptor = await self._registry.get(
            self._space.provider, self._space.model, ModelType.EMBEDDING
        )
        context = context or AIRequestContext()
        options = dict(descriptor.provider_options) if descriptor else None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                result = await provider.embed(
                    model=self._space.model, texts=list(texts), options=options
                )
                break
            except (ProviderUnavailable, ProviderTimeout) as exc:
                # Transient (e.g. runner restarted under memory pressure): embedding calls
                # are idempotent and cheap, so retry once before giving up.
                self._breaker.record_failure(key)
                await self._record(task, descriptor, context, started, None, exc.code, attempt)
                if attempt >= self._max_retries:
                    raise EmbeddingError("embedding request failed") from exc
                logger.warning("embedding call failed (%s), retrying", exc.code)
            except ProviderError as exc:
                self._breaker.record_failure(key)
                await self._record(task, descriptor, context, started, None, exc.code, attempt)
                raise EmbeddingError("embedding request failed") from exc
            except Exception as exc:
                logger.exception("unexpected embedding provider error")
                self._breaker.record_failure(key)
                await self._record(
                    task, descriptor, context, started, None, "provider_error", attempt
                )
                raise EmbeddingError("embedding request failed") from exc

        if len(result.vectors) != len(texts):
            await self._record(
                task, descriptor, context, started, result, "embedding_mismatch", attempt
            )
            raise EmbeddingError("embedding count does not match input count")
        self._breaker.record_success(key)
        await self._record(task, descriptor, context, started, result, None, attempt)
        return result

    async def _record(
        self,
        task: str,
        descriptor: ModelDescriptor | None,
        context: AIRequestContext,
        started: float,
        result: EmbeddingResult | None,
        error_type: str | None,
        retry_count: int = 0,
    ) -> None:
        latency = result.latency_ms if result else int((time.perf_counter() - started) * 1000)
        tokens = result.input_tokens if result else None
        cost = estimate_api_cost(tokens, 0, descriptor.pricing) if descriptor else None
        record = AIRequestRecord(
            task=task,
            provider=self._space.provider,
            model=self._space.model,
            is_local=bool(descriptor and descriptor.is_local),
            success=error_type is None,
            latency_ms=latency,
            run_id=context.run_id,
            input_tokens=tokens,
            api_estimated_cost=cost,
            currency=descriptor.pricing.currency if descriptor else None,
            error_type=error_type,
            retry_count=retry_count,
        )
        try:
            await self._usage.record_request(record)
        except Exception:
            logger.exception("failed to record embedding telemetry")
