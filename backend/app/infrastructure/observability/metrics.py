"""In-process AI metrics + a recorder decorator that logs every AI request.

Logs contain only metadata (task, provider, model, tokens, latency, outcome); prompts,
answers and API keys are never logged.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.domain.ai.telemetry import AIRequestRecord, AIUsageRecorder, GenerationRunRecord

logger = logging.getLogger("app.ai.telemetry")


@dataclass(slots=True)
class _Counter:
    requests: int = 0
    successes: int = 0
    parse_failures: int = 0
    fallbacks: int = 0
    retries: int = 0
    latency_ms_total: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    api_estimated_cost: Decimal = Decimal(0)


class InMemoryMetrics:
    def __init__(self) -> None:
        self._requests: dict[tuple[str, str, str], _Counter] = defaultdict(_Counter)
        self._errors: dict[str, int] = defaultdict(int)
        self._runs = 0
        self._drafts_generated = 0
        self._drafts_accepted = 0

    def observe_request(self, record: AIRequestRecord) -> None:
        counter = self._requests[(record.task, record.provider, record.model)]
        counter.requests += 1
        counter.successes += int(record.success)
        counter.parse_failures += int(record.parse_failed)
        counter.fallbacks += int(record.fallback_used)
        counter.retries += record.retry_count
        counter.latency_ms_total += record.latency_ms
        counter.input_tokens += record.input_tokens or 0
        counter.output_tokens += record.output_tokens or 0
        counter.api_estimated_cost += record.api_estimated_cost or Decimal(0)
        if record.error_type:
            self._errors[record.error_type] += 1

    def observe_run(self, run: GenerationRunRecord) -> None:
        self._runs += 1
        self._drafts_generated += run.drafts_generated
        self._drafts_accepted += run.drafts_accepted

    def snapshot(self) -> dict[str, Any]:
        return {
            "requests": [
                {
                    "task": task,
                    "provider": provider,
                    "model": model,
                    "requests": c.requests,
                    "success_rate": c.successes / c.requests if c.requests else None,
                    "parse_failures": c.parse_failures,
                    "fallbacks": c.fallbacks,
                    "retries": c.retries,
                    "avg_latency_ms": c.latency_ms_total / c.requests if c.requests else None,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "api_estimated_cost": float(c.api_estimated_cost),
                }
                for (task, provider, model), c in sorted(self._requests.items())
            ],
            "errors": dict(self._errors),
            "runs": self._runs,
            "card_acceptance_rate": (
                self._drafts_accepted / self._drafts_generated if self._drafts_generated else None
            ),
        }


class ObservedUsageRecorder:
    """Decorates a recorder with structured logging and in-process metrics."""

    def __init__(self, inner: AIUsageRecorder, metrics: InMemoryMetrics) -> None:
        self._inner = inner
        self.metrics = metrics

    async def record_request(self, record: AIRequestRecord) -> None:
        self.metrics.observe_request(record)
        logger.info(
            "ai_request",
            extra={
                "task": record.task,
                "provider": record.provider,
                "model": record.model,
                "is_local": record.is_local,
                "success": record.success,
                "error_type": record.error_type,
                "latency_ms": record.latency_ms,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "api_estimated_cost": (
                    str(record.api_estimated_cost)
                    if record.api_estimated_cost is not None
                    else None
                ),
                "retry_count": record.retry_count,
                "provider_attempt": record.provider_attempt,
                "fallback_used": record.fallback_used,
                "parse_failed": record.parse_failed,
                "prompt": f"{record.prompt_name}@{record.prompt_version}"
                if record.prompt_name
                else None,
            },
        )
        await self._inner.record_request(record)

    async def annotate_request(
        self,
        request_id: uuid.UUID,
        *,
        validation_failures: int | None = None,
        items_accepted: int | None = None,
    ) -> None:
        await self._inner.annotate_request(
            request_id, validation_failures=validation_failures, items_accepted=items_accepted
        )

    async def record_run(self, run: GenerationRunRecord) -> None:
        self.metrics.observe_run(run)
        await self._inner.record_run(run)
