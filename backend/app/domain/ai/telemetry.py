"""Telemetry records and the recorder ports used by the application layer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class AIRequestContext:
    """Per-call metadata the caller wants attached to telemetry."""

    run_id: uuid.UUID | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None


@dataclass(slots=True)
class AIRequestRecord:
    """One call to one provider/model (every parse retry is its own record)."""

    task: str
    provider: str
    model: str
    is_local: bool
    success: bool
    latency_ms: int
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    run_id: uuid.UUID | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    api_estimated_cost: Decimal | None = None
    currency: str | None = None
    error_type: str | None = None
    retry_count: int = 0
    provider_attempt: int = 1
    fallback_used: bool = False
    parse_failed: bool = False
    validation_failures: int | None = None
    items_accepted: int | None = None
    created_at: datetime = field(default_factory=_now)


@dataclass(slots=True)
class GenerationRunRecord:
    """One end-to-end use-case execution (e.g. one /cards/generate request)."""

    task: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    requested_items: int = 0
    returned_items: int = 0
    drafts_generated: int = 0
    drafts_accepted: int = 0
    validation_failures: int = 0
    duplicates_rejected: int = 0
    repair_attempts: int = 0
    generation_rounds: int = 0
    rag_enabled: bool = False
    context_documents: int = 0
    prompt_name: str | None = None
    prompt_version: str | None = None
    success: bool = False
    error_type: str | None = None
    duration_ms: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)

    @property
    def acceptance_rate(self) -> float | None:
        if not self.drafts_generated:
            return None
        return self.drafts_accepted / self.drafts_generated


class AIUsageRecorder(Protocol):
    async def record_request(self, record: AIRequestRecord) -> None: ...

    async def annotate_request(
        self,
        request_id: uuid.UUID,
        *,
        validation_failures: int | None = None,
        items_accepted: int | None = None,
    ) -> None: ...

    async def record_run(self, run: GenerationRunRecord) -> None: ...
