from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.ai.routing import ModelDescriptor
from app.domain.cards.entities import Difficulty


class ModelOut(BaseModel):
    id: uuid.UUID
    provider: str
    model_name: str
    model_type: str
    enabled: bool
    is_local: bool
    priority: int
    input_cost_per_million: Decimal | None
    output_cost_per_million: Decimal | None
    currency: str
    supports_structured_output: bool
    supports_json_mode: bool
    supports_tools: bool
    supports_vision: bool
    supports_streaming: bool
    context_window: int | None
    languages: list[str] | None
    quality_score: float | None
    avg_latency_ms: int | None
    health_status: str
    provider_options: dict[str, Any]

    @classmethod
    def from_descriptor(cls, m: ModelDescriptor) -> ModelOut:
        return cls(
            id=m.id,
            provider=m.provider,
            model_name=m.model_name,
            model_type=m.model_type.value,
            enabled=m.enabled,
            is_local=m.is_local,
            priority=m.priority,
            input_cost_per_million=m.pricing.input_cost_per_million,
            output_cost_per_million=m.pricing.output_cost_per_million,
            currency=m.pricing.currency,
            supports_structured_output=m.capabilities.structured_output,
            supports_json_mode=m.capabilities.json_mode,
            supports_tools=m.capabilities.tool_calling,
            supports_vision=m.capabilities.vision,
            supports_streaming=m.capabilities.streaming,
            context_window=m.capabilities.max_context_tokens,
            languages=list(m.languages) if m.languages else None,
            quality_score=m.quality_score,
            avg_latency_ms=m.avg_latency_ms,
            health_status=m.health_status.value,
            provider_options=dict(m.provider_options),
        )


class ModelPatch(BaseModel):
    """Only fields that are set are changed."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    priority: int | None = Field(default=None, ge=0, le=10_000)
    is_local: bool | None = None
    quality_score: float | None = Field(default=None, ge=0, le=100)
    input_cost_per_million: Decimal | None = Field(default=None, ge=0)
    output_cost_per_million: Decimal | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    languages: list[str] | None = None
    provider_options: dict[str, Any] | None = None
    supports_structured_output: bool | None = None
    supports_json_mode: bool | None = None
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_streaming: bool | None = None
    context_window: int | None = Field(default=None, ge=1)


class RagDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str = Field(pattern=r"^[a-z]{2}$")
    word: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=4000)
    category: str | None = Field(default=None, max_length=64)
    related_terms: list[str] = Field(default_factory=list, max_length=50)
    source: str = Field(min_length=1, max_length=100)
    source_license: str | None = Field(default=None, max_length=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagIngestRequest(BaseModel):
    documents: list[RagDocumentIn] = Field(min_length=1, max_length=500)


class RagSearchRequest(BaseModel):
    language: str = Field(pattern=r"^[a-z]{2}$")
    query: str = Field(min_length=1, max_length=200)
    top_k: int | None = Field(default=None, ge=1, le=50)


class GenerateForTargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1, max_length=60)
    language: str = Field(pattern=r"^[a-z]{2}$")
    difficulty: Difficulty | None = None
    category: str | None = Field(default=None, max_length=50)
    persist: bool = True
