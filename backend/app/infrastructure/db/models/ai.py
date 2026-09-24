from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base, CreatedAtMixin, TimestampMixin


class AIModelRow(TimestampMixin, Base):
    """Model registry."""

    __tablename__ = "ai_models"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    provider: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(200))
    model_type: Mapped[str] = mapped_column(String(16))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    is_local: Mapped[bool] = mapped_column(Boolean, default=False)
    priority: Mapped[int] = mapped_column(Integer, default=100)
    input_cost_per_million: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    output_cost_per_million: Mapped[Decimal | None] = mapped_column(Numeric(14, 6))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    supports_structured_output: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_json_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_tools: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_vision: Mapped[bool] = mapped_column(Boolean, default=False)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, default=False)
    context_window: Mapped[int | None] = mapped_column(Integer)
    languages: Mapped[list[str] | None] = mapped_column(ARRAY(String(8)))
    quality_score: Mapped[float | None] = mapped_column(Float)
    avg_latency_ms: Mapped[int | None] = mapped_column(Integer)
    health_status: Mapped[str] = mapped_column(String(16), default="unknown")
    provider_options: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    __table_args__ = (UniqueConstraint("provider", "model_name", "model_type"),)


class GenerationRunRow(CreatedAtMixin, Base):
    __tablename__ = "generation_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    task: Mapped[str] = mapped_column(String(64))
    requested_items: Mapped[int] = mapped_column(Integer)
    returned_items: Mapped[int] = mapped_column(Integer)
    drafts_generated: Mapped[int] = mapped_column(Integer)
    drafts_accepted: Mapped[int] = mapped_column(Integer)
    validation_failures: Mapped[int] = mapped_column(Integer)
    duplicates_rejected: Mapped[int] = mapped_column(Integer)
    repair_attempts: Mapped[int] = mapped_column(Integer)
    generation_rounds: Mapped[int] = mapped_column(Integer)
    rag_enabled: Mapped[bool] = mapped_column(Boolean)
    context_documents: Mapped[int] = mapped_column(Integer)
    prompt_name: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    success: Mapped[bool] = mapped_column(Boolean)
    error_type: Mapped[str | None] = mapped_column(String(64))
    duration_ms: Mapped[int] = mapped_column(Integer)
    attributes: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    __table_args__ = (Index("ix_generation_runs_task_created_at", "task", "created_at"),)


class AIRequestRow(CreatedAtMixin, Base):
    """AI request telemetry. Never stores prompts, answers or credentials."""

    __tablename__ = "ai_requests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, index=True)
    task: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(200))
    is_local: Mapped[bool] = mapped_column(Boolean)
    prompt_name: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    api_estimated_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 8))
    currency: Mapped[str | None] = mapped_column(String(3))
    latency_ms: Mapped[int] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean)
    error_type: Mapped[str | None] = mapped_column(String(64))
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    provider_attempt: Mapped[int] = mapped_column(Integer, default=1)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    parse_failed: Mapped[bool] = mapped_column(Boolean, default=False)
    validation_failures: Mapped[int | None] = mapped_column(Integer)
    items_accepted: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_ai_requests_created_at", "created_at"),
        Index("ix_ai_requests_provider_model", "provider", "model"),
        Index("ix_ai_requests_task", "task"),
    )
