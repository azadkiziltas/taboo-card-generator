"""Persists AI request / generation run telemetry and answers aggregate queries."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Integer, case, cast, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.ai.telemetry import AIRequestRecord, GenerationRunRecord
from app.infrastructure.db.models import AIRequestRow, GenerationRunRow
from app.infrastructure.db.session import translate_db_errors


class SqlAIUsageRecorder:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    @translate_db_errors
    async def record_request(self, record: AIRequestRecord) -> None:
        async with self._sessions() as session, session.begin():
            session.add(
                AIRequestRow(
                    id=record.id,
                    run_id=record.run_id,
                    task=record.task,
                    provider=record.provider,
                    model=record.model,
                    is_local=record.is_local,
                    prompt_name=record.prompt_name,
                    prompt_version=record.prompt_version,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    api_estimated_cost=record.api_estimated_cost,
                    currency=record.currency,
                    latency_ms=record.latency_ms,
                    success=record.success,
                    error_type=record.error_type,
                    retry_count=record.retry_count,
                    provider_attempt=record.provider_attempt,
                    fallback_used=record.fallback_used,
                    parse_failed=record.parse_failed,
                    validation_failures=record.validation_failures,
                    items_accepted=record.items_accepted,
                    created_at=record.created_at,
                )
            )

    @translate_db_errors
    async def annotate_request(
        self,
        request_id: uuid.UUID,
        *,
        validation_failures: int | None = None,
        items_accepted: int | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if validation_failures is not None:
            values["validation_failures"] = (
                func.coalesce(AIRequestRow.validation_failures, 0) + validation_failures
            )
        if items_accepted is not None:
            values["items_accepted"] = (
                func.coalesce(AIRequestRow.items_accepted, 0) + items_accepted
            )
        if not values:
            return
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(AIRequestRow).where(AIRequestRow.id == request_id).values(**values)
            )

    @translate_db_errors
    async def record_run(self, run: GenerationRunRecord) -> None:
        async with self._sessions() as session, session.begin():
            session.add(
                GenerationRunRow(
                    id=run.id,
                    task=run.task,
                    requested_items=run.requested_items,
                    returned_items=run.returned_items,
                    drafts_generated=run.drafts_generated,
                    drafts_accepted=run.drafts_accepted,
                    validation_failures=run.validation_failures,
                    duplicates_rejected=run.duplicates_rejected,
                    repair_attempts=run.repair_attempts,
                    generation_rounds=run.generation_rounds,
                    rag_enabled=run.rag_enabled,
                    context_documents=run.context_documents,
                    prompt_name=run.prompt_name,
                    prompt_version=run.prompt_version,
                    success=run.success,
                    error_type=run.error_type,
                    duration_ms=run.duration_ms,
                    attributes=run.attributes,
                    created_at=run.created_at,
                )
            )


class SqlAIUsageQuery:
    """Aggregations for the admin telemetry endpoint."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    @translate_db_errors
    async def request_stats(self, since: datetime) -> list[dict[str, Any]]:
        r = AIRequestRow
        as_int = lambda cond: func.sum(cast(case((cond, 1), else_=0), Integer))  # noqa: E731
        query = (
            select(
                r.task,
                r.provider,
                r.model,
                r.is_local,
                r.prompt_version,
                func.count().label("requests"),
                as_int(r.success).label("successes"),
                as_int(r.parse_failed).label("parse_failures"),
                as_int(r.fallback_used).label("fallbacks"),
                func.sum(r.retry_count).label("retries"),
                func.avg(r.latency_ms).label("avg_latency_ms"),
                func.sum(r.input_tokens).label("input_tokens"),
                func.sum(r.output_tokens).label("output_tokens"),
                func.sum(r.api_estimated_cost).label("api_estimated_cost"),
                func.sum(r.validation_failures).label("validation_failures"),
                func.sum(r.items_accepted).label("items_accepted"),
            )
            .where(r.created_at >= since)
            .group_by(r.task, r.provider, r.model, r.is_local, r.prompt_version)
            .order_by(r.task, r.provider, r.model)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).mappings().all()
        return [_jsonable(dict(row)) for row in rows]

    @translate_db_errors
    async def run_stats(self, since: datetime) -> list[dict[str, Any]]:
        g = GenerationRunRow
        query = (
            select(
                g.task,
                g.prompt_version,
                func.count().label("runs"),
                func.sum(cast(g.success, Integer)).label("successful_runs"),
                func.sum(g.requested_items).label("requested_items"),
                func.sum(g.returned_items).label("returned_items"),
                func.sum(g.drafts_generated).label("drafts_generated"),
                func.sum(g.drafts_accepted).label("drafts_accepted"),
                func.sum(g.validation_failures).label("validation_failures"),
                func.sum(g.duplicates_rejected).label("duplicates_rejected"),
                func.sum(g.repair_attempts).label("repair_attempts"),
                func.avg(g.duration_ms).label("avg_duration_ms"),
            )
            .where(g.created_at >= since)
            .group_by(g.task, g.prompt_version)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).mappings().all()
        result = []
        for row in rows:
            item = _jsonable(dict(row))
            drafts = item.get("drafts_generated") or 0
            item["acceptance_rate"] = (item["drafts_accepted"] or 0) / drafts if drafts else None
            result.append(item)
        return result


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or isinstance(value, bool | int | str):
            out[key] = value
        else:
            out[key] = float(value)
    return out
