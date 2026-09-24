"""Helpers shared by the card use cases: persistence and run telemetry."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from app.application.cards.engine import AcceptedCard
from app.domain.ai.telemetry import AIUsageRecorder, GenerationRunRecord
from app.domain.cards.entities import Card, CardProvenance, CardSource, CardStatus
from app.domain.cards.repositories import CardRepository, NewCardRecord
from app.domain.cards.validator import CardValidator

logger = logging.getLogger(__name__)


def build_cards(
    accepted: Sequence[AcceptedCard],
    validator: CardValidator,
    run: GenerationRunRecord,
) -> list[NewCardRecord]:
    records: list[NewCardRecord] = []
    for item in accepted:
        status = (
            CardStatus.DUPLICATE_CANDIDATE if item.is_duplicate_candidate else CardStatus.ACTIVE
        )
        card = Card.from_draft(item.draft, source=CardSource.AI, status=status)
        records.append(
            NewCardRecord(
                card=card,
                target_key=validator.target_key(item.draft),
                forbidden_keys=validator.forbidden_keys(item.draft),
                provenance=CardProvenance(
                    ai_request_id=item.ai_request_id, generation_run_id=run.id
                ),
            )
        )
    return records


async def persist(repository: CardRepository, records: Sequence[NewCardRecord]) -> list[Card]:
    if records:
        await repository.add_many(records)
    return [record.card for record in records]


@asynccontextmanager
async def tracked_run(
    usage: AIUsageRecorder, run: GenerationRunRecord
) -> AsyncIterator[GenerationRunRecord]:
    """Measure the run and always persist its record, whether it succeeds or fails."""
    started = time.perf_counter()
    try:
        yield run
    except Exception as exc:
        run.success = False
        run.error_type = getattr(exc, "code", type(exc).__name__)
        raise
    finally:
        run.duration_ms = int((time.perf_counter() - started) * 1000)
        try:
            await usage.record_run(run)
        except Exception:
            logger.exception("failed to record generation run")
        logger.info(
            "generation run finished",
            extra={
                "task": run.task,
                "run_id": str(run.id),
                "requested": run.requested_items,
                "returned": run.returned_items,
                "drafts": run.drafts_generated,
                "acceptance_rate": run.acceptance_rate,
                "rounds": run.generation_rounds,
                "repairs": run.repair_attempts,
                "success": run.success,
                "error_type": run.error_type,
                "duration_ms": run.duration_ms,
            },
        )
