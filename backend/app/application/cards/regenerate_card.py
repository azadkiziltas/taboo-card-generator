"""Use case: replace an existing card with a fresh set of forbidden words (same target)."""

from __future__ import annotations

import uuid

from app.application.cards.engine import CardGenerationEngine, CardSpec
from app.application.cards.support import build_cards, persist, tracked_run
from app.domain.ai.exceptions import ValidationExhausted
from app.domain.ai.telemetry import AIUsageRecorder, GenerationRunRecord
from app.domain.cards.entities import Card, CardStatus
from app.domain.cards.exceptions import CardNotFound
from app.domain.cards.repositories import CardRepository


class RegenerateCardUseCase:
    TASK = "card_regeneration"

    def __init__(
        self,
        *,
        engine: CardGenerationEngine,
        repository: CardRepository,
        usage_recorder: AIUsageRecorder,
    ) -> None:
        self._engine = engine
        self._repository = repository
        self._usage = usage_recorder

    async def execute(self, card_id: uuid.UUID) -> Card:
        existing = await self._repository.get(card_id)
        if existing is None or existing.status in (CardStatus.REPLACED, CardStatus.RETIRED):
            raise CardNotFound(str(card_id))

        spec = CardSpec(
            language=existing.language,
            difficulty=existing.difficulty,
            category=existing.category,
        )
        run = GenerationRunRecord(
            task=self.TASK,
            requested_items=1,
            attributes={"language": spec.language, "replaces": str(existing.id)},
        )
        async with tracked_run(self._usage, run):
            accepted = await self._engine.produce(
                spec=spec,
                count=1,
                run=run,
                targets=[existing.target],
                avoid_forbidden=existing.forbidden_words,
                ignore_card_ids=frozenset({existing.id}),
            )
            if not accepted:
                raise ValidationExhausted(f"could not regenerate card {card_id}")
            [new_card] = await persist(
                self._repository, build_cards(accepted, self._engine.validator, run)
            )
            await self._repository.mark_replaced(existing.id, new_card.id)
            run.returned_items = 1
            run.success = True
            return new_card
