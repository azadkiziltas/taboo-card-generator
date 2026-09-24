"""Use case: generate the forbidden words for one given target word."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from app.application.cards.engine import CardGenerationEngine, CardSpec
from app.application.cards.support import build_cards, persist, tracked_run
from app.domain.ai.exceptions import ValidationExhausted
from app.domain.ai.telemetry import AIUsageRecorder, GenerationRunRecord
from app.domain.cards.entities import Card, Difficulty
from app.domain.cards.exceptions import InvalidCardRequest
from app.domain.cards.repositories import CardRepository


@dataclass(frozen=True, slots=True)
class GenerateCardCommand:
    target: str
    language: str
    difficulty: Difficulty | None = None
    category: str | None = None
    persist: bool = True


class GenerateCardUseCase:
    TASK = "card_generation_for_target"

    def __init__(
        self,
        *,
        engine: CardGenerationEngine,
        repository: CardRepository,
        usage_recorder: AIUsageRecorder,
        supported_languages: Collection[str],
    ) -> None:
        self._engine = engine
        self._repository = repository
        self._usage = usage_recorder
        self._languages = frozenset(supported_languages)

    async def execute(self, command: GenerateCardCommand) -> Card:
        if command.language not in self._languages:
            raise InvalidCardRequest(f"language '{command.language}' is not supported")
        target = self._engine.validator.normalizer(command.language).clean(command.target)
        if not target:
            raise InvalidCardRequest("target must not be empty")

        spec = CardSpec(
            language=command.language,
            difficulty=command.difficulty,
            category=self._engine.validator.normalize_category(command.language, command.category),
        )
        run = GenerationRunRecord(
            task=self.TASK,
            requested_items=1,
            attributes={
                "language": spec.language,
                "difficulty": spec.difficulty.value if spec.difficulty else None,
            },
        )
        async with tracked_run(self._usage, run):
            accepted = await self._engine.produce(spec=spec, count=1, run=run, targets=[target])
            if not accepted:
                raise ValidationExhausted(f"could not generate a valid card for '{target}'")
            records = build_cards(accepted, self._engine.validator, run)
            cards = (
                await persist(self._repository, records)
                if command.persist
                else [r.card for r in records]
            )
            run.returned_items = len(cards)
            run.success = True
            return cards[0]
