"""Use case: generate a batch of new cards for a language/category/difficulty."""

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
class GenerateCardsCommand:
    language: str
    difficulty: Difficulty | None  # None: generate without a difficulty level
    count: int
    category: str | None = None
    topic: str | None = None


class GenerateCardsUseCase:
    TASK = "card_generation"

    def __init__(
        self,
        *,
        engine: CardGenerationEngine,
        repository: CardRepository,
        usage_recorder: AIUsageRecorder,
        supported_languages: Collection[str],
        max_cards_per_request: int,
        allow_partial_results: bool = True,
    ) -> None:
        self._engine = engine
        self._repository = repository
        self._usage = usage_recorder
        self._languages = frozenset(supported_languages)
        self._max_cards = max_cards_per_request
        self._allow_partial = allow_partial_results

    async def execute(self, command: GenerateCardsCommand) -> list[Card]:
        if command.language not in self._languages:
            raise InvalidCardRequest(f"language '{command.language}' is not supported")
        if not 1 <= command.count <= self._max_cards:
            raise InvalidCardRequest(f"count must be between 1 and {self._max_cards}")

        spec = CardSpec(
            language=command.language,
            difficulty=command.difficulty,
            category=self._engine.validator.normalize_category(command.language, command.category),
            topic=command.topic.strip() if command.topic and command.topic.strip() else None,
        )
        run = GenerationRunRecord(
            task=self.TASK,
            requested_items=command.count,
            attributes={
                "language": spec.language,
                "category": spec.category,
                "difficulty": spec.difficulty.value if spec.difficulty else None,
                "has_topic": spec.topic is not None,
            },
        )
        async with tracked_run(self._usage, run):
            accepted = await self._engine.produce(spec=spec, count=command.count, run=run)
            if not accepted or (len(accepted) < command.count and not self._allow_partial):
                raise ValidationExhausted(
                    f"only {len(accepted)} of {command.count} cards passed validation"
                )
            cards = await persist(
                self._repository, build_cards(accepted, self._engine.validator, run)
            )
            run.returned_items = len(cards)
            run.success = True
            return cards
