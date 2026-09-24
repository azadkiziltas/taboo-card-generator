"""Persistence ports for cards (implemented in infrastructure)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.domain.cards.entities import Card, CardFeedback, CardProvenance, CardStatus


@dataclass(frozen=True, slots=True)
class NewCardRecord:
    """A validated card plus the precomputed comparison keys used for duplicate lookups."""

    card: Card
    target_key: str
    forbidden_keys: tuple[str, ...]
    provenance: CardProvenance = CardProvenance()


@dataclass(frozen=True, slots=True)
class ExistingCardSummary:
    id: uuid.UUID
    language: str
    target_key: str
    forbidden_keys: frozenset[str]
    status: CardStatus


class CardRepository(Protocol):
    async def add_many(self, records: Sequence[NewCardRecord]) -> None: ...

    async def get(self, card_id: uuid.UUID) -> Card | None: ...

    async def find_existing(
        self, language: str, target_keys: Sequence[str]
    ) -> list[ExistingCardSummary]: ...

    async def recent_targets(
        self, language: str, category: str | None, limit: int
    ) -> list[str]: ...

    async def mark_replaced(self, card_id: uuid.UUID, replaced_by: uuid.UUID) -> None: ...


class FeedbackRepository(Protocol):
    async def add(self, feedback: CardFeedback) -> None: ...
