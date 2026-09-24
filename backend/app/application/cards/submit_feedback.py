"""Use case: store player feedback for a card (future benchmark / fine-tuning data)."""

from __future__ import annotations

import uuid

from app.domain.cards.entities import CardFeedback, DifficultyFeedback, FeedbackRating
from app.domain.cards.exceptions import CardNotFound, InvalidCardRequest
from app.domain.cards.repositories import CardRepository, FeedbackRepository


class SubmitFeedbackUseCase:
    def __init__(self, *, cards: CardRepository, feedback: FeedbackRepository) -> None:
        self._cards = cards
        self._feedback = feedback

    async def execute(
        self,
        card_id: uuid.UUID,
        *,
        rating: FeedbackRating | None,
        difficulty_feedback: DifficultyFeedback | None,
    ) -> CardFeedback:
        if rating is None and difficulty_feedback is None:
            raise InvalidCardRequest("rating or difficulty_feedback is required")
        if await self._cards.get(card_id) is None:
            raise CardNotFound(str(card_id))
        feedback = CardFeedback(
            card_id=card_id, rating=rating, difficulty_feedback=difficulty_feedback
        )
        await self._feedback.add(feedback)
        return feedback
