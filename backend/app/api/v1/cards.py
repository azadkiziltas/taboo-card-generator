from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status

from app.api.dependencies import ContainerDep, rate_limit
from app.api.v1.schemas import (
    CardResponse,
    FeedbackRequest,
    FeedbackResponse,
    GenerateCardsRequest,
    GenerateCardsResponse,
)
from app.application.cards.generate_cards import GenerateCardsCommand

router = APIRouter(prefix="/cards", tags=["cards"])

_ERRORS = {
    422: {"description": "Invalid request"},
    429: {"description": "Rate limited"},
    502: {"description": "The model could not produce valid cards"},
    503: {"description": "Generation temporarily unavailable"},
    504: {"description": "Generation timed out"},
}


@router.post(
    "/generate",
    response_model=GenerateCardsResponse,
    dependencies=[Depends(rate_limit)],
    responses=_ERRORS,
    summary="Generate new validated cards",
)
async def generate_cards(
    body: GenerateCardsRequest, container: ContainerDep
) -> GenerateCardsResponse:
    cards = await container.generate_cards.execute(
        GenerateCardsCommand(
            language=body.language,
            category=body.category,
            difficulty=body.difficulty,
            count=body.count,
            topic=body.topic,
        )
    )
    return GenerateCardsResponse(cards=[CardResponse.from_card(card) for card in cards])


@router.post(
    "/{card_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit)],
    responses={404: {"description": "Card not found"}, 422: {"description": "Invalid request"}},
    summary="Record player feedback for a card",
)
async def submit_feedback(
    card_id: uuid.UUID, body: FeedbackRequest, container: ContainerDep
) -> FeedbackResponse:
    feedback = await container.submit_feedback.execute(
        card_id, rating=body.rating, difficulty_feedback=body.difficulty_feedback
    )
    return FeedbackResponse(id=feedback.id, card_id=feedback.card_id)


@router.post(
    "/{card_id}/regenerate",
    response_model=CardResponse,
    dependencies=[Depends(rate_limit)],
    responses={404: {"description": "Card not found"}, **_ERRORS},
    summary="Replace a card with a new set of forbidden words (same target)",
)
async def regenerate_card(card_id: uuid.UUID, container: ContainerDep) -> CardResponse:
    return CardResponse.from_card(await container.regenerate_card.execute(card_id))
