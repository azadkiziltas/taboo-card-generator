"""Public request/response DTOs. Deliberately contain no provider/model fields."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.cards.entities import Card, Difficulty, DifficultyFeedback, FeedbackRating


class GenerateCardsRequest(BaseModel):
    # Unknown fields (e.g. "model", "provider") are rejected: clients never pick a model.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "language": "tr",
                    "category": "technology",
                    "difficulty": "medium",
                    "count": 10,
                    "topic": None,
                }
            ]
        },
    )

    language: str = Field(pattern=r"^[a-z]{2}$", description="ISO-639-1 code, e.g. tr, en, de")
    category: str | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        pattern=r"^[\w \-'’.,&]+$",
        description="Category or free-text theme, e.g. 'technology' or 'uzay yolculuğu'",
    )
    difficulty: Difficulty | None = Field(
        default=None,
        description="Optional. When omitted the cards are generated without any difficulty level "
        "(nothing about difficulty is sent to the model) and `difficulty` is null.",
    )
    count: int = Field(default=10, ge=1, le=100)
    topic: str | None = Field(default=None, max_length=100)

    @field_validator("topic")
    @classmethod
    def _clean_topic(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join("".join(ch for ch in value if ch.isprintable()).split())
        return cleaned or None


class CardResponse(BaseModel):
    id: uuid.UUID
    target: str
    forbidden_words: list[str]
    language: str
    category: str | None
    difficulty: Difficulty | None

    @classmethod
    def from_card(cls, card: Card) -> CardResponse:
        return cls(
            id=card.id,
            target=card.target,
            forbidden_words=list(card.forbidden_words),
            language=card.language,
            category=card.category,
            difficulty=card.difficulty,
        )


class GenerateCardsResponse(BaseModel):
    cards: list[CardResponse]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"rating": "good", "difficulty_feedback": "too_easy"}]},
    )

    rating: FeedbackRating | None = None
    difficulty_feedback: DifficultyFeedback | None = None

    @model_validator(mode="after")
    def _needs_value(self) -> FeedbackRequest:
        if self.rating is None and self.difficulty_feedback is None:
            raise ValueError("rating or difficulty_feedback is required")
        return self


class FeedbackResponse(BaseModel):
    id: uuid.UUID
    card_id: uuid.UUID
    status: str = "recorded"


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]
