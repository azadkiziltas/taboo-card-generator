"""Card domain entities. Deliberately free of any provider/model information."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from app.domain.cards.languages import is_iso_639_1

FORBIDDEN_WORD_COUNT = 5


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CardSource(StrEnum):
    AI = "ai"
    MANUAL = "manual"
    IMPORT = "import"


class CardStatus(StrEnum):
    ACTIVE = "active"
    DUPLICATE_CANDIDATE = "duplicate_candidate"  # same target exists with other words
    REPLACED = "replaced"  # superseded by a regenerated card
    RETIRED = "retired"


class FeedbackRating(StrEnum):
    GOOD = "good"
    BAD = "bad"


class DifficultyFeedback(StrEnum):
    TOO_EASY = "too_easy"
    CORRECT = "correct"
    TOO_HARD = "too_hard"


class CardInvariantError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CardDraft:
    """A card candidate that has not passed validation yet (e.g. straight from an LLM)."""

    target: str
    forbidden_words: tuple[str, ...]
    language: str
    difficulty: Difficulty | None  # None: generated without a difficulty level
    category: str | None = None


@dataclass(frozen=True, slots=True)
class Card:
    target: str
    forbidden_words: tuple[str, ...]
    language: str
    difficulty: Difficulty | None  # None: the request asked for no difficulty level
    source: CardSource
    category: str | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    status: CardStatus = CardStatus.ACTIVE
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise CardInvariantError("target must not be empty")
        if len(self.forbidden_words) != FORBIDDEN_WORD_COUNT:
            raise CardInvariantError(f"exactly {FORBIDDEN_WORD_COUNT} forbidden words required")
        if any(not word.strip() for word in self.forbidden_words):
            raise CardInvariantError("forbidden words must not be empty")
        if not is_iso_639_1(self.language):
            raise CardInvariantError("language must be an ISO-639-1 code")

    @classmethod
    def from_draft(
        cls,
        draft: CardDraft,
        *,
        source: CardSource,
        status: CardStatus = CardStatus.ACTIVE,
    ) -> Card:
        return cls(
            target=draft.target,
            forbidden_words=draft.forbidden_words,
            language=draft.language,
            difficulty=draft.difficulty,
            category=draft.category,
            source=source,
            status=status,
        )


@dataclass(frozen=True, slots=True)
class CardFeedback:
    card_id: uuid.UUID
    rating: FeedbackRating | None = None
    difficulty_feedback: DifficultyFeedback | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.rating is None and self.difficulty_feedback is None:
            raise CardInvariantError("feedback needs a rating or difficulty_feedback")


@dataclass(frozen=True, slots=True)
class CardProvenance:
    """Where a persisted card came from (kept out of the Card entity on purpose)."""

    ai_request_id: uuid.UUID | None = None
    generation_run_id: uuid.UUID | None = None
