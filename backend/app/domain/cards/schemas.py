"""Structured-output schema expected from the generation model.

Counts are intentionally *not* enforced at parse time: a card with 4 forbidden words is
a validation error that can be repaired, not a parse error that discards the batch.
The JSON schema still advertises `minItems/maxItems` so that providers with native
schema support constrain decoding.

There is no difficulty field: difficulty is a request parameter, never something the
model estimates. A model that adds one anyway is ignored (`extra="ignore"`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.domain.cards.entities import FORBIDDEN_WORD_COUNT


class GeneratedCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target: str
    forbidden_words: list[str] = Field(
        json_schema_extra={"minItems": FORBIDDEN_WORD_COUNT, "maxItems": FORBIDDEN_WORD_COUNT}
    )
    category: str | None = None

    @field_validator("forbidden_words", mode="before")
    @classmethod
    def _coerce_words(cls, value: object) -> object:
        if isinstance(value, list):
            return [str(item) if isinstance(item, int | float) else item for item in value]
        return value


class GeneratedCardBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cards: list[GeneratedCard]
