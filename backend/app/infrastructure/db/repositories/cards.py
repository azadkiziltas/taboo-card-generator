from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.cards.entities import (
    Card,
    CardFeedback,
    CardSource,
    CardStatus,
    Difficulty,
)
from app.domain.cards.repositories import ExistingCardSummary, NewCardRecord
from app.infrastructure.db.models import CardFeedbackRow, CardForbiddenWordRow, CardRow
from app.infrastructure.db.session import translate_db_errors

_LIVE = (CardStatus.ACTIVE.value, CardStatus.DUPLICATE_CANDIDATE.value)


def _to_card(row: CardRow) -> Card:
    return Card(
        id=row.id,
        target=row.target,
        forbidden_words=tuple(w.word for w in row.forbidden_words),
        language=row.language,
        category=row.category,
        difficulty=Difficulty(row.difficulty) if row.difficulty else None,
        source=CardSource(row.source),
        status=CardStatus(row.status),
        created_at=row.created_at,
    )


class SqlCardRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    @translate_db_errors
    async def add_many(self, records: Sequence[NewCardRecord]) -> None:
        async with self._sessions() as session, session.begin():
            for record in records:
                card = record.card
                session.add(
                    CardRow(
                        id=card.id,
                        target=card.target,
                        target_key=record.target_key,
                        language=card.language,
                        category=card.category,
                        difficulty=card.difficulty.value if card.difficulty else None,
                        source=card.source.value,
                        status=card.status.value,
                        ai_request_id=record.provenance.ai_request_id,
                        generation_run_id=record.provenance.generation_run_id,
                        forbidden_words=[
                            CardForbiddenWordRow(word=word, word_key=key, position=position)
                            for position, (word, key) in enumerate(
                                zip(card.forbidden_words, record.forbidden_keys, strict=True),
                                start=1,
                            )
                        ],
                    )
                )

    @translate_db_errors
    async def get(self, card_id: uuid.UUID) -> Card | None:
        async with self._sessions() as session:
            row = await session.get(CardRow, card_id)
            return _to_card(row) if row else None

    @translate_db_errors
    async def find_existing(
        self, language: str, target_keys: Sequence[str]
    ) -> list[ExistingCardSummary]:
        if not target_keys:
            return []
        async with self._sessions() as session:
            rows = await session.scalars(
                select(CardRow).where(
                    CardRow.language == language,
                    CardRow.target_key.in_(list(target_keys)),
                    CardRow.status.in_(_LIVE),
                )
            )
            return [
                ExistingCardSummary(
                    id=row.id,
                    language=row.language,
                    target_key=row.target_key,
                    forbidden_keys=frozenset(w.word_key for w in row.forbidden_words),
                    status=CardStatus(row.status),
                )
                for row in rows
            ]

    @translate_db_errors
    async def recent_targets(self, language: str, category: str | None, limit: int) -> list[str]:
        if limit <= 0:
            return []
        query = select(CardRow.target).where(
            CardRow.language == language, CardRow.status.in_(_LIVE)
        )
        if category:
            query = query.where(CardRow.category == category)
        query = query.order_by(CardRow.created_at.desc()).limit(limit)
        async with self._sessions() as session:
            return list(dict.fromkeys((await session.scalars(query)).all()))

    @translate_db_errors
    async def mark_replaced(self, card_id: uuid.UUID, replaced_by: uuid.UUID) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(CardRow)
                .where(CardRow.id == card_id)
                .values(status=CardStatus.REPLACED.value, replaced_by_id=replaced_by)
            )


class SqlFeedbackRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    @translate_db_errors
    async def add(self, feedback: CardFeedback) -> None:
        async with self._sessions() as session, session.begin():
            session.add(
                CardFeedbackRow(
                    id=feedback.id,
                    card_id=feedback.card_id,
                    rating=feedback.rating.value if feedback.rating else None,
                    difficulty_feedback=(
                        feedback.difficulty_feedback.value if feedback.difficulty_feedback else None
                    ),
                )
            )
