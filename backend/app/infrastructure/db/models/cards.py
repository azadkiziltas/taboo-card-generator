from __future__ import annotations

import uuid

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Identity,
    Index,
    SmallInteger,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.db.base import Base, CreatedAtMixin, TimestampMixin


class CardRow(TimestampMixin, Base):
    __tablename__ = "cards"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    target: Mapped[str] = mapped_column(String(100))
    target_key: Mapped[str] = mapped_column(String(100))
    language: Mapped[str] = mapped_column(String(8))
    category: Mapped[str | None] = mapped_column(String(64))
    difficulty: Mapped[str | None] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="active")
    # Soft references to telemetry: cards never depend on telemetry being written.
    ai_request_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, index=True)
    generation_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, index=True)
    replaced_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("cards.id", ondelete="SET NULL")
    )

    forbidden_words: Mapped[list[CardForbiddenWordRow]] = relationship(
        back_populates="card",
        order_by="CardForbiddenWordRow.position",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_cards_language_target_key", "language", "target_key"),
        Index("ix_cards_language_category_created_at", "language", "category", "created_at"),
        CheckConstraint("difficulty IN ('easy', 'medium', 'hard')", name="difficulty"),
    )


class CardForbiddenWordRow(Base):
    __tablename__ = "card_forbidden_words"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    card_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("cards.id", ondelete="CASCADE"), index=True
    )
    word: Mapped[str] = mapped_column(String(100))
    word_key: Mapped[str] = mapped_column(String(100))
    position: Mapped[int] = mapped_column(SmallInteger)

    card: Mapped[CardRow] = relationship(back_populates="forbidden_words")

    __table_args__ = (UniqueConstraint("card_id", "position"),)


class CardFeedbackRow(CreatedAtMixin, Base):
    __tablename__ = "card_feedback"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    card_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("cards.id", ondelete="CASCADE"), index=True
    )
    rating: Mapped[str | None] = mapped_column(String(16))
    difficulty_feedback: Mapped[str | None] = mapped_column(String(16))

    __table_args__ = (
        CheckConstraint("rating IS NOT NULL OR difficulty_feedback IS NOT NULL", name="has_value"),
    )
