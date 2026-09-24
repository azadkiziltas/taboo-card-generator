from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base, TimestampMixin


class RagDocumentRow(TimestampMixin, Base):
    __tablename__ = "rag_documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    language: Mapped[str] = mapped_column(String(8), index=True)
    word: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(64))
    related_terms: Mapped[list[str]] = mapped_column(ARRAY(String(200)), default=list)
    source: Mapped[str] = mapped_column(String(100))
    source_license: Mapped[str | None] = mapped_column(String(100))
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64))

    __table_args__ = (UniqueConstraint("language", "word", "source"),)


class RagEmbeddingRow(Base):
    """One vector of one document in one embedding space (provider+model+version).

    The column has no fixed dimension so several spaces can coexist during a
    re-index; similarity queries always filter on one space and cast to its dimension.
    """

    __tablename__ = "rag_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("rag_documents.id", ondelete="CASCADE")
    )
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(200))
    embedding_version: Mapped[str] = mapped_column(String(32))
    dimension: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    content_hash: Mapped[str] = mapped_column(String(64))
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("document_id", "provider", "model", "embedding_version"),
        Index("ix_rag_embeddings_space", "provider", "model", "embedding_version", "dimension"),
    )
