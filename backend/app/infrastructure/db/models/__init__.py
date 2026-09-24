"""Importing this package registers every table on `Base.metadata` (used by Alembic)."""

from app.infrastructure.db.models.ai import AIModelRow, AIRequestRow, GenerationRunRow
from app.infrastructure.db.models.cards import CardFeedbackRow, CardForbiddenWordRow, CardRow
from app.infrastructure.db.models.rag import RagDocumentRow, RagEmbeddingRow

__all__ = [
    "AIModelRow",
    "AIRequestRow",
    "CardFeedbackRow",
    "CardForbiddenWordRow",
    "CardRow",
    "GenerationRunRow",
    "RagDocumentRow",
    "RagEmbeddingRow",
]
