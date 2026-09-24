"""RAG documents and retrieval results (independent from the generation model)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.domain.ai.embeddings import EmbeddingSpace


@dataclass(frozen=True, slots=True)
class RagDocument:
    language: str
    word: str
    content: str  # definition / description
    source: str
    category: str | None = None
    related_terms: tuple[str, ...] = ()
    source_license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    def embedding_text(self) -> str:
        """Text that represents this document in vector space."""
        parts = [f"{self.word}: {self.content}"]
        if self.category:
            parts.append(f"({self.category})")
        if self.related_terms:
            parts.append(", ".join(self.related_terms))
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class ScoredDocument:
    document: RagDocument
    similarity: float


@dataclass(frozen=True, slots=True)
class RagContext:
    documents: tuple[ScoredDocument, ...] = ()
    query: str | None = None
    enabled: bool = True
    error: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.documents


@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    document_id: uuid.UUID
    vector: list[float]


class VectorStore(Protocol):
    async def upsert_documents(self, documents: Sequence[RagDocument]) -> list[uuid.UUID]:
        """Insert or update documents (identity: language + word + source). Returns ids."""
        ...

    async def save_embeddings(
        self, space: EmbeddingSpace, dimension: int, embeddings: Sequence[StoredEmbedding]
    ) -> None: ...

    async def documents_missing_embeddings(
        self, space: EmbeddingSpace, *, limit: int, language: str | None = None
    ) -> list[RagDocument]: ...

    async def search(
        self,
        space: EmbeddingSpace,
        vector: Sequence[float],
        *,
        language: str,
        top_k: int,
        min_similarity: float | None = None,
    ) -> list[ScoredDocument]: ...

    async def ensure_index(self, space: EmbeddingSpace, dimension: int) -> None: ...
