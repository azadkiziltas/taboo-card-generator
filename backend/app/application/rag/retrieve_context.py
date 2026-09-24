"""Retrieve same-language reference context for prompts (target/topic -> vectors -> top-k)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.application.ai.embedding_router import EmbeddingRouter
from app.domain.ai.exceptions import AIError, RAGUnavailable
from app.domain.ai.telemetry import AIRequestContext
from app.domain.rag.entities import RagContext, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    enabled: bool = True
    top_k: int = 6
    min_similarity: float | None = 0.3
    fail_open: bool = True


def build_rag_query(
    *,
    topic: str | None = None,
    category: str | None = None,
    targets: list[str] | None = None,
) -> str | None:
    """Build the retrieval query. Returns None when there is nothing meaningful to search."""
    parts: list[str] = []
    if targets:
        parts.append(", ".join(targets))
    if topic:
        parts.append(topic)
    if category:
        parts.append(category)
    query = " - ".join(p.strip() for p in parts if p and p.strip())
    return query or None


class RetrieveContextUseCase:
    def __init__(
        self,
        *,
        embeddings: EmbeddingRouter,
        store: VectorStore,
        settings: RetrievalSettings,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    async def execute(
        self,
        *,
        language: str,
        query: str | None,
        top_k: int | None = None,
        context: AIRequestContext | None = None,
    ) -> RagContext:
        if not self._settings.enabled:
            return RagContext(enabled=False)
        if not query:
            return RagContext(query=None)
        try:
            embedded = await self._embeddings.embed([query], task="rag_query", context=context)
            documents = await self._store.search(
                self._embeddings.space,
                embedded.vectors[0],
                language=language,
                top_k=top_k or self._settings.top_k,
                min_similarity=self._settings.min_similarity,
            )
        except AIError as exc:
            return self._degrade(query, exc)
        except Exception as exc:  # database/vector store failure
            logger.exception("vector search failed")
            return self._degrade(query, RAGUnavailable("vector search failed"), cause=exc)
        return RagContext(documents=tuple(documents), query=query)

    def _degrade(
        self, query: str, error: AIError, cause: BaseException | None = None
    ) -> RagContext:
        if not self._settings.fail_open:
            raise RAGUnavailable("reference context is unavailable") from (cause or error)
        logger.warning("RAG unavailable, continuing without context: %s", error.code)
        return RagContext(query=query, error=error.code)
