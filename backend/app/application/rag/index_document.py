"""Document ingestion: store RAG documents and embed them into the active space."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from app.application.ai.embedding_router import EmbeddingRouter
from app.domain.rag.entities import RagDocument, StoredEmbedding, VectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexReport:
    documents_upserted: int = 0
    embeddings_written: int = 0
    dimension: int | None = None
    space: str | None = None


class IndexDocumentsUseCase:
    def __init__(
        self,
        *,
        embeddings: EmbeddingRouter,
        store: VectorStore,
        batch_size: int = 32,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._batch_size = batch_size

    async def index(self, documents: Sequence[RagDocument]) -> IndexReport:
        """Upsert documents and (re)embed them into the active embedding space."""
        report = IndexReport(space=self._embeddings.space.key)
        for start in range(0, len(documents), self._batch_size):
            chunk = documents[start : start + self._batch_size]
            ids = await self._store.upsert_documents(chunk)
            report.documents_upserted += len(ids)
            stored = [replace(doc, id=doc_id) for doc, doc_id in zip(chunk, ids, strict=True)]
            report.embeddings_written += await self._embed_and_save(stored, report)
        if report.dimension:
            await self._store.ensure_index(self._embeddings.space, report.dimension)
        return report

    async def reindex_missing(self, *, language: str | None = None) -> IndexReport:
        """Embed every document that has no vector in the active space yet.

        Used after switching to a new embedding model/version.
        """
        report = IndexReport(space=self._embeddings.space.key)
        while True:
            batch = await self._store.documents_missing_embeddings(
                self._embeddings.space, limit=self._batch_size, language=language
            )
            if not batch:
                break
            written = await self._embed_and_save(batch, report)
            if written == 0:
                break
            report.embeddings_written += written
        if report.dimension:
            await self._store.ensure_index(self._embeddings.space, report.dimension)
        return report

    async def _embed_and_save(self, documents: Sequence[RagDocument], report: IndexReport) -> int:
        if not documents:
            return 0
        result = await self._embeddings.embed(
            [doc.embedding_text() for doc in documents], task="rag_index"
        )
        report.dimension = result.dimension
        await self._store.save_embeddings(
            self._embeddings.space,
            result.dimension,
            [
                StoredEmbedding(document_id=doc.id, vector=vector)
                for doc, vector in zip(documents, result.vectors, strict=True)
            ],
        )
        logger.info("indexed %d documents into %s", len(documents), self._embeddings.space.key)
        return len(documents)
