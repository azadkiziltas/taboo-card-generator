"""pgvector implementation of the RAG vector store."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import and_, cast, literal, or_, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.ai.embeddings import EmbeddingSpace
from app.domain.rag.entities import RagDocument, ScoredDocument, StoredEmbedding
from app.infrastructure.db.models import RagDocumentRow, RagEmbeddingRow
from app.infrastructure.db.session import translate_db_errors


def content_hash(document: RagDocument) -> str:
    return hashlib.sha256(document.embedding_text().encode("utf-8")).hexdigest()


def _to_document(row: RagDocumentRow) -> RagDocument:
    return RagDocument(
        id=row.id,
        language=row.language,
        word=row.word,
        content=row.content,
        category=row.category,
        related_terms=tuple(row.related_terms or ()),
        source=row.source,
        source_license=row.source_license,
        metadata=dict(row.metadata_ or {}),
    )


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class PgVectorStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    @translate_db_errors
    async def upsert_documents(self, documents: Sequence[RagDocument]) -> list[uuid.UUID]:
        if not documents:
            return []
        ids: list[uuid.UUID] = []
        async with self._sessions() as session, session.begin():
            for doc in documents:
                values = {
                    "id": doc.id,
                    "language": doc.language,
                    "word": doc.word,
                    "content": doc.content,
                    "category": doc.category,
                    "related_terms": list(doc.related_terms),
                    "source": doc.source,
                    "source_license": doc.source_license,
                    "metadata": doc.metadata,
                    "content_hash": content_hash(doc),
                }
                statement = insert(RagDocumentRow.__table__).values(**values)
                statement = statement.on_conflict_do_update(
                    index_elements=["language", "word", "source"],
                    set_={
                        key: statement.excluded[key]
                        for key in (
                            "content",
                            "category",
                            "related_terms",
                            "source_license",
                            "metadata",
                            "content_hash",
                        )
                    }
                    | {"updated_at": text("now()")},
                ).returning(RagDocumentRow.__table__.c.id)
                ids.append((await session.execute(statement)).scalar_one())
        return ids

    @translate_db_errors
    async def save_embeddings(
        self, space: EmbeddingSpace, dimension: int, embeddings: Sequence[StoredEmbedding]
    ) -> None:
        if not embeddings:
            return
        async with self._sessions() as session, session.begin():
            hashes = dict(
                (
                    await session.execute(
                        select(RagDocumentRow.id, RagDocumentRow.content_hash).where(
                            RagDocumentRow.id.in_([e.document_id for e in embeddings])
                        )
                    )
                ).all()
            )
            for item in embeddings:
                if len(item.vector) != dimension:
                    raise ValueError("embedding dimension mismatch")
                statement = insert(RagEmbeddingRow.__table__).values(
                    id=uuid.uuid4(),
                    document_id=item.document_id,
                    provider=space.provider,
                    model=space.model,
                    embedding_version=space.version,
                    dimension=dimension,
                    embedding=item.vector,
                    content_hash=hashes.get(item.document_id, ""),
                )
                statement = statement.on_conflict_do_update(
                    index_elements=["document_id", "provider", "model", "embedding_version"],
                    set_={
                        "embedding": statement.excluded.embedding,
                        "dimension": statement.excluded.dimension,
                        "content_hash": statement.excluded.content_hash,
                        "indexed_at": text("now()"),
                    },
                )
                await session.execute(statement)

    @translate_db_errors
    async def documents_missing_embeddings(
        self, space: EmbeddingSpace, *, limit: int, language: str | None = None
    ) -> list[RagDocument]:
        """Documents without a vector in `space`, or whose content changed since indexing."""
        join = and_(
            RagEmbeddingRow.document_id == RagDocumentRow.id,
            RagEmbeddingRow.provider == space.provider,
            RagEmbeddingRow.model == space.model,
            RagEmbeddingRow.embedding_version == space.version,
        )
        query = (
            select(RagDocumentRow)
            .outerjoin(RagEmbeddingRow, join)
            .where(
                or_(
                    RagEmbeddingRow.id.is_(None),
                    RagEmbeddingRow.content_hash != RagDocumentRow.content_hash,
                )
            )
            .order_by(RagDocumentRow.created_at)
            .limit(limit)
        )
        if language:
            query = query.where(RagDocumentRow.language == language)
        async with self._sessions() as session:
            return [_to_document(row) for row in await session.scalars(query)]

    @translate_db_errors
    async def search(
        self,
        space: EmbeddingSpace,
        vector: Sequence[float],
        *,
        language: str,
        top_k: int,
        min_similarity: float | None = None,
    ) -> list[ScoredDocument]:
        dimension = len(vector)
        distance = cast(RagEmbeddingRow.embedding, Vector(dimension)).cosine_distance(
            cast(literal(list(vector), Vector(dimension)), Vector(dimension))
        )
        query = (
            select(RagDocumentRow, (1 - distance).label("similarity"))
            .join(RagEmbeddingRow, RagEmbeddingRow.document_id == RagDocumentRow.id)
            .where(
                RagEmbeddingRow.provider == space.provider,
                RagEmbeddingRow.model == space.model,
                RagEmbeddingRow.embedding_version == space.version,
                RagEmbeddingRow.dimension == dimension,
                RagDocumentRow.language == language,
            )
            .order_by(distance)
            .limit(top_k)
        )
        async with self._sessions() as session:
            rows = (await session.execute(query)).all()
        results = [ScoredDocument(_to_document(row[0]), float(row[1])) for row in rows]
        if min_similarity is not None:
            results = [r for r in results if r.similarity >= min_similarity]
        return results

    @translate_db_errors
    async def ensure_index(self, space: EmbeddingSpace, dimension: int) -> None:
        """Create an HNSW index for one embedding space (partial expression index)."""
        if dimension > 2000:  # pgvector HNSW limit for `vector`
            return
        digest = hashlib.sha1(f"{space.key}:{dimension}".encode()).hexdigest()[:12]
        ddl = (
            f"CREATE INDEX IF NOT EXISTS ix_rag_embeddings_hnsw_{digest} ON rag_embeddings "
            f"USING hnsw ((embedding::vector({int(dimension)})) vector_cosine_ops) "
            f"WHERE provider = {_sql_literal(space.provider)} "
            f"AND model = {_sql_literal(space.model)} "
            f"AND embedding_version = {_sql_literal(space.version)} "
            f"AND dimension = {int(dimension)}"
        )
        async with self._sessions() as session, session.begin():
            await session.execute(text(ddl))
