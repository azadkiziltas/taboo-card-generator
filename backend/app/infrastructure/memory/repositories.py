"""In-memory adapters for tests, benchmarks and DB-less tooling."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import replace

from app.domain.ai.embeddings import EmbeddingSpace
from app.domain.ai.telemetry import AIRequestRecord, GenerationRunRecord
from app.domain.cards.entities import Card, CardFeedback, CardStatus
from app.domain.cards.repositories import ExistingCardSummary, NewCardRecord
from app.domain.rag.entities import RagDocument, ScoredDocument, StoredEmbedding

_LIVE = (CardStatus.ACTIVE, CardStatus.DUPLICATE_CANDIDATE)


class InMemoryCardRepository:
    def __init__(self) -> None:
        self.records: dict[uuid.UUID, NewCardRecord] = {}
        self._order: list[uuid.UUID] = []

    async def add_many(self, records: Sequence[NewCardRecord]) -> None:
        for record in records:
            self.records[record.card.id] = record
            self._order.append(record.card.id)

    async def get(self, card_id: uuid.UUID) -> Card | None:
        record = self.records.get(card_id)
        return record.card if record else None

    async def find_existing(
        self, language: str, target_keys: Sequence[str]
    ) -> list[ExistingCardSummary]:
        keys = set(target_keys)
        return [
            ExistingCardSummary(
                id=r.card.id,
                language=r.card.language,
                target_key=r.target_key,
                forbidden_keys=frozenset(r.forbidden_keys),
                status=r.card.status,
            )
            for r in self.records.values()
            if r.card.language == language and r.target_key in keys and r.card.status in _LIVE
        ]

    async def recent_targets(self, language: str, category: str | None, limit: int) -> list[str]:
        targets: list[str] = []
        for card_id in reversed(self._order):
            card = self.records[card_id].card
            if card.language != language or card.status not in _LIVE:
                continue
            if category and card.category != category:
                continue
            targets.append(card.target)
        return list(dict.fromkeys(targets))[:limit]

    async def mark_replaced(self, card_id: uuid.UUID, replaced_by: uuid.UUID) -> None:
        record = self.records[card_id]
        self.records[card_id] = replace(
            record, card=replace(record.card, status=CardStatus.REPLACED)
        )


class InMemoryFeedbackRepository:
    def __init__(self) -> None:
        self.items: list[CardFeedback] = []

    async def add(self, feedback: CardFeedback) -> None:
        self.items.append(feedback)


class InMemoryUsageRecorder:
    def __init__(self) -> None:
        self.requests: dict[uuid.UUID, AIRequestRecord] = {}
        self.runs: list[GenerationRunRecord] = []

    async def record_request(self, record: AIRequestRecord) -> None:
        self.requests[record.id] = record

    async def annotate_request(
        self,
        request_id: uuid.UUID,
        *,
        validation_failures: int | None = None,
        items_accepted: int | None = None,
    ) -> None:
        record = self.requests.get(request_id)
        if record is None:
            return
        if validation_failures is not None:
            record.validation_failures = (record.validation_failures or 0) + validation_failures
        if items_accepted is not None:
            record.items_accepted = (record.items_accepted or 0) + items_accepted

    async def record_run(self, run: GenerationRunRecord) -> None:
        self.runs.append(run)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, RagDocument] = {}
        self.vectors: dict[tuple[str, uuid.UUID], list[float]] = {}

    async def upsert_documents(self, documents: Sequence[RagDocument]) -> list[uuid.UUID]:
        ids = []
        for doc in documents:
            existing = next(
                (
                    d
                    for d in self.documents.values()
                    if (d.language, d.word, d.source) == (doc.language, doc.word, doc.source)
                ),
                None,
            )
            stored = replace(doc, id=existing.id) if existing else doc
            self.documents[stored.id] = stored
            ids.append(stored.id)
        return ids

    async def save_embeddings(
        self, space: EmbeddingSpace, dimension: int, embeddings: Sequence[StoredEmbedding]
    ) -> None:
        for item in embeddings:
            self.vectors[(space.key, item.document_id)] = list(item.vector)

    async def documents_missing_embeddings(
        self, space: EmbeddingSpace, *, limit: int, language: str | None = None
    ) -> list[RagDocument]:
        missing = [
            doc
            for doc in self.documents.values()
            if (space.key, doc.id) not in self.vectors and language in (None, doc.language)
        ]
        return missing[:limit]

    async def search(
        self,
        space: EmbeddingSpace,
        vector: Sequence[float],
        *,
        language: str,
        top_k: int,
        min_similarity: float | None = None,
    ) -> list[ScoredDocument]:
        scored = []
        for (space_key, doc_id), stored in self.vectors.items():
            doc = self.documents.get(doc_id)
            if space_key != space.key or doc is None or doc.language != language:
                continue
            if len(stored) != len(vector):
                continue
            similarity = _cosine(stored, vector)
            if min_similarity is None or similarity >= min_similarity:
                scored.append(ScoredDocument(doc, similarity))
        scored.sort(key=lambda s: s.similarity, reverse=True)
        return scored[:top_k]

    async def ensure_index(self, space: EmbeddingSpace, dimension: int) -> None:
        return None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
