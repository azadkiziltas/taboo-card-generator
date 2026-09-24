"""Repositories, registry, telemetry and pgvector against a real PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select, text

from app.domain.ai.embeddings import EmbeddingSpace
from app.domain.ai.routing import HealthStatus, ModelType
from app.domain.ai.telemetry import AIRequestRecord, GenerationRunRecord
from app.domain.cards.entities import (
    Card,
    CardDraft,
    CardFeedback,
    CardProvenance,
    CardSource,
    CardStatus,
    Difficulty,
    FeedbackRating,
)
from app.domain.cards.repositories import NewCardRecord
from app.domain.cards.validator import CardValidator
from app.domain.errors import StorageError
from app.domain.rag.entities import RagDocument, StoredEmbedding
from app.infrastructure.ai.registry.database_registry import DatabaseModelRegistry, sync_registry
from app.infrastructure.db.models import CardForbiddenWordRow
from app.infrastructure.db.repositories.cards import SqlCardRepository, SqlFeedbackRepository
from app.infrastructure.observability.ai_usage_repository import (
    SqlAIUsageQuery,
    SqlAIUsageRecorder,
)
from app.infrastructure.rag.pgvector_store import PgVectorStore
from tests.helpers import embedding_model, generation_model

pytestmark = pytest.mark.integration

VALIDATOR = CardValidator()


def record_for(target, words, language="tr", status=CardStatus.ACTIVE, category="technology"):
    draft = CardDraft(target, tuple(words), language, Difficulty.MEDIUM, category)
    card = Card.from_draft(draft, source=CardSource.AI, status=status)
    return NewCardRecord(
        card, VALIDATOR.target_key(draft), VALIDATOR.forbidden_keys(draft), CardProvenance()
    )


async def test_card_repository_roundtrip(sessions):
    repo = SqlCardRepository(sessions)
    first = record_for("Işık", ["lamba", "güneş", "karanlık", "ampul", "parlak"])
    second = record_for("Kahve", ["fincan", "kafein", "çekirdek", "telve", "sabah"])
    await repo.add_many([first, second])

    loaded = await repo.get(first.card.id)
    assert loaded.target == "Işık" and loaded.forbidden_words == first.card.forbidden_words
    assert loaded.created_at is not None

    [existing] = await repo.find_existing("tr", ["ışık"])  # Turkish-aware key
    assert existing.forbidden_keys == frozenset(first.forbidden_keys)
    assert await repo.find_existing("en", ["ışık"]) == []

    assert await repo.recent_targets("tr", "technology", 10) == ["Kahve", "Işık"]
    await repo.mark_replaced(second.card.id, first.card.id)
    assert (await repo.get(second.card.id)).status is CardStatus.REPLACED
    assert await repo.recent_targets("tr", None, 10) == ["Işık"]

    async with sessions() as session:
        positions = (
            await session.scalars(
                select(CardForbiddenWordRow.position)
                .where(CardForbiddenWordRow.card_id == first.card.id)
                .order_by(CardForbiddenWordRow.position)
            )
        ).all()
    assert positions == [1, 2, 3, 4, 5]


async def test_card_without_difficulty_roundtrip(sessions):
    repo = SqlCardRepository(sessions)
    draft = CardDraft("Robot", ("makine", "motor", "sensör", "fabrika", "pil"), "tr", None)
    card = Card.from_draft(draft, source=CardSource.AI)
    await repo.add_many(
        [NewCardRecord(card, VALIDATOR.target_key(draft), VALIDATOR.forbidden_keys(draft))]
    )
    assert (await repo.get(card.id)).difficulty is None


async def test_arabic_text_is_stored_unchanged(sessions):
    repo = SqlCardRepository(sessions)
    record = record_for("حاسوب", ["شاشة", "لوحة مفاتيح", "إنترنت", "معالج", "برنامج"], "ar")
    await repo.add_many([record])
    loaded = await repo.get(record.card.id)
    assert loaded.target == "حاسوب" and loaded.forbidden_words[1] == "لوحة مفاتيح"


async def test_feedback_repository(sessions):
    repo = SqlCardRepository(sessions)
    record = record_for("Kahve", ["fincan", "kafein", "çekirdek", "telve", "sabah"])
    await repo.add_many([record])
    await SqlFeedbackRepository(sessions).add(
        CardFeedback(card_id=record.card.id, rating=FeedbackRating.GOOD)
    )
    async with sessions() as session:
        count = await session.scalar(text("SELECT count(*) FROM card_feedback"))
    assert count == 1


async def test_database_registry_sync_update_and_health(sessions):
    registry = DatabaseModelRegistry(sessions, cache_seconds=0)
    model = generation_model("ollama", "gen:7b", input_price="0", languages=("tr", "en"))
    await sync_registry(registry, [model, embedding_model("ollama", "embed")], overwrite=False)

    stored = await registry.get("ollama", "gen:7b", ModelType.GENERATION)
    assert stored.languages == ("tr", "en") and stored.capabilities.structured_output

    updated = await registry.update(stored.id, {"priority": 5, "quality_score": 0.8})
    assert (updated.priority, updated.quality_score) == (5, 0.8)

    # insert_missing keeps admin edits; overwrite replaces config fields but keeps benchmarks
    await sync_registry(registry, [model], overwrite=False)
    assert (await registry.get_by_id(stored.id)).priority == 5
    await sync_registry(registry, [model], overwrite=True)
    after = await registry.get_by_id(stored.id)
    assert after.priority == model.priority and after.quality_score == 0.8

    await registry.record_health(stored.id, HealthStatus.DEGRADED)
    await registry.record_latency(stored.id, 1000)
    await registry.record_latency(stored.id, 2000)
    final = await registry.get_by_id(stored.id)
    assert final.health_status is HealthStatus.DEGRADED and final.avg_latency_ms == 1200


async def test_usage_recorder_and_stats(sessions):
    recorder, query = SqlAIUsageRecorder(sessions), SqlAIUsageQuery(sessions)
    run = GenerationRunRecord(
        task="card_generation",
        drafts_generated=10,
        drafts_accepted=8,
        success=True,
        prompt_version="v1",
    )
    ok = AIRequestRecord(
        task="card_generation",
        provider="ollama",
        model="m",
        is_local=True,
        success=True,
        latency_ms=900,
        run_id=run.id,
        input_tokens=100,
        output_tokens=50,
        api_estimated_cost=Decimal(0),
        prompt_version="v1",
    )
    failed = AIRequestRecord(
        task="card_generation",
        provider="ollama",
        model="m",
        is_local=True,
        success=False,
        latency_ms=100,
        error_type="x",
        parse_failed=True,
        retry_count=1,
        prompt_version="v1",
    )
    await recorder.record_run(run)
    await recorder.record_request(ok)
    await recorder.record_request(failed)
    await recorder.annotate_request(ok.id, validation_failures=2, items_accepted=8)
    await recorder.annotate_request(ok.id, items_accepted=1)

    since = datetime.now(UTC) - timedelta(hours=1)
    [stats] = await query.request_stats(since)
    assert (stats["requests"], stats["successes"], stats["parse_failures"]) == (2, 1, 1)
    assert stats["items_accepted"] == 9 and stats["validation_failures"] == 2
    assert stats["avg_latency_ms"] == 500
    [runs] = await query.run_stats(since)
    assert runs["acceptance_rate"] == 0.8


async def test_pgvector_store_search_is_language_and_space_scoped(sessions):
    store = PgVectorStore(sessions)
    space = EmbeddingSpace("ollama", "embed", "v1")
    other_space = EmbeddingSpace("ollama", "embed", "v2")
    docs = [
        RagDocument(language="tr", word="kahve", content="sıcak içecek", source="t"),
        RagDocument(language="tr", word="çay", content="demlenen içecek", source="t"),
        RagDocument(language="en", word="coffee", content="hot drink", source="t"),
    ]
    ids = await store.upsert_documents(docs)
    await store.save_embeddings(
        space,
        3,
        [
            StoredEmbedding(ids[0], [1.0, 0.0, 0.0]),
            StoredEmbedding(ids[1], [0.0, 1.0, 0.0]),
            StoredEmbedding(ids[2], [1.0, 0.0, 0.0]),
        ],
    )
    await store.save_embeddings(other_space, 2, [StoredEmbedding(ids[1], [1.0, 0.0])])
    await store.ensure_index(space, 3)

    results = await store.search(space, [0.9, 0.1, 0.0], language="tr", top_k=5)
    assert [r.document.word for r in results] == ["kahve", "çay"]
    assert results[0].similarity > 0.99
    filtered = await store.search(
        space, [0.9, 0.1, 0.0], language="tr", top_k=5, min_similarity=0.5
    )
    assert [r.document.word for r in filtered] == ["kahve"]
    assert [d.word for d in await store.documents_missing_embeddings(other_space, limit=10)] == [
        "kahve",
        "coffee",
    ]


async def test_changed_document_needs_reembedding(sessions):
    store = PgVectorStore(sessions)
    space = EmbeddingSpace("ollama", "embed", "v1")
    doc = RagDocument(language="tr", word="kahve", content="sıcak içecek", source="t")
    [doc_id] = await store.upsert_documents([doc])
    await store.save_embeddings(space, 2, [StoredEmbedding(doc_id, [1.0, 0.0])])
    assert await store.documents_missing_embeddings(space, limit=10) == []

    changed = RagDocument(language="tr", word="kahve", content="kavrulmuş çekirdek", source="t")
    assert await store.upsert_documents([changed]) == [doc_id]  # same identity, same id
    [missing] = await store.documents_missing_embeddings(space, limit=10)
    assert missing.content == "kavrulmuş çekirdek"
    async with sessions() as session:
        assert await session.scalar(text("SELECT count(*) FROM rag_documents")) == 1
        assert await session.scalar(select(func.count()).select_from(text("rag_embeddings"))) == 1


async def test_storage_errors_are_translated(database_url):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine("postgresql+asyncpg://nobody:wrong@127.0.0.1:1/none")
    repo = SqlCardRepository(async_sessionmaker(engine))
    with pytest.raises(StorageError):
        await repo.recent_targets("tr", None, 5)
    await engine.dispose()
