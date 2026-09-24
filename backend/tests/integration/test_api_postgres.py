"""`/v1/cards/generate` end-to-end with PostgreSQL storage and a fake LLM."""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy import text

from app.bootstrap import Adapters, wire
from app.cli import load_jsonl
from app.config.settings import BACKEND_ROOT
from app.domain.ai.exceptions import ProviderUnavailable
from app.domain.ai.routing import RoutingPolicySet
from app.infrastructure.ai.providers.fake import FakeEmbeddingProvider, FakeGenerationProvider
from app.infrastructure.ai.registry.database_registry import DatabaseModelRegistry, sync_registry
from app.infrastructure.db.repositories.cards import SqlCardRepository, SqlFeedbackRepository
from app.infrastructure.observability.ai_usage_repository import (
    SqlAIUsageQuery,
    SqlAIUsageRecorder,
)
from app.infrastructure.rag.pgvector_store import PgVectorStore
from app.main import create_app
from tests.helpers import (
    batch,
    card,
    embedding_model,
    generation_model,
    make_settings,
    policy,
    tr_batch,
)

pytestmark = pytest.mark.integration


async def build_app(sessions, primary, secondary):
    settings = make_settings()
    registry = DatabaseModelRegistry(sessions, cache_seconds=0)
    await sync_registry(
        registry,
        [
            generation_model("primary", "model-a", priority=10),
            generation_model("secondary", "model-b", priority=20),
            embedding_model(),
        ],
        overwrite=False,
    )

    async def db_ok() -> bool:
        return True

    container = wire(
        settings,
        Adapters(
            registry=registry,
            card_repository=SqlCardRepository(sessions),
            feedback_repository=SqlFeedbackRepository(sessions),
            usage_recorder=SqlAIUsageRecorder(sessions),
            vector_store=PgVectorStore(sessions),
            generation_providers={"primary": primary, "secondary": secondary},
            embedding_providers={"fake": FakeEmbeddingProvider(name="fake", dimension=32)},
            policies=RoutingPolicySet(
                [policy("card_generation", ("primary", "model-a"), ("secondary", "model-b"))]
            ),
            database_check=db_ok,
            usage_query=SqlAIUsageQuery(sessions),
        ),
    )
    app = create_app(settings, container=container)
    app.state.container = container  # ASGITransport does not run the lifespan
    return app, container


def client_for(app):
    # Same event loop as the asyncpg pool (TestClient would use a separate loop).
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def scalar(sessions, sql):
    async with sessions() as session:
        return await session.scalar(text(sql))


async def test_generate_persists_valid_cards_and_telemetry_with_fallback(sessions):
    primary = FakeGenerationProvider(provider_name="primary", responses=[ProviderUnavailable()])
    secondary = FakeGenerationProvider(
        provider_name="secondary",
        responses=[
            batch(
                card("Bilgisayar"),
                card("Kahve", ["kahve", "fincan", "kafein", "çekirdek", "telve"]),  # invalid
                card("Uçak"),
            ),
            batch(card("Kahve", ["sabah", "fincan", "kafein", "çekirdek", "telve"])),  # repair
        ],
    )
    app, container = await build_app(sessions, primary, secondary)
    await container.index_documents.index(load_jsonl(BACKEND_ROOT / "data/rag/seed_tr.jsonl"))

    async with client_for(app) as client:
        response = await client.post(
            "/v1/cards/generate",
            json={"language": "tr", "category": "technology", "difficulty": "medium", "count": 3},
        )
        assert response.status_code == 200, response.text
        targets = sorted(c["target"] for c in response.json()["cards"])
        assert targets == ["Bilgisayar", "Kahve", "Uçak"]

        telemetry = (
            await client.get("/admin/v1/telemetry", headers={"X-Admin-Token": "test-admin-token"})
        ).json()

    assert await scalar(sessions, "SELECT count(*) FROM cards") == 3
    assert await scalar(sessions, "SELECT count(*) FROM card_forbidden_words") == 15
    assert (
        await scalar(
            sessions,
            "SELECT count(*) FROM card_forbidden_words w JOIN cards c ON c.id = w.card_id "
            "WHERE c.target = 'Kahve' AND w.word = 'kahve'",
        )
        == 0
    )  # the invalid version never reached the database
    assert await scalar(sessions, "SELECT count(*) FROM ai_requests WHERE fallback_used") >= 1
    assert (
        await scalar(
            sessions,
            "SELECT count(*) FROM ai_requests "
            "WHERE task = 'card_generation' AND error_type = 'provider_unavailable'",
        )
        == 1
    )
    assert await scalar(sessions, "SELECT count(*) FROM generation_runs WHERE success") == 1
    assert await scalar(sessions, "SELECT max(context_documents) FROM generation_runs") > 0
    assert (
        await scalar(
            sessions,
            "SELECT count(*) FROM cards c JOIN ai_requests r ON r.id = c.ai_request_id "
            "WHERE r.provider = 'secondary'",
        )
        == 3
    )
    tasks = {row["task"] for row in telemetry["requests"]}
    assert {"card_generation", "card_repair", "rag_query", "rag_index"} <= tasks


async def test_second_request_avoids_existing_targets(sessions):
    primary = FakeGenerationProvider(
        provider_name="primary", responses=[tr_batch("Kahve"), tr_batch("Kahve"), tr_batch("Tren")]
    )
    app, _ = await build_app(sessions, primary, FakeGenerationProvider(provider_name="secondary"))
    async with client_for(app) as client:
        body = {"language": "tr", "category": "technology", "count": 1}
        first = (await client.post("/v1/cards/generate", json=body)).json()["cards"][0]
        second = (await client.post("/v1/cards/generate", json=body)).json()["cards"][0]
    assert first["target"] == "Kahve"
    assert second["target"] == "Tren"  # exact duplicate rejected, new round generated
    assert "Kahve" in primary.calls[1].messages[-1].content  # avoid list from the database
