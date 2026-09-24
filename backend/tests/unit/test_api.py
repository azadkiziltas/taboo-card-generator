"""HTTP contract tests with a fake-provider container (no LLM, no database)."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.domain.ai.exceptions import ProviderUnavailable
from app.main import create_app
from tests.helpers import TR_WORDS, build_env, make_settings, tr_batch

ADMIN = {"X-Admin-Token": "test-admin-token"}


@pytest.fixture
def harness():
    return build_env()


@pytest.fixture
def client(harness):
    app = create_app(harness.container.settings, container=harness.container)
    with TestClient(app) as test_client:
        yield test_client


def generate(client, **body):
    payload = {"language": "tr", "category": "technology", "difficulty": "medium", "count": 2}
    payload.update(body)
    return client.post("/v1/cards/generate", json=payload)


def test_generate_returns_the_public_contract(client, harness):
    harness.primary.enqueue(tr_batch("Bilgisayar", "Uçak"))
    response = generate(client)
    assert response.status_code == 200
    cards = response.json()["cards"]
    assert len(cards) == 2
    assert set(cards[0]) == {
        "id",
        "target",
        "forbidden_words",
        "language",
        "category",
        "difficulty",
    }
    assert cards[0]["target"] == "Bilgisayar"
    assert cards[0]["forbidden_words"] == TR_WORDS["Bilgisayar"]
    uuid.UUID(cards[0]["id"])
    body = response.text.lower()
    assert "primary" not in body and "model-a" not in body  # no provider/model leakage


def test_difficulty_is_optional(client, harness):
    harness.primary.enqueue(tr_batch("Robot"))
    response = client.post("/v1/cards/generate", json={"language": "tr", "count": 1})
    assert response.status_code == 200
    assert response.json()["cards"][0]["difficulty"] is None
    assert "difficulty" not in harness.primary.calls[0].prompt.lower()


def test_client_cannot_choose_a_model(client, harness):
    response = generate(client, model="qwen")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert harness.primary.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"language": "turkish"},
        {"count": 0},
        {"count": 101},
        {"difficulty": "extreme"},
        {"category": "a" * 51},
        {"topic": "x" * 101},
        {"category": "tech; DROP TABLE cards"},
    ],
)
def test_request_validation(client, body):
    assert generate(client, **body).status_code == 422


@pytest.mark.parametrize(
    "category", ["uzay yolculuğu", "90'lar müziği", "yemek & içecek", "Dr. Who"]
)
def test_category_accepts_theme_phrases(client, harness, category):
    harness.primary.enqueue(tr_batch("Robot"))
    assert generate(client, category=category, count=1).status_code == 200


@pytest.mark.parametrize("category", ["<script>", "a;b", "x" * 51])
def test_category_rejects_markup_and_overlong_values(client, category):
    assert generate(client, category=category).status_code == 422


def test_unsupported_but_well_formed_language(client):
    response = generate(client, language="xx")
    assert response.status_code == 422
    assert "not supported" in response.json()["error"]["message"]


def test_provider_outage_is_a_controlled_error_without_details(client, harness):
    harness.primary.enqueue(ProviderUnavailable("connection refused to http://10.0.0.5:11434"))
    response = generate(client)
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "generation_unavailable"
    assert "11434" not in response.text and "primary" not in response.text


def test_invalid_model_output_is_a_502(client, harness):
    harness.primary.default = "I cannot help with that."
    response = generate(client)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "generation_failed"


def test_feedback_flow(client, harness):
    harness.primary.enqueue(tr_batch("Kahve"))
    card_id = generate(client, count=1).json()["cards"][0]["id"]

    ok = client.post(
        f"/v1/cards/{card_id}/feedback", json={"rating": "good", "difficulty_feedback": "too_easy"}
    )
    assert ok.status_code == 201 and ok.json()["status"] == "recorded"
    assert harness.feedback.items[0].rating.value == "good"

    assert client.post(f"/v1/cards/{card_id}/feedback", json={}).status_code == 422
    assert client.post(f"/v1/cards/{card_id}/feedback", json={"rating": "meh"}).status_code == 422
    missing = client.post(f"/v1/cards/{uuid.uuid4()}/feedback", json={"rating": "bad"})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "card_not_found"


def test_regenerate_endpoint(client, harness):
    harness.primary.enqueue(tr_batch("Kahve"))
    card_id = generate(client, count=1).json()["cards"][0]["id"]
    harness.primary.enqueue(
        '{"cards": [{"target": "Kahve", "forbidden_words": ["espresso", "barista", "kupa", '
        '"süt", "köpük"], "difficulty": "medium"}]}'
    )
    response = client.post(f"/v1/cards/{card_id}/regenerate")
    assert response.status_code == 200
    assert response.json()["target"] == "Kahve" and response.json()["id"] != card_id


def test_public_health_hides_provider_details(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["components"] == {"database": "ok", "generation": "ok", "embedding": "ok"}
    assert "primary" not in response.text and "fake" not in response.text


def test_health_reports_generation_outage(harness):
    harness.primary.healthy = False
    app = create_app(harness.container.settings, container=harness.container)
    with TestClient(app) as client:
        data = client.get("/v1/health").json()
    assert data["status"] == "degraded" and data["components"]["generation"] == "unavailable"


def test_admin_requires_token(client):
    assert client.get("/admin/v1/models").status_code == 401
    assert client.get("/admin/v1/models", headers={"X-Admin-Token": "wrong"}).status_code == 401


def test_admin_models_and_routing(client, harness):
    models = client.get("/admin/v1/models", headers=ADMIN).json()
    generation = next(m for m in models if m["model_type"] == "generation")
    assert generation["provider"] == "primary"

    patched = client.patch(
        f"/admin/v1/models/{generation['id']}", headers=ADMIN, json={"enabled": False}
    )
    assert patched.status_code == 200 and patched.json()["enabled"] is False
    routing = client.get("/admin/v1/routing/card_generation", headers=ADMIN).json()
    assert routing["candidates"] == []

    health = client.get("/admin/v1/health", headers=ADMIN).json()
    assert health["generation_providers"] == {"primary": True}


def test_admin_rag_ingest_and_search(client):
    ingest = client.post(
        "/admin/v1/rag/documents",
        headers=ADMIN,
        json={
            "documents": [
                {"language": "tr", "word": "kahve", "content": "sıcak içecek", "source": "test"}
            ]
        },
    )
    assert ingest.status_code == 200 and ingest.json()["embeddings_written"] == 1
    search = client.post(
        "/admin/v1/rag/search", headers=ADMIN, json={"language": "tr", "query": "kahve içecek"}
    )
    assert search.json()["documents"][0]["word"] == "kahve"


def test_admin_disabled_without_configured_token():
    harness = build_env(settings=make_settings(admin_api_token=None))
    app = create_app(harness.container.settings, container=harness.container)
    with TestClient(app) as client:
        assert client.get("/admin/v1/models", headers=ADMIN).status_code == 403


def test_openapi_is_available(client):
    schema = client.get("/openapi.json").json()
    assert "/v1/cards/generate" in schema["paths"]
    request_schema = schema["components"]["schemas"]["GenerateCardsRequest"]
    assert "model" not in request_schema["properties"]
    assert "provider" not in request_schema["properties"]
