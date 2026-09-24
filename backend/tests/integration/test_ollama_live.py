"""Live tests against a local Ollama (opt-in: RUN_OLLAMA_TESTS=1).

They use the models configured in .env (DEFAULT_GENERATION_MODEL / DEFAULT_EMBEDDING_MODEL)
and in-memory storage. Large local models can take a minute per call.
"""

from __future__ import annotations

import os

import httpx
import pytest

from app.application.cards.generate_card import GenerateCardCommand
from app.application.cards.generate_cards import GenerateCardsCommand
from app.bootstrap import build_in_memory_container
from app.config.settings import BACKEND_ROOT, Settings
from app.domain.ai.generation import ChatMessage
from app.domain.cards.entities import Difficulty
from app.domain.cards.schemas import GeneratedCardBatch
from app.infrastructure.ai.providers.ollama_embedding import OllamaEmbeddingProvider
from app.infrastructure.ai.providers.ollama_generation import OllamaGenerationProvider

pytestmark = [pytest.mark.integration, pytest.mark.ollama]


def _pulled_models(base_url: str) -> set[str]:
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=3)
        return {m["name"] for m in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        return set()


@pytest.fixture(scope="module")
def settings() -> Settings:
    if os.environ.get("RUN_OLLAMA_TESTS") != "1":
        pytest.skip("set RUN_OLLAMA_TESTS=1 to run live Ollama tests")
    settings = Settings()
    pulled = _pulled_models(settings.ollama_base_url)
    if not pulled:
        pytest.skip(f"Ollama not reachable at {settings.ollama_base_url}")
    for model in (settings.default_generation_model, settings.default_embedding_model):
        if model not in pulled:
            pytest.skip(f"model '{model}' not pulled (ollama pull {model})")
    return settings


async def test_embedding_adapter(settings):
    async with httpx.AsyncClient() as client:
        provider = OllamaEmbeddingProvider(client=client, base_url=settings.ollama_base_url)
        result = await provider.embed(
            model=settings.default_embedding_model, texts=["kahve", "coffee", "Bilgisayar"]
        )
    assert len(result.vectors) == 3 and result.dimension > 0


async def test_generation_adapter_structured_output(settings):
    async with httpx.AsyncClient() as client:
        provider = OllamaGenerationProvider(
            client=client, base_url=settings.ollama_base_url, timeout_seconds=300
        )
        assert await provider.health_check()
        result = await provider.generate_structured(
            model=settings.default_generation_model,
            messages=[
                ChatMessage.system("You create Taboo-style game cards. Answer in JSON only."),
                ChatMessage.user(
                    "Create 1 card in Turkish for the target 'uçak' with exactly 5 forbidden "
                    'words. Shape: {"cards": [{"target": "...", "forbidden_words": [...], '
                    '"difficulty": "easy"}]}'
                ),
            ],
            schema=GeneratedCardBatch,
            temperature=0.2,
            max_tokens=512,
            options={"think": False},
        )
    assert result.parsed.cards and result.input_tokens and result.output_tokens


async def test_full_pipeline_generates_valid_cards(settings):
    container = build_in_memory_container(settings)
    try:
        from app.cli import load_jsonl

        await container.index_documents.index(load_jsonl(BACKEND_ROOT / "data/rag/seed_tr.jsonl"))
        cards = await container.generate_cards.execute(
            GenerateCardsCommand(
                language="tr", difficulty=Difficulty.MEDIUM, count=3, category="technology"
            )
        )
        single = await container.generate_card.execute(
            GenerateCardCommand(target="uçak", language="tr", difficulty=Difficulty.EASY)
        )
    finally:
        await container.aclose()

    assert 1 <= len(cards) <= 3
    for card in [*cards, single]:
        assert len(card.forbidden_words) == 5
        assert card.language == "tr"
    assert single.target == "uçak"
