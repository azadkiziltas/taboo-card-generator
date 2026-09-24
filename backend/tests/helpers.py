"""Test helpers: fake-provider containers and card JSON builders (no LLM, no DB)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.bootstrap import Adapters, Container, wire
from app.config.settings import Settings
from app.domain.ai.capabilities import ModelCapabilities
from app.domain.ai.routing import (
    ModelDescriptor,
    ModelPricing,
    ModelType,
    RouteCandidate,
    RoutingPolicySet,
    TaskRoutingPolicy,
)
from app.infrastructure.ai.providers.fake import FakeEmbeddingProvider, FakeGenerationProvider
from app.infrastructure.ai.registry.in_memory_registry import InMemoryModelRegistry
from app.infrastructure.ai.routing.circuit_breaker import FailureCounterCircuitBreaker
from app.infrastructure.memory.repositories import (
    InMemoryCardRepository,
    InMemoryFeedbackRepository,
    InMemoryUsageRecorder,
    InMemoryVectorStore,
)

TR_WORDS = {
    "Bilgisayar": ["ekran", "klavye", "internet", "işlemci", "yazılım"],
    "Uçak": ["pilot", "havalimanı", "kanat", "bilet", "uçmak"],
    "Kahve": ["fincan", "kafein", "çekirdek", "telve", "sabah"],
    "Kitap": ["okumak", "sayfa", "yazar", "kütüphane", "roman"],
    "Futbol": ["gol", "top", "kaleci", "hakem", "stadyum"],
    "Penguen": ["buz", "kuş", "soğuk", "Antarktika", "yüzmek"],
    "Robot": ["makine", "yapay zeka", "fabrika", "sensör", "otomasyon"],
    "Tren": ["ray", "vagon", "istasyon", "lokomotif", "makinist"],
    "Gemi": ["deniz", "kaptan", "liman", "dümen", "çapa"],
    "Volkan": ["lav", "patlama", "dağ", "kül", "magma"],
    "Satranç": ["şah", "vezir", "tahta", "mat", "piyon"],
    "Arı": ["bal", "kovan", "iğne", "çiçek", "kraliçe"],
}


def card(
    target: str,
    words: list[str] | None = None,
    *,
    difficulty: str = "medium",
    category: str | None = "technology",
) -> dict[str, Any]:
    return {
        "target": target,
        "forbidden_words": words if words is not None else TR_WORDS[target],
        "category": category,
        "difficulty": difficulty,
    }


def batch(*cards: dict[str, Any]) -> str:
    return json.dumps({"cards": list(cards)}, ensure_ascii=False)


def tr_batch(*targets: str) -> str:
    return batch(*(card(t) for t in targets))


def generation_model(
    provider: str,
    model: str,
    *,
    priority: int = 10,
    structured_output: bool = True,
    json_mode: bool = True,
    is_local: bool = True,
    enabled: bool = True,
    languages: tuple[str, ...] | None = None,
    input_price: str | None = "0",
    output_price: str | None = "0",
    quality: float | None = None,
    latency: int | None = None,
) -> ModelDescriptor:
    return ModelDescriptor(
        provider=provider,
        model_name=model,
        model_type=ModelType.GENERATION,
        enabled=enabled,
        is_local=is_local,
        priority=priority,
        pricing=ModelPricing(
            Decimal(input_price) if input_price is not None else None,
            Decimal(output_price) if output_price is not None else None,
        ),
        capabilities=ModelCapabilities(structured_output=structured_output, json_mode=json_mode),
        languages=languages,
        quality_score=quality,
        avg_latency_ms=latency,
    )


def embedding_model(provider: str = "fake", model: str = "fake-embed") -> ModelDescriptor:
    return ModelDescriptor(
        provider=provider,
        model_name=model,
        model_type=ModelType.EMBEDDING,
        is_local=True,
        pricing=ModelPricing(Decimal(0), Decimal(0)),
    )


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "app_env": "test",
        "database_url": "postgresql+asyncpg://unused:unused@localhost:1/unused",
        "default_generation_provider": "primary",
        "default_generation_model": "model-a",
        "default_embedding_provider": "fake",
        "default_embedding_model": "fake-embed",
        "rag_enabled": True,
        "rag_min_similarity": 0.0,
        "max_parse_retries": 1,
        "max_repair_attempts": 2,
        "max_generation_rounds": 3,
        "card_batch_size": 10,
        "card_overgenerate_extra": 0,
        "admin_api_token": "test-admin-token",
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@dataclass
class Harness:
    container: Container
    providers: dict[str, FakeGenerationProvider]
    embedding: FakeEmbeddingProvider
    cards: InMemoryCardRepository
    feedback: InMemoryFeedbackRepository
    usage: InMemoryUsageRecorder
    vectors: InMemoryVectorStore
    registry: InMemoryModelRegistry
    breaker: FailureCounterCircuitBreaker
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> FakeGenerationProvider:
        return self.providers["primary"]


def build_env(
    *,
    settings: Settings | None = None,
    providers: Mapping[str, FakeGenerationProvider] | None = None,
    models: Iterable[ModelDescriptor] | None = None,
    policies: Iterable[TaskRoutingPolicy] | None = None,
    database_ok: bool = True,
) -> Harness:
    settings = settings or make_settings()
    providers = dict(providers or {"primary": FakeGenerationProvider(provider_name="primary")})
    if models is None:
        models = [
            generation_model(name, f"model-{chr(ord('a') + i)}", priority=10 * (i + 1))
            for i, name in enumerate(providers)
        ]
    models = [*models, embedding_model()]
    registry = InMemoryModelRegistry(models)
    embedding = FakeEmbeddingProvider(name="fake")
    breaker = FailureCounterCircuitBreaker(failure_threshold=3, reset_seconds=30)
    cards = InMemoryCardRepository()
    feedback = InMemoryFeedbackRepository()
    usage = InMemoryUsageRecorder()
    vectors = InMemoryVectorStore()

    async def db_check() -> bool:
        return database_ok

    container = wire(
        settings,
        Adapters(
            registry=registry,
            card_repository=cards,
            feedback_repository=feedback,
            usage_recorder=usage,
            vector_store=vectors,
            generation_providers=providers,
            embedding_providers={"fake": embedding},
            policies=RoutingPolicySet(policies or [], default_max_attempts=3),
            database_check=db_check,
            circuit_breaker=breaker,
        ),
    )
    return Harness(
        container, providers, embedding, cards, feedback, usage, vectors, registry, breaker
    )


def policy(task: str, *pairs: tuple[str, str], attempts: int = 3) -> TaskRoutingPolicy:
    return TaskRoutingPolicy(
        task=task,
        candidates=tuple(RouteCandidate(p, m, i + 1) for i, (p, m) in enumerate(pairs)),
        max_provider_attempts=attempts,
    )
