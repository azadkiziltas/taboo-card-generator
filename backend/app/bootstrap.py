"""Composition root: the only place that knows every concrete implementation.

`build_container()` wires production adapters (PostgreSQL, Ollama, ...). Tests and the
benchmark runner call `wire()` directly with in-memory/fake adapters.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from app.application.ai.embedding_router import EmbeddingRouter
from app.application.ai.model_router import ModelRouter, RouterLimits
from app.application.cards.engine import CardGenerationEngine, CardGenerationSettings
from app.application.cards.generate_card import GenerateCardUseCase
from app.application.cards.generate_cards import GenerateCardsUseCase
from app.application.cards.regenerate_card import RegenerateCardUseCase
from app.application.cards.submit_feedback import SubmitFeedbackUseCase
from app.application.health import HealthService
from app.application.prompts.service import PromptService
from app.application.rag.index_document import IndexDocumentsUseCase
from app.application.rag.retrieve_context import RetrievalSettings, RetrieveContextUseCase
from app.application.rate_limit import NoopRateLimiter, RateLimiter
from app.config.settings import Settings
from app.domain.ai.embeddings import EmbeddingProvider, EmbeddingSpace
from app.domain.ai.generation import GenerationProvider
from app.domain.ai.routing import CircuitBreaker, ModelRegistry, RoutingPolicySet
from app.domain.ai.telemetry import AIUsageRecorder
from app.domain.cards.duplicates import DuplicateDetector
from app.domain.cards.normalization import NormalizerRegistry
from app.domain.cards.repositories import CardRepository, FeedbackRepository
from app.domain.cards.safety import AllowAllSafetyFilter, BlocklistSafetyFilter, ContentSafetyFilter
from app.domain.cards.validator import CardValidator, ValidationConfig
from app.domain.errors import StorageError
from app.domain.rag.entities import VectorStore
from app.infrastructure.ai.routing.circuit_breaker import FailureCounterCircuitBreaker
from app.infrastructure.content_safety import load_blocklists
from app.infrastructure.observability.metrics import InMemoryMetrics, ObservedUsageRecorder
from app.infrastructure.prompts.file_store import FilePromptStore

logger = logging.getLogger(__name__)


class UsageQuery(Protocol):
    async def request_stats(self, since: Any) -> list[dict[str, Any]]: ...

    async def run_stats(self, since: Any) -> list[dict[str, Any]]: ...


@dataclass(slots=True)
class Adapters:
    """Infrastructure implementations to wire into the application."""

    registry: ModelRegistry
    card_repository: CardRepository
    feedback_repository: FeedbackRepository
    usage_recorder: AIUsageRecorder
    vector_store: VectorStore
    generation_providers: Mapping[str, GenerationProvider]
    embedding_providers: Mapping[str, EmbeddingProvider]
    policies: RoutingPolicySet
    database_check: Callable[[], Awaitable[bool]]
    usage_query: UsageQuery | None = None
    safety_filter: ContentSafetyFilter | None = None
    rate_limiter: RateLimiter | None = None
    circuit_breaker: CircuitBreaker | None = None


@dataclass(slots=True)
class Container:
    settings: Settings
    registry: ModelRegistry
    router: ModelRouter
    embeddings: EmbeddingRouter
    prompts: PromptService
    engine: CardGenerationEngine
    generate_cards: GenerateCardsUseCase
    generate_card: GenerateCardUseCase
    regenerate_card: RegenerateCardUseCase
    submit_feedback: SubmitFeedbackUseCase
    index_documents: IndexDocumentsUseCase
    retrieve_context: RetrieveContextUseCase
    health: HealthService
    metrics: InMemoryMetrics
    rate_limiter: RateLimiter
    usage_query: UsageQuery | None
    disabled_providers: dict[str, str] = field(default_factory=dict)
    # Underlying telemetry store; only set by `build_in_memory_container` (benchmarks).
    usage_store: Any = None
    _closers: list[Callable[[], Awaitable[Any]]] = field(default_factory=list)

    def add_closer(self, closer: Callable[[], Awaitable[Any]]) -> None:
        self._closers.append(closer)

    async def aclose(self) -> None:
        for closer in reversed(self._closers):
            try:
                await closer()
            except Exception:
                logger.exception("error while closing resource")


def wire(settings: Settings, adapters: Adapters) -> Container:
    metrics = InMemoryMetrics()
    usage = ObservedUsageRecorder(adapters.usage_recorder, metrics)
    breaker = adapters.circuit_breaker or FailureCounterCircuitBreaker(
        failure_threshold=settings.circuit_breaker_failure_threshold,
        reset_seconds=settings.circuit_breaker_reset_seconds,
    )
    prompts = PromptService(FilePromptStore(settings.prompts_dir))
    normalizers = NormalizerRegistry.with_defaults()
    validator = CardValidator(
        normalizers,
        ValidationConfig(
            max_target_length=settings.card_max_target_length,
            max_forbidden_word_length=settings.card_max_forbidden_word_length,
            supported_languages=frozenset(settings.supported_languages),
        ),
    )
    safety = adapters.safety_filter
    if safety is None:
        blocklists = load_blocklists(settings.blocklist_dir)
        safety = (
            BlocklistSafetyFilter(blocklists, normalizers) if blocklists else AllowAllSafetyFilter()
        )

    router = ModelRouter(
        registry=adapters.registry,
        providers=adapters.generation_providers,
        policies=adapters.policies,
        circuit_breaker=breaker,
        usage_recorder=usage,
        prompts=prompts,
        limits=RouterLimits(max_parse_retries=settings.max_parse_retries),
    )
    embeddings = EmbeddingRouter(
        providers=adapters.embedding_providers,
        space=EmbeddingSpace(
            provider=settings.default_embedding_provider,
            model=settings.default_embedding_model,
            version=settings.embedding_version,
        ),
        registry=adapters.registry,
        usage_recorder=usage,
        circuit_breaker=breaker,
        max_retries=settings.embedding_max_retries,
    )
    retrieve = RetrieveContextUseCase(
        embeddings=embeddings,
        store=adapters.vector_store,
        settings=RetrievalSettings(
            enabled=settings.rag_enabled,
            top_k=settings.rag_top_k,
            min_similarity=settings.rag_min_similarity,
            fail_open=settings.rag_fail_open,
        ),
    )
    engine = CardGenerationEngine(
        router=router,
        prompts=prompts,
        validator=validator,
        duplicates=DuplicateDetector(
            adapters.card_repository,
            validator,
            overlap_threshold=settings.duplicate_forbidden_overlap_threshold,
        ),
        safety=safety,
        repository=adapters.card_repository,
        retrieve_context=retrieve,
        usage_recorder=usage,
        settings=CardGenerationSettings(
            batch_size=settings.card_batch_size,
            overgenerate_extra=settings.card_overgenerate_extra,
            max_generation_rounds=settings.max_generation_rounds,
            max_repair_attempts=settings.max_repair_attempts,
            generation_temperature=settings.card_generation_temperature,
            repair_temperature=settings.card_repair_temperature,
            max_output_tokens=settings.card_max_output_tokens,
            prompt_version=settings.card_prompt_version,
            avoid_existing_targets_limit=settings.avoid_existing_targets_limit,
        ),
    )
    return Container(
        settings=settings,
        registry=adapters.registry,
        router=router,
        embeddings=embeddings,
        prompts=prompts,
        engine=engine,
        generate_cards=GenerateCardsUseCase(
            engine=engine,
            repository=adapters.card_repository,
            usage_recorder=usage,
            supported_languages=settings.supported_languages,
            max_cards_per_request=settings.max_cards_per_request,
            allow_partial_results=settings.allow_partial_results,
        ),
        generate_card=GenerateCardUseCase(
            engine=engine,
            repository=adapters.card_repository,
            usage_recorder=usage,
            supported_languages=settings.supported_languages,
        ),
        regenerate_card=RegenerateCardUseCase(
            engine=engine, repository=adapters.card_repository, usage_recorder=usage
        ),
        submit_feedback=SubmitFeedbackUseCase(
            cards=adapters.card_repository, feedback=adapters.feedback_repository
        ),
        index_documents=IndexDocumentsUseCase(
            embeddings=embeddings,
            store=adapters.vector_store,
            batch_size=settings.rag_embed_batch_size,
        ),
        retrieve_context=retrieve,
        health=HealthService(
            database_check=adapters.database_check,
            router=router,
            embeddings=embeddings,
            circuit_breaker=breaker,
            rag_enabled=settings.rag_enabled,
        ),
        metrics=metrics,
        rate_limiter=adapters.rate_limiter or NoopRateLimiter(),
        usage_query=adapters.usage_query,
    )


def build_in_memory_container(
    settings: Settings,
    *,
    policies: RoutingPolicySet | None = None,
    models: list[Any] | None = None,
) -> Container:
    """Real providers from config/ai.yaml, but in-memory storage (benchmarks, live tests).

    Nothing is written to PostgreSQL. `models`/`policies` override the configured ones.
    """
    from app.infrastructure.ai.config import build_environment, load_ai_config
    from app.infrastructure.ai.providers.factory import build_providers
    from app.infrastructure.ai.registry.in_memory_registry import InMemoryModelRegistry
    from app.infrastructure.memory.repositories import (
        InMemoryCardRepository,
        InMemoryFeedbackRepository,
        InMemoryUsageRecorder,
        InMemoryVectorStore,
    )

    env = build_environment(settings)
    ai_config = load_ai_config(settings.ai_config_path, env)
    http = httpx.AsyncClient()
    providers = build_providers(ai_config.providers, http, env)
    usage = InMemoryUsageRecorder()

    async def no_database() -> bool:
        return False

    container = wire(
        settings,
        Adapters(
            registry=InMemoryModelRegistry(models if models is not None else ai_config.models),
            card_repository=InMemoryCardRepository(),
            feedback_repository=InMemoryFeedbackRepository(),
            usage_recorder=usage,
            vector_store=InMemoryVectorStore(),
            generation_providers=providers.generation,
            embedding_providers=providers.embedding,
            policies=policies
            or RoutingPolicySet(ai_config.routing, settings.default_max_provider_attempts),
            database_check=no_database,
        ),
    )
    container.disabled_providers = providers.disabled
    container.add_closer(http.aclose)
    container.usage_store = usage
    return container


async def build_container(settings: Settings) -> Container:
    """Production wiring: PostgreSQL + pgvector + providers from config/ai.yaml."""
    from app.infrastructure.ai.config import build_environment, load_ai_config
    from app.infrastructure.ai.providers.factory import build_providers
    from app.infrastructure.ai.registry.database_registry import (
        DatabaseModelRegistry,
        sync_registry,
    )
    from app.infrastructure.db.repositories.cards import SqlCardRepository, SqlFeedbackRepository
    from app.infrastructure.db.session import (
        check_database,
        create_engine,
        create_session_factory,
    )
    from app.infrastructure.observability.ai_usage_repository import (
        SqlAIUsageQuery,
        SqlAIUsageRecorder,
    )
    from app.infrastructure.rag.pgvector_store import PgVectorStore

    env = build_environment(settings)
    ai_config = load_ai_config(settings.ai_config_path, env)
    engine = create_engine(settings)
    sessions = create_session_factory(engine)
    http = httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=10))
    providers = build_providers(ai_config.providers, http, env)

    registry = DatabaseModelRegistry(sessions, cache_seconds=settings.model_registry_cache_seconds)
    if settings.model_registry_sync != "off":
        try:
            await sync_registry(
                registry, ai_config.models, overwrite=settings.model_registry_sync == "overwrite"
            )
        except StorageError:
            logger.error("model registry sync skipped: database unavailable")

    container = wire(
        settings,
        Adapters(
            registry=registry,
            card_repository=SqlCardRepository(sessions),
            feedback_repository=SqlFeedbackRepository(sessions),
            usage_recorder=SqlAIUsageRecorder(sessions),
            vector_store=PgVectorStore(sessions),
            generation_providers=providers.generation,
            embedding_providers=providers.embedding,
            policies=RoutingPolicySet(
                ai_config.routing, default_max_attempts=settings.default_max_provider_attempts
            ),
            database_check=lambda: check_database(engine),
            usage_query=SqlAIUsageQuery(sessions),
        ),
    )
    container.disabled_providers = providers.disabled
    container.add_closer(engine.dispose)
    container.add_closer(http.aclose)
    for name, reason in providers.disabled.items():
        logger.warning("provider %s disabled: %s", name, reason)
    logger.info(
        "container ready",
        extra={
            "generation_providers": ",".join(sorted(providers.generation)) or "-",
            "embedding_space": container.embeddings.space.key,
            "rag_enabled": settings.rag_enabled,
        },
    )
    return container
