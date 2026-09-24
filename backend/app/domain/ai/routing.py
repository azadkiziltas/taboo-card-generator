"""Model registry entries, routing policies and pure candidate selection logic."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from app.domain.ai.capabilities import ModelCapabilities


class ModelType(StrEnum):
    GENERATION = "generation"
    EMBEDDING = "embedding"


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class RoutingStrategy(StrEnum):
    PRIORITY = "priority"  # explicit priority order (MVP default)
    COST = "cost"  # cheapest first
    QUALITY = "quality"  # highest benchmark quality_score first
    LATENCY = "latency"  # lowest observed latency first
    LOCAL_FIRST = "local_first"  # local models first, then priority


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_cost_per_million: Decimal | None = None
    output_cost_per_million: Decimal | None = None
    currency: str = "USD"


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """One model registry entry, independent from storage."""

    provider: str
    model_name: str
    model_type: ModelType
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    enabled: bool = True
    is_local: bool = False
    priority: int = 100
    pricing: ModelPricing = field(default_factory=ModelPricing)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    languages: tuple[str, ...] | None = None  # None = every language
    quality_score: float | None = None
    avg_latency_ms: int | None = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def supports_language(self, language: str | None) -> bool:
        return language is None or self.languages is None or language in self.languages


@dataclass(frozen=True, slots=True)
class RoutingRequirements:
    """Hard filters a candidate model must satisfy.

    `structured_output=True` means the model must reliably emit JSON by itself
    (native schema support or JSON mode). Prompt-only models are excluded then.
    """

    language: str | None = None
    structured_output: bool = False
    tool_calling: bool = False
    vision: bool = False
    local_only: bool = False
    min_context_tokens: int | None = None
    max_input_cost_per_million: Decimal | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> RoutingRequirements:
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown routing requirement(s): {sorted(unknown)}")
        return cls(**dict(data))


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    provider: str
    model: str
    priority: int = 100


@dataclass(frozen=True, slots=True)
class TaskRoutingPolicy:
    task: str
    candidates: tuple[RouteCandidate, ...] = ()
    strategy: RoutingStrategy = RoutingStrategy.PRIORITY
    max_provider_attempts: int = 3


class RoutingPolicySet:
    """Routing policies per task with a registry-driven default."""

    DEFAULT_TASK = "default"

    def __init__(self, policies: Iterable[TaskRoutingPolicy], default_max_attempts: int = 3):
        self._policies = {policy.task: policy for policy in policies}
        self._default_max_attempts = default_max_attempts

    def for_task(self, task: str) -> TaskRoutingPolicy:
        if task in self._policies:
            return self._policies[task]
        if self.DEFAULT_TASK in self._policies:
            return self._policies[self.DEFAULT_TASK]
        return TaskRoutingPolicy(task=task, max_provider_attempts=self._default_max_attempts)

    @property
    def policies(self) -> Mapping[str, TaskRoutingPolicy]:
        return dict(self._policies)


def meets_requirements(model: ModelDescriptor, req: RoutingRequirements) -> bool:
    caps = model.capabilities
    if req.structured_output and not (caps.structured_output or caps.json_mode):
        return False
    if req.tool_calling and not caps.tool_calling:
        return False
    if req.vision and not caps.vision:
        return False
    if req.local_only and not model.is_local:
        return False
    if (
        req.min_context_tokens is not None
        and caps.max_context_tokens is not None
        and caps.max_context_tokens < req.min_context_tokens
    ):
        return False
    if req.max_input_cost_per_million is not None:
        price = model.pricing.input_cost_per_million
        if price is not None and price > req.max_input_cost_per_million:
            return False
    return model.supports_language(req.language)


def _strategy_key(strategy: RoutingStrategy, priority: int, model: ModelDescriptor) -> tuple:
    inf = float("inf")
    match strategy:
        case RoutingStrategy.COST:
            cost = model.pricing.input_cost_per_million
            return (float(cost) if cost is not None else inf, priority)
        case RoutingStrategy.QUALITY:
            quality = model.quality_score
            return (-(quality if quality is not None else -inf), priority)
        case RoutingStrategy.LATENCY:
            latency = model.avg_latency_ms
            return (latency if latency is not None else inf, priority)
        case RoutingStrategy.LOCAL_FIRST:
            return (0 if model.is_local else 1, priority)
        case _:
            return (priority,)


def select_candidates(
    *,
    models: Sequence[ModelDescriptor],
    policy: TaskRoutingPolicy,
    requirements: RoutingRequirements,
    available_providers: Iterable[str],
    is_allowed: Mapping[str, bool] | None = None,
) -> list[ModelDescriptor]:
    """Return eligible generation models in the order they should be tried.

    - When the policy lists candidates, only those (provider, model) pairs are used and the
      policy priority overrides the registry priority.
    - Otherwise every enabled generation model in the registry is eligible.
    - Disabled models, models whose provider adapter is not registered, models that fail
      the requirements and models whose circuit is open (`is_allowed[key] is False`) are
      filtered out. The persisted registry `health_status` is informational only; the live
      circuit breaker decides availability (a stale DB flag must not block a model forever).
    - The result is truncated to `policy.max_provider_attempts`.
    """
    providers = set(available_providers)
    allowed = is_allowed or {}
    by_key = {m.key: m for m in models if m.model_type is ModelType.GENERATION and m.enabled}

    ranked: list[tuple[int, ModelDescriptor]] = []
    if policy.candidates:
        for candidate in policy.candidates:
            model = by_key.get(f"{candidate.provider}:{candidate.model}")
            if model is not None:
                ranked.append((candidate.priority, model))
    else:
        ranked = [(m.priority, m) for m in by_key.values()]

    eligible = [
        (priority, model)
        for priority, model in ranked
        if model.provider in providers
        and allowed.get(model.key, True)
        and meets_requirements(model, requirements)
    ]
    eligible.sort(key=lambda item: _strategy_key(policy.strategy, item[0], item[1]))

    seen: set[str] = set()
    ordered: list[ModelDescriptor] = []
    for _, model in eligible:
        if model.key not in seen:
            seen.add(model.key)
            ordered.append(model)
    return ordered[: max(policy.max_provider_attempts, 1)]


class ModelRegistry(Protocol):
    async def list_models(self, model_type: ModelType | None = None) -> list[ModelDescriptor]: ...

    async def get(
        self, provider: str, model_name: str, model_type: ModelType
    ) -> ModelDescriptor | None: ...

    async def get_by_id(self, model_id: uuid.UUID) -> ModelDescriptor | None: ...

    async def upsert(self, model: ModelDescriptor, *, overwrite: bool) -> ModelDescriptor: ...

    async def update(self, model_id: uuid.UUID, changes: Mapping[str, Any]) -> ModelDescriptor: ...

    async def record_health(self, model_id: uuid.UUID, status: HealthStatus) -> None: ...

    async def record_latency(self, model_id: uuid.UUID, latency_ms: int) -> None: ...


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker(Protocol):
    """Keeps continuously failing models out of rotation for a while."""

    def allow(self, key: str) -> bool: ...

    def record_success(self, key: str) -> HealthStatus: ...

    def record_failure(self, key: str) -> HealthStatus: ...

    def status(self, key: str) -> HealthStatus: ...

    def snapshot(self) -> dict[str, dict[str, Any]]: ...
