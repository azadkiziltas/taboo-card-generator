"""Loads `config/ai.yaml`: provider definitions, model registry seed and routing policies.

`${VAR}` and `${VAR:-default}` placeholders are resolved from the environment so model
names stay out of the code and out of the committed config.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

from app.config.settings import Settings
from app.domain.ai.capabilities import ModelCapabilities
from app.domain.ai.routing import (
    ModelDescriptor,
    ModelPricing,
    ModelType,
    RouteCandidate,
    RoutingStrategy,
    TaskRoutingPolicy,
)

logger = logging.getLogger(__name__)

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class AIConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: str
    type: str
    enabled: bool = True
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    options: Mapping[str, Any] = field(default_factory=dict)
    extra_headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AIConfig:
    providers: dict[str, ProviderConfig]
    models: list[ModelDescriptor]
    routing: list[TaskRoutingPolicy]


def build_environment(settings: Settings, env_file: str | Path = ".env") -> dict[str, str]:
    """Settings values < .env file < real environment variables."""
    values: dict[str, str] = {}
    for key, value in settings.model_dump().items():
        if isinstance(value, str | int | float | bool) and value is not None:
            values[key.upper()] = str(value).lower() if isinstance(value, bool) else str(value)
    if Path(env_file).is_file():
        values.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    values.update(os.environ)
    return values


def interpolate(text: str, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = env.get(name)
        if value is None or value == "":
            return default if default is not None else ""
        return value

    return _ENV_REF.sub(replace, text)


def load_ai_config(path: Path, env: Mapping[str, str]) -> AIConfig:
    if not path.is_file():
        raise AIConfigError(f"AI config not found: {path}")
    raw = yaml.safe_load(interpolate(path.read_text(encoding="utf-8"), env)) or {}
    return AIConfig(
        providers=_parse_providers(raw.get("providers") or {}),
        models=_parse_models(raw.get("models") or []),
        routing=_parse_routing(raw.get("routing") or {}),
    )


def _parse_providers(data: Mapping[str, Any]) -> dict[str, ProviderConfig]:
    providers: dict[str, ProviderConfig] = {}
    for name, item in data.items():
        if not isinstance(item, Mapping) or "type" not in item:
            raise AIConfigError(f"provider '{name}' needs a 'type'")
        providers[name] = ProviderConfig(
            name=name,
            type=str(item["type"]),
            enabled=_bool(item.get("enabled", True)),
            base_url=item.get("base_url") or None,
            api_key_env=item.get("api_key_env") or None,
            timeout_seconds=float(item.get("timeout_seconds") or 60.0),
            options=dict(item.get("options") or {}),
            extra_headers=dict(item.get("extra_headers") or {}),
        )
    return providers


def _parse_models(items: list[Mapping[str, Any]]) -> list[ModelDescriptor]:
    models: list[ModelDescriptor] = []
    for item in items:
        provider = str(item.get("provider") or "").strip()
        model_name = str(item.get("model_name") or "").strip()
        if not provider or not model_name:
            logger.info("skipping model registry entry without model_name (provider=%s)", provider)
            continue
        caps = item.get("capabilities") or {}
        pricing = item.get("pricing") or {}
        languages = item.get("languages")
        models.append(
            ModelDescriptor(
                provider=provider,
                model_name=model_name,
                model_type=ModelType(item.get("model_type", "generation")),
                enabled=_bool(item.get("enabled", True)),
                is_local=_bool(item.get("is_local", False)),
                priority=int(item.get("priority", 100)),
                pricing=ModelPricing(
                    input_cost_per_million=_decimal(pricing.get("input_cost_per_million")),
                    output_cost_per_million=_decimal(pricing.get("output_cost_per_million")),
                    currency=str(pricing.get("currency", "USD")),
                ),
                capabilities=ModelCapabilities(
                    structured_output=_bool(caps.get("structured_output", False)),
                    json_mode=_bool(caps.get("json_mode", False)),
                    tool_calling=_bool(caps.get("tool_calling", False)),
                    vision=_bool(caps.get("vision", False)),
                    streaming=_bool(caps.get("streaming", False)),
                    max_context_tokens=_int(item.get("context_window")),
                ),
                languages=tuple(languages) if languages else None,
                quality_score=None,  # only set from our own benchmarks, never invented
                provider_options=dict(item.get("provider_options") or {}),
            )
        )
    return models


def _parse_routing(data: Mapping[str, Any]) -> list[TaskRoutingPolicy]:
    policies: list[TaskRoutingPolicy] = []
    for task, item in data.items():
        item = item or {}
        candidates = []
        for entry in item.get("providers") or []:
            provider = str(entry.get("provider") or "").strip()
            model = str(entry.get("model") or "").strip()
            if provider and model:
                candidates.append(RouteCandidate(provider, model, int(entry.get("priority", 100))))
        policies.append(
            TaskRoutingPolicy(
                task=task,
                candidates=tuple(candidates),
                strategy=RoutingStrategy(item.get("strategy", "priority")),
                max_provider_attempts=int(item.get("max_provider_attempts", 3)),
            )
        )
    return policies


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int(value: Any) -> int | None:
    return int(value) if value not in (None, "") else None


def _decimal(value: Any) -> Decimal | None:
    return Decimal(str(value)) if value not in (None, "") else None
