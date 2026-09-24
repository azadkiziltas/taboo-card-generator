"""Provider factory: turns `ai.yaml` provider entries into adapter instances.

Adding a new provider type = implement the adapter + add one entry to the maps below.
Card generation code is not touched.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import httpx

from app.domain.ai.embeddings import EmbeddingProvider
from app.domain.ai.generation import GenerationProvider
from app.infrastructure.ai.config import ProviderConfig
from app.infrastructure.ai.providers.ollama_embedding import OllamaEmbeddingProvider
from app.infrastructure.ai.providers.ollama_generation import OllamaGenerationProvider
from app.infrastructure.ai.providers.openai_compatible import OpenAICompatibleGenerationProvider

logger = logging.getLogger(__name__)


class ProviderConfigError(ValueError):
    pass


def _require_base_url(config: ProviderConfig) -> str:
    if not config.base_url:
        raise ProviderConfigError(f"provider '{config.name}' needs base_url")
    return config.base_url


def _api_key(config: ProviderConfig, env: Mapping[str, str]) -> str:
    if not config.api_key_env:
        raise ProviderConfigError(f"provider '{config.name}' needs api_key_env")
    key = env.get(config.api_key_env, "").strip()
    if not key:
        raise ProviderConfigError(
            f"provider '{config.name}' disabled: {config.api_key_env} is not set"
        )
    return key


def _ollama_generation(cfg, client, env) -> GenerationProvider:
    return OllamaGenerationProvider(
        client=client,
        base_url=_require_base_url(cfg),
        timeout_seconds=cfg.timeout_seconds,
        name=cfg.name,
        default_options=cfg.options,
    )


def _ollama_embedding(cfg, client, env) -> EmbeddingProvider:
    return OllamaEmbeddingProvider(
        client=client,
        base_url=_require_base_url(cfg),
        timeout_seconds=cfg.timeout_seconds,
        name=cfg.name,
    )


def _openai_compatible_generation(cfg, client, env) -> GenerationProvider:
    return OpenAICompatibleGenerationProvider(
        name=cfg.name,
        client=client,
        base_url=_require_base_url(cfg),
        api_key=_api_key(cfg, env),
        timeout_seconds=cfg.timeout_seconds,
        extra_headers=cfg.extra_headers,
    )


Factory = Callable[[ProviderConfig, httpx.AsyncClient, Mapping[str, str]], object]

GENERATION_FACTORIES: dict[str, Factory] = {
    "ollama": _ollama_generation,
    "openai_compatible": _openai_compatible_generation,
}
EMBEDDING_FACTORIES: dict[str, Factory] = {
    "ollama": _ollama_embedding,
}


@dataclass(slots=True)
class ProviderSet:
    generation: dict[str, GenerationProvider] = field(default_factory=dict)
    embedding: dict[str, EmbeddingProvider] = field(default_factory=dict)
    disabled: dict[str, str] = field(default_factory=dict)  # name -> reason


def build_providers(
    configs: Mapping[str, ProviderConfig],
    client: httpx.AsyncClient,
    env: Mapping[str, str],
) -> ProviderSet:
    """Instantiate every enabled provider. Misconfigured providers are disabled, not fatal."""
    result = ProviderSet()
    for name, config in configs.items():
        if not config.enabled:
            result.disabled[name] = "disabled in config"
            continue
        gen_factory = GENERATION_FACTORIES.get(config.type)
        emb_factory = EMBEDDING_FACTORIES.get(config.type)
        if gen_factory is None and emb_factory is None:
            result.disabled[name] = f"unknown provider type '{config.type}'"
            logger.warning("provider %s disabled: unknown type %s", name, config.type)
            continue
        try:
            if gen_factory is not None:
                result.generation[name] = gen_factory(config, client, env)  # type: ignore[assignment]
            if emb_factory is not None:
                result.embedding[name] = emb_factory(config, client, env)  # type: ignore[assignment]
        except ProviderConfigError as exc:
            result.generation.pop(name, None)
            result.disabled[name] = str(exc)
            logger.warning("%s", exc)
    return result
