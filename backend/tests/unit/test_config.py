import httpx

from app.config.settings import BACKEND_ROOT
from app.domain.ai.routing import ModelType, RoutingStrategy
from app.infrastructure.ai.config import ProviderConfig, interpolate, load_ai_config
from app.infrastructure.ai.providers.factory import build_providers
from tests.helpers import make_settings

ENV = {
    "DEFAULT_GENERATION_PROVIDER": "ollama",
    "DEFAULT_GENERATION_MODEL": "gen-model:7b",
    "DEFAULT_EMBEDDING_PROVIDER": "ollama",
    "DEFAULT_EMBEDDING_MODEL": "embed-model",
    "OLLAMA_BASE_URL": "http://ollama.local:11434",
}


def test_interpolation_with_defaults():
    assert interpolate("${A}-${B:-x}-${C:-}", {"A": "1"}) == "1-x-"
    assert interpolate("${A:-fallback}", {"A": ""}) == "fallback"


def test_shipped_ai_yaml_loads_without_hardcoded_models():
    config = load_ai_config(BACKEND_ROOT / "config" / "ai.yaml", ENV)

    assert config.providers["ollama"].base_url == "http://ollama.local:11434"
    keys = {(m.provider, m.model_name, m.model_type) for m in config.models}
    assert ("ollama", "gen-model:7b", ModelType.GENERATION) in keys
    assert ("ollama", "embed-model", ModelType.EMBEDDING) in keys
    assert len(config.models) == 2  # empty FALLBACK_GENERATION_MODEL is skipped

    gen = next(m for m in config.models if m.model_type is ModelType.GENERATION)
    assert gen.is_local and gen.capabilities.structured_output
    assert gen.provider_options["think"] is False
    assert gen.quality_score is None  # never invented

    routing = {p.task: p for p in config.routing}
    card = routing["card_generation"]
    assert card.strategy is RoutingStrategy.PRIORITY
    assert [(c.provider, c.model) for c in card.candidates] == [("ollama", "gen-model:7b")]


def test_fallback_model_from_env_is_routed_second():
    config = load_ai_config(
        BACKEND_ROOT / "config" / "ai.yaml", {**ENV, "FALLBACK_GENERATION_MODEL": "small:3b"}
    )
    card = next(p for p in config.routing if p.task == "card_generation")
    assert [c.model for c in card.candidates] == ["gen-model:7b", "small:3b"]


def test_cloud_provider_without_api_key_is_disabled_not_fatal():
    configs = {
        "ollama": ProviderConfig(name="ollama", type="ollama", base_url="http://o"),
        "cloud_a": ProviderConfig(
            name="cloud_a", type="openai_compatible", base_url="https://api", api_key_env="CLOUD_A"
        ),
        "weird": ProviderConfig(name="weird", type="does-not-exist"),
    }
    client = httpx.AsyncClient()
    providers = build_providers(configs, client, env={})
    assert set(providers.generation) == {"ollama"}
    assert set(providers.embedding) == {"ollama"}
    assert "CLOUD_A is not set" in providers.disabled["cloud_a"]
    assert "unknown provider type" in providers.disabled["weird"]

    with_key = build_providers(configs, client, env={"CLOUD_A": "secret"})
    assert set(with_key.generation) == {"ollama", "cloud_a"}


def test_settings_parse_csv_lists():
    settings = make_settings(supported_languages="TR, en ,de", cors_allow_origins="")
    assert settings.supported_languages == ["tr", "en", "de"]
    assert settings.cors_allow_origins == []
