"""Application settings.

All values come from environment variables (or a local `.env` file). Model names are
never hard-coded in code: they are provided through the environment and `config/ai.yaml`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = BACKEND_ROOT / "app"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- application -------------------------------------------------------------------
    app_env: Literal["development", "test", "staging", "production"] = "development"
    app_name: str = "taboo-card-backend"
    log_level: str = "INFO"
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # --- database ----------------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://taboo:taboo@localhost:5432/taboo"
    db_echo: bool = False
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # --- AI defaults (names only come from env / ai.yaml) --------------------------------
    default_generation_provider: str = "ollama"
    default_generation_model: str = ""
    default_embedding_provider: str = "ollama"
    default_embedding_model: str = ""
    embedding_version: str = "v1"

    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 180.0

    ai_config_path: Path = BACKEND_ROOT / "config" / "ai.yaml"
    prompts_dir: Path = APP_ROOT / "prompts"
    blocklist_dir: Path = BACKEND_ROOT / "config" / "blocklists"

    model_registry_sync: Literal["insert_missing", "overwrite", "off"] = "insert_missing"
    model_registry_cache_seconds: float = 15.0

    # --- routing / retry limits ----------------------------------------------------------
    max_parse_retries: int = Field(default=1, ge=0, le=5)
    max_repair_attempts: int = Field(default=2, ge=0, le=5)
    max_generation_rounds: int = Field(default=4, ge=1, le=10)
    default_max_provider_attempts: int = Field(default=3, ge=1, le=10)
    circuit_breaker_failure_threshold: int = Field(default=3, ge=1)
    circuit_breaker_reset_seconds: float = Field(default=30.0, gt=0)

    # --- cards ---------------------------------------------------------------------------
    supported_languages: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["tr", "en", "de", "fr", "es", "ar"]
    )
    card_batch_size: int = Field(default=10, ge=1, le=25)
    card_overgenerate_extra: int = Field(default=2, ge=0, le=10)
    max_cards_per_request: int = Field(default=30, ge=1, le=100)
    allow_partial_results: bool = True
    card_generation_temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    card_repair_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    card_max_output_tokens: int | None = Field(default=4096, ge=256)
    card_prompt_version: str = "v3"
    card_max_target_length: int = Field(default=40, ge=2)
    card_max_forbidden_word_length: int = Field(default=40, ge=2)
    avoid_existing_targets_limit: int = Field(default=100, ge=0, le=500)
    duplicate_forbidden_overlap_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # --- RAG -----------------------------------------------------------------------------
    rag_enabled: bool = True
    rag_top_k: int = Field(default=6, ge=1, le=50)
    rag_min_similarity: float = Field(default=0.3, ge=-1.0, le=1.0)
    rag_fail_open: bool = True
    rag_embed_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_max_retries: int = Field(default=1, ge=0, le=3)

    # --- security ------------------------------------------------------------------------
    admin_api_token: SecretStr | None = None

    @field_validator("supported_languages", "cors_allow_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("supported_languages")
    @classmethod
    def _lower_languages(cls, value: list[str]) -> list[str]:
        return [item.lower() for item in value]

    @field_validator("card_max_output_tokens", mode="before")
    @classmethod
    def _empty_as_none(cls, value: object) -> object:
        if value == "" or value == "none":
            return None
        return value

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
