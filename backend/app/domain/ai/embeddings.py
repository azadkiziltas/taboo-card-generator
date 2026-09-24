"""Embedding provider contract, kept fully separate from text generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EmbeddingSpace:
    """Identifies one vector space. Vectors from different spaces must never be mixed.

    Changing the provider, model or version creates a new space that needs re-indexing.
    """

    provider: str
    model: str
    version: str

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}:{self.version}"


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    provider: str
    model: str
    dimension: int
    input_tokens: int | None
    latency_ms: int


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def embed(
        self,
        *,
        model: str,
        texts: Sequence[str],
        options: Mapping[str, Any] | None = None,
    ) -> EmbeddingResult: ...

    async def health_check(self) -> bool: ...
