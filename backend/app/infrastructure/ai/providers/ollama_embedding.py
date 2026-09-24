"""Ollama embedding adapter (/api/embed)."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from app.domain.ai.embeddings import EmbeddingResult
from app.domain.ai.exceptions import ProviderError
from app.infrastructure.ai.providers.base import send_json


class OllamaEmbeddingProvider:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        timeout_seconds: float = 60.0,
        name: str = "ollama",
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def embed(
        self,
        *,
        model: str,
        texts: Sequence[str],
        options: Mapping[str, Any] | None = None,
    ) -> EmbeddingResult:
        """`options` are registry `provider_options`, e.g. {"keep_alive": "0"} to unload the
        embedding model right away on memory-constrained machines."""
        started = time.perf_counter()
        payload: dict[str, Any] = {"model": model, "input": list(texts), "truncate": True}
        if options and "keep_alive" in options:
            payload["keep_alive"] = options["keep_alive"]
        data = await send_json(
            self._client,
            "POST",
            f"{self._base_url}/api/embed",
            provider=self._name,
            model=model,
            json=payload,
            request_timeout=self._timeout,
        )
        vectors = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vectors, list) or not vectors or not isinstance(vectors[0], list):
            raise ProviderError(
                "unexpected Ollama embed response", provider=self._name, model=model
            )
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ProviderError(
                "inconsistent embedding dimensions", provider=self._name, model=model
            )
        return EmbeddingResult(
            vectors=[[float(x) for x in vector] for vector in vectors],
            provider=self._name,
            model=model,
            dimension=dimension,
            input_tokens=data.get("prompt_eval_count"),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/version", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False
