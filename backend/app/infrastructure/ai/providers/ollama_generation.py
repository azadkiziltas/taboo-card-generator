"""Ollama chat adapter (local models, e.g. Qwen). Talks to the REST API via httpx."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from app.domain.ai.exceptions import ProviderError, ProviderUnavailable
from app.domain.ai.generation import ChatMessage
from app.infrastructure.ai.providers.base import (
    BaseGenerationProvider,
    Completion,
    ResponseFormat,
    ResponseFormatKind,
    send_json,
)

# Registry `provider_options` keys that are top-level request fields; every other key is
# passed through as an Ollama model option (num_ctx, top_p, seed, ...).
_TOP_LEVEL_KEYS = frozenset({"think", "keep_alive"})


class OllamaGenerationProvider(BaseGenerationProvider):
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        timeout_seconds: float = 180.0,
        name: str = "ollama",
        default_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(name)
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._default_options = dict(default_options or {})

    async def _complete(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        max_tokens: int | None,
        response_format: ResponseFormat,
        options: Mapping[str, Any],
    ) -> Completion:
        merged = {**self._default_options, **options}
        model_options: dict[str, Any] = {
            key: value for key, value in merged.items() if key not in _TOP_LEVEL_KEYS
        }
        model_options["temperature"] = temperature
        if max_tokens is not None:
            model_options["num_predict"] = max_tokens

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": False,
            "options": model_options,
        }
        for key in _TOP_LEVEL_KEYS & merged.keys():
            payload[key] = merged[key]
        if response_format.kind is ResponseFormatKind.JSON_SCHEMA:
            payload["format"] = response_format.schema
        elif response_format.kind is ResponseFormatKind.JSON_OBJECT:
            payload["format"] = "json"

        data = await send_json(
            self._client,
            "POST",
            f"{self._base_url}/api/chat",
            provider=self.name,
            model=model,
            json=payload,
            request_timeout=self._timeout,
        )
        # A crashed runner (e.g. GPU out of memory) answers HTTP 200 with an `error` field or
        # an unfinished, empty message. Treat that as an outage so fallback/circuit apply.
        if not isinstance(data, dict) or data.get("error") or data.get("done") is False:
            raise ProviderUnavailable(
                "Ollama returned an incomplete response", provider=self.name, model=model
            )
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(
                "unexpected Ollama response", provider=self.name, model=model
            ) from exc
        return Completion(
            content=content or "",
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/version", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_local_models(self) -> list[str]:
        data = await send_json(
            self._client, "GET", f"{self._base_url}/api/tags", provider=self.name, model=None
        )
        return [item.get("name", "") for item in data.get("models", [])]
