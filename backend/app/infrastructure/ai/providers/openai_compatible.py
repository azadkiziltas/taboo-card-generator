"""Generic cloud adapter for providers that expose an OpenAI-compatible
`/chat/completions` endpoint (many hosted LLM vendors do).

This is the extension point for cloud models: configure it in `config/ai.yaml` with a
`base_url` and the *name* of the environment variable that holds the API key. When the
key is missing the provider is simply not registered (no startup crash).

Plain httpx is used on purpose - no vendor SDK is required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from app.domain.ai.exceptions import ProviderError
from app.domain.ai.generation import ChatMessage
from app.infrastructure.ai.providers.base import (
    BaseGenerationProvider,
    Completion,
    ResponseFormat,
    ResponseFormatKind,
    send_json,
)


class OpenAICompatibleGenerationProvider(BaseGenerationProvider):
    def __init__(
        self,
        *,
        name: str,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(name)
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._headers = {"Authorization": f"Bearer {api_key}", **(extra_headers or {})}

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
        payload: dict[str, Any] = {
            **options,
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format.kind is ResponseFormatKind.JSON_SCHEMA:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.schema_name or "response",
                    "schema": response_format.schema,
                },
            }
        elif response_format.kind is ResponseFormatKind.JSON_OBJECT:
            payload["response_format"] = {"type": "json_object"}

        data = await send_json(
            self._client,
            "POST",
            f"{self._base_url}/chat/completions",
            provider=self.name,
            model=model,
            json=payload,
            headers=self._headers,
            request_timeout=self._timeout,
        )
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError(
                "unexpected provider response", provider=self.name, model=model
            ) from exc
        usage = data.get("usage") or {}
        return Completion(
            content=content,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get(
                f"{self._base_url}/models", headers=self._headers, timeout=5.0
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False
