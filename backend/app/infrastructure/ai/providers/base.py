"""Shared adapter plumbing.

Concrete adapters only implement `_complete()` (one chat request in a given response
format) and `health_check()`. Structured output, JSON parsing and HTTP error
normalization are handled here once for every provider.
"""

from __future__ import annotations

import abc
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx
from pydantic import BaseModel

from app.domain.ai.capabilities import StructuredOutputMode
from app.domain.ai.exceptions import (
    ProviderError,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitExceeded,
    StructuredOutputError,
)
from app.domain.ai.generation import ChatMessage, GenerationResult
from app.domain.ai.structured import parse_structured


class ResponseFormatKind(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    kind: ResponseFormatKind = ResponseFormatKind.TEXT
    schema: dict[str, Any] | None = None
    schema_name: str | None = None

    @classmethod
    def for_mode(cls, mode: StructuredOutputMode, schema: type[BaseModel]) -> ResponseFormat:
        if mode is StructuredOutputMode.NATIVE_SCHEMA:
            return cls(ResponseFormatKind.JSON_SCHEMA, schema.model_json_schema(), schema.__name__)
        if mode is StructuredOutputMode.JSON_MODE:
            return cls(ResponseFormatKind.JSON_OBJECT)
        return cls(ResponseFormatKind.TEXT)


@dataclass(frozen=True, slots=True)
class Completion:
    """What an adapter returns from one raw call (already provider-independent)."""

    content: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None


class BaseGenerationProvider(abc.ABC):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    async def _complete(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        max_tokens: int | None,
        response_format: ResponseFormat,
        options: Mapping[str, Any],
    ) -> Completion: ...

    @abc.abstractmethod
    async def health_check(self) -> bool: ...

    async def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        max_tokens: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        started = time.perf_counter()
        completion = await self._complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=ResponseFormat(),
            options=options or {},
        )
        return self._result(model, completion, started, parsed=None)

    async def generate_structured(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        schema: type[BaseModel],
        temperature: float,
        max_tokens: int | None = None,
        mode: StructuredOutputMode = StructuredOutputMode.NATIVE_SCHEMA,
        options: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        started = time.perf_counter()
        completion = await self._complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=ResponseFormat.for_mode(mode, schema),
            options=options or {},
        )
        result = self._result(model, completion, started, parsed=None)
        if completion.finish_reason == "length":
            raise StructuredOutputError(
                "answer was truncated (max tokens reached)",
                raw_content=completion.content,
                result=result,
            )
        try:
            parsed = parse_structured(completion.content, schema)
        except StructuredOutputError as exc:
            exc.result = result
            raise
        return self._result(model, completion, started, parsed=parsed)

    def _result(
        self, model: str, completion: Completion, started: float, *, parsed: Any
    ) -> GenerationResult:
        return GenerationResult(
            content=completion.content,
            parsed=parsed,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=int((time.perf_counter() - started) * 1000),
            provider=self._name,
            model=model,
            raw_finish_reason=completion.finish_reason,
        )


async def send_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    provider: str,
    model: str | None,
    json: Any | None = None,
    headers: Mapping[str, str] | None = None,
    request_timeout: float | None = None,
) -> Any:
    """Perform an HTTP call and translate every failure into a normalized ProviderError.

    Provider error bodies are deliberately not copied into exception messages that
    could reach clients; only the status code is kept.
    """
    try:
        response = await client.request(
            method, url, json=json, headers=dict(headers or {}), timeout=request_timeout
        )
    except httpx.TimeoutException as exc:
        raise ProviderTimeout("provider timed out", provider=provider, model=model) from exc
    except httpx.TransportError as exc:
        raise ProviderUnavailable(
            "provider is unreachable", provider=provider, model=model
        ) from exc

    status = response.status_code
    if status == 429:
        retry_after = _retry_after(response.headers.get("retry-after"))
        raise RateLimitExceeded(
            "provider rate limit", provider=provider, model=model, retry_after=retry_after
        )
    if status in (408, 504):
        raise ProviderTimeout(f"provider timeout ({status})", provider=provider, model=model)
    if status >= 500:
        raise ProviderUnavailable(f"provider error ({status})", provider=provider, model=model)
    if status >= 400:
        raise ProviderRequestError(
            f"provider rejected request ({status})", provider=provider, model=model, status=status
        )
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(
            "provider returned invalid JSON", provider=provider, model=model
        ) from exc


def _retry_after(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None
