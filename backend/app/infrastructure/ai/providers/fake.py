"""Deterministic fake providers so application logic can be tested without any LLM."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.domain.ai.embeddings import EmbeddingResult
from app.domain.ai.exceptions import ProviderUnavailable
from app.domain.ai.generation import ChatMessage
from app.infrastructure.ai.providers.base import (
    BaseGenerationProvider,
    Completion,
    ResponseFormat,
)


@dataclass(frozen=True, slots=True)
class FakeCall:
    model: str
    messages: tuple[ChatMessage, ...]
    temperature: float
    max_tokens: int | None
    response_format: ResponseFormat
    options: Mapping[str, Any]

    @property
    def prompt(self) -> str:
        return "\n".join(m.content for m in self.messages)


FakeResponse = str | BaseException | Callable[[FakeCall], str]


@dataclass
class FakeGenerationProvider(BaseGenerationProvider):
    """Returns scripted responses in order; falls back to `default` when exhausted.

    A response may be a string (the model answer), an exception instance (raised, e.g.
    `ProviderUnavailable()`), or a callable that builds the answer from the call.
    """

    provider_name: str = "fake"
    responses: Iterable[FakeResponse] = ()
    default: FakeResponse | None = None
    healthy: bool = True
    calls: list[FakeCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        BaseGenerationProvider.__init__(self, self.provider_name)
        self._queue: deque[FakeResponse] = deque(self.responses)

    def enqueue(self, *responses: FakeResponse) -> None:
        self._queue.extend(responses)

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
        call = FakeCall(model, tuple(messages), temperature, max_tokens, response_format, options)
        self.calls.append(call)
        response = self._queue.popleft() if self._queue else self.default
        if response is None:
            raise ProviderUnavailable("fake provider has no scripted response", provider=self.name)
        if isinstance(response, BaseException):
            if hasattr(response, "provider") and response.provider is None:
                response.provider = self.name  # type: ignore[attr-defined]
            raise response
        content = response(call) if callable(response) else response
        return Completion(
            content=content,
            input_tokens=len(call.prompt.split()),
            output_tokens=len(content.split()),
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        return self.healthy


_TOKEN = re.compile(r"\w+", re.UNICODE)


class FakeEmbeddingProvider:
    """Hashing bag-of-words embeddings: texts sharing words get similar vectors."""

    def __init__(self, *, name: str = "fake", dimension: int = 64, healthy: bool = True) -> None:
        self._name = name
        self.dimension = dimension
        self.healthy = healthy
        self.fail = False
        self.fail_times = 0  # fail only the next N calls (transient errors)
        self.calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return self._name

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall(text.casefold()):
            digest = hashlib.sha256(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    async def embed(
        self, *, model: str, texts: Sequence[str], options: Mapping[str, Any] | None = None
    ) -> EmbeddingResult:
        started = time.perf_counter()
        self.calls.append(list(texts))
        if self.fail or self.fail_times > 0:
            self.fail_times = max(self.fail_times - 1, 0)
            raise ProviderUnavailable("fake embedding failure", provider=self._name, model=model)
        return EmbeddingResult(
            vectors=[self.vector(text) for text in texts],
            provider=self._name,
            model=model,
            dimension=self.dimension,
            input_tokens=sum(len(t.split()) for t in texts),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def health_check(self) -> bool:
        return self.healthy
