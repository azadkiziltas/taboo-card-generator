"""Generation provider contract and normalized result types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from app.domain.ai.capabilities import StructuredOutputMode


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(MessageRole.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(MessageRole.USER, content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(MessageRole.ASSISTANT, content)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Provider-independent result. Raw provider responses never leave the adapter."""

    content: str
    parsed: Any | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    provider: str
    model: str
    raw_finish_reason: str | None = None


@runtime_checkable
class GenerationProvider(Protocol):
    """Every text-generation backend (local or cloud) implements this contract."""

    @property
    def name(self) -> str: ...

    async def generate_text(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        max_tokens: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> GenerationResult: ...

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
        """Return a result whose `parsed` is an instance of `schema`.

        Raises `StructuredOutputError` when the answer cannot be parsed/validated.
        """
        ...

    async def health_check(self) -> bool: ...
