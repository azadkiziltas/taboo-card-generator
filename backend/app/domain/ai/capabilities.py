"""Model capability description and structured-output mode selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    structured_output: bool = False
    json_mode: bool = False
    tool_calling: bool = False
    vision: bool = False
    streaming: bool = False
    max_context_tokens: int | None = None


class StructuredOutputMode(StrEnum):
    """How a structured (schema-bound) answer is requested from a model."""

    NATIVE_SCHEMA = "native_schema"  # provider enforces the JSON schema
    JSON_MODE = "json_mode"  # provider guarantees JSON, schema described in prompt
    PROMPT_ONLY = "prompt_only"  # strict prompt + tolerant JSON parser


def select_structured_mode(capabilities: ModelCapabilities) -> StructuredOutputMode:
    """Pick the strongest structured-output mechanism a model supports."""
    if capabilities.structured_output:
        return StructuredOutputMode.NATIVE_SCHEMA
    if capabilities.json_mode:
        return StructuredOutputMode.JSON_MODE
    return StructuredOutputMode.PROMPT_ONLY
