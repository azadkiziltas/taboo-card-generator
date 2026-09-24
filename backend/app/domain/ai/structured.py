"""Tolerant JSON extraction + schema validation for model answers."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from app.domain.ai.exceptions import StructuredOutputError

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Extract the first JSON value from a model answer.

    Handles plain JSON, markdown code fences, leading/trailing prose and
    `<think>` blocks emitted by reasoning models.
    """
    if not text or not text.strip():
        raise StructuredOutputError("empty response", raw_content=text)

    cleaned = _THINK_RE.sub("", text).strip()
    candidates: list[str] = [cleaned]
    candidates.extend(match.strip() for match in _FENCE_RE.findall(cleaned))

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for index, char in enumerate(candidate):
            if char in "{[":
                try:
                    value, _ = decoder.raw_decode(candidate, index)
                    return value
                except json.JSONDecodeError:
                    continue
    raise StructuredOutputError("response does not contain valid JSON", raw_content=text)


def parse_structured[T: BaseModel](text: str, schema: type[T]) -> T:
    """Parse `text` into `schema` or raise `StructuredOutputError`."""
    data = extract_json(text)
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        summary = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()[:10]
        )
        raise StructuredOutputError(
            f"schema validation failed: {summary}", raw_content=text
        ) from exc


def json_schema_for(schema: type[BaseModel]) -> dict[str, Any]:
    return schema.model_json_schema()
