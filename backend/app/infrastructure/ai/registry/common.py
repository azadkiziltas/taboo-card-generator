"""Helpers shared by registry implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from decimal import Decimal
from typing import Any

from app.domain.ai.capabilities import ModelCapabilities
from app.domain.ai.routing import ModelDescriptor, ModelPricing

# Fields an administrator may change at runtime.
EDITABLE_FIELDS = frozenset(
    {
        "enabled",
        "priority",
        "is_local",
        "quality_score",
        "input_cost_per_million",
        "output_cost_per_million",
        "currency",
        "languages",
        "provider_options",
        "supports_structured_output",
        "supports_json_mode",
        "supports_tools",
        "supports_vision",
        "supports_streaming",
        "context_window",
    }
)

LATENCY_EMA_WEIGHT = 0.2


def apply_changes(model: ModelDescriptor, changes: Mapping[str, Any]) -> ModelDescriptor:
    unknown = set(changes) - EDITABLE_FIELDS
    if unknown:
        raise ValueError(f"fields not editable: {sorted(unknown)}")
    c = dict(changes)
    pricing = model.pricing
    if {"input_cost_per_million", "output_cost_per_million", "currency"} & c.keys():
        pricing = ModelPricing(
            input_cost_per_million=_dec(
                c.get("input_cost_per_million", pricing.input_cost_per_million)
            ),
            output_cost_per_million=_dec(
                c.get("output_cost_per_million", pricing.output_cost_per_million)
            ),
            currency=c.get("currency", pricing.currency),
        )
    caps = model.capabilities
    caps = ModelCapabilities(
        structured_output=c.get("supports_structured_output", caps.structured_output),
        json_mode=c.get("supports_json_mode", caps.json_mode),
        tool_calling=c.get("supports_tools", caps.tool_calling),
        vision=c.get("supports_vision", caps.vision),
        streaming=c.get("supports_streaming", caps.streaming),
        max_context_tokens=c.get("context_window", caps.max_context_tokens),
    )
    languages = c.get("languages", model.languages)
    return replace(
        model,
        enabled=c.get("enabled", model.enabled),
        priority=c.get("priority", model.priority),
        is_local=c.get("is_local", model.is_local),
        quality_score=c.get("quality_score", model.quality_score),
        provider_options=c.get("provider_options", model.provider_options),
        languages=tuple(languages) if languages else None,
        pricing=pricing,
        capabilities=caps,
    )


def next_latency(current: int | None, observed: int) -> int:
    if current is None:
        return observed
    return int(current * (1 - LATENCY_EMA_WEIGHT) + observed * LATENCY_EMA_WEIGHT)


def _dec(value: Any) -> Decimal | None:
    return Decimal(str(value)) if value is not None else None
