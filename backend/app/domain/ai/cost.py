"""API cost estimation. Prices always come from the model registry."""

from __future__ import annotations

from decimal import Decimal

from app.domain.ai.routing import ModelPricing

_MILLION = Decimal(1_000_000)


def estimate_api_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    pricing: ModelPricing,
) -> Decimal | None:
    """estimated = in/1M * input_price + out/1M * output_price.

    Returns None when the price is unknown or no token counts are available. For local
    models the registry price is 0, so the *API* cost is 0 - which says nothing about the
    real infrastructure cost (hence the `api_estimated_cost` naming in storage).
    """
    if pricing.input_cost_per_million is None and pricing.output_cost_per_million is None:
        return None
    if input_tokens is None and output_tokens is None:
        return None
    cost = Decimal(0)
    if input_tokens and pricing.input_cost_per_million is not None:
        cost += Decimal(input_tokens) / _MILLION * pricing.input_cost_per_million
    if output_tokens and pricing.output_cost_per_million is not None:
        cost += Decimal(output_tokens) / _MILLION * pricing.output_cost_per_million
    return cost.quantize(Decimal("0.00000001"))
