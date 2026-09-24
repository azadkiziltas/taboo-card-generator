from decimal import Decimal

import pytest

from app.domain.ai.exceptions import (
    NoEligibleModel,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    StructuredOutputError,
)
from app.domain.ai.generation import ChatMessage, MessageRole
from app.domain.ai.routing import HealthStatus
from app.domain.ai.telemetry import AIRequestContext
from app.domain.cards.schemas import GeneratedCardBatch
from app.infrastructure.ai.providers.base import ResponseFormatKind
from app.infrastructure.ai.providers.fake import FakeGenerationProvider
from tests.helpers import build_env, generation_model, policy, tr_batch

MESSAGES = [ChatMessage.system("sys"), ChatMessage.user("make cards")]


def two_provider_env(**kwargs):
    primary = FakeGenerationProvider(provider_name="primary", **kwargs.pop("primary", {}))
    secondary = FakeGenerationProvider(provider_name="secondary", **kwargs.pop("secondary", {}))
    return build_env(
        providers={"primary": primary, "secondary": secondary},
        models=kwargs.pop(
            "models",
            [
                generation_model("primary", "model-a", priority=10),
                generation_model(
                    "secondary",
                    "model-b",
                    priority=20,
                    is_local=False,
                    input_price="1.0",
                    output_price="2.0",
                ),
            ],
        ),
        policies=[policy("card_generation", ("primary", "model-a"), ("secondary", "model-b"))],
    )


async def route(env, **kwargs):
    return await env.container.router.generate_structured(
        task="card_generation",
        messages=MESSAGES,
        schema=GeneratedCardBatch,
        requirements={"language": "tr", "structured_output": True},
        context=AIRequestContext(prompt_name="card.generate", prompt_version="v1"),
        **kwargs,
    )


async def test_primary_success_uses_native_schema():
    env = two_provider_env(primary={"responses": [tr_batch("Kahve")]})
    routed = await route(env)
    assert routed.result.parsed.cards[0].target == "Kahve"
    assert routed.model.key == "primary:model-a" and not routed.fallback_used
    call = env.primary.calls[0]
    assert call.response_format.kind is ResponseFormatKind.JSON_SCHEMA
    assert call.messages == tuple(MESSAGES)  # no extra schema instruction needed
    [record] = env.usage.requests.values()
    assert record.success and record.prompt_name == "card.generate"
    assert record.api_estimated_cost == Decimal(0)


async def test_fallback_to_second_provider_when_primary_is_down():
    env = two_provider_env(
        primary={"responses": [ProviderUnavailable("down")]},
        secondary={"responses": [tr_batch("Uçak")]},
    )
    routed = await route(env)
    assert routed.model.key == "secondary:model-b"
    assert routed.fallback_used and routed.provider_attempt == 2
    records = sorted(env.usage.requests.values(), key=lambda r: r.provider_attempt)
    assert [(r.provider, r.success, r.error_type) for r in records] == [
        ("primary", False, "provider_unavailable"),
        ("secondary", True, None),
    ]
    assert records[1].fallback_used
    # cloud pricing comes from the registry: in=(words) out=(words) tokens > 0 -> cost > 0
    assert records[1].api_estimated_cost > 0
    assert env.breaker.status("primary:model-a") is HealthStatus.DEGRADED


async def test_parse_retry_on_same_model_appends_repair_messages():
    env = two_provider_env(primary={"responses": ["not json at all", tr_batch("Kahve")]})
    routed = await route(env)
    assert routed.model.key == "primary:model-a" and routed.parse_retries == 1
    retry_call = env.primary.calls[1]
    assert retry_call.messages[-2].role is MessageRole.ASSISTANT
    assert retry_call.messages[-2].content == "not json at all"
    assert "could not be used" in retry_call.messages[-1].content
    failed = [r for r in env.usage.requests.values() if not r.success]
    assert failed[0].parse_failed and failed[0].error_type == "structured_output_error"


async def test_parse_retries_exhausted_then_fallback():
    env = two_provider_env(
        primary={"responses": ["garbage", "still garbage"]},
        secondary={"responses": [tr_batch("Tren")]},
    )
    routed = await route(env)
    assert routed.model.key == "secondary:model-b"
    assert len(env.primary.calls) == 2  # 1 try + max_parse_retries(1)
    # parse failures are a quality problem, not a health problem
    assert env.breaker.status("primary:model-a") is HealthStatus.UNKNOWN


async def test_all_providers_fail_raises_normalized_error():
    env = two_provider_env(
        primary={"responses": [ProviderUnavailable("down")]},
        secondary={"responses": [ProviderTimeout("slow")]},
    )
    with pytest.raises(ProviderTimeout):
        await route(env)


async def test_no_eligible_model():
    env = two_provider_env(
        models=[generation_model("primary", "model-a", structured_output=False, json_mode=False)]
    )
    with pytest.raises(NoEligibleModel):
        await route(env)


async def test_open_circuit_skips_model_without_calling_it():
    env = two_provider_env(secondary={"default": tr_batch("Gemi")})
    for _ in range(3):
        env.breaker.record_failure("primary:model-a")
    routed = await route(env)
    assert routed.model.key == "secondary:model-b"
    assert env.primary.calls == []


async def test_prompt_only_model_gets_schema_instruction():
    env = two_provider_env(
        primary={"responses": [f"Here you go:\n```json\n{tr_batch('Kahve')}\n```"]},
        models=[
            generation_model("primary", "model-a", structured_output=False, json_mode=False),
        ],
    )
    routed = await env.container.router.generate_structured(
        task="card_generation", messages=MESSAGES, schema=GeneratedCardBatch
    )
    assert routed.result.parsed.cards[0].target == "Kahve"
    call = env.primary.calls[0]
    assert call.response_format.kind is ResponseFormatKind.TEXT
    assert "JSON Schema" in call.messages[0].content
    assert call.messages[0].content.startswith("sys")


async def test_json_mode_model():
    env = two_provider_env(
        primary={"responses": [tr_batch("Kahve")]},
        models=[generation_model("primary", "model-a", structured_output=False, json_mode=True)],
    )
    await route(env)
    assert env.primary.calls[0].response_format.kind is ResponseFormatKind.JSON_OBJECT


async def test_unexpected_adapter_exception_is_normalized():
    env = two_provider_env(primary={"responses": [KeyError("boom")]})
    env.providers["secondary"].enqueue(ProviderUnavailable("down"))
    with pytest.raises(ProviderError) as info:
        await route(env)
    assert type(info.value) is ProviderUnavailable  # last error re-raised


async def test_generate_text_and_latency_recorded_in_registry():
    env = two_provider_env(primary={"responses": ["plain answer"]})
    routed = await env.container.router.generate_text(task="card_generation", messages=MESSAGES)
    assert routed.result.content == "plain answer" and routed.result.parsed is None
    model = await env.registry.get_by_id(routed.model.id)
    assert model.avg_latency_ms is not None
    assert model.health_status is HealthStatus.HEALTHY


async def test_structured_output_error_keeps_usage_for_telemetry():
    env = two_provider_env(primary={"responses": ["x", "y"]}, secondary={"responses": ["z", "w"]})
    with pytest.raises(StructuredOutputError):
        await route(env)
    assert all(r.input_tokens for r in env.usage.requests.values())
    assert len(env.usage.requests) == 4
