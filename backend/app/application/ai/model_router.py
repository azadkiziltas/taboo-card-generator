"""Model router: the only entry point the application uses for text generation.

Callers describe *what* they need (task + requirements); the router decides *which*
provider/model runs it, normalizes capability differences, retries unparsable answers,
falls back to the next model and records telemetry for every call.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.application.prompts.service import PromptService
from app.domain.ai.capabilities import StructuredOutputMode, select_structured_mode
from app.domain.ai.cost import estimate_api_cost
from app.domain.ai.exceptions import (
    AIError,
    NoEligibleModel,
    ProviderError,
    StructuredOutputError,
)
from app.domain.ai.generation import ChatMessage, GenerationProvider, GenerationResult, MessageRole
from app.domain.ai.routing import (
    CircuitBreaker,
    HealthStatus,
    ModelDescriptor,
    ModelRegistry,
    ModelType,
    RoutingPolicySet,
    RoutingRequirements,
    select_candidates,
)
from app.domain.ai.telemetry import AIRequestContext, AIRequestRecord, AIUsageRecorder

logger = logging.getLogger(__name__)

_MAX_ECHOED_ANSWER_CHARS = 6000


@dataclass(frozen=True, slots=True)
class RouterLimits:
    max_parse_retries: int = 1
    json_prompt_version: str = "v1"


@dataclass(frozen=True, slots=True)
class RoutedResult:
    result: GenerationResult
    request_id: uuid.UUID
    model: ModelDescriptor
    provider_attempt: int
    parse_retries: int

    @property
    def fallback_used(self) -> bool:
        return self.provider_attempt > 1


class ModelRouter:
    def __init__(
        self,
        *,
        registry: ModelRegistry,
        providers: Mapping[str, GenerationProvider],
        policies: RoutingPolicySet,
        circuit_breaker: CircuitBreaker,
        usage_recorder: AIUsageRecorder,
        prompts: PromptService,
        limits: RouterLimits | None = None,
    ) -> None:
        self._registry = registry
        self._providers = dict(providers)
        self._policies = policies
        self._breaker = circuit_breaker
        self._usage = usage_recorder
        self._prompts = prompts
        self._limits = limits or RouterLimits()

    @property
    def provider_names(self) -> list[str]:
        return sorted(self._providers)

    def provider(self, name: str) -> GenerationProvider | None:
        return self._providers.get(name)

    async def candidates(
        self, task: str, requirements: RoutingRequirements | Mapping[str, Any] | None = None
    ) -> list[ModelDescriptor]:
        req = _as_requirements(requirements)
        models = await self._registry.list_models(ModelType.GENERATION)
        allowed = {m.key: self._breaker.allow(m.key) for m in models}
        return select_candidates(
            models=models,
            policy=self._policies.for_task(task),
            requirements=req,
            available_providers=self._providers.keys(),
            is_allowed=allowed,
        )

    async def generate_structured(
        self,
        *,
        task: str,
        messages: Sequence[ChatMessage],
        schema: type[BaseModel],
        requirements: RoutingRequirements | Mapping[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        context: AIRequestContext | None = None,
    ) -> RoutedResult:
        return await self._route(
            task=task,
            messages=messages,
            schema=schema,
            requirements=requirements,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context or AIRequestContext(),
        )

    async def generate_text(
        self,
        *,
        task: str,
        messages: Sequence[ChatMessage],
        requirements: RoutingRequirements | Mapping[str, Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        context: AIRequestContext | None = None,
    ) -> RoutedResult:
        return await self._route(
            task=task,
            messages=messages,
            schema=None,
            requirements=requirements,
            temperature=temperature,
            max_tokens=max_tokens,
            context=context or AIRequestContext(),
        )

    # -- internals ------------------------------------------------------------------------

    async def _route(
        self,
        *,
        task: str,
        messages: Sequence[ChatMessage],
        schema: type[BaseModel] | None,
        requirements: RoutingRequirements | Mapping[str, Any] | None,
        temperature: float,
        max_tokens: int | None,
        context: AIRequestContext,
    ) -> RoutedResult:
        candidates = await self.candidates(task, requirements)
        if not candidates:
            raise NoEligibleModel(f"no eligible model for task '{task}'", task=task)

        last_error: AIError | None = None
        for attempt, model in enumerate(candidates, start=1):
            provider = self._providers[model.provider]
            try:
                return await self._call_model(
                    task=task,
                    provider=provider,
                    model=model,
                    messages=messages,
                    schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=context,
                    attempt=attempt,
                )
            except StructuredOutputError as exc:
                last_error = exc
                logger.warning(
                    "structured output failed, trying next model",
                    extra={"task": task, "route": model.key, "attempt": attempt},
                )
            except ProviderError as exc:
                last_error = exc
                await self._on_failure(model)
                logger.warning(
                    "provider call failed, trying next model",
                    extra={
                        "task": task,
                        "route": model.key,
                        "attempt": attempt,
                        "error_type": exc.code,
                    },
                )
        assert last_error is not None
        raise last_error

    async def _call_model(
        self,
        *,
        task: str,
        provider: GenerationProvider,
        model: ModelDescriptor,
        messages: Sequence[ChatMessage],
        schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int | None,
        context: AIRequestContext,
        attempt: int,
    ) -> RoutedResult:
        mode = select_structured_mode(model.capabilities)
        conversation = list(messages)
        if schema is not None and mode is not StructuredOutputMode.NATIVE_SCHEMA:
            conversation = self._with_schema_instruction(conversation, schema)

        max_parse_retries = self._limits.max_parse_retries if schema is not None else 0
        for parse_try in range(max_parse_retries + 1):
            started = time.perf_counter()
            try:
                result = await self._invoke(
                    provider, model, conversation, schema, temperature, max_tokens, mode
                )
            except StructuredOutputError as exc:
                await self._record(
                    task,
                    model,
                    context,
                    attempt,
                    parse_try,
                    started,
                    result=exc.result,
                    error=exc,
                    parse_failed=True,
                )
                if parse_try >= max_parse_retries:
                    raise
                conversation = self._with_json_repair(conversation, exc)
                continue
            except ProviderError as exc:
                await self._record(task, model, context, attempt, parse_try, started, error=exc)
                raise
            except Exception as exc:  # adapter bug: never leak raw exceptions
                logger.exception("unexpected provider error", extra={"route": model.key})
                error = ProviderError(
                    "unexpected provider error", provider=model.provider, model=model.model_name
                )
                await self._record(task, model, context, attempt, parse_try, started, error=error)
                raise error from exc

            request_id = await self._record(
                task, model, context, attempt, parse_try, started, result=result
            )
            await self._on_success(model, result.latency_ms)
            return RoutedResult(
                result=result,
                request_id=request_id,
                model=model,
                provider_attempt=attempt,
                parse_retries=parse_try,
            )
        raise AssertionError("unreachable")  # pragma: no cover

    async def _invoke(
        self,
        provider: GenerationProvider,
        model: ModelDescriptor,
        messages: list[ChatMessage],
        schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int | None,
        mode: StructuredOutputMode,
    ) -> GenerationResult:
        options = dict(model.provider_options)
        if schema is None:
            return await provider.generate_text(
                model=model.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                options=options,
            )
        return await provider.generate_structured(
            model=model.model_name,
            messages=messages,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            mode=mode,
            options=options,
        )

    def _with_schema_instruction(
        self, messages: list[ChatMessage], schema: type[BaseModel]
    ) -> list[ChatMessage]:
        instruction = self._prompts.render(
            "common.json_schema_instruction",
            self._limits.json_prompt_version,
            schema=json.dumps(schema.model_json_schema(), ensure_ascii=False),
        ).text
        if messages and messages[0].role is MessageRole.SYSTEM:
            merged = ChatMessage.system(f"{messages[0].content}\n\n{instruction}")
            return [merged, *messages[1:]]
        return [ChatMessage.system(instruction), *messages]

    def _with_json_repair(
        self, messages: list[ChatMessage], error: StructuredOutputError
    ) -> list[ChatMessage]:
        repair = self._prompts.render(
            "common.json_repair",
            self._limits.json_prompt_version,
            error=error.message[:500],
        ).text
        previous = (error.raw_content or "")[:_MAX_ECHOED_ANSWER_CHARS]
        return [*messages, ChatMessage.assistant(previous), ChatMessage.user(repair)]

    async def _record(
        self,
        task: str,
        model: ModelDescriptor,
        context: AIRequestContext,
        attempt: int,
        parse_try: int,
        started: float,
        *,
        result: GenerationResult | None = None,
        error: AIError | None = None,
        parse_failed: bool = False,
    ) -> uuid.UUID:
        latency_ms = result.latency_ms if result else int((time.perf_counter() - started) * 1000)
        input_tokens = result.input_tokens if result else None
        output_tokens = result.output_tokens if result else None
        record = AIRequestRecord(
            task=task,
            provider=model.provider,
            model=model.model_name,
            is_local=model.is_local,
            success=error is None,
            latency_ms=latency_ms,
            run_id=context.run_id,
            prompt_name=context.prompt_name,
            prompt_version=context.prompt_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_estimated_cost=estimate_api_cost(input_tokens, output_tokens, model.pricing),
            currency=model.pricing.currency,
            error_type=error.code if error else None,
            retry_count=parse_try,
            provider_attempt=attempt,
            fallback_used=attempt > 1,
            parse_failed=parse_failed,
        )
        try:
            await self._usage.record_request(record)
        except Exception:  # telemetry must never break generation
            logger.exception("failed to record AI request telemetry")
        return record.id

    async def _on_success(self, model: ModelDescriptor, latency_ms: int) -> None:
        status = self._breaker.record_success(model.key)
        await self._sync_registry(model, status, latency_ms)

    async def _on_failure(self, model: ModelDescriptor) -> None:
        status = self._breaker.record_failure(model.key)
        await self._sync_registry(model, status, None)

    async def _sync_registry(
        self, model: ModelDescriptor, status: HealthStatus, latency_ms: int | None
    ) -> None:
        try:
            if status is not model.health_status:
                await self._registry.record_health(model.id, status)
            if latency_ms is not None:
                await self._registry.record_latency(model.id, latency_ms)
        except Exception:
            logger.exception("failed to update model registry health/latency")


def _as_requirements(
    value: RoutingRequirements | Mapping[str, Any] | None,
) -> RoutingRequirements:
    if isinstance(value, RoutingRequirements):
        return value
    return RoutingRequirements.from_mapping(value)
