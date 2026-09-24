"""Normalized AI errors.

Provider adapters translate SDK/HTTP specific failures into these types so that
provider-specific exceptions never leak into the application layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.ai.generation import GenerationResult


class AIError(Exception):
    """Base class for every normalized AI error."""

    code: str = "ai_error"

    def __init__(self, message: str = "", **context: object) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.context = context


class ProviderError(AIError):
    """A provider call failed. `provider`/`model` are for internal telemetry only."""

    code = "provider_error"

    def __init__(
        self,
        message: str = "",
        *,
        provider: str | None = None,
        model: str | None = None,
        **context: object,
    ) -> None:
        super().__init__(message, **context)
        self.provider = provider
        self.model = model


class ProviderUnavailable(ProviderError):
    code = "provider_unavailable"


class ProviderTimeout(ProviderError):
    code = "provider_timeout"


class RateLimitExceeded(ProviderError):
    code = "rate_limit_exceeded"

    def __init__(self, message: str = "", *, retry_after: float | None = None, **kw: object):
        super().__init__(message, **kw)  # type: ignore[arg-type]
        self.retry_after = retry_after


class ProviderRequestError(ProviderError):
    """Provider rejected the request (bad parameters, unknown model, auth, ...)."""

    code = "provider_request_error"


class StructuredOutputError(AIError):
    """The model answered but the answer could not be parsed into the expected schema."""

    code = "structured_output_error"

    def __init__(
        self,
        message: str = "",
        *,
        raw_content: str | None = None,
        result: GenerationResult | None = None,
        **context: object,
    ) -> None:
        super().__init__(message, **context)
        self.raw_content = raw_content
        self.result = result


class ValidationExhausted(AIError):
    """Generated content kept failing validation after all repair/regeneration attempts."""

    code = "validation_exhausted"


class NoEligibleModel(AIError):
    """No enabled/healthy model satisfies the routing requirements."""

    code = "no_eligible_model"


class EmbeddingError(AIError):
    code = "embedding_error"


class RAGUnavailable(AIError):
    code = "rag_unavailable"
