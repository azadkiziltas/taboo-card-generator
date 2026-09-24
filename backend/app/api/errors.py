"""Maps internal errors to stable, non-leaky public error responses.

Provider names, URLs and provider error bodies never reach the client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.application.prompts.service import PromptError
from app.application.rate_limit import RateLimited
from app.domain.ai.exceptions import (
    AIError,
    EmbeddingError,
    NoEligibleModel,
    ProviderRequestError,
    ProviderTimeout,
    ProviderUnavailable,
    RAGUnavailable,
    RateLimitExceeded,
    StructuredOutputError,
    ValidationExhausted,
)
from app.domain.cards.exceptions import CardNotFound, InvalidCardRequest
from app.domain.errors import StorageError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PublicError:
    status: int
    code: str
    message: str


GENERATION_UNAVAILABLE = PublicError(
    503, "generation_unavailable", "Card generation is temporarily unavailable. Please retry."
)
GENERATION_FAILED = PublicError(
    502, "generation_failed", "Could not generate valid cards. Please retry."
)

# Order matters: the first matching (most specific) class wins.
_ERROR_MAP: list[tuple[type[BaseException], PublicError]] = [
    (CardNotFound, PublicError(404, "card_not_found", "Card not found.")),
    (RateLimited, PublicError(429, "rate_limited", "Too many requests. Please slow down.")),
    (NoEligibleModel, GENERATION_UNAVAILABLE),
    (ProviderTimeout, PublicError(504, "generation_timeout", "Card generation timed out.")),
    (
        RateLimitExceeded,
        PublicError(503, "generation_busy", "Card generation is busy. Retry soon."),
    ),
    (ProviderUnavailable, GENERATION_UNAVAILABLE),
    (ProviderRequestError, GENERATION_FAILED),
    (StructuredOutputError, GENERATION_FAILED),
    (ValidationExhausted, GENERATION_FAILED),
    (EmbeddingError, PublicError(503, "context_unavailable", "Reference data is unavailable.")),
    (RAGUnavailable, PublicError(503, "context_unavailable", "Reference data is unavailable.")),
    (StorageError, PublicError(503, "storage_unavailable", "Service temporarily unavailable.")),
    (AIError, GENERATION_FAILED),
    (PromptError, PublicError(500, "internal_error", "Internal server error.")),
]


def error_body(code: str, message: str, details: list[dict] | None = None) -> dict:
    error: dict = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"error": error}


def to_public(exc: BaseException) -> PublicError:
    for exc_type, public in _ERROR_MAP:
        if isinstance(exc, exc_type):
            return public
    return PublicError(500, "internal_error", "Internal server error.")


async def _handle_known(request: Request, exc: Exception) -> JSONResponse:
    public = to_public(exc)
    log = (
        logger.warning if public.status < 500 or public.status in (502, 503, 504) else logger.error
    )
    log(
        "request failed",
        extra={"path": request.url.path, "error_type": getattr(exc, "code", type(exc).__name__)},
    )
    headers = {}
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        headers["Retry-After"] = str(int(retry_after))
    return JSONResponse(
        status_code=public.status,
        content=error_body(public.code, public.message),
        headers=headers,
    )


async def _handle_invalid_request(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=422, content=error_body("invalid_request", str(exc)))


async def _handle_validation(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        {"field": ".".join(str(p) for p in err["loc"][1:]) or "body", "message": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422, content=error_body("invalid_request", "Invalid request.", details)
    )


async def _handle_http(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = {401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed"}
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code.get(exc.status_code, "http_error"), str(exc.detail)),
        headers=getattr(exc, "headers", None),
    )


async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500, content=error_body("internal_error", "Internal server error.")
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _handle_validation)
    app.add_exception_handler(StarletteHTTPException, _handle_http)
    app.add_exception_handler(InvalidCardRequest, _handle_invalid_request)
    for exc_type in {t for t, _ in _ERROR_MAP}:
        app.add_exception_handler(exc_type, _handle_known)
    app.add_exception_handler(Exception, _handle_unexpected)
