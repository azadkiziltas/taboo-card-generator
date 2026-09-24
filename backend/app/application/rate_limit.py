"""Rate limiting extension point (MVP: no-op; later Redis/token bucket)."""

from __future__ import annotations

from typing import Protocol


class RateLimited(Exception):
    code = "rate_limited"

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("rate limited")
        self.retry_after = retry_after


class RateLimiter(Protocol):
    async def check(self, key: str, *, cost: int = 1) -> None:
        """Raise `RateLimited` when `key` exceeded its budget."""
        ...


class NoopRateLimiter:
    async def check(self, key: str, *, cost: int = 1) -> None:
        return None
