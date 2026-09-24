"""In-process failure-counter circuit breaker (MVP).

The `CircuitBreaker` protocol allows swapping this for a shared (e.g. Redis) breaker
when the API runs with several workers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.domain.ai.routing import CircuitState, HealthStatus


@dataclass(slots=True)
class _Entry:
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    opened_at: float = 0.0


class FailureCounterCircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        reset_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._reset = reset_seconds
        self._clock = clock
        self._entries: dict[str, _Entry] = {}

    def allow(self, key: str) -> bool:
        entry = self._entries.get(key)
        if entry is None or entry.state is CircuitState.CLOSED:
            return True
        if entry.state is CircuitState.OPEN and self._clock() - entry.opened_at >= self._reset:
            entry.state = CircuitState.HALF_OPEN  # let one trial request through
            return True
        return entry.state is CircuitState.HALF_OPEN

    def record_success(self, key: str) -> HealthStatus:
        self._entries[key] = _Entry()
        return HealthStatus.HEALTHY

    def record_failure(self, key: str) -> HealthStatus:
        entry = self._entries.setdefault(key, _Entry())
        entry.failures += 1
        if entry.state is CircuitState.HALF_OPEN or entry.failures >= self._threshold:
            entry.state = CircuitState.OPEN
            entry.opened_at = self._clock()
        return self.status(key)

    def status(self, key: str) -> HealthStatus:
        entry = self._entries.get(key)
        if entry is None:
            return HealthStatus.UNKNOWN
        if entry.state is CircuitState.OPEN:
            return HealthStatus.UNAVAILABLE
        if entry.state is CircuitState.HALF_OPEN or entry.failures:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            key: {"state": entry.state.value, "failures": entry.failures}
            for key, entry in self._entries.items()
        }
