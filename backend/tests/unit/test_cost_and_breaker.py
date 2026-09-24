from decimal import Decimal

from app.domain.ai.cost import estimate_api_cost
from app.domain.ai.routing import HealthStatus, ModelPricing
from app.infrastructure.ai.routing.circuit_breaker import FailureCounterCircuitBreaker


def test_cost_formula():
    pricing = ModelPricing(Decimal("0.50"), Decimal("1.50"))
    # 2000/1M*0.5 + 1000/1M*1.5 = 0.001 + 0.0015
    assert estimate_api_cost(2000, 1000, pricing) == Decimal("0.0025")


def test_local_model_has_zero_api_cost():
    assert estimate_api_cost(1234, 567, ModelPricing(Decimal(0), Decimal(0))) == Decimal(0)


def test_unknown_price_or_tokens_give_none():
    assert estimate_api_cost(100, 100, ModelPricing()) is None
    assert estimate_api_cost(None, None, ModelPricing(Decimal(1), Decimal(1))) is None


def test_partial_tokens():
    assert estimate_api_cost(1_000_000, None, ModelPricing(Decimal(2), Decimal(8))) == Decimal(2)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_circuit_opens_after_threshold_and_recovers():
    clock = Clock()
    breaker = FailureCounterCircuitBreaker(failure_threshold=2, reset_seconds=10, clock=clock)
    key = "p:m"
    assert breaker.status(key) is HealthStatus.UNKNOWN and breaker.allow(key)

    assert breaker.record_failure(key) is HealthStatus.DEGRADED
    assert breaker.allow(key)
    assert breaker.record_failure(key) is HealthStatus.UNAVAILABLE
    assert not breaker.allow(key)

    clock.now = 11  # after the reset window one trial request is allowed
    assert breaker.allow(key)
    assert breaker.record_failure(key) is HealthStatus.UNAVAILABLE  # trial failed: open again
    assert not breaker.allow(key)

    clock.now = 25
    assert breaker.allow(key)
    assert breaker.record_success(key) is HealthStatus.HEALTHY
    assert breaker.allow(key)
    assert breaker.snapshot()[key] == {"state": "closed", "failures": 0}
