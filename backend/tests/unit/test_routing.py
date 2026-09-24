import pytest

from app.domain.ai.routing import (
    RouteCandidate,
    RoutingPolicySet,
    RoutingRequirements,
    RoutingStrategy,
    TaskRoutingPolicy,
    select_candidates,
)
from tests.helpers import generation_model

LOCAL = generation_model("ollama", "local-a", priority=10, is_local=True)
LOCAL_B = generation_model("ollama", "local-b", priority=30, is_local=True, quality=0.7)
CLOUD = generation_model(
    "cloud_x", "model-x", priority=20, is_local=False, input_price="0.2", quality=0.9, latency=300
)
PROMPT_ONLY = generation_model(
    "cloud_y", "plain", priority=5, structured_output=False, json_mode=False, is_local=False
)
MODELS = [LOCAL, LOCAL_B, CLOUD, PROMPT_ONLY]
ALL_PROVIDERS = {"ollama", "cloud_x", "cloud_y"}


def keys(models):
    return [m.key for m in models]


def run(policy=None, req=None, providers=ALL_PROVIDERS, allowed=None, models=MODELS):
    return keys(
        select_candidates(
            models=models,
            policy=policy or TaskRoutingPolicy(task="t", max_provider_attempts=10),
            requirements=req or RoutingRequirements(),
            available_providers=providers,
            is_allowed=allowed,
        )
    )


def test_registry_priority_when_policy_has_no_candidates():
    assert run() == ["cloud_y:plain", "ollama:local-a", "cloud_x:model-x", "ollama:local-b"]


def test_policy_candidates_define_order_and_membership():
    policy = TaskRoutingPolicy(
        task="t",
        candidates=(
            RouteCandidate("cloud_x", "model-x", 1),
            RouteCandidate("ollama", "local-a", 2),
            RouteCandidate("unknown", "ghost", 3),
        ),
    )
    assert run(policy) == ["cloud_x:model-x", "ollama:local-a"]


def test_structured_output_requirement_excludes_prompt_only_models():
    assert "cloud_y:plain" not in run(req=RoutingRequirements(structured_output=True))


def test_language_and_local_only_filters():
    tr_only = generation_model("ollama", "tr-only", priority=1, languages=("tr",))
    assert "ollama:tr-only" not in run(req=RoutingRequirements(language="en"), models=[tr_only])
    assert run(req=RoutingRequirements(language="tr"), models=[tr_only]) == ["ollama:tr-only"]
    assert run(req=RoutingRequirements(local_only=True)) == ["ollama:local-a", "ollama:local-b"]


def test_disabled_models_missing_adapters_and_open_circuits_are_skipped():
    disabled = generation_model("ollama", "off", priority=1, enabled=False)
    result = run(
        models=[*MODELS, disabled],
        providers={"ollama", "cloud_x"},  # cloud_y adapter not registered (no API key)
        allowed={"ollama:local-a": False},  # circuit open
    )
    assert result == ["cloud_x:model-x", "ollama:local-b"]


@pytest.mark.parametrize(
    ("strategy", "first"),
    [
        (RoutingStrategy.COST, "cloud_y:plain"),  # 0 < 0.2; None price sorts last
        (RoutingStrategy.QUALITY, "cloud_x:model-x"),
        (RoutingStrategy.LATENCY, "cloud_x:model-x"),
        (RoutingStrategy.LOCAL_FIRST, "ollama:local-a"),
    ],
)
def test_strategies(strategy, first):
    policy = TaskRoutingPolicy(task="t", strategy=strategy, max_provider_attempts=10)
    assert run(policy)[0] == first


def test_max_provider_attempts_truncates():
    assert len(run(TaskRoutingPolicy(task="t", max_provider_attempts=2))) == 2


def test_policy_set_falls_back_to_default_policy():
    specific = TaskRoutingPolicy(task="card_generation", max_provider_attempts=1)
    default = TaskRoutingPolicy(task="default", max_provider_attempts=5)
    policies = RoutingPolicySet([specific, default])
    assert policies.for_task("card_generation") is specific
    assert policies.for_task("other") is default
    assert RoutingPolicySet([], default_max_attempts=4).for_task("x").max_provider_attempts == 4


def test_requirements_from_mapping_rejects_unknown_keys():
    assert RoutingRequirements.from_mapping({"language": "tr"}).language == "tr"
    with pytest.raises(ValueError):
        RoutingRequirements.from_mapping({"model": "qwen"})
