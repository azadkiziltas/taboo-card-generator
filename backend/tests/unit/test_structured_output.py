import pytest

from app.domain.ai.capabilities import (
    ModelCapabilities,
    StructuredOutputMode,
    select_structured_mode,
)
from app.domain.ai.exceptions import StructuredOutputError
from app.domain.ai.structured import extract_json, parse_structured
from app.domain.cards.schemas import GeneratedCardBatch

VALID = (
    '{"cards": [{"target": "Kahve", "forbidden_words": ["a","b","c","d","e"], '
    '"difficulty": "medium"}]}'
)


@pytest.mark.parametrize(
    "text",
    [
        VALID,
        f"```json\n{VALID}\n```",
        f"Sure! Here are your cards:\n{VALID}\nEnjoy.",
        f"<think>let me think about {{this}}</think>{VALID}",
    ],
)
def test_extract_json_tolerates_wrappers(text):
    assert extract_json(text)["cards"][0]["target"] == "Kahve"


@pytest.mark.parametrize("text", ["", "   ", "no json here", '{"cards": [1, 2'])
def test_extract_json_failures(text):
    with pytest.raises(StructuredOutputError):
        extract_json(text)


def test_parse_structured_validates_schema():
    batch = parse_structured(VALID, GeneratedCardBatch)
    assert batch.cards[0].target == "Kahve"
    assert not hasattr(batch.cards[0], "difficulty")  # a model-added difficulty is ignored
    with pytest.raises(StructuredOutputError, match="schema validation failed"):
        parse_structured('{"cards": [{"target": "x"}]}', GeneratedCardBatch)


def test_wrong_forbidden_count_is_not_a_parse_error():
    # Count errors must reach the validator (repairable), not fail the whole batch.
    batch = parse_structured(
        '{"cards": [{"target": "X", "forbidden_words": ["a"], "difficulty": "easy"}]}',
        GeneratedCardBatch,
    )
    assert batch.cards[0].forbidden_words == ["a"]


def test_json_schema_advertises_exact_forbidden_count():
    schema = GeneratedCardBatch.model_json_schema()
    words = schema["$defs"]["GeneratedCard"]["properties"]["forbidden_words"]
    assert words["minItems"] == 5 and words["maxItems"] == 5
    assert "difficulty" not in schema["$defs"]["GeneratedCard"]["properties"]


@pytest.mark.parametrize(
    ("caps", "mode"),
    [
        (
            ModelCapabilities(structured_output=True, json_mode=True),
            StructuredOutputMode.NATIVE_SCHEMA,
        ),
        (ModelCapabilities(json_mode=True), StructuredOutputMode.JSON_MODE),
        (ModelCapabilities(), StructuredOutputMode.PROMPT_ONLY),
    ],
)
def test_capability_based_mode_selection(caps, mode):
    assert select_structured_mode(caps) is mode
