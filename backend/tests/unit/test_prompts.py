import pytest

from app.application.prompts.service import PromptError, PromptService, PromptTemplate
from app.config.settings import APP_ROOT
from app.infrastructure.prompts.file_store import FilePromptStore, parse_prompt_file

STORE = FilePromptStore(APP_ROOT / "prompts")


def test_every_prompt_file_has_consistent_metadata():
    available = STORE.available()
    assert ("card.system", "v1") in available
    assert ("card.generate", "v1") in available
    assert ("card.repair", "v1") in available
    for name, version in available:
        template = STORE.load(name, version)
        assert (template.name, template.version) == (name, version)
        assert template.language_strategy
        assert template.created_at


def test_prompts_do_not_mention_providers_or_models():
    forbidden = ("ollama", "qwen", "openai", "gpt", "claude", "gemini", "llama")
    for name, version in STORE.available():
        body = STORE.load(name, version).body.lower()
        assert not any(term in body for term in forbidden), name


def test_render_substitutes_all_placeholders():
    service = PromptService(STORE)
    rendered = service.render(
        "card.generate",
        "v1",
        count=3,
        language_name="Turkish",
        language_code="tr",
        category="technology",
        difficulty="medium",
        topic="uzay",
        avoid_targets="none",
        reference_context="- roket: uzay aracı",
    )
    assert "Create 3 new" in rendered.text
    assert "{{" not in rendered.text
    assert "<user_topic>uzay</user_topic>" in rendered.text
    assert (rendered.name, rendered.version) == ("card.generate", "v1")


def test_v3_generate_prompt_has_no_difficulty_unless_given():
    service = PromptService(STORE)
    common = dict(
        count=3,
        language_name="Turkish",
        language_code="tr",
        category="food",
        topic="none",
        avoid_targets="none",
        reference_context="-",
    )
    without = service.render("card.generate", "v3", difficulty_section="", **common).text
    assert "difficulty" not in without.lower()
    section = service.render("card.difficulty", "v3", difficulty="easy").text
    with_difficulty = service.render(
        "card.generate", "v3", difficulty_section=section, **common
    ).text
    assert "Difficulty: easy" in with_difficulty


def test_render_rejects_missing_and_unexpected_variables():
    template = PromptTemplate(name="x.y", version="v1", body="Hello {{ name }}")
    with pytest.raises(PromptError, match="missing"):
        template.render({})
    with pytest.raises(PromptError, match="unexpected"):
        template.render({"name": "a", "other": 1})
    assert template.render({"name": "Ada"}) == "Hello Ada"


def test_json_braces_in_templates_are_left_alone():
    template = PromptTemplate(name="x.y", version="v1", body='{"cards": []} {{ n }}')
    assert template.render({"n": 1}) == '{"cards": []} 1'


def test_front_matter_is_required():
    with pytest.raises(PromptError):
        parse_prompt_file("no front matter")
    with pytest.raises(PromptError):
        parse_prompt_file("---\nversion: v1\n---\nbody")


def test_unknown_prompt_and_path_traversal():
    service = PromptService(STORE)
    with pytest.raises(PromptError):
        service.get("card.generate", "v999")
    with pytest.raises(PromptError):
        STORE.load("../etc.passwd", "v1")
