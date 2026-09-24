"""Guards the dependency rules of the architecture."""

import ast
from pathlib import Path

import pytest

from app.config.settings import APP_ROOT

FORBIDDEN_IN_DOMAIN = (
    "fastapi",
    "sqlalchemy",
    "httpx",
    "ollama",
    "openai",
    "anthropic",
    "asyncpg",
    "pgvector",
    "app.infrastructure",
    "app.api",
    "app.application",
)
FORBIDDEN_IN_APPLICATION = (
    "fastapi",
    "sqlalchemy",
    "httpx",
    "ollama",
    "openai",
    "anthropic",
    "asyncpg",
    "pgvector",
    "app.infrastructure",
    "app.api",
)


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def python_files(layer: str) -> list[Path]:
    return sorted((APP_ROOT / layer).rglob("*.py"))


@pytest.mark.parametrize(
    ("layer", "forbidden"),
    [("domain", FORBIDDEN_IN_DOMAIN), ("application", FORBIDDEN_IN_APPLICATION)],
)
def test_layer_does_not_import_frameworks_or_sdks(layer, forbidden):
    violations = [
        f"{path.relative_to(APP_ROOT)} imports {name}"
        for path in python_files(layer)
        for name in imports_of(path)
        if any(name == f or name.startswith(f + ".") for f in forbidden)
    ]
    assert violations == []


def test_no_model_names_hardcoded_in_domain_or_application():
    suspicious = ("qwen", "llama", "gpt-", "mistral", "gemma", "claude", "deepseek", "phi3")
    hits = [
        f"{path.relative_to(APP_ROOT)}: {term}"
        for layer in ("domain", "application")
        for path in python_files(layer)
        for term in suspicious
        if term in path.read_text(encoding="utf-8").lower()
    ]
    assert hits == []


def test_prompts_are_not_embedded_in_python():
    # Card prompts must live in versioned files, not in Python string literals.
    engine = (APP_ROOT / "application" / "cards" / "engine.py").read_text(encoding="utf-8")
    assert "Taboo" not in engine and "forbidden words a player" not in engine
