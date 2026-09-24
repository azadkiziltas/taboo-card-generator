"""Versioned prompt templates.

Templates live in files (`app/prompts/<group>/<name>_<version>.md`) with YAML front
matter metadata. Placeholders use `{{ name }}` so JSON examples inside prompts need no
escaping. Every placeholder must be provided; unused variables are an error too, which
catches template/code drift early.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


class PromptError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    name: str  # e.g. "card.generate"
    version: str  # e.g. "v1"
    body: str
    language_strategy: str | None = None
    created_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def variables(self) -> frozenset[str]:
        return frozenset(_PLACEHOLDER.findall(self.body))

    def render(self, variables: Mapping[str, Any]) -> str:
        missing = self.variables - set(variables)
        if missing:
            raise PromptError(f"prompt {self.name}@{self.version} missing: {sorted(missing)}")
        extra = set(variables) - self.variables
        if extra:
            raise PromptError(f"prompt {self.name}@{self.version} unexpected: {sorted(extra)}")
        return _PLACEHOLDER.sub(lambda m: str(variables[m.group(1)]), self.body).strip()


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    name: str
    version: str
    text: str


class PromptStore(Protocol):
    def load(self, name: str, version: str) -> PromptTemplate: ...


class PromptService:
    def __init__(self, store: PromptStore) -> None:
        self._store = store
        self._cache: dict[tuple[str, str], PromptTemplate] = {}

    def get(self, name: str, version: str) -> PromptTemplate:
        key = (name, version)
        if key not in self._cache:
            template = self._store.load(name, version)
            if template.name != name or template.version != version:
                raise PromptError(
                    f"prompt file metadata ({template.name}@{template.version}) "
                    f"does not match requested {name}@{version}"
                )
            self._cache[key] = template
        return self._cache[key]

    def render(self, name: str, version: str, **variables: Any) -> RenderedPrompt:
        template = self.get(name, version)
        return RenderedPrompt(name=name, version=version, text=template.render(variables))
