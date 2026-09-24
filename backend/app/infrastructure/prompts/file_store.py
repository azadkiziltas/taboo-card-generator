"""Loads prompt templates from `<dir>/<group>/<name>_<version>.md` with YAML front matter."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.application.prompts.service import PromptError, PromptTemplate


class FilePromptStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def path_for(self, name: str, version: str) -> Path:
        group, _, basename = name.partition(".")
        if not group or not basename or "/" in name or ".." in name:
            raise PromptError(f"invalid prompt name '{name}'")
        return self._directory / group / f"{basename}_{version}.md"

    def load(self, name: str, version: str) -> PromptTemplate:
        path = self.path_for(name, version)
        if not path.is_file():
            raise PromptError(f"prompt file not found: {path.relative_to(self._directory)}")
        return parse_prompt_file(path.read_text(encoding="utf-8"), source=str(path))

    def available(self) -> list[tuple[str, str]]:
        """All (name, version) pairs present on disk."""
        found: list[tuple[str, str]] = []
        for path in sorted(self._directory.glob("*/*_v*.md")):
            basename, _, version = path.stem.rpartition("_")
            found.append((f"{path.parent.name}.{basename}", version))
        return found


def parse_prompt_file(text: str, *, source: str = "<prompt>") -> PromptTemplate:
    if not text.startswith("---"):
        raise PromptError(f"{source}: missing front matter")
    try:
        _, header, body = text.split("---", 2)
    except ValueError as exc:
        raise PromptError(f"{source}: malformed front matter") from exc
    meta = yaml.safe_load(header) or {}
    for key in ("name", "version"):
        if not meta.get(key):
            raise PromptError(f"{source}: front matter needs '{key}'")
    return PromptTemplate(
        name=str(meta["name"]),
        version=str(meta["version"]),
        body=body.strip(),
        language_strategy=meta.get("language_strategy"),
        created_at=str(meta["created_at"]) if meta.get("created_at") else None,
        metadata=meta,
    )
