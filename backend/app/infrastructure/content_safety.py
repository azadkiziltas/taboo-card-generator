"""Loads optional per-language blocklists: `<dir>/<language>.txt`, one term per line."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_blocklists(directory: Path) -> dict[str, frozenset[str]]:
    blocklists: dict[str, frozenset[str]] = {}
    if not directory.is_dir():
        return blocklists
    for path in sorted(directory.glob("*.txt")):
        terms = {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if terms:
            blocklists[path.stem.lower()] = frozenset(terms)
    if blocklists:
        logger.info("loaded content blocklists for: %s", ", ".join(sorted(blocklists)))
    return blocklists
