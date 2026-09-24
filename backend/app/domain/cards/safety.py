"""Content safety extension point."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from app.domain.cards.entities import CardDraft
from app.domain.cards.normalization import NormalizerRegistry
from app.domain.cards.validator import IssueCode, ValidationIssue


class ContentSafetyFilter(Protocol):
    async def check(self, draft: CardDraft) -> list[ValidationIssue]: ...


class AllowAllSafetyFilter:
    async def check(self, draft: CardDraft) -> list[ValidationIssue]:
        return []


class BlocklistSafetyFilter:
    """Rejects cards whose target or forbidden words hit a per-language blocklist."""

    def __init__(
        self, blocklists: Mapping[str, frozenset[str]], normalizers: NormalizerRegistry
    ) -> None:
        self._normalizers = normalizers
        self._blocklists = {
            language: frozenset(normalizers.get(language).key(term) for term in terms)
            for language, terms in blocklists.items()
        }

    async def check(self, draft: CardDraft) -> list[ValidationIssue]:
        blocked = self._blocklists.get(draft.language)
        if not blocked:
            return []
        normalizer = self._normalizers.get(draft.language)
        for text in (draft.target, *draft.forbidden_words):
            key = normalizer.key(text)
            if key in blocked or any(token in blocked for token in key.split()):
                return [
                    ValidationIssue(
                        IssueCode.UNSAFE_CONTENT, "card contains blocked content", "target"
                    )
                ]
        return []
