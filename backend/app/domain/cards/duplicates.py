"""Duplicate detection against already persisted cards.

Exact layer : same (language, normalized target) *and* a similar forbidden-word set
              -> the draft is a duplicate and is rejected.
              Same target with a different word set -> accepted but flagged as a
              duplicate *candidate* (warning).
Semantic    : optional pluggable checker; it may only add warnings, never reject.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from app.domain.cards.entities import CardDraft, CardStatus
from app.domain.cards.repositories import CardRepository, ExistingCardSummary
from app.domain.cards.validator import (
    CardValidator,
    IssueCode,
    ValidationIssue,
    ValidationResult,
)

_LIVE_STATUSES = frozenset({CardStatus.ACTIVE, CardStatus.DUPLICATE_CANDIDATE})


def forbidden_overlap(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Jaccard similarity of two forbidden-word key sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class SemanticDuplicateChecker(Protocol):
    """Future extension: embedding based similarity between cards/targets."""

    async def find_similar(self, draft: CardDraft) -> list[ValidationIssue]: ...


class DuplicateDetector:
    def __init__(
        self,
        repository: CardRepository,
        validator: CardValidator,
        *,
        overlap_threshold: float = 0.6,
        semantic_checker: SemanticDuplicateChecker | None = None,
    ) -> None:
        self._repository = repository
        self._validator = validator
        self._threshold = overlap_threshold
        self._semantic = semantic_checker

    async def assess(
        self,
        drafts: Sequence[CardDraft],
        *,
        ignore_card_ids: frozenset[uuid.UUID] = frozenset(),
    ) -> list[ValidationResult]:
        """`ignore_card_ids`: cards being replaced (e.g. regeneration) are not duplicates."""
        results = [ValidationResult() for _ in drafts]
        by_language: dict[str, list[int]] = defaultdict(list)
        for index, draft in enumerate(drafts):
            by_language[draft.language].append(index)

        for language, indexes in by_language.items():
            keys = [self._validator.target_key(drafts[i]) for i in indexes]
            existing = await self._repository.find_existing(language, sorted(set(keys)))
            matches: dict[str, list[ExistingCardSummary]] = defaultdict(list)
            for summary in existing:
                if summary.status in _LIVE_STATUSES and summary.id not in ignore_card_ids:
                    matches[summary.target_key].append(summary)

            for index, key in zip(indexes, keys, strict=True):
                draft = drafts[index]
                candidates = matches.get(key, [])
                if not candidates:
                    continue
                draft_keys = frozenset(self._validator.forbidden_keys(draft))
                best = max(forbidden_overlap(draft_keys, c.forbidden_keys) for c in candidates)
                if best >= self._threshold:
                    results[index].errors.append(
                        ValidationIssue(
                            IssueCode.DUPLICATE_EXISTING_CARD,
                            f"a card for '{draft.target}' with nearly the same forbidden "
                            "words already exists",
                            "target",
                        )
                    )
                else:
                    results[index].warnings.append(
                        ValidationIssue(
                            IssueCode.EXISTING_TARGET,
                            f"target '{draft.target}' already exists with other forbidden words",
                            "target",
                        )
                    )

        if self._semantic is not None:
            for index, draft in enumerate(drafts):
                if results[index].valid:
                    results[index].warnings.extend(await self._semantic.find_similar(draft))
        return results
