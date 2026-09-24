"""Deterministic card validation.

The validator is a list of small rules. New rules (e.g. morphological checks for a
specific language) implement `CardRule` and are passed to `CardValidator`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol

from app.domain.cards.entities import FORBIDDEN_WORD_COUNT, CardDraft
from app.domain.cards.languages import LanguageProfile, get_language_profile, is_iso_639_1
from app.domain.cards.normalization import NormalizerRegistry, TextNormalizer


class IssueCode(StrEnum):
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    EMPTY_TARGET = "empty_target"
    TARGET_TOO_LONG = "target_too_long"
    INVALID_TARGET_CHARACTERS = "invalid_target_characters"
    FORBIDDEN_COUNT = "forbidden_count"
    EMPTY_FORBIDDEN_WORD = "empty_forbidden_word"
    FORBIDDEN_WORD_TOO_LONG = "forbidden_word_too_long"
    INVALID_FORBIDDEN_CHARACTERS = "invalid_forbidden_characters"
    DUPLICATE_FORBIDDEN_WORD = "duplicate_forbidden_word"
    TARGET_IN_FORBIDDEN = "target_in_forbidden"
    FORBIDDEN_CONTAINS_TARGET = "forbidden_contains_target"
    DUPLICATE_TARGET_IN_BATCH = "duplicate_target_in_batch"
    DUPLICATE_EXISTING_CARD = "duplicate_existing_card"
    EXISTING_TARGET = "existing_target"
    SEMANTIC_DUPLICATE_CANDIDATE = "semantic_duplicate_candidate"
    UNSAFE_CONTENT = "unsafe_content"


# Errors a model can fix while keeping the same target word.
REPAIRABLE_CODES = frozenset(
    {
        IssueCode.FORBIDDEN_COUNT,
        IssueCode.EMPTY_FORBIDDEN_WORD,
        IssueCode.FORBIDDEN_WORD_TOO_LONG,
        IssueCode.INVALID_FORBIDDEN_CHARACTERS,
        IssueCode.DUPLICATE_FORBIDDEN_WORD,
        IssueCode.TARGET_IN_FORBIDDEN,
        IssueCode.FORBIDDEN_CONTAINS_TARGET,
    }
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: IssueCode
    message: str
    field: str | None = None

    @property
    def repairable(self) -> bool:
        return self.code in REPAIRABLE_CODES

    def __str__(self) -> str:
        return self.message


@dataclass(slots=True)
class ValidationResult:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def repairable(self) -> bool:
        return bool(self.errors) and all(issue.repairable for issue in self.errors)

    @property
    def error_messages(self) -> list[str]:
        return [issue.message for issue in self.errors]

    def extend(self, other: ValidationResult) -> ValidationResult:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self

    def has(self, code: IssueCode) -> bool:
        return any(i.code == code for i in (*self.errors, *self.warnings))


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    forbidden_word_count: int = FORBIDDEN_WORD_COUNT
    max_target_length: int = 40
    max_forbidden_word_length: int = 40
    max_tokens_per_forbidden_word: int = 3
    min_containment_length: int = 4
    supported_languages: frozenset[str] | None = None
    allowed_punctuation: str = "-'’."


@dataclass(frozen=True, slots=True)
class RuleContext:
    normalizer: TextNormalizer
    config: ValidationConfig
    profile: LanguageProfile | None


class CardRule(Protocol):
    def check(self, draft: CardDraft, ctx: RuleContext) -> Iterable[ValidationIssue]: ...


def invalid_characters(text: str, ctx: RuleContext) -> list[str]:
    """Characters that do not belong to the language's script (digits are allowed)."""
    if ctx.profile is None:
        return []
    bad: list[str] = []
    for ch in text:
        if ch.isspace() or ch in ctx.config.allowed_punctuation:
            continue
        category = unicodedata.category(ch)
        if category == "Nd" or category.startswith("M"):
            continue
        if category.startswith("L") and unicodedata.name(ch, "").startswith(ctx.profile.script):
            continue
        if ch not in bad:
            bad.append(ch)
    return bad


class LanguageRule:
    def check(self, draft: CardDraft, ctx: RuleContext) -> Iterable[ValidationIssue]:
        supported = ctx.config.supported_languages
        if not is_iso_639_1(draft.language) or (
            supported is not None and draft.language not in supported
        ):
            yield ValidationIssue(
                IssueCode.UNSUPPORTED_LANGUAGE,
                f"language '{draft.language}' is not supported",
                "language",
            )


class TargetRule:
    def check(self, draft: CardDraft, ctx: RuleContext) -> Iterable[ValidationIssue]:
        target = draft.target.strip()
        if not target:
            yield ValidationIssue(IssueCode.EMPTY_TARGET, "target is empty", "target")
            return
        if len(target) > ctx.config.max_target_length:
            yield ValidationIssue(
                IssueCode.TARGET_TOO_LONG,
                f"target is longer than {ctx.config.max_target_length} characters",
                "target",
            )
        bad = invalid_characters(target, ctx)
        if bad:
            yield ValidationIssue(
                IssueCode.INVALID_TARGET_CHARACTERS,
                f"target contains characters not valid for language "
                f"'{draft.language}': {''.join(bad)}",
                "target",
            )


class ForbiddenWordsRule:
    def check(self, draft: CardDraft, ctx: RuleContext) -> Iterable[ValidationIssue]:
        expected = ctx.config.forbidden_word_count
        words = draft.forbidden_words
        if len(words) != expected:
            yield ValidationIssue(
                IssueCode.FORBIDDEN_COUNT,
                f"forbidden_words has {len(words)} items, exactly {expected} required",
                "forbidden_words",
            )
        for position, word in enumerate(words, start=1):
            if not word.strip():
                yield ValidationIssue(
                    IssueCode.EMPTY_FORBIDDEN_WORD,
                    f"forbidden word #{position} is empty",
                    "forbidden_words",
                )
                continue
            if len(word) > ctx.config.max_forbidden_word_length:
                yield ValidationIssue(
                    IssueCode.FORBIDDEN_WORD_TOO_LONG,
                    f"forbidden word '{word}' is longer than "
                    f"{ctx.config.max_forbidden_word_length} characters",
                    "forbidden_words",
                )
            elif len(word.split()) > ctx.config.max_tokens_per_forbidden_word:
                yield ValidationIssue(
                    IssueCode.FORBIDDEN_WORD_TOO_LONG,
                    f"forbidden word '{word}' has more than "
                    f"{ctx.config.max_tokens_per_forbidden_word} words",
                    "forbidden_words",
                )
            bad = invalid_characters(word, ctx)
            if bad:
                yield ValidationIssue(
                    IssueCode.INVALID_FORBIDDEN_CHARACTERS,
                    f"forbidden word '{word}' contains characters not valid for language "
                    f"'{draft.language}': {''.join(bad)}",
                    "forbidden_words",
                )


class ForbiddenUniquenessRule:
    def check(self, draft: CardDraft, ctx: RuleContext) -> Iterable[ValidationIssue]:
        seen: set[str] = set()
        reported: set[str] = set()
        for word in draft.forbidden_words:
            key = ctx.normalizer.key(word)
            if not key:
                continue
            if key in seen and key not in reported:
                reported.add(key)
                yield ValidationIssue(
                    IssueCode.DUPLICATE_FORBIDDEN_WORD,
                    f"forbidden word '{word}' is repeated",
                    "forbidden_words",
                )
            seen.add(key)


class TargetNotForbiddenRule:
    def check(self, draft: CardDraft, ctx: RuleContext) -> Iterable[ValidationIssue]:
        target_key = ctx.normalizer.key(draft.target)
        if not target_key:
            return
        for word in draft.forbidden_words:
            if ctx.normalizer.key(word) == target_key:
                yield ValidationIssue(
                    IssueCode.TARGET_IN_FORBIDDEN,
                    f"the target '{draft.target}' itself appears in forbidden_words",
                    "forbidden_words",
                )
                return


_WORD_PARTS = re.compile(r"[\s\-'’.]+")


def _parts(key: str) -> list[str]:
    return [part for part in _WORD_PARTS.split(key) if part]


class ForbiddenContainsTargetRule:
    """Rejects forbidden words that overlap with the target, in either direction.

    Any part of the target is off-limits in the game anyway, so listing it wastes one of
    the five forbidden words:
    - the target (or one of its parts) inside a forbidden word: 'kahve' -> 'kahveci',
      'E-posta' -> 'postacı';
    - a forbidden word that is a part of the target: 'E-posta' -> 'posta',
      'Spor salonu' -> 'spor'.
    Short substrings are only matched as whole parts ('at' stays valid for 'Satranç').
    This is a cheap heuristic; real morphological analysis can replace it.
    """

    def check(self, draft: CardDraft, ctx: RuleContext) -> Iterable[ValidationIssue]:
        minimum = ctx.config.min_containment_length
        target_key = ctx.normalizer.key(draft.target)
        target_parts = set(_parts(target_key))
        long_target_parts = [part for part in target_parts if len(part) >= minimum]
        for word in draft.forbidden_words:
            word_key = ctx.normalizer.key(word)
            if not word_key or word_key == target_key:
                continue  # exact match is reported by TargetNotForbiddenRule
            overlaps = (
                any(part in word_key for part in long_target_parts)
                or any(len(part) >= 2 and part in target_parts for part in _parts(word_key))
                or (len(word_key) >= minimum and word_key in target_key)
            )
            if overlaps:
                yield ValidationIssue(
                    IssueCode.FORBIDDEN_CONTAINS_TARGET,
                    f"forbidden word '{word}' repeats part of the target '{draft.target}'; "
                    "parts of the target are forbidden anyway, use a different word",
                    "forbidden_words",
                )


DEFAULT_RULES: tuple[CardRule, ...] = (
    LanguageRule(),
    TargetRule(),
    ForbiddenWordsRule(),
    ForbiddenUniquenessRule(),
    TargetNotForbiddenRule(),
    ForbiddenContainsTargetRule(),
)


class CardValidator:
    def __init__(
        self,
        normalizers: NormalizerRegistry | None = None,
        config: ValidationConfig | None = None,
        rules: Sequence[CardRule] | None = None,
    ) -> None:
        self._normalizers = normalizers or NormalizerRegistry.with_defaults()
        self.config = config or ValidationConfig()
        self._rules = tuple(rules) if rules is not None else DEFAULT_RULES

    def normalizer(self, language: str) -> TextNormalizer:
        return self._normalizers.get(language)

    def normalize_category(self, language: str, category: str | None) -> str | None:
        """Categories are free text (a label or a theme); stored lowercased per language."""
        if not category:
            return None
        return self.normalizer(language).lower(category) or None

    def target_key(self, draft: CardDraft) -> str:
        return self.normalizer(draft.language).key(draft.target)

    def forbidden_keys(self, draft: CardDraft) -> tuple[str, ...]:
        normalizer = self.normalizer(draft.language)
        return tuple(normalizer.key(word) for word in draft.forbidden_words)

    def normalize(self, draft: CardDraft) -> CardDraft:
        """Unicode/whitespace normalization of the display values."""
        normalizer = self.normalizer(draft.language)
        category = self.normalize_category(draft.language, draft.category)
        return replace(
            draft,
            target=normalizer.clean(draft.target),
            forbidden_words=tuple(normalizer.clean(word) for word in draft.forbidden_words),
            category=category or None,
        )

    def validate(self, draft: CardDraft) -> ValidationResult:
        ctx = RuleContext(
            normalizer=self.normalizer(draft.language),
            config=self.config,
            profile=get_language_profile(draft.language),
        )
        result = ValidationResult()
        for rule in self._rules:
            result.errors.extend(rule.check(draft, ctx))
        return result

    def validate_batch(
        self, drafts: Sequence[CardDraft], *, taken_target_keys: Iterable[str] = ()
    ) -> list[ValidationResult]:
        """Validate each card and reject targets repeated inside the batch.

        `taken_target_keys` are targets already accepted earlier in the same request.
        """
        taken = set(taken_target_keys)
        results: list[ValidationResult] = []
        for draft in drafts:
            result = self.validate(draft)
            key = self.target_key(draft)
            if key and key in taken:
                result.errors.append(
                    ValidationIssue(
                        IssueCode.DUPLICATE_TARGET_IN_BATCH,
                        f"target '{draft.target}' is repeated in this batch",
                        "target",
                    )
                )
            elif key:
                taken.add(key)
            results.append(result)
        return results
