"""Unicode normalization with language-specific comparison rules.

Two operations are distinguished:

* `clean`  - produces the display form that is stored (NFC, trimmed, single spaces).
* `key`    - produces a comparison key (case-folded, language aware) that is used for
             duplicate and "target in forbidden words" checks.

New languages register their own normalizer in `NormalizerRegistry` without touching
the validator.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

_WHITESPACE = re.compile(r"\s+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_WRAPPING = "\"'“”„«»‹›()[]{}"
_TRAILING = ".,;:!?،؛"


class TextNormalizer(Protocol):
    def clean(self, text: str) -> str: ...

    def key(self, text: str) -> str: ...

    def lower(self, text: str) -> str:
        """Language-aware lowercase for display values such as categories."""
        ...


class DefaultNormalizer:
    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch.isspace())
        text = _WHITESPACE.sub(" ", text).strip()
        text = text.strip(_WRAPPING).rstrip(_TRAILING).strip()
        return text

    def key(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", self.clean(text)).translate(_APOSTROPHES)
        return self._lower(text)

    def lower(self, text: str) -> str:
        return self.clean(text).lower()

    def _lower(self, text: str) -> str:
        return text.casefold()


class TurkicNormalizer(DefaultNormalizer):
    """Turkish/Azerbaijani casing: I -> ı and İ -> i (Python's casefold gets both wrong)."""

    _MAP = str.maketrans({"I": "ı", "İ": "i"})

    def lower(self, text: str) -> str:
        return self._lower(self.clean(text))

    def _lower(self, text: str) -> str:
        return text.translate(self._MAP).lower()


class ArabicNormalizer(DefaultNormalizer):
    """Ignores diacritics (tashkeel), tatweel and hamza-carrying alef variants in keys."""

    _TASHKEEL = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
    _MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ـ": None})

    def key(self, text: str) -> str:
        base = super().key(text)
        return self._TASHKEEL.sub("", base).translate(self._MAP)


class NormalizerRegistry:
    def __init__(self, default: TextNormalizer | None = None) -> None:
        self._default = default or DefaultNormalizer()
        self._by_language: dict[str, TextNormalizer] = {}

    def register(self, language: str, normalizer: TextNormalizer) -> None:
        self._by_language[language] = normalizer

    def get(self, language: str) -> TextNormalizer:
        return self._by_language.get(language, self._default)

    @classmethod
    def with_defaults(cls) -> NormalizerRegistry:
        registry = cls()
        turkic = TurkicNormalizer()
        registry.register("tr", turkic)
        registry.register("az", turkic)
        registry.register("ar", ArabicNormalizer())
        return registry
