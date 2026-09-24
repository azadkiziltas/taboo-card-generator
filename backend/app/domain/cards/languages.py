"""Language profiles used by prompts and validators (ISO-639-1 codes)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_ISO_639_1 = re.compile(r"^[a-z]{2}$")


class Script(StrEnum):
    LATIN = "LATIN"
    ARABIC = "ARABIC"
    CYRILLIC = "CYRILLIC"
    GREEK = "GREEK"


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    code: str
    english_name: str
    native_name: str
    script: Script
    rtl: bool = False


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    p.code: p
    for p in (
        LanguageProfile("tr", "Turkish", "Türkçe", Script.LATIN),
        LanguageProfile("en", "English", "English", Script.LATIN),
        LanguageProfile("de", "German", "Deutsch", Script.LATIN),
        LanguageProfile("fr", "French", "Français", Script.LATIN),
        LanguageProfile("es", "Spanish", "Español", Script.LATIN),
        LanguageProfile("ar", "Arabic", "العربية", Script.ARABIC, rtl=True),
    )
}


def is_iso_639_1(code: str) -> bool:
    return bool(_ISO_639_1.match(code))


def get_language_profile(code: str) -> LanguageProfile | None:
    return LANGUAGE_PROFILES.get(code)
