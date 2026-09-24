import unicodedata

from app.domain.cards.normalization import (
    ArabicNormalizer,
    DefaultNormalizer,
    NormalizerRegistry,
    TurkicNormalizer,
)


def test_clean_applies_nfc_and_collapses_whitespace():
    decomposed = unicodedata.normalize("NFD", "  Çiçek   bahçesi \n")
    cleaned = DefaultNormalizer().clean(decomposed)
    assert cleaned == "Çiçek bahçesi"
    assert unicodedata.is_normalized("NFC", cleaned)


def test_clean_strips_wrapping_quotes_and_trailing_punctuation():
    assert DefaultNormalizer().clean('"Kahve."') == "Kahve"
    assert DefaultNormalizer().clean("«Wi-Fi»") == "Wi-Fi"


def test_turkish_dotted_and_dotless_i():
    tr = TurkicNormalizer()
    assert tr.key("IŞIK") == "ışık"
    assert tr.key("İstanbul") == "istanbul"
    assert tr.key("ISPARTA") != tr.key("İSPARTA")
    # The default (non-Turkish) casefold would wrongly turn "I" into "i".
    assert DefaultNormalizer().key("IŞIK") == "işik"


def test_display_lowercase_is_language_aware():
    registry = NormalizerRegistry.with_defaults()
    assert registry.get("tr").lower("  IŞIK  Oyunları ") == "ışık oyunları"
    assert registry.get("tr").lower("İstanbul Lezzetleri") == "istanbul lezzetleri"
    assert registry.get("en").lower("Harry Potter") == "harry potter"


def test_german_sharp_s_casefold():
    assert DefaultNormalizer().key("STRASSE") == DefaultNormalizer().key("Straße")


def test_arabic_key_ignores_diacritics_tatweel_and_alef_variants():
    ar = ArabicNormalizer()
    assert ar.key("كِتَاب") == ar.key("كتاب")
    assert ar.key("كتــاب") == ar.key("كتاب")
    assert ar.key("أحمد") == ar.key("احمد")
    # Display form keeps the original text untouched (RTL rendering is the client's job).
    assert ar.clean("كِتَاب") == "كِتَاب"


def test_registry_defaults_and_extension():
    registry = NormalizerRegistry.with_defaults()
    assert isinstance(registry.get("tr"), TurkicNormalizer)
    assert isinstance(registry.get("ar"), ArabicNormalizer)
    assert isinstance(registry.get("en"), DefaultNormalizer)

    class Upper(DefaultNormalizer):
        def key(self, text: str) -> str:
            return text.upper()

    registry.register("xx", Upper())
    assert registry.get("xx").key("a") == "A"
