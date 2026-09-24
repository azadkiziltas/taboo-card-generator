import pytest

from app.domain.cards.entities import (
    Card,
    CardDraft,
    CardInvariantError,
    CardSource,
    Difficulty,
)
from app.domain.cards.validator import CardValidator, IssueCode, ValidationConfig


def draft(target="Bilgisayar", words=None, language="tr", difficulty=Difficulty.MEDIUM):
    words = words if words is not None else ["ekran", "klavye", "internet", "işlemci", "yazılım"]
    return CardDraft(target, tuple(words), language, difficulty, "technology")


@pytest.fixture
def validator():
    return CardValidator(config=ValidationConfig(supported_languages=frozenset({"tr", "en", "ar"})))


def codes(result):
    return {issue.code for issue in result.errors}


def test_valid_card(validator):
    result = validator.validate(draft())
    assert result.valid
    assert result.errors == [] and result.warnings == []


@pytest.mark.parametrize("words", [["a1", "b", "c", "d"], ["a", "b", "c", "d", "e", "f"]])
def test_exactly_five_forbidden_words(validator, words):
    words = ["ekran", "klavye", "internet", "işlemci", "yazılım", "fare"][: len(words)]
    result = validator.validate(draft(words=words))
    assert IssueCode.FORBIDDEN_COUNT in codes(result)
    assert result.repairable


def test_duplicate_forbidden_words_use_language_aware_keys(validator):
    result = validator.validate(draft(words=["IŞIK", "ışık", "klavye", "internet", "işlemci"]))
    assert IssueCode.DUPLICATE_FORBIDDEN_WORD in codes(result)


def test_target_in_forbidden_words_case_insensitive(validator):
    result = validator.validate(
        draft(words=["BİLGİSAYAR", "klavye", "internet", "işlemci", "fare"])
    )
    assert IssueCode.TARGET_IN_FORBIDDEN in codes(result)
    assert result.repairable


@pytest.mark.parametrize(
    ("target", "word"),
    [
        ("E-posta", "posta"),  # forbidden word is a part of a hyphenated target
        ("E-posta", "postacı"),
        ("Spor salonu", "spor"),  # part of a multi-word target
        ("Spor salonu", "salon"),
        ("Kütüphane", "hane"),
        ("Kahve", "kahveci"),  # target inside the forbidden word
    ],
)
def test_forbidden_word_overlapping_the_target(validator, target, word):
    result = validator.validate(
        draft(target=target, words=[word, "fincan", "mesaj", "çekirdek", "telve"])
    )
    assert IssueCode.FORBIDDEN_CONTAINS_TARGET in codes(result)
    assert result.repairable


@pytest.mark.parametrize(
    ("target", "word"),
    [
        ("Satranç", "at"),  # a chess piece, only a coincidental substring
        ("E-posta", "e-devlet"),  # single-letter parts are ignored
        ("Buzdolabı", "buz"),  # short substring without a separator
    ],
)
def test_short_or_coincidental_substrings_are_allowed(validator, target, word):
    result = validator.validate(
        draft(target=target, words=[word, "fincan", "mesaj", "çekirdek", "telve"])
    )
    assert IssueCode.FORBIDDEN_CONTAINS_TARGET not in codes(result)


def test_forbidden_word_containing_target(validator):
    result = validator.validate(
        draft(target="Kahve", words=["kahveci", "fincan", "kafein", "çekirdek", "telve"])
    )
    assert IssueCode.FORBIDDEN_CONTAINS_TARGET in codes(result)


def test_empty_values(validator):
    result = validator.validate(
        draft(target="  ", words=["", "klavye", "internet", "işlemci", "x"])
    )
    assert {IssueCode.EMPTY_TARGET, IssueCode.EMPTY_FORBIDDEN_WORD} <= codes(result)
    assert not result.repairable  # an empty target cannot be repaired


def test_max_lengths(validator):
    result = validator.validate(
        draft(target="a" * 41, words=["b" * 41, "klavye", "internet", "işlemci", "fare"])
    )
    assert {IssueCode.TARGET_TOO_LONG, IssueCode.FORBIDDEN_WORD_TOO_LONG} <= codes(result)


def test_forbidden_word_with_too_many_tokens(validator):
    result = validator.validate(
        draft(words=["çok uzun bir ifade", "klavye", "internet", "işlemci", "fare"])
    )
    assert IssueCode.FORBIDDEN_WORD_TOO_LONG in codes(result)


def test_script_check_rejects_foreign_letters(validator):
    cyrillic = validator.validate(draft(target="Компьютер"))
    assert IssueCode.INVALID_TARGET_CHARACTERS in codes(cyrillic)

    emoji = validator.validate(draft(words=["ekran 😀", "klavye", "internet", "işlemci", "fare"]))
    assert IssueCode.INVALID_FORBIDDEN_CHARACTERS in codes(emoji)


def test_script_check_accepts_digits_hyphen_and_apostrophe(validator):
    result = validator.validate(
        draft(target="3D yazıcı", words=["Wi-Fi", "plastik", "model", "katman", "rock'n'roll"])
    )
    assert result.valid, result.errors


def test_arabic_card(validator):
    ok = validator.validate(
        CardDraft(
            "حاسوب", ("شاشة", "لوحة مفاتيح", "إنترنت", "معالج", "برنامج"), "ar", Difficulty.EASY
        )
    )
    assert ok.valid, ok.errors
    latin = validator.validate(
        CardDraft("حاسوب", ("screen", "لوحة", "إنترنت", "معالج", "برنامج"), "ar", Difficulty.EASY)
    )
    assert IssueCode.INVALID_FORBIDDEN_CHARACTERS in codes(latin)


def test_unsupported_language(validator):
    result = validator.validate(draft(language="xx"))
    assert IssueCode.UNSUPPORTED_LANGUAGE in codes(result)


def test_batch_rejects_repeated_targets_and_taken_targets(validator):
    results = validator.validate_batch(
        [
            draft(),
            draft(target="BİLGİSAYAR"),
            draft(target="Uçak", words=["a", "b", "c", "d", "e"]),
        ],
        taken_target_keys={"uçak"},
    )
    assert results[0].valid
    assert IssueCode.DUPLICATE_TARGET_IN_BATCH in codes(results[1])
    assert IssueCode.DUPLICATE_TARGET_IN_BATCH in codes(results[2])


def test_normalize_produces_nfc_trimmed_display_values(validator):
    normalized = validator.normalize(
        CardDraft(
            " Bilgisayar ",
            (" ekran", "klavye ", "internet", "işlemci", "yazılım"),
            "tr",
            Difficulty.MEDIUM,
            " Technology ",
        )
    )
    assert normalized.target == "Bilgisayar"
    assert normalized.forbidden_words[0] == "ekran"
    assert normalized.category == "technology"


def test_card_entity_invariants():
    with pytest.raises(CardInvariantError):
        Card("x", ("a", "b"), "tr", Difficulty.EASY, CardSource.AI)
    with pytest.raises(CardInvariantError):
        Card("x", ("a", "b", "c", "d", "e"), "tur", Difficulty.EASY, CardSource.AI)
