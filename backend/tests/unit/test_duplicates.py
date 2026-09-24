from app.domain.cards.duplicates import DuplicateDetector, forbidden_overlap
from app.domain.cards.entities import Card, CardDraft, CardSource, Difficulty
from app.domain.cards.repositories import NewCardRecord
from app.domain.cards.validator import CardValidator, IssueCode
from app.infrastructure.memory.repositories import InMemoryCardRepository

WORDS = ("ekran", "klavye", "internet", "işlemci", "yazılım")


async def _store(repo, validator, target, words):
    d = CardDraft(target, words, "tr", Difficulty.MEDIUM)
    card = Card.from_draft(d, source=CardSource.AI)
    await repo.add_many([NewCardRecord(card, validator.target_key(d), validator.forbidden_keys(d))])
    return card


def test_forbidden_overlap_is_jaccard():
    assert forbidden_overlap({"a", "b"}, {"a", "b"}) == 1.0
    assert forbidden_overlap({"a", "b"}, {"c"}) == 0.0
    assert forbidden_overlap({"a", "b", "c"}, {"a", "b", "d"}) == 0.5


async def test_exact_duplicate_is_rejected_and_similar_target_is_flagged():
    repo, validator = InMemoryCardRepository(), CardValidator()
    existing = await _store(repo, validator, "Bilgisayar", WORDS)
    detector = DuplicateDetector(repo, validator, overlap_threshold=0.6)

    same = CardDraft(
        "BİLGİSAYAR", ("Ekran", "klavye", "internet", "işlemci", "fare"), "tr", Difficulty.MEDIUM
    )
    different = CardDraft(
        "bilgisayar", ("monitör", "fare", "kasa", "anakart", "program"), "tr", Difficulty.MEDIUM
    )
    new = CardDraft(
        "Uçak", ("pilot", "kanat", "bilet", "uçuş", "havalimanı"), "tr", Difficulty.MEDIUM
    )

    same_r, different_r, new_r = await detector.assess([same, different, new])
    assert [i.code for i in same_r.errors] == [IssueCode.DUPLICATE_EXISTING_CARD]
    assert different_r.valid and [w.code for w in different_r.warnings] == [
        IssueCode.EXISTING_TARGET
    ]
    assert new_r.valid and not new_r.warnings

    ignored = await detector.assess([same], ignore_card_ids=frozenset({existing.id}))
    assert ignored[0].valid and not ignored[0].warnings


async def test_other_language_is_not_a_duplicate():
    repo, validator = InMemoryCardRepository(), CardValidator()
    await _store(repo, validator, "Robot", ("makine", "yapay zeka", "fabrika", "sensör", "motor"))
    detector = DuplicateDetector(repo, validator)
    [result] = await detector.assess(
        [
            CardDraft(
                "Robot",
                ("machine", "factory", "sensor", "motor", "android"),
                "en",
                Difficulty.MEDIUM,
            )
        ]
    )
    assert result.valid and not result.warnings
