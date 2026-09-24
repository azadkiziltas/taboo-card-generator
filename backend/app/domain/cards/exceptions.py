from __future__ import annotations


class CardDomainError(Exception):
    code = "card_error"


class CardNotFound(CardDomainError):
    code = "card_not_found"


class InvalidCardRequest(CardDomainError):
    code = "invalid_card_request"
