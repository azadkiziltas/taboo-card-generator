"""Card generation engine shared by the card use cases.

LLM output -> schema (router) -> normalization -> rule validators -> safety filter
-> duplicate checks -> accept | repair (same target) | drop.

Nothing here knows which provider or model is used; that is the router's job.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.application.ai.model_router import ModelRouter, RoutedResult
from app.application.prompts.service import PromptService
from app.application.rag.retrieve_context import RetrieveContextUseCase, build_rag_query
from app.domain.ai.exceptions import AIError
from app.domain.ai.generation import ChatMessage
from app.domain.ai.telemetry import AIRequestContext, AIUsageRecorder, GenerationRunRecord
from app.domain.cards.duplicates import DuplicateDetector
from app.domain.cards.entities import CardDraft, Difficulty
from app.domain.cards.languages import get_language_profile
from app.domain.cards.repositories import CardRepository
from app.domain.cards.safety import ContentSafetyFilter
from app.domain.cards.schemas import GeneratedCard, GeneratedCardBatch
from app.domain.cards.validator import CardValidator, IssueCode, ValidationIssue, ValidationResult
from app.domain.rag.entities import RagContext

logger = logging.getLogger(__name__)

TASK_GENERATION = "card_generation"
TASK_REPAIR = "card_repair"

PROMPT_SYSTEM = "card.system"
PROMPT_GENERATE = "card.generate"
PROMPT_GENERATE_FOR_TARGETS = "card.generate_for_targets"
PROMPT_REPAIR = "card.repair"
PROMPT_DIFFICULTY = "card.difficulty"

_DUPLICATE_CODES = frozenset(
    {IssueCode.DUPLICATE_EXISTING_CARD, IssueCode.DUPLICATE_TARGET_IN_BATCH}
)
_CONTEXT_CHARS_PER_DOC = 300


@dataclass(frozen=True, slots=True)
class CardGenerationSettings:
    batch_size: int = 10
    overgenerate_extra: int = 2
    max_generation_rounds: int = 4
    max_repair_attempts: int = 2
    generation_temperature: float = 0.8
    repair_temperature: float = 0.3
    max_output_tokens: int | None = 4096
    prompt_version: str = "v3"
    avoid_existing_targets_limit: int = 100


@dataclass(frozen=True, slots=True)
class CardSpec:
    language: str
    difficulty: Difficulty | None = None  # None: the model is told nothing about difficulty
    category: str | None = None
    topic: str | None = None


@dataclass(slots=True)
class AcceptedCard:
    draft: CardDraft
    ai_request_id: uuid.UUID
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_duplicate_candidate(self) -> bool:
        return any(w.code is IssueCode.EXISTING_TARGET for w in self.warnings)


@dataclass(slots=True)
class _Pending:
    draft: CardDraft
    result: ValidationResult


class CardGenerationEngine:
    def __init__(
        self,
        *,
        router: ModelRouter,
        prompts: PromptService,
        validator: CardValidator,
        duplicates: DuplicateDetector,
        safety: ContentSafetyFilter,
        repository: CardRepository,
        retrieve_context: RetrieveContextUseCase,
        usage_recorder: AIUsageRecorder,
        settings: CardGenerationSettings,
    ) -> None:
        self._router = router
        self._prompts = prompts
        self._validator = validator
        self._duplicates = duplicates
        self._safety = safety
        self._repository = repository
        self._retrieve = retrieve_context
        self._usage = usage_recorder
        self.settings = settings

    @property
    def validator(self) -> CardValidator:
        return self._validator

    # -- public -------------------------------------------------------------------------

    async def produce(
        self,
        *,
        spec: CardSpec,
        count: int,
        run: GenerationRunRecord,
        targets: Sequence[str] | None = None,
        avoid_forbidden: Sequence[str] = (),
        ignore_card_ids: frozenset[uuid.UUID] = frozenset(),
    ) -> list[AcceptedCard]:
        """Generate up to `count` valid cards (one per target when `targets` is given)."""
        run.prompt_name = PROMPT_GENERATE_FOR_TARGETS if targets else PROMPT_GENERATE
        run.prompt_version = self.settings.prompt_version

        rag = await self._retrieve.execute(
            language=spec.language,
            query=build_rag_query(
                topic=spec.topic, category=spec.category, targets=list(targets or [])
            ),
            context=AIRequestContext(run_id=run.id),
        )
        run.rag_enabled = rag.enabled
        run.context_documents = len(rag.documents)

        avoid_targets: list[str] = []
        if not targets and self.settings.avoid_existing_targets_limit:
            avoid_targets = await self._repository.recent_targets(
                spec.language, spec.category, self.settings.avoid_existing_targets_limit
            )

        accepted: list[AcceptedCard] = []
        taken: set[str] = set()
        remaining_targets = list(targets or [])

        for _ in range(self.settings.max_generation_rounds):
            need = count - len(accepted)
            if need <= 0 or (targets is not None and not remaining_targets):
                break
            run.generation_rounds += 1

            try:
                if targets:
                    round_targets = remaining_targets[: self.settings.batch_size]
                    routed = await self._generate(
                        spec,
                        rag,
                        run,
                        prompt=PROMPT_GENERATE_FOR_TARGETS,
                        variables={
                            "targets_json": json.dumps(round_targets, ensure_ascii=False),
                            "avoid_forbidden": _join_or_none(avoid_forbidden),
                        },
                    )
                    drafts = self._match_targets(routed, spec, round_targets)
                else:
                    request_n = min(
                        self.settings.batch_size, need + self.settings.overgenerate_extra
                    )
                    routed = await self._generate(
                        spec,
                        rag,
                        run,
                        prompt=PROMPT_GENERATE,
                        variables={
                            "count": request_n,
                            "avoid_targets": _join_or_none(
                                [*avoid_targets, *(a.draft.target for a in accepted)]
                            ),
                        },
                    )
                    drafts = [self._to_draft(card, spec) for card in routed.result.parsed.cards]

            except AIError as exc:
                if not accepted:
                    raise
                # A later round failing (timeout, outage) must not discard valid cards
                # already produced: return them as a partial result.
                logger.warning(
                    "generation round failed after %d accepted cards: %s", len(accepted), exc.code
                )
                run.error_type = exc.code
                break

            run.drafts_generated += len(drafts)
            new_cards = await self._evaluate(drafts, routed, spec, run, taken, ignore_card_ids)
            # Acceptance = passed validation; surplus valid cards beyond `need` are dropped.
            run.drafts_accepted += len(new_cards)
            for card in new_cards[:need]:
                accepted.append(card)
                taken.add(self._validator.target_key(card.draft))

            if targets:
                done = {self._validator.target_key(a.draft) for a in accepted}
                attempted = {self._validator.target_key(d) for d in drafts}
                # Targets the model answered are not retried (one that failed after repairs
                # would most likely fail again); targets it skipped get another round.
                remaining_targets = [
                    t
                    for t in remaining_targets
                    if self._key(spec, t) not in done and self._key(spec, t) not in attempted
                ]

        return accepted

    # -- generation -----------------------------------------------------------------------

    async def _generate(
        self,
        spec: CardSpec,
        rag: RagContext,
        run: GenerationRunRecord,
        *,
        prompt: str,
        variables: dict[str, object],
    ) -> RoutedResult:
        version = self.settings.prompt_version
        profile = get_language_profile(spec.language)
        rendered = self._prompts.render(
            prompt,
            version,
            language_name=profile.english_name if profile else spec.language,
            language_code=spec.language,
            category=spec.category or "general",
            difficulty_section=self._difficulty_section(spec),
            topic=_sanitize(spec.topic) if spec.topic else "none",
            reference_context=format_reference_context(rag),
            **variables,
        )
        return await self._router.generate_structured(
            task=TASK_GENERATION,
            messages=self._messages(rendered.text),
            schema=GeneratedCardBatch,
            requirements={"language": spec.language, "structured_output": True},
            temperature=self.settings.generation_temperature,
            max_tokens=self.settings.max_output_tokens,
            context=AIRequestContext(run_id=run.id, prompt_name=prompt, prompt_version=version),
        )

    def _difficulty_section(self, spec: CardSpec) -> str:
        """Difficulty instructions, or nothing at all when no difficulty was requested."""
        if spec.difficulty is None:
            return ""
        return self._prompts.render(
            PROMPT_DIFFICULTY, self.settings.prompt_version, difficulty=spec.difficulty.value
        ).text

    def _messages(self, user_prompt: str) -> list[ChatMessage]:
        system = self._prompts.render(
            PROMPT_SYSTEM,
            self.settings.prompt_version,
            forbidden_word_count=self._validator.config.forbidden_word_count,
        ).text
        return [ChatMessage.system(system), ChatMessage.user(user_prompt)]

    def _to_draft(
        self, card: GeneratedCard, spec: CardSpec, target: str | None = None
    ) -> CardDraft:
        return CardDraft(
            target=target if target is not None else card.target,
            forbidden_words=tuple(str(word) for word in card.forbidden_words),
            language=spec.language,
            difficulty=spec.difficulty,
            category=spec.category or card.category,
        )

    def _key(self, spec: CardSpec, text: str) -> str:
        return self._validator.normalizer(spec.language).key(text)

    def _match_targets(
        self, routed: RoutedResult, spec: CardSpec, targets: Sequence[str]
    ) -> list[CardDraft]:
        """Keep one generated card per requested target (the model must not change it)."""
        by_key = {self._key(spec, t): t for t in targets}
        drafts: list[CardDraft] = []
        for card in routed.result.parsed.cards:
            key = self._key(spec, card.target)
            if key in by_key:
                drafts.append(self._to_draft(card, spec, target=by_key.pop(key)))
        return drafts

    # -- validation / repair --------------------------------------------------------------

    async def _check(
        self,
        drafts: list[CardDraft],
        taken: set[str],
        ignore_card_ids: frozenset[uuid.UUID],
    ) -> list[ValidationResult]:
        results = self._validator.validate_batch(drafts, taken_target_keys=taken)
        for draft, result in zip(drafts, results, strict=True):
            result.errors.extend(await self._safety.check(draft))
        duplicate_results = await self._duplicates.assess(drafts, ignore_card_ids=ignore_card_ids)
        for result, extra in zip(results, duplicate_results, strict=True):
            result.extend(extra)
        return results

    async def _evaluate(
        self,
        drafts: list[CardDraft],
        routed: RoutedResult,
        spec: CardSpec,
        run: GenerationRunRecord,
        taken: set[str],
        ignore_card_ids: frozenset[uuid.UUID],
    ) -> list[AcceptedCard]:
        accepted: list[AcceptedCard] = []
        current = [self._validator.normalize(d) for d in drafts]
        request_id = routed.request_id
        attempt = 0

        while current:
            results = await self._check(current, taken, ignore_card_ids)
            to_repair: list[_Pending] = []
            failures = 0
            round_accepted = 0
            for draft, result in zip(current, results, strict=True):
                if result.valid:
                    accepted.append(AcceptedCard(draft, request_id, list(result.warnings)))
                    taken.add(self._validator.target_key(draft))
                    round_accepted += 1
                    continue
                failures += 1
                run.duplicates_rejected += sum(
                    1 for e in result.errors if e.code in _DUPLICATE_CODES
                )
                if result.repairable:
                    to_repair.append(_Pending(draft, result))
            run.validation_failures += failures
            await self._annotate(request_id, failures, round_accepted)

            if not to_repair or attempt >= self.settings.max_repair_attempts:
                break
            attempt += 1
            run.repair_attempts += 1
            try:
                repaired, request_id = await self._repair(to_repair, spec, run)
            except AIError as exc:
                logger.warning("card repair call failed: %s", exc.code)
                break
            current = [self._validator.normalize(d) for d in repaired]
        return accepted

    async def _repair(
        self, pending: list[_Pending], spec: CardSpec, run: GenerationRunRecord
    ) -> tuple[list[CardDraft], uuid.UUID]:
        version = self.settings.prompt_version
        profile = get_language_profile(spec.language)
        failures = [
            {
                "card": {
                    "target": item.draft.target,
                    "forbidden_words": list(item.draft.forbidden_words),
                },
                "errors": item.result.error_messages,
            }
            for item in pending
        ]
        rendered = self._prompts.render(
            PROMPT_REPAIR,
            version,
            language_name=profile.english_name if profile else spec.language,
            language_code=spec.language,
            difficulty_section=self._difficulty_section(spec),
            failures_json=json.dumps(failures, ensure_ascii=False, indent=2),
        )
        routed = await self._router.generate_structured(
            task=TASK_REPAIR,
            messages=self._messages(rendered.text),
            schema=GeneratedCardBatch,
            requirements={"language": spec.language, "structured_output": True},
            temperature=self.settings.repair_temperature,
            max_tokens=self.settings.max_output_tokens,
            context=AIRequestContext(
                run_id=run.id, prompt_name=PROMPT_REPAIR, prompt_version=version
            ),
        )
        originals = {self._validator.target_key(p.draft): p.draft for p in pending}
        repaired: list[CardDraft] = []
        for card in routed.result.parsed.cards:
            original = originals.pop(self._key(spec, card.target), None)
            if original is None:
                continue  # the model changed the target: not a repair
            repaired.append(
                CardDraft(
                    target=original.target,
                    forbidden_words=tuple(str(w) for w in card.forbidden_words),
                    language=original.language,
                    difficulty=original.difficulty,
                    category=original.category,
                )
            )
        return repaired, routed.request_id

    async def _annotate(self, request_id: uuid.UUID, failures: int, accepted: int) -> None:
        try:
            await self._usage.annotate_request(
                request_id, validation_failures=failures, items_accepted=accepted
            )
        except Exception:
            logger.exception("failed to annotate AI request telemetry")


def _sanitize(text: str) -> str:
    """Neutralize text that is inserted into prompts inside delimiters."""
    cleaned = " ".join(text.replace("<", " ").replace(">", " ").split())
    return cleaned


def _join_or_none(items: Sequence[str]) -> str:
    unique = list(dict.fromkeys(i for i in items if i))
    return ", ".join(_sanitize(i) for i in unique) if unique else "none"


def format_reference_context(rag: RagContext) -> str:
    if rag.is_empty:
        return "No reference entries available."
    lines = []
    for scored in rag.documents:
        doc = scored.document
        content = _sanitize(doc.content)[:_CONTEXT_CHARS_PER_DOC]
        line = f"- {_sanitize(doc.word)}: {content}"
        if doc.related_terms:
            line += f" (related: {', '.join(_sanitize(t) for t in doc.related_terms[:8])})"
        lines.append(line)
    return "\n".join(lines)
