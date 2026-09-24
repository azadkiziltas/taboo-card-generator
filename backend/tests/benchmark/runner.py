"""Model benchmark runner.

Runs a dataset of target words through the *real* card pipeline (prompts, structured
output, validation, repair) once per model, pinning the router to that model only.
Storage is in-memory: nothing is written to PostgreSQL.

    uv run python -m tests.benchmark.runner \
        --dataset tests/benchmark/datasets/tr_basic.json \
        --model ollama:qwen3.5:27b [--model cloud_a:model-x] \
        [--rag-seed data/rag/seed_tr.jsonl] [--limit 5] [--out benchmark-reports] \
        [--batch tr:technology:5]

`--batch LANG:CATEGORY:COUNT` (repeatable) also times the category batch generation the
UI uses (`POST /v1/cards/generate`). `--dataset` may be omitted when batches are given.

Outputs `<out>/<timestamp>/report.json` (summary + items + batches), `items.csv`
(with an empty `human_rating` column to fill in later) and `batches.csv`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.application.cards.generate_card import GenerateCardCommand
from app.application.cards.generate_cards import GenerateCardsCommand
from app.bootstrap import Container, build_in_memory_container
from app.config.settings import Settings, get_settings
from app.domain.ai.capabilities import ModelCapabilities
from app.domain.ai.routing import (
    ModelDescriptor,
    ModelType,
    RouteCandidate,
    RoutingPolicySet,
    TaskRoutingPolicy,
)
from app.domain.cards.entities import CardStatus, Difficulty
from app.infrastructure.memory.repositories import InMemoryUsageRecorder

GENERATION_TASKS = ("card_generation", "card_repair")


@dataclass(frozen=True, slots=True)
class DatasetItem:
    target: str
    language: str
    difficulty: Difficulty
    category: str | None = None


@dataclass(slots=True)
class ItemResult:
    model: str
    target: str
    language: str
    difficulty: str | None
    success: bool
    forbidden_words: list[str] = field(default_factory=list)
    status: str | None = None
    error_type: str | None = None
    latency_ms: int = 0
    human_rating: str | None = None  # filled in later by a human reviewer


@dataclass(frozen=True, slots=True)
class BatchSpec:
    language: str
    category: str | None
    count: int


@dataclass(slots=True)
class BatchResult:
    model: str
    language: str
    category: str | None
    requested: int
    returned: int = 0
    latency_ms: int = 0
    error_type: str | None = None
    targets: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelSummary:
    model: str
    items: int
    final_success_rate: float | None
    json_schema_success_rate: float | None
    first_pass_validation_rate: float | None
    duplicate_rate: float | None
    retry_rate: float | None
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    input_tokens: int
    output_tokens: int
    api_estimated_cost: float
    batch_runs: int = 0
    batch_avg_latency_ms: float | None = None
    batch_return_rate: float | None = None
    human_rating: float | None = None
    quality_score: float | None = None


class QualityScorer(Protocol):
    """Extension point: turn a benchmark summary into a registry `quality_score`."""

    def score(self, summary: ModelSummary, items: list[ItemResult]) -> float | None: ...


class NoQualityScore:
    """Default: no automatic score is invented. Set it from human ratings instead."""

    def score(self, summary: ModelSummary, items: list[ItemResult]) -> float | None:
        return None


def load_dataset(path: Path) -> list[DatasetItem]:
    return [
        DatasetItem(
            target=item["target"],
            language=item["language"],
            difficulty=Difficulty(item.get("difficulty", "medium")),
            category=item.get("category"),
        )
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]


def parse_batch_spec(value: str) -> BatchSpec:
    language, _, rest = value.partition(":")
    category, _, count = rest.rpartition(":")
    if len(language) != 2 or not count.isdigit():
        raise argparse.ArgumentTypeError("use LANG:CATEGORY:COUNT, e.g. tr:technology:5")
    return BatchSpec(language=language, category=category or None, count=int(count))


def parse_model_key(value: str) -> tuple[str, str]:
    provider, _, model = value.partition(":")  # model names may contain ':' themselves
    if not provider or not model:
        raise argparse.ArgumentTypeError("use provider:model, e.g. ollama:qwen3.5:27b")
    return provider, model


def pinned_policies(provider: str, model: str) -> RoutingPolicySet:
    candidate = (RouteCandidate(provider, model, 1),)
    return RoutingPolicySet(
        [
            TaskRoutingPolicy(task=t, candidates=candidate, max_provider_attempts=1)
            for t in GENERATION_TASKS
        ]
    )


def registry_models(configured: list[ModelDescriptor], provider: str, model: str):
    """Configured models plus the benchmarked one if it is not configured yet."""
    if any((m.provider, m.model_name) == (provider, model) for m in configured):
        return configured
    template = next(
        (m for m in configured if m.provider == provider and m.model_type is ModelType.GENERATION),
        None,
    )
    extra = (
        replace(template, model_name=model, quality_score=None)
        if template
        else ModelDescriptor(
            provider=provider,
            model_name=model,
            model_type=ModelType.GENERATION,
            is_local=provider == "ollama",
            capabilities=ModelCapabilities(structured_output=True, json_mode=True),
        )
    )
    return [*configured, extra]


def summarize(
    model_key: str,
    results: list[ItemResult],
    usage: InMemoryUsageRecorder,
    scorer: QualityScorer | None = None,
    batches: list[BatchResult] | None = None,
) -> ModelSummary:
    requests = [r for r in usage.requests.values() if r.task in GENERATION_TASKS]
    first_pass = [r for r in requests if r.task == "card_generation" and r.success]
    runs = usage.runs
    latencies = sorted(r.latency_ms for r in results)
    batches = batches or []

    def ratio(numerator: float, denominator: float) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    checked = sum((r.items_accepted or 0) + (r.validation_failures or 0) for r in first_pass)
    summary = ModelSummary(
        model=model_key,
        items=len(results),
        final_success_rate=ratio(sum(r.success for r in results), len(results)),
        json_schema_success_rate=ratio(sum(not r.parse_failed for r in requests), len(requests)),
        first_pass_validation_rate=ratio(sum(r.items_accepted or 0 for r in first_pass), checked),
        duplicate_rate=ratio(
            sum(run.duplicates_rejected for run in runs)
            + sum(r.status == CardStatus.DUPLICATE_CANDIDATE.value for r in results),
            sum(run.drafts_generated for run in runs),
        ),
        retry_rate=ratio(
            sum(r.retry_count > 0 for r in requests) + sum(run.repair_attempts for run in runs),
            len(runs),
        ),
        avg_latency_ms=round(statistics.fmean(latencies), 1) if latencies else None,
        p95_latency_ms=(
            float(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))])
            if latencies
            else None
        ),
        input_tokens=sum(r.input_tokens or 0 for r in requests),
        output_tokens=sum(r.output_tokens or 0 for r in requests),
        api_estimated_cost=float(sum(r.api_estimated_cost or 0 for r in requests)),
        batch_runs=len(batches),
        batch_avg_latency_ms=(
            round(statistics.fmean(b.latency_ms for b in batches), 1) if batches else None
        ),
        batch_return_rate=ratio(
            sum(b.returned for b in batches), sum(b.requested for b in batches)
        ),
    )
    summary.quality_score = (scorer or NoQualityScore()).score(summary, results)
    return summary


async def run_items(
    container: Container, model_key: str, dataset: list[DatasetItem]
) -> list[ItemResult]:
    results: list[ItemResult] = []
    for item in dataset:
        started = time.perf_counter()
        result = ItemResult(
            model=model_key,
            target=item.target,
            language=item.language,
            difficulty=item.difficulty.value if item.difficulty else None,
            success=False,
        )
        try:
            card = await container.generate_card.execute(
                GenerateCardCommand(
                    target=item.target,
                    language=item.language,
                    difficulty=item.difficulty,
                    category=item.category,
                )
            )
            result.success = True
            result.forbidden_words = list(card.forbidden_words)
            result.status = card.status.value
        except Exception as exc:  # benchmark keeps going; the error type is the data point
            result.error_type = getattr(exc, "code", type(exc).__name__)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        results.append(result)
        print(
            f"  [{model_key}] {item.target}: "
            f"{'ok ' + ', '.join(result.forbidden_words) if result.success else result.error_type}"
            f" ({result.latency_ms} ms)",
            file=sys.stderr,
        )
    return results


async def run_batches(
    container: Container, model_key: str, specs: list[BatchSpec]
) -> list[BatchResult]:
    results: list[BatchResult] = []
    for spec in specs:
        result = BatchResult(
            model=model_key, language=spec.language, category=spec.category, requested=spec.count
        )
        started = time.perf_counter()
        try:
            cards = await container.generate_cards.execute(
                GenerateCardsCommand(
                    language=spec.language,
                    difficulty=None,  # the UI default: no difficulty sent to the model
                    count=spec.count,
                    category=spec.category,
                )
            )
            result.returned = len(cards)
            result.targets = [card.target for card in cards]
        except Exception as exc:
            result.error_type = getattr(exc, "code", type(exc).__name__)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        results.append(result)
        print(
            f"  [{model_key}] batch {spec.language}/{spec.category}: "
            f"{result.returned}/{spec.count} "
            f"{', '.join(result.targets) if result.targets else result.error_type or ''}"
            f" ({result.latency_ms} ms)",
            file=sys.stderr,
        )
    return results


async def benchmark_model(
    settings: Settings,
    model_key: str,
    dataset: list[DatasetItem],
    rag_seed: list[Path],
    batch_specs: list[BatchSpec] | None = None,
) -> tuple[ModelSummary, list[ItemResult], list[BatchResult]]:
    from app.cli import load_jsonl
    from app.infrastructure.ai.config import build_environment, load_ai_config

    provider, model = parse_model_key(model_key)
    configured = load_ai_config(settings.ai_config_path, build_environment(settings)).models
    container = build_in_memory_container(
        settings,
        policies=pinned_policies(provider, model),
        models=registry_models(configured, provider, model),
    )
    try:
        if model_key.split(":", 1)[0] not in container.router.provider_names:
            raise SystemExit(
                f"provider '{provider}' is not available: {container.disabled_providers}"
            )
        for path in rag_seed:
            await container.index_documents.index(load_jsonl(path))
        results = await run_items(container, model_key, dataset)
        batches = await run_batches(container, model_key, batch_specs or [])
        summary = summarize(model_key, results, container.usage_store, batches=batches)
        return summary, results, batches
    finally:
        await container.aclose()


def write_reports(
    out_dir: Path,
    summaries: list[ModelSummary],
    items: list[ItemResult],
    batches: list[BatchResult] | None = None,
) -> Path:
    batches = batches or []
    target = out_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "summaries": [asdict(s) for s in summaries],
        "items": [asdict(i) for i in items],
        "batches": [asdict(b) for b in batches],
    }
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (target / "items.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(items[0]).keys()) if items else [])
        writer.writeheader()
        for item in items:
            row = asdict(item)
            row["forbidden_words"] = "|".join(item.forbidden_words)
            writer.writerow(row)
    with (target / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(s) for s in summaries)
    if batches:
        with (target / "batches.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(batches[0]).keys()))
            writer.writeheader()
            for batch in batches:
                row = asdict(batch)
                row["targets"] = "|".join(batch.targets)
                writer.writerow(row)
    return target


async def main_async(args: argparse.Namespace) -> None:
    settings = get_settings()
    dataset = load_dataset(Path(args.dataset))[: args.limit or None] if args.dataset else []
    if not dataset and not args.batch:
        raise SystemExit("give --dataset and/or --batch")
    summaries: list[ModelSummary] = []
    all_items: list[ItemResult] = []
    all_batches: list[BatchResult] = []
    for model_key in args.model:
        print(
            f"benchmarking {model_key} on {len(dataset)} items, {len(args.batch)} batches",
            file=sys.stderr,
        )
        summary, items, batches = await benchmark_model(
            settings, model_key, dataset, [Path(p) for p in args.rag_seed], args.batch
        )
        summaries.append(summary)
        all_items.extend(items)
        all_batches.extend(batches)
    path = write_reports(Path(args.out), summaries, all_items, all_batches)
    print(json.dumps([asdict(s) for s in summaries], ensure_ascii=False, indent=2))
    print(f"reports written to {path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m tests.benchmark.runner")
    parser.add_argument("--dataset")
    parser.add_argument("--model", action="append", required=True, help="provider:model")
    parser.add_argument("--rag-seed", action="append", default=[], help="JSONL RAG documents")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="benchmark-reports")
    parser.add_argument(
        "--batch", action="append", default=[], type=parse_batch_spec, help="LANG:CATEGORY:COUNT"
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
