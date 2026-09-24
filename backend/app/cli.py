"""Operational commands.

python -m app.cli seed-models [--overwrite]
python -m app.cli rag-ingest data/rag/seed_tr.jsonl [more.jsonl ...]
python -m app.cli rag-reindex [--language tr]
python -m app.cli rag-search --language tr "mutfak"
python -m app.cli routing [--task card_generation]
python -m app.cli generate --language tr --category technology --count 3
python -m app.cli generate-for-target --language tr --target uçak [--no-persist]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.application.cards.generate_card import GenerateCardCommand
from app.application.cards.generate_cards import GenerateCardsCommand
from app.bootstrap import Container, build_container
from app.config.settings import get_settings
from app.domain.cards.entities import Card, Difficulty
from app.domain.rag.entities import RagDocument
from app.infrastructure.observability.logging import configure_logging


def _card_json(card: Card) -> dict[str, Any]:
    return {
        "id": str(card.id),
        "target": card.target,
        "forbidden_words": list(card.forbidden_words),
        "language": card.language,
        "category": card.category,
        "difficulty": card.difficulty.value if card.difficulty else None,
        "status": card.status.value,
    }


def load_jsonl(path: Path) -> list[RagDocument]:
    documents = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            documents.append(
                RagDocument(
                    language=item["language"],
                    word=item["word"],
                    content=item["content"],
                    category=item.get("category"),
                    related_terms=tuple(item.get("related_terms") or ()),
                    source=item["source"],
                    source_license=item.get("source_license"),
                    metadata=item.get("metadata") or {},
                )
            )
        except (KeyError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{path}:{number}: invalid document ({exc})") from exc
    return documents


async def _run(args: argparse.Namespace) -> Any:
    settings = get_settings()
    configure_logging(settings.log_level)
    container: Container = await build_container(settings)
    try:
        match args.command:
            case "seed-models":
                from app.infrastructure.ai.config import build_environment, load_ai_config
                from app.infrastructure.ai.registry.database_registry import sync_registry

                config = load_ai_config(settings.ai_config_path, build_environment(settings))
                models = await sync_registry(
                    container.registry, config.models, overwrite=args.overwrite
                )
                return [m.key + f" ({m.model_type.value})" for m in models]
            case "rag-ingest":
                documents = [doc for path in args.paths for doc in load_jsonl(Path(path))]
                report = await container.index_documents.index(documents)
                return asdict(report)
            case "rag-reindex":
                report = await container.index_documents.reindex_missing(language=args.language)
                return asdict(report)
            case "rag-search":
                context = await container.retrieve_context.execute(
                    language=args.language, query=args.query, top_k=args.top_k
                )
                return {
                    "error": context.error,
                    "documents": [
                        {"word": s.document.word, "similarity": round(s.similarity, 4)}
                        for s in context.documents
                    ],
                }
            case "routing":
                candidates = await container.router.candidates(args.task)
                return {"task": args.task, "candidates": [m.key for m in candidates]}
            case "generate":
                cards = await container.generate_cards.execute(
                    GenerateCardsCommand(
                        language=args.language,
                        category=args.category,
                        difficulty=Difficulty(args.difficulty) if args.difficulty else None,
                        count=args.count,
                        topic=args.topic,
                    )
                )
                return [_card_json(card) for card in cards]
            case "generate-for-target":
                card = await container.generate_card.execute(
                    GenerateCardCommand(
                        target=args.target,
                        language=args.language,
                        difficulty=Difficulty(args.difficulty) if args.difficulty else None,
                        category=args.category,
                        persist=not args.no_persist,
                    )
                )
                return _card_json(card)
    finally:
        await container.aclose()
    raise SystemExit(f"unknown command {args.command}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser("seed-models", help="sync config/ai.yaml models into the registry")
    seed.add_argument("--overwrite", action="store_true", help="overwrite existing entries")

    ingest = sub.add_parser("rag-ingest", help="upsert + embed JSONL RAG documents")
    ingest.add_argument("paths", nargs="+")

    reindex = sub.add_parser("rag-reindex", help="embed documents missing in the active space")
    reindex.add_argument("--language")

    search = sub.add_parser("rag-search", help="debug retrieval")
    search.add_argument("--language", required=True)
    search.add_argument("--top-k", type=int)
    search.add_argument("query")

    routing = sub.add_parser("routing", help="show routing candidates for a task")
    routing.add_argument("--task", default="card_generation")

    generate = sub.add_parser("generate", help="generate cards (same as the public API)")
    generate.add_argument("--language", required=True)
    generate.add_argument("--category")
    generate.add_argument("--difficulty", choices=[d.value for d in Difficulty], help="optional")
    generate.add_argument("--count", type=int, default=5)
    generate.add_argument("--topic")

    single = sub.add_parser("generate-for-target", help="generate one card for a given target")
    single.add_argument("--language", required=True)
    single.add_argument("--target", required=True)
    single.add_argument("--category")
    single.add_argument("--difficulty", choices=[d.value for d in Difficulty], help="optional")
    single.add_argument("--no-persist", action="store_true")

    args = parser.parse_args(argv)
    result = asyncio.run(_run(args))
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
