"""Admin/debug API. Separate prefix, token protected, never used by the game client."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.admin.schemas import (
    GenerateForTargetRequest,
    ModelOut,
    ModelPatch,
    RagIngestRequest,
    RagSearchRequest,
)
from app.api.dependencies import ContainerDep, require_admin
from app.api.v1.schemas import CardResponse
from app.application.cards.generate_card import GenerateCardCommand
from app.domain.rag.entities import RagDocument

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])


# -- model registry ---------------------------------------------------------------------


@router.get("/models", response_model=list[ModelOut], summary="List the model registry")
async def list_models(container: ContainerDep) -> list[ModelOut]:
    return [ModelOut.from_descriptor(m) for m in await container.registry.list_models()]


@router.patch("/models/{model_id}", response_model=ModelOut, summary="Update a registry entry")
async def update_model(model_id: uuid.UUID, body: ModelPatch, container: ContainerDep) -> ModelOut:
    changes = body.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=422, detail="No changes given.")
    if await container.registry.get_by_id(model_id) is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    return ModelOut.from_descriptor(await container.registry.update(model_id, changes))


@router.get("/routing/{task}", summary="Models the router would try for a task, in order")
async def routing_preview(
    task: str, container: ContainerDep, language: str | None = None
) -> dict[str, Any]:
    requirements = {"language": language, "structured_output": True} if language else None
    candidates = await container.router.candidates(task, requirements)
    return {"task": task, "candidates": [m.key for m in candidates]}


# -- health / telemetry -----------------------------------------------------------------


@router.get("/health", summary="Detailed provider health and circuit breaker state")
async def detailed_health(container: ContainerDep) -> dict[str, Any]:
    status = await container.health.detailed_status()
    status["disabled_providers"] = container.disabled_providers
    return status


@router.get("/telemetry", summary="Aggregated AI request and generation-run statistics")
async def telemetry(
    container: ContainerDep, hours: int = Query(default=24, ge=1, le=24 * 90)
) -> dict[str, Any]:
    if container.usage_query is None:
        raise HTTPException(status_code=404, detail="Telemetry storage is not configured.")
    since = datetime.now(UTC) - timedelta(hours=hours)
    return {
        "since": since.isoformat(),
        "requests": await container.usage_query.request_stats(since),
        "runs": await container.usage_query.run_stats(since),
    }


@router.get("/metrics", summary="In-process counters since this worker started")
async def metrics(container: ContainerDep) -> dict[str, Any]:
    return container.metrics.snapshot()


# -- RAG ----------------------------------------------------------------------------------


@router.post("/rag/documents", summary="Upsert and embed RAG documents")
async def ingest_documents(body: RagIngestRequest, container: ContainerDep) -> dict[str, Any]:
    documents = [
        RagDocument(
            language=d.language,
            word=d.word.strip(),
            content=d.content.strip(),
            category=d.category,
            related_terms=tuple(d.related_terms),
            source=d.source,
            source_license=d.source_license,
            metadata=d.metadata,
        )
        for d in body.documents
    ]
    report = await container.index_documents.index(documents)
    return {
        "documents_upserted": report.documents_upserted,
        "embeddings_written": report.embeddings_written,
        "dimension": report.dimension,
        "embedding_space": report.space,
    }


@router.post("/rag/reindex", summary="Embed documents missing from the active embedding space")
async def reindex(container: ContainerDep, language: str | None = None) -> dict[str, Any]:
    report = await container.index_documents.reindex_missing(language=language)
    return {"embeddings_written": report.embeddings_written, "embedding_space": report.space}


@router.post("/rag/search", summary="Debug retrieval for a query")
async def rag_search(body: RagSearchRequest, container: ContainerDep) -> dict[str, Any]:
    context = await container.retrieve_context.execute(
        language=body.language, query=body.query, top_k=body.top_k
    )
    return {
        "enabled": context.enabled,
        "error": context.error,
        "documents": [
            {
                "word": s.document.word,
                "similarity": round(s.similarity, 4),
                "source": s.document.source,
            }
            for s in context.documents
        ],
    }


# -- cards --------------------------------------------------------------------------------


@router.post(
    "/cards/generate-for-target",
    response_model=CardResponse,
    summary="Generate forbidden words for a given target word",
)
async def generate_for_target(
    body: GenerateForTargetRequest, container: ContainerDep
) -> CardResponse:
    card = await container.generate_card.execute(
        GenerateCardCommand(
            target=body.target,
            language=body.language,
            difficulty=body.difficulty,
            category=body.category,
            persist=body.persist,
        )
    )
    return CardResponse.from_card(card)
