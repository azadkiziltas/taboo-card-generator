"""Initial schema: cards, feedback, RAG documents/embeddings, model registry, telemetry.

Revision ID: 0001
Revises:
Create Date: 2026-09-24 04:13:17.860023
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "ai_models",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model_name", sa.String(length=200), nullable=False),
        sa.Column("model_type", sa.String(length=16), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("is_local", sa.Boolean(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("input_cost_per_million", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("output_cost_per_million", sa.Numeric(precision=14, scale=6), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("supports_structured_output", sa.Boolean(), nullable=False),
        sa.Column("supports_json_mode", sa.Boolean(), nullable=False),
        sa.Column("supports_tools", sa.Boolean(), nullable=False),
        sa.Column("supports_vision", sa.Boolean(), nullable=False),
        sa.Column("supports_streaming", sa.Boolean(), nullable=False),
        sa.Column("context_window", sa.Integer(), nullable=True),
        sa.Column("languages", postgresql.ARRAY(sa.String(length=8)), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("avg_latency_ms", sa.Integer(), nullable=True),
        sa.Column("health_status", sa.String(length=16), nullable=False),
        sa.Column("provider_options", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ai_models")),
        sa.UniqueConstraint(
            "provider",
            "model_name",
            "model_type",
            name=op.f("uq_ai_models_provider_model_name_model_type"),
        ),
    )
    op.create_table(
        "ai_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("task", sa.String(length=64), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("is_local", sa.Boolean(), nullable=False),
        sa.Column("prompt_name", sa.String(length=100), nullable=True),
        sa.Column("prompt_version", sa.String(length=32), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("api_estimated_cost", sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_type", sa.String(length=64), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("provider_attempt", sa.Integer(), nullable=False),
        sa.Column("fallback_used", sa.Boolean(), nullable=False),
        sa.Column("parse_failed", sa.Boolean(), nullable=False),
        sa.Column("validation_failures", sa.Integer(), nullable=True),
        sa.Column("items_accepted", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ai_requests")),
    )
    op.create_index("ix_ai_requests_created_at", "ai_requests", ["created_at"], unique=False)
    op.create_index(
        "ix_ai_requests_provider_model", "ai_requests", ["provider", "model"], unique=False
    )
    op.create_index(op.f("ix_ai_requests_run_id"), "ai_requests", ["run_id"], unique=False)
    op.create_index("ix_ai_requests_task", "ai_requests", ["task"], unique=False)
    op.create_table(
        "cards",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("target", sa.String(length=100), nullable=False),
        sa.Column("target_key", sa.String(length=100), nullable=False),
        sa.Column("language", sa.String(length=8), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("difficulty", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("ai_request_id", sa.Uuid(), nullable=True),
        sa.Column("generation_run_id", sa.Uuid(), nullable=True),
        sa.Column("replaced_by_id", sa.Uuid(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "difficulty IN ('easy', 'medium', 'hard')", name=op.f("ck_cards_difficulty")
        ),
        sa.ForeignKeyConstraint(
            ["replaced_by_id"],
            ["cards.id"],
            name=op.f("fk_cards_replaced_by_id_cards"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cards")),
    )
    op.create_index(op.f("ix_cards_ai_request_id"), "cards", ["ai_request_id"], unique=False)
    op.create_index(
        op.f("ix_cards_generation_run_id"), "cards", ["generation_run_id"], unique=False
    )
    op.create_index(
        "ix_cards_language_category_created_at",
        "cards",
        ["language", "category", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_cards_language_target_key", "cards", ["language", "target_key"], unique=False
    )
    op.create_table(
        "generation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task", sa.String(length=64), nullable=False),
        sa.Column("requested_items", sa.Integer(), nullable=False),
        sa.Column("returned_items", sa.Integer(), nullable=False),
        sa.Column("drafts_generated", sa.Integer(), nullable=False),
        sa.Column("drafts_accepted", sa.Integer(), nullable=False),
        sa.Column("validation_failures", sa.Integer(), nullable=False),
        sa.Column("duplicates_rejected", sa.Integer(), nullable=False),
        sa.Column("repair_attempts", sa.Integer(), nullable=False),
        sa.Column("generation_rounds", sa.Integer(), nullable=False),
        sa.Column("rag_enabled", sa.Boolean(), nullable=False),
        sa.Column("context_documents", sa.Integer(), nullable=False),
        sa.Column("prompt_name", sa.String(length=100), nullable=True),
        sa.Column("prompt_version", sa.String(length=32), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_type", sa.String(length=64), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_generation_runs")),
    )
    op.create_index(
        "ix_generation_runs_task_created_at",
        "generation_runs",
        ["task", "created_at"],
        unique=False,
    )
    op.create_table(
        "rag_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("language", sa.String(length=8), nullable=False),
        sa.Column("word", sa.String(length=200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("related_terms", postgresql.ARRAY(sa.String(length=200)), nullable=False),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("source_license", sa.String(length=100), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_documents")),
        sa.UniqueConstraint(
            "language", "word", "source", name=op.f("uq_rag_documents_language_word_source")
        ),
    )
    op.create_index(op.f("ix_rag_documents_language"), "rag_documents", ["language"], unique=False)
    op.create_table(
        "card_feedback",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("card_id", sa.Uuid(), nullable=False),
        sa.Column("rating", sa.String(length=16), nullable=True),
        sa.Column("difficulty_feedback", sa.String(length=16), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "rating IS NOT NULL OR difficulty_feedback IS NOT NULL",
            name=op.f("ck_card_feedback_has_value"),
        ),
        sa.ForeignKeyConstraint(
            ["card_id"],
            ["cards.id"],
            name=op.f("fk_card_feedback_card_id_cards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_card_feedback")),
    )
    op.create_index(op.f("ix_card_feedback_card_id"), "card_feedback", ["card_id"], unique=False)
    op.create_table(
        "card_forbidden_words",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("card_id", sa.Uuid(), nullable=False),
        sa.Column("word", sa.String(length=100), nullable=False),
        sa.Column("word_key", sa.String(length=100), nullable=False),
        sa.Column("position", sa.SmallInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["card_id"],
            ["cards.id"],
            name=op.f("fk_card_forbidden_words_card_id_cards"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_card_forbidden_words")),
        sa.UniqueConstraint(
            "card_id", "position", name=op.f("uq_card_forbidden_words_card_id_position")
        ),
    )
    op.create_index(
        op.f("ix_card_forbidden_words_card_id"), "card_forbidden_words", ["card_id"], unique=False
    )
    op.create_table(
        "rag_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("embedding_version", sa.String(length=32), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["rag_documents.id"],
            name=op.f("fk_rag_embeddings_document_id_rag_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_embeddings")),
        sa.UniqueConstraint(
            "document_id",
            "provider",
            "model",
            "embedding_version",
            name=op.f("uq_rag_embeddings_document_id_provider_model_embedding_version"),
        ),
    )
    op.create_index(
        "ix_rag_embeddings_space",
        "rag_embeddings",
        ["provider", "model", "embedding_version", "dimension"],
        unique=False,
    )


def downgrade() -> None:
    # The `vector` extension is intentionally kept (other schemas may use it).
    op.drop_index("ix_rag_embeddings_space", table_name="rag_embeddings")
    op.drop_table("rag_embeddings")
    op.drop_index(op.f("ix_card_forbidden_words_card_id"), table_name="card_forbidden_words")
    op.drop_table("card_forbidden_words")
    op.drop_index(op.f("ix_card_feedback_card_id"), table_name="card_feedback")
    op.drop_table("card_feedback")
    op.drop_index(op.f("ix_rag_documents_language"), table_name="rag_documents")
    op.drop_table("rag_documents")
    op.drop_index("ix_generation_runs_task_created_at", table_name="generation_runs")
    op.drop_table("generation_runs")
    op.drop_index("ix_cards_language_target_key", table_name="cards")
    op.drop_index("ix_cards_language_category_created_at", table_name="cards")
    op.drop_index(op.f("ix_cards_generation_run_id"), table_name="cards")
    op.drop_index(op.f("ix_cards_ai_request_id"), table_name="cards")
    op.drop_table("cards")
    op.drop_index("ix_ai_requests_task", table_name="ai_requests")
    op.drop_index(op.f("ix_ai_requests_run_id"), table_name="ai_requests")
    op.drop_index("ix_ai_requests_provider_model", table_name="ai_requests")
    op.drop_index("ix_ai_requests_created_at", table_name="ai_requests")
    op.drop_table("ai_requests")
    op.drop_table("ai_models")
