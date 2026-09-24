"""Card difficulty becomes optional (cards generated without a difficulty level).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-24 15:00:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The existing CHECK (difficulty IN (...)) already accepts NULL.
    op.alter_column("cards", "difficulty", existing_type=sa.String(length=16), nullable=True)


def downgrade() -> None:
    op.execute("UPDATE cards SET difficulty = 'medium' WHERE difficulty IS NULL")
    op.alter_column("cards", "difficulty", existing_type=sa.String(length=16), nullable=False)
