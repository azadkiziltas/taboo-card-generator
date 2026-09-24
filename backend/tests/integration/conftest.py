"""Integration fixtures.

A dedicated database (`<name>_test`) is created next to the configured one and migrated
with Alembic, so the development data is never touched. Tests are skipped when
PostgreSQL is not reachable (e.g. `docker compose up -d postgres` was not run).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator

import asyncpg
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import BACKEND_ROOT, Settings

pytestmark = pytest.mark.integration

TABLES = (
    "card_feedback",
    "card_forbidden_words",
    "cards",
    "rag_embeddings",
    "rag_documents",
    "ai_requests",
    "generation_runs",
    "ai_models",
)


def _test_database_url() -> str:
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    base = make_url(Settings().database_url)
    return base.set(database=f"{base.database}_test").render_as_string(hide_password=False)


async def _ensure_database(url: str) -> None:
    parsed = make_url(url)
    admin = await asyncpg.connect(
        host=parsed.host,
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        database="postgres",
        timeout=3,
    )
    try:
        exists = await admin.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", parsed.database
        )
        if not exists:
            await admin.execute(f'CREATE DATABASE "{parsed.database}"')
    finally:
        await admin.close()


@pytest.fixture(scope="session")
def database_url() -> str:
    url = _test_database_url()
    try:
        asyncio.run(_ensure_database(url))
    except (OSError, asyncpg.PostgresError, TimeoutError) as exc:
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_ROOT,
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"migrations failed:\n{result.stderr}")
    return url


@pytest.fixture
async def sessions(database_url: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()
