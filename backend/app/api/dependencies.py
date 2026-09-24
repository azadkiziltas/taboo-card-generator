from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from app.bootstrap import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


async def rate_limit(request: Request, container: ContainerDep) -> None:
    client = request.client.host if request.client else "unknown"
    await container.rate_limiter.check(f"{request.url.path}:{client}")


async def require_admin(
    container: ContainerDep,
    x_admin_token: Annotated[str | None, Header()] = None,
) -> None:
    configured = container.settings.admin_api_token
    if configured is None or not configured.get_secret_value():
        raise HTTPException(status_code=403, detail="Admin API is disabled.")
    if not x_admin_token or not secrets.compare_digest(
        x_admin_token.encode(), configured.get_secret_value().encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid admin token.")
