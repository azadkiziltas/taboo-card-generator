"""The internal UI is served from the same origin and only uses public endpoints."""

import re

from fastapi.testclient import TestClient

from app.config.settings import APP_ROOT
from app.main import create_app
from tests.helpers import build_env

UI = APP_ROOT / "ui" / "static"


def test_ui_is_served_and_root_redirects():
    harness = build_env()
    app = create_app(harness.container.settings, container=harness.container)
    with TestClient(app) as client:
        root = client.get("/", follow_redirects=False)
        assert root.status_code in (302, 307) and root.headers["location"] == "/ui/"
        page = client.get("/ui/")
        assert page.status_code == 200 and "Kart Üretimi" in page.text
        assert 'name="topic"' not in page.text  # the category field also takes a theme
        assert client.get("/ui/app.js").status_code == 200
        assert "text/css" in client.get("/ui/app.css").headers["content-type"]
        assert "/ui/" not in client.get("/openapi.json").json()["paths"]


def test_ui_uses_only_public_card_endpoints_and_never_sends_a_model():
    script = (UI / "app.js").read_text(encoding="utf-8")
    paths = set(re.findall(r'"(/(?:v1|admin)[^"`$]*)', script)) | set(
        re.findall(r"`(/(?:v1|admin)[^`$]*)", script)
    )
    assert paths and all(p.startswith("/v1/") for p in paths)
    assert not re.search(r"\b(model|provider)\s*:", script)
    assert "innerHTML" not in script  # all card text goes through textContent
