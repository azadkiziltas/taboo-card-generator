from __future__ import annotations

import pytest

from tests.helpers import Harness, build_env


@pytest.fixture
def env() -> Harness:
    return build_env()
