"""Every test starts from the built-in model list, whatever llm.toml sits in the cwd."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import meeting_minutes as mm


@pytest.fixture(autouse=True)
def _no_llm_config() -> Iterator[None]:
    saved = mm._llm_config
    mm._llm_config = {}
    yield
    mm._llm_config = saved
