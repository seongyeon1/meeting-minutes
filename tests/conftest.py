"""Every test starts from the built-in model list and empty user settings, whatever llm.toml sits in the
cwd and whatever you saved under ~/.config/voicekit on this machine."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import meeting_minutes as mm


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    root = tmp_path_factory.mktemp("voicekit-config")
    monkeypatch.setenv("VOICEKIT_CONFIG_DIR", str(root))
    monkeypatch.setenv("VOICEKIT_MEMORY_DIR", str(root / "memory"))
    saved = (mm._llm_config, mm._model_sources, mm._project_llm_path)
    mm._llm_config, mm._model_sources, mm._project_llm_path = {}, {}, None
    yield
    mm._llm_config, mm._model_sources, mm._project_llm_path = saved


@pytest.fixture
def config_dir() -> Path:
    import os

    return Path(os.environ["VOICEKIT_CONFIG_DIR"])
