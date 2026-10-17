"""LLM settings saved from the web UI: model aliases, defaults and API keys, layered over llm.toml and the
environment. No network: the connectivity test is stubbed."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import meeting_minutes as mm
import meeting_minutes_web as web
from voicekit import llm_settings, manage

TEAM = {"provider": "compat", "base_url": "https://gw.example.com/v1", "model": "team-1", "api_key_env": "TEAM_LLM_KEY"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(manage, "USER_PROFILE_DIR", tmp_path / "profiles")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TEAM_LLM_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return TestClient(web.create_app(tmp_path / "jobs", None))


def test_save_and_remove_a_model(client: TestClient, config_dir: Path) -> None:
    assert client.put("/api/llm/models/team", json=TEAM).status_code == 200
    assert "team" in client.get("/api/config").json()["llms"]
    spec = mm.resolve_llm("team")
    assert (spec.provider, spec.model, spec.base_url, spec.api_key_env) == ("compat", "team-1", TEAM["base_url"], "TEAM_LLM_KEY")
    saved = (config_dir / "llm.toml").read_text(encoding="utf-8")
    assert '[models."team"]' in saved

    row = next(m for m in client.get("/api/llm").json()["models"] if m["name"] == "team")
    assert row["source"] == "user" and row["key_name"] == "TEAM_LLM_KEY" and "TEAM_LLM_KEY" in row["problem"]

    assert client.delete("/api/llm/models/team").status_code == 200
    assert "team" not in client.get("/api/config").json()["llms"]
    assert client.delete("/api/llm/models/team").status_code == 404
    assert client.delete("/api/llm/models/claude").status_code == 404  # built-ins cannot be removed


def test_bad_models_are_refused(client: TestClient) -> None:
    put = client.put
    assert put("/api/llm/models/bad name", json=TEAM).status_code == 400
    assert put("/api/llm/models/a:b", json=TEAM).status_code in (400, 404)
    assert put("/api/llm/models/x", json={**TEAM, "provider": "nope"}).status_code == 400
    assert put("/api/llm/models/x", json={**TEAM, "model": " "}).status_code == 400
    assert put("/api/llm/models/x", json={**TEAM, "base_url": "file:///etc/passwd"}).status_code == 400
    assert put("/api/llm/models/x", json={**TEAM, "api_key_env": "lower case"}).status_code == 400
    assert put("/api/llm/models/x", json=TEAM, headers={"Origin": "http://evil.example"}).status_code == 403


def test_user_models_layer_over_the_project_file(tmp_path: Path, client: TestClient) -> None:
    project = tmp_path / "llm.toml"
    project.write_text('default = "proj"\n[models.proj]\nprovider = "ollama"\nmodel = "qwen3:14b"\n'
                       '[models.team]\nprovider = "ollama"\nmodel = "from-project"\n', encoding="utf-8")
    mm.configure_llms(project)
    sources = {m["name"]: m["source"] for m in mm.llm_models()}
    assert sources["proj"] == "file" and sources["claude"] == "builtin"

    llm_settings.upsert_model("team", TEAM, {"compat", "ollama"})
    mm.reload_llms()  # remembers the project path
    assert mm.resolve_llm("team").model == "team-1"  # same name: the one you saved wins
    assert {m["name"]: m["source"] for m in mm.llm_models()}["team"] == "user"
    assert mm.default_llm() == "proj"


def test_defaults_can_be_set_and_a_stale_one_is_ignored(client: TestClient, config_dir: Path) -> None:
    client.put("/api/llm/models/team", json=TEAM)
    r = client.put("/api/llm/defaults", json={"default": "team", "analysis_default": "gpt"})
    assert r.json() == {"default": "team", "analysis_default": "gpt"}
    assert mm.default_llm() == "team" and client.get("/api/llm").json()["analysis_default"] == "gpt"
    assert client.put("/api/llm/defaults", json={"default": "nope"}).status_code == 400

    # Removing the model that was the default leaves a working server, not a broken startup.
    client.delete("/api/llm/models/team")
    assert mm.default_llm() == mm.DEFAULT_LLM
    (config_dir / "llm.toml").write_text('default = "ghost"\n', encoding="utf-8")
    mm.configure_llms(None)
    assert mm.default_llm() == mm.DEFAULT_LLM


def test_keys_are_saved_privately_and_never_returned(client: TestClient, config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-test-1234567890abcdef"
    assert client.put("/api/llm/keys/OPENAI_API_KEY", json={"value": secret}).status_code == 200
    path = config_dir / "secrets.env"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert f"OPENAI_API_KEY={secret}" in path.read_text()

    listing = client.get("/api/llm")
    assert secret not in listing.text
    key = next(k for k in listing.json()["keys"] if k["name"] == "OPENAI_API_KEY")
    assert key == {"name": "OPENAI_API_KEY", "source": "saved", "hint": "…cdef"}
    gpt = next(m for m in listing.json()["models"] if m["name"] == "gpt")
    assert gpt["problem"] is None and gpt["key_source"] == "saved"

    monkeypatch.setenv("OPENAI_API_KEY", "from-env")  # an exported variable keeps winning
    assert mm._secret("OPENAI_API_KEY", None) == "from-env"
    assert mm.secret_source("OPENAI_API_KEY", None) == "env"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert mm._secret("OPENAI_API_KEY", None) == secret

    assert client.delete("/api/llm/keys/OPENAI_API_KEY").status_code == 200
    assert mm._secret("OPENAI_API_KEY", None) is None
    assert client.delete("/api/llm/keys/OPENAI_API_KEY").status_code == 404


def test_bad_keys_are_refused(client: TestClient) -> None:
    assert client.put("/api/llm/keys/lowercase", json={"value": "x"}).status_code == 400
    assert client.put("/api/llm/keys/MY_KEY", json={"value": ""}).status_code == 400
    assert client.put("/api/llm/keys/MY_KEY", json={"value": "a\nB=c"}).status_code == 400
    assert client.put("/api/llm/keys/../x", json={"value": "x"}).status_code in (404, 405)
    assert llm_settings.saved_keys() == {}


def test_connection_test_reports_success_and_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(user: str, *, system: str, llm: str, env_file: Any) -> str:
        calls.append(llm)
        if llm == "gpt":
            raise SystemExit("OPENAI_API_KEY 가 없습니다.")
        return " 확인 "

    monkeypatch.setattr(mm, "_complete", fake)
    ok = client.post("/api/llm/test", json={"name": "claude"}).json()
    assert ok["ok"] is True and ok["reply"] == "확인" and calls == ["claude"]
    bad = client.post("/api/llm/test", json={"name": "gpt"}).json()
    assert bad["ok"] is False and "OPENAI_API_KEY" in bad["error"]
    assert client.post("/api/llm/test", json={"name": "unknown-model"}).status_code == 400


def test_a_broken_user_file_does_not_stop_startup(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "llm.toml").write_text("this is [not toml", encoding="utf-8")
    mm.configure_llms(None)
    assert "claude" in mm.llm_choices()
