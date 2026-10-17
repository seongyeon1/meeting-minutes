"""Model resolution and the OpenAI-compatible streaming client (network faked)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

import meeting_minutes as mm


def test_builtin_and_provider_names() -> None:
    assert mm.resolve_llm("claude") == mm.LLMSpec("anthropic:claude-opus-5-5", "anthropic", mm.CLAUDE_MODEL)
    gpt = mm.resolve_llm("gpt")
    assert (gpt.provider, gpt.model, gpt.reasoning) == ("openai", "gpt-5.2", True)
    local = mm.resolve_llm("ollama:qwen3:14b")  # the model id itself may contain a colon
    assert (local.provider, local.model, local.reasoning) == ("ollama", "qwen3:14b", False)
    for bad in ("nope", "mystery:model", "openai:"):
        with pytest.raises(ValueError):
            mm.resolve_llm(bad)


def test_config_aliases_and_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.toml"
    cfg.write_text(
        'default = "team"\nanalysis_default = "big"\n'
        '[models.team]\nbase_url = "https://gw.example.com/api/chat/completions"\nmodel = "team-model"\n'
        'api_key_env = "TEAM_KEY"\n'
        '[models.big]\nbase_url = "https://gw.example.com/api/chat/completions"\nmodel = "big-model"\n'
        'api_key_env = "BIG_KEY"\nreasoning = true\n',
        encoding="utf-8",
    )
    mm.configure_llms(cfg)
    assert (mm.default_llm(), mm.default_analysis_llm()) == ("team", "big")
    assert mm.llm_choices() == ["team", "big", "claude", "gpt", "claude-code", "codex"]
    team = mm.resolve_llm("team")
    assert (team.provider, team.api_key_env, team.reasoning) == ("compat", "TEAM_KEY", False)
    assert mm.resolve_llm("big").reasoning


def test_config_with_unknown_default_fails_at_startup(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.toml"
    cfg.write_text('default = "missing"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        mm.configure_llms(cfg)


class FakeStream:
    def __init__(self, lines: list[str], status: int = 200) -> None:
        self.lines, self.status_code = lines, status

    def iter_lines(self) -> list[str]:
        return self.lines

    def read(self) -> bytes:
        return b"bad request"


def fake_stream(monkeypatch: pytest.MonkeyPatch, lines: list[str], status: int = 200) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    @contextmanager
    def stream(method: str, url: str, **kw: Any) -> Any:
        calls.append({"url": url, **kw})
        yield FakeStream(lines, status)

    monkeypatch.setattr(httpx, "stream", stream)
    return calls


def sse(*texts: str, finish: str = "stop") -> list[str]:
    out = [f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}" for t in texts]
    out.append("data:" + json.dumps({"choices": [{"delta": {}, "finish_reason": finish}]}))  # no space after colon
    return [*out, "", "data: [DONE]"]


def test_openai_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    calls = fake_stream(monkeypatch, sse("# 회의", "록"))
    assert mm._complete("녹취록", system="sys", llm="gpt", env_file=None) == "# 회의록"
    call = calls[0]
    assert call["url"] == "https://api.openai.com/v1/chat/completions"
    assert call["headers"] == {"Authorization": "Bearer sk-test"}
    body = call["json"]
    assert body["stream"] and body["max_completion_tokens"] and "temperature" not in body
    assert body["messages"][0] == {"role": "system", "content": "sys"}


def test_ollama_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1/")
    calls = fake_stream(monkeypatch, sse("ok"))
    assert mm._complete("x", system="s", llm="ollama:qwen3:14b", env_file=None) == "ok"
    assert calls[0]["url"] == "http://gpu-box:11434/v1/chat/completions" and calls[0]["headers"] == {}
    assert calls[0]["json"]["max_tokens"] == 8192


def test_key_from_env_file_and_full_endpoint_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mm._llm_config = {"models": {"team": {"base_url": "https://gw.example.com/v2/chat/completions",
                                          "model": "m", "api_key_env": "TEAM_KEY"}}}
    monkeypatch.delenv("TEAM_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("TEAM_KEY='k-1'\n", encoding="utf-8")
    calls = fake_stream(monkeypatch, sse("ok"))
    mm._complete("x", system="s", llm="team", env_file=env)
    assert calls[0]["url"] == "https://gw.example.com/v2/chat/completions"
    assert calls[0]["headers"] == {"Authorization": "Bearer k-1"}

    with pytest.raises(SystemExit, match="TEAM_KEY"):
        mm._complete("x", system="s", llm="team", env_file=None)


@pytest.mark.parametrize(
    ("lines", "status", "match"),
    [
        (sse(), 200, "empty response"),
        (['data:{"state": "fail", "res": {"msg": "quota"}}'], 200, "quota"),
        (['data: {"error": {"message": "overloaded"}}'], 200, "overloaded"),
        ([], 400, "HTTP 400"),
    ],
)
def test_stream_errors(monkeypatch: pytest.MonkeyPatch, lines: list[str], status: int, match: str) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_stream(monkeypatch, lines, status)
    with pytest.raises(RuntimeError, match=match):
        mm._complete("x", system="s", llm="gpt", env_file=None)


def test_compat_without_base_url_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="LLM_BASE_URL"):
        mm._complete("x", system="s", llm="compat:any", env_file=None)
