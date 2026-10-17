"""Profiles, agent-CLI hand-off, the shared job flow in local-only mode, and the MCP tool surface."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

import meeting_minutes as mm
from voicekit import agent_cli
from voicekit.profiles import get_profile, load_profiles, with_profile
from voicekit.service import Service

SEGMENTS = [{"start": 0.0, "end": 3.0, "text": "안녕하세요 반갑습니다", "speaker": "SPEAKER_00"}]


# ---- profiles ----------------------------------------------------------------------------------------------


def test_builtin_profiles_cover_recording_types() -> None:
    profiles = load_profiles()
    assert {"meeting", "interview", "call", "lecture", "podcast", "memo", "general"} <= set(profiles)
    assert all(p.system.strip() and p.document for p in profiles.values())
    # one-voice recordings skip diarization by default
    assert not profiles["lecture"].diarize and not profiles["memo"].diarize
    assert profiles["meeting"].diarize


def test_meeting_profile_keeps_the_original_prompt() -> None:
    assert mm.MINUTES_SYSTEM == get_profile("meeting").system
    assert "# 회의록: <회의 주제 한 줄>" in mm.MINUTES_SYSTEM


def test_user_profile_overrides_and_bad_file_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "sales.toml").write_text(
        'label = "영업 통화"\ndocument = "영업 노트"\ndiarize = true\nsystem_file = "sales.md"\n', encoding="utf-8"
    )
    (tmp_path / "sales.md").write_text("영업 통화를 정리하세요.", encoding="utf-8")
    (tmp_path / "broken.toml").write_text('label = "x"\n', encoding="utf-8")  # no system prompt
    (tmp_path / "memo.toml").write_text('label = "내 메모"\nsystem = "짧게"\n', encoding="utf-8")
    monkeypatch.setenv("VOICEKIT_PROFILES", str(tmp_path))
    profiles = load_profiles()
    assert profiles["sales"].document == "영업 노트" and profiles["sales"].source == "env"
    assert "broken" not in profiles
    assert profiles["memo"].label == "내 메모"  # user file wins over the built-in with the same id


def test_unknown_profile_lists_choices() -> None:
    with pytest.raises(ValueError, match="meeting"):
        get_profile("nope")


def test_profile_changes_labels_and_prompts() -> None:
    turns = [mm.Turn("SPEAKER_00", 0, 3, "안녕하세요")]
    interview = get_profile("interview")
    prompt = mm.build_user_prompt(turns, names="홍길동", context="신제품 취재", vocab=None, profile=interview)
    assert f"{interview.names_label}: 홍길동" in prompt and f"{interview.context_label}: 신제품 취재" in prompt
    default = mm.build_user_prompt(turns, names="홍길동", context="x", vocab=None)
    assert "참석자 이름 후보: 홍길동" in default and "회의 배경: x" in default
    assert with_profile(mm.CORRECTION_SYSTEM, get_profile("meeting")) == mm.CORRECTION_SYSTEM
    assert "인터뷰" in with_profile(mm.CORRECTION_SYSTEM, interview)


# ---- agent CLI hand-off ------------------------------------------------------------------------------------


def test_agent_commands_disable_tools_and_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_cli.shutil, "which", lambda name: f"/bin/{name}")
    claude = agent_cli.build_command("claude", system="SYS", model="opus", out_file=tmp_path / "o")
    assert claude[claude.index("--tools") + 1] == "" and "--no-session-persistence" in claude
    assert claude[claude.index("--system-prompt") + 1] == "SYS" and claude[-2:] == ["--model", "opus"]
    codex = agent_cli.build_command("codex", system="SYS", model=None, out_file=tmp_path / "o")
    assert codex[:2] == ["/bin/codex", "exec"] and "read-only" in codex and codex[-1] == "-"


def test_agent_missing_binary_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_cli.shutil, "which", lambda name: None)
    monkeypatch.delenv("VOICEKIT_CODEX_BIN", raising=False)
    with pytest.raises(SystemExit, match="codex"):
        agent_cli.agent_binary("codex")


def test_agent_run_passes_prompt_on_stdin_and_reads_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        seen.update(cmd=cmd, **kw)
        Path(cmd[cmd.index("-o") + 1]).write_text("답변\n", encoding="utf-8")

        class P:
            returncode, stdout, stderr = 0, "", ""

        return P()

    monkeypatch.setattr(agent_cli.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(agent_cli.subprocess, "run", fake_run)
    assert agent_cli.complete_via_agent("U", system="S", agent="codex") == "답변"
    assert seen["input"] == "S\n\n---\n\nU"
    assert seen["cwd"] != str(Path.cwd())  # runs in an empty temp dir, not the project


def test_agent_failure_and_empty_answers_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_cli.shutil, "which", lambda name: f"/bin/{name}")

    def run(rc: int, out: str) -> Any:
        class P:
            returncode, stdout, stderr = rc, out, "boom"

        return lambda *a, **k: P()

    monkeypatch.setattr(agent_cli.subprocess, "run", run(1, ""))
    with pytest.raises(RuntimeError, match="exit 1"):
        agent_cli.complete_via_agent("U", system="S", agent="claude")
    monkeypatch.setattr(agent_cli.subprocess, "run", run(0, "  "))
    with pytest.raises(RuntimeError, match="empty"):
        agent_cli.complete_via_agent("U", system="S", agent="claude")


def test_cli_provider_names_resolve_and_route(monkeypatch: pytest.MonkeyPatch) -> None:
    mm.configure_llms(Path("/nonexistent/llm.toml"))
    assert mm.resolve_llm("claude-code").provider == "cli"
    assert mm.resolve_llm("cli:codex:gpt-5.2").model == "codex:gpt-5.2"
    with pytest.raises(ValueError):
        mm.resolve_llm("cli:bash")
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(mm, "complete_via_agent", lambda user, **kw: calls.append((user, kw)) or "ok")
    assert mm._complete("hi", system="s", llm="cli:codex:gpt-5.2", env_file=None) == "ok"
    assert calls == [("hi", {"system": "s", "agent": "codex", "model": "gpt-5.2"})]


# ---- local-only job flow + service -------------------------------------------------------------------------


@pytest.fixture
def svc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Service:
    seen: dict[str, Any] = {}

    def fake_stt(audio: Path, **kw: Any) -> list[dict[str, Any]]:
        seen.update(kw)
        return SEGMENTS

    monkeypatch.setattr(mm, "transcribe_and_diarize", fake_stt)
    monkeypatch.chdir(tmp_path)
    service = Service(tmp_path / "jobs")
    service.seen = seen  # type: ignore[attr-defined]
    return service


def finish(svc: Service, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        st = svc.status(job_id)
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_local_only_job_stops_after_transcript_and_uses_profile_defaults(svc: Service, tmp_path: Path) -> None:
    audio = tmp_path / "강의.m4a"
    audio.write_bytes(b"x" * 64)
    job = svc.start_transcription(str(audio), profile="lecture", language="auto", vocab="LLM")
    st = finish(svc, job["id"])

    assert st["status"] == "done" and st["local_only"] and st["has_transcript"] and not st["has_minutes"]
    assert st["diarize"] is False and st["document"] == "강의 노트"
    kw = svc.seen  # type: ignore[attr-defined]
    assert kw["diarize"] is False and kw["language"] is None and kw["vocab"] == "LLM"
    assert svc.status(job["id"])["analysis_status"] == "none"  # no LLM stage ran


def test_agent_writes_document_through_the_prompt_and_save_cycle(svc: Service, tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 64)
    job_id = svc.start_transcription(str(audio), profile="interview")["id"]
    finish(svc, job_id)

    page = svc.read_transcript(job_id)
    assert page["total_turns"] == 1 and page["next_offset"] is None and "안녕하세요" in page["text"]
    svc.set_speaker_names(job_id, {"SPEAKER_00": "김기자"})
    prompt = svc.document_prompt(job_id, names="김기자,박교수", context="신제품 취재")
    assert prompt["document"] == "인터뷰 정리" and prompt["system"] == get_profile("interview").system
    assert "김기자" in prompt["user"] and "신제품 취재" in prompt["user"]

    svc.save_document(job_id, "# 인터뷰 정리: 테스트")
    st = svc.status(job_id)
    assert st["has_minutes"] and (Path(st and svc.job(job_id).out_dir) / "minutes.md").read_text(encoding="utf-8").startswith("# 인터뷰")


def test_transcript_pages_and_guards(svc: Service, tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 8)
    job_id = svc.start_transcription(str(audio))["id"]
    finish(svc, job_id)
    out = svc.job(job_id).out_dir
    turns = [{"speaker": "SPEAKER_00", "start": i, "end": i + 1, "text": "가" * 50} for i in range(5)]
    (out / "transcript.json").write_text(json.dumps({"source": "x", "turns": turns}), encoding="utf-8")

    first = svc.read_transcript(job_id, max_chars=120)
    assert (first["from_turn"], first["to_turn"], first["next_offset"]) == (0, 2, 2)
    last = svc.read_transcript(job_id, offset_turn=first["next_offset"], max_chars=1000)
    assert last["next_offset"] is None and last["to_turn"] == 5

    with pytest.raises(KeyError):
        svc.status("../etc")
    with pytest.raises(ValueError, match="unknown speaker"):
        svc.set_speaker_names(job_id, {"SPEAKER_09": "x"})
    with pytest.raises(FileNotFoundError):
        svc.start_transcription(str(tmp_path / "missing.wav"))
    with pytest.raises(ValueError, match="empty"):
        svc.save_document(job_id, "  ")


def test_generate_document_uses_chosen_llm(svc: Service, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 8)
    job_id = svc.start_transcription(str(audio), profile="call")["id"]
    finish(svc, job_id)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(mm, "_complete", lambda user, **kw: seen.update(kw) or "# 상담 요약")
    result = svc.generate_document(job_id, llm="claude-code")
    assert seen["llm"] == "claude-code" and seen["system"] == get_profile("call").system
    assert result["chars"] == len("# 상담 요약") and svc.status(job_id)["has_minutes"]


# ---- MCP surface -------------------------------------------------------------------------------------------


def test_mcp_exposes_the_expected_tools_with_schemas() -> None:
    pytest.importorskip("mcp")
    from voicekit.mcp_server import mcp

    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert set(tools) == {
        "list_profiles", "transcribe_audio", "wait_for_job", "job_status", "list_jobs", "read_transcript",
        "get_document_prompt", "save_document", "generate_document", "set_speaker_names",
    }
    tool = tools["transcribe_audio"]
    schema = getattr(tool, "input_schema", None) or tool.inputSchema  # mcp 2.x / 1.x
    assert schema["required"] == ["path"] and {"profile", "language", "speakers", "diarize"} <= set(schema["properties"])


def test_mcp_tool_errors_are_readable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    import voicekit.mcp_server as server

    monkeypatch.setenv("VOICEKIT_HOME", str(tmp_path / "jobs"))
    monkeypatch.setattr(server, "_service", None)
    with pytest.raises(server.ToolError, match="job not found: nope"):
        server.job_status("nope")
    with pytest.raises(server.ToolError, match="audio file not found"):
        server.transcribe_audio(str(tmp_path / "missing.wav"))
