"""Running a local agent on one job from the UI: the command it gets, the limits on what it may touch,
and the run lifecycle. The agent CLI is replaced by a small script that prints stream-json."""

from __future__ import annotations

import json
import stat
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import meeting_minutes as mm
import meeting_minutes_web as web
from voicekit import agent_run
from voicekit.agent_run import AgentRunner
from voicekit.jobs import Job, JobNotFound
from voicekit.service import ONLY_JOB_ENV, Service

FAKE_CLAUDE = """#!/usr/bin/env python3
import json, os, sys, time
prompt = sys.stdin.read()
if os.environ.get("FAKE_AGENT_FAIL"):
    sys.stderr.write("login required"); sys.exit(3)
if os.environ.get("FAKE_AGENT_SLEEP"):
    time.sleep(30)
for name in ("read_transcript", "save_document"):
    print(json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__voicekit__" + name}]}}), flush=True)
print(json.dumps({"type": "result", "result": "끝: " + prompt.splitlines()[0]}), flush=True)
"""


def make_job(root: Path, job_id: str = "job000000001") -> Path:
    out = root / job_id
    out.mkdir(parents=True)
    turns = [mm.Turn("SPEAKER_00", 0.0, 3.0, "안녕하세요 김팀장입니다")]
    mm.save_transcript_json(turns, out, source="a.m4a")
    mm.save_transcript_md(turns, out)
    Job(id=job_id, filename="a.m4a", out_dir=out, llm="claude", status="done", local_only=True).save_meta()
    return out


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "fake-claude"
    script.write_text(FAKE_CLAUDE)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("VOICEKIT_CLAUDE_BIN", str(script))
    return script


def wait_idle(runner: AgentRunner) -> None:
    for _ in range(100):
        if not runner.active_job():
            return
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_claude_command_allows_only_the_job_tools(tmp_path: Path, fake_claude: Path) -> None:
    cmd = agent_run.build_command("claude", home=tmp_path, job_id="abc", model=None, out_file=tmp_path / "o")
    allowed = cmd[cmd.index("--allowedTools") + 1 : cmd.index("--mcp-config")]
    assert "mcp__voicekit__save_document" in allowed
    assert not {"mcp__voicekit__transcribe_audio", "mcp__voicekit__generate_document"} & set(allowed)
    assert cmd[cmd.index("--tools") + 1] == ""  # no shell, file or web tools
    assert "--strict-mcp-config" in cmd and cmd[cmd.index("--setting-sources") + 1] == ""
    server = json.loads(cmd[cmd.index("--mcp-config") + 1])["mcpServers"]["voicekit"]
    assert server["env"] == {"VOICEKIT_HOME": str(tmp_path), ONLY_JOB_ENV: "abc"}


def test_codex_command_config_is_valid_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICEKIT_CODEX_BIN", "/bin/true")
    cmd = agent_run.build_command("codex", home=tmp_path, job_id="abc", model=None, out_file=tmp_path / "o")
    overrides = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-c"]
    parsed = tomllib.loads("\n".join(overrides))["mcp_servers"]["voicekit"]
    assert parsed["env"][ONLY_JOB_ENV] == "abc"
    assert "transcribe_audio" not in parsed["enabled_tools"] and "generate_document" not in parsed["enabled_tools"]
    assert parsed["default_tools_approval_mode"] == "approve"
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"


def test_service_limited_to_one_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_job(tmp_path, "mine00000001")
    make_job(tmp_path, "other0000001")
    monkeypatch.setenv(ONLY_JOB_ENV, "mine00000001")
    svc = Service(tmp_path)
    assert [j["id"] for j in svc.list_jobs()] == ["mine00000001"]
    assert svc.read_transcript("mine00000001")["total_turns"] == 1
    with pytest.raises(JobNotFound):
        svc.read_transcript("other0000001")
    with pytest.raises(JobNotFound):
        svc.save_document("other0000001", "x")
    with pytest.raises(RuntimeError, match="limited to one"):
        svc.start_transcription(str(tmp_path / "x.wav"))


def test_run_lifecycle_logs_tools_and_notifies(tmp_path: Path, fake_claude: Path) -> None:
    out = make_job(tmp_path)
    finished: list[str] = []
    runner = AgentRunner(on_finish=finished.append)
    runner.start(job_dir=out, agent="claude", instruction="document")
    with pytest.raises(RuntimeError, match="in progress"):
        runner.start(job_dir=out, agent="claude", instruction="document")
    wait_idle(runner)

    state = runner.state(out) or {}
    assert state["status"] == "done" and state["result"].startswith("끝: 작업 id: job000000001")
    assert [line.split(" ", 1)[1] for line in state["log"]] == ["도구: read_transcript", "도구: save_document"]
    assert finished == ["job000000001"]
    assert json.loads((out / "job.json").read_text())["id"] == "job000000001"


def test_chat_keeps_history_and_replays_it(tmp_path: Path, fake_claude: Path) -> None:
    out = make_job(tmp_path)
    runner = AgentRunner()
    runner.start(job_dir=out, agent="claude", instruction="첫 질문")
    wait_idle(runner)
    runner.start(job_dir=out, agent="claude", instruction="두번째 질문")
    wait_idle(runner)
    state = runner.state(out) or {}
    assert state["instruction"] == "두번째 질문" and [t["instruction"] for t in state["history"]] == ["첫 질문"]
    assert "history" not in state["history"][0]

    last = {k: v for k, v in state.items() if k != "history"}
    prompt = agent_run._prompt("claude", "j1", "세번째", [*state["history"], last])
    assert "사용자: 첫 질문" in prompt and "사용자: 두번째 질문" in prompt and prompt.endswith("요청: 세번째")
    assert "이전 대화" not in agent_run._prompt("claude", "j1", "x", [])

    runner.start(job_dir=out, agent="claude", instruction="새로", reset=True)
    wait_idle(runner)
    assert (runner.state(out) or {})["history"] == []
    runner.clear(out)
    assert runner.state(out) is None


def test_failure_cancel_and_bad_input(tmp_path: Path, fake_claude: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = make_job(tmp_path)
    runner = AgentRunner()
    monkeypatch.setenv("FAKE_AGENT_FAIL", "1")
    runner.start(job_dir=out, agent="claude", instruction="x")
    wait_idle(runner)
    assert (runner.state(out) or {})["status"] == "error" and "login required" in (runner.state(out) or {})["error"]

    monkeypatch.delenv("FAKE_AGENT_FAIL")
    monkeypatch.setenv("FAKE_AGENT_SLEEP", "1")
    runner.start(job_dir=out, agent="claude", instruction="x")
    time.sleep(0.3)
    runner.cancel(out)
    wait_idle(runner)
    assert (runner.state(out) or {})["status"] == "cancelled"

    with pytest.raises(ValueError, match="unknown agent"):
        runner.start(job_dir=out, agent="vim", instruction="x")
    with pytest.raises(ValueError, match="empty"):
        runner.start(job_dir=out, agent="claude", instruction="  ")
    with pytest.raises(ValueError, match="longer"):
        runner.start(job_dir=out, agent="claude", instruction="x" * 5000)
    with pytest.raises(RuntimeError, match="no agent run"):
        runner.cancel(out)


def test_recover_marks_dead_runs(tmp_path: Path) -> None:
    out = make_job(tmp_path)
    (out / agent_run.RUN_FILE).write_text(json.dumps({"status": "running", "log": []}))
    AgentRunner().recover(tmp_path)
    assert (AgentRunner.state(out) or {})["status"] == "error"


@pytest.fixture
def client(tmp_path: Path, fake_claude: Path) -> TestClient:
    make_job(tmp_path / "jobs")
    return TestClient(web.create_app(tmp_path / "jobs", None))


def test_web_runs_agent_and_refreshes_job(client: TestClient) -> None:
    cfg = client.get("/api/config").json()
    assert "claude" in cfg["agents"] and cfg["agent_presets"][0] == "document"
    assert client.get("/api/jobs/job000000001/agent").json() == {"status": "none"}

    r = client.post("/api/jobs/job000000001/agent", data={"agent": "claude", "instruction": "document"})
    assert r.status_code == 200 and r.json()["instruction"].startswith("이 작업의 녹취록으로")
    assert client.delete("/api/jobs/job000000001").status_code == 409  # not while an agent works on it
    for _ in range(100):
        job: dict[str, Any] = client.get("/api/jobs/job000000001").json()
        if job["agent_status"] != "running":
            break
        time.sleep(0.05)
    assert job["agent_status"] == "done"
    assert client.get("/api/jobs/job000000001/agent").json()["log"]

    client.post("/api/jobs/job000000001/agent", data={"agent": "claude", "instruction": "again"})
    for _ in range(100):
        again = client.get("/api/jobs/job000000001/agent").json()
        if again["status"] != "running":
            break
        time.sleep(0.05)
    assert again["history"][0]["instruction"].startswith("이 작업의 녹취록으로")
    assert client.delete("/api/jobs/job000000001/agent").status_code == 200
    assert client.get("/api/jobs/job000000001/agent").json() == {"status": "none"}


def test_web_rejects_bad_agent_requests(client: TestClient, tmp_path: Path) -> None:
    url = "/api/jobs/job000000001/agent"
    assert client.post(url, data={"agent": "vim", "instruction": "x"}).status_code == 400
    assert client.post(url, data={"agent": "claude", "instruction": " "}).status_code == 400
    assert client.post("/api/jobs/nope/agent", data={"agent": "claude", "instruction": "x"}).status_code == 404
    assert client.post(url + "/cancel").status_code == 409
    assert client.post(url, data={"agent": "claude", "instruction": "x"},
                       headers={"Origin": "http://evil.example"}).status_code == 403
