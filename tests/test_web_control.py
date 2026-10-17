"""Controls the web UI exposes for the local pipeline: local-only runs, pasted documents, deletion,
jobs created by other processes, agent connection and profile editing. No models, no network."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import meeting_minutes as mm
import meeting_minutes_web as web
from voicekit import manage
from voicekit.jobs import Job, JobStore

SEGMENTS = [{"start": 0.0, "end": 3.0, "text": "안녕하세요 반갑습니다", "speaker": "SPEAKER_00"}]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(mm, "transcribe_and_diarize", lambda audio, **kw: SEGMENTS)
    monkeypatch.setattr(manage, "USER_PROFILE_DIR", tmp_path / "profiles")
    monkeypatch.setenv("VOICEKIT_PROFILES", str(tmp_path / "profiles"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return TestClient(web.create_app(tmp_path / "jobs", None))


def wait(c: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def upload(c: TestClient, **form: str) -> dict[str, Any]:
    r = c.post("/api/jobs", files={"file": ("a.wav", b"RIFF" + b"\0" * 64, "audio/wav")}, data=form)
    assert r.status_code == 200, r.text
    return r.json()


def test_local_only_upload_skips_every_llm_stage(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **kw: Any) -> Any:
        raise AssertionError("LLM must not be called")

    for name in ("write_minutes", "analyze_transcript", "correct_transcript"):
        monkeypatch.setattr(mm, name, boom)
    job = wait(client, upload(client, local_only="true", correct="true")["id"])

    assert job["status"] == "done" and job["local_only"] is True
    assert job["has_transcript"] and not job["has_minutes"]
    assert job["dir"].endswith(job["id"])


def test_pasted_document_is_stored_and_empty_is_refused(client: TestClient) -> None:
    job = wait(client, upload(client, local_only="true")["id"])
    assert client.put(f"/api/jobs/{job['id']}/document", json={"markdown": "  "}).status_code == 400

    r = client.put(f"/api/jobs/{job['id']}/document", json={"markdown": "# 요약\n\n- 결정"})
    assert r.status_code == 200 and r.json()["has_minutes"]
    assert client.get(f"/api/jobs/{job['id']}/files/minutes.md").text.startswith("# 요약")


def test_delete_removes_job_and_folder(client: TestClient, tmp_path: Path) -> None:
    job = wait(client, upload(client, local_only="true")["id"])
    assert (tmp_path / "jobs" / job["id"]).is_dir()
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 200
    assert not (tmp_path / "jobs" / job["id"]).exists()
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 404


def test_jobs_made_by_another_process_appear_and_progress(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    out = root / "agentjob0001"
    out.mkdir(parents=True)
    job = Job(id="agentjob0001", filename="x.m4a", out_dir=out, llm="claude", status="running", stage="ASR")
    job.save_meta()

    listed = client.get("/api/jobs").json()
    assert [j["id"] for j in listed] == ["agentjob0001"]
    assert listed[0]["status"] == "running"  # not mislabelled as interrupted: another process owns it

    job.status, job.stage = "done", "녹취록 완료"
    job.save_meta()
    assert client.get("/api/jobs/agentjob0001").json()["status"] == "done"


def test_dead_job_found_at_startup_stays_interrupted(tmp_path: Path) -> None:
    out = tmp_path / "jobs" / "deadjob00001"
    out.mkdir(parents=True)
    Job(id="deadjob00001", filename="x.m4a", out_dir=out, llm="claude", status="running").save_meta()
    store = JobStore(tmp_path / "jobs", None, "claude")
    store.rescan()
    assert store.get("deadjob00001").status == "error"


def test_cross_origin_changes_are_refused(client: TestClient) -> None:
    r = client.post("/api/profiles", json={"id": "x", "system": "s"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/agents/claude/install", headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    ok = client.post("/api/profiles", json={"id": "x", "system": "s"}, headers={"Origin": "http://testserver"})
    assert ok.status_code == 200


def test_profile_create_edit_delete_roundtrip(client: TestClient, tmp_path: Path) -> None:
    body = {
        "id": "sales-call", "label": "영업 통화", "document": "영업 통화 요약", "diarize": True, "speakers": 2,
        "system": '고객 "요구"를 정리하세요.\n- 다음 단계', "description": "고객 통화",
    }
    saved = client.post("/api/profiles", json=body)
    assert saved.status_code == 200, saved.text
    assert saved.json()["system"] == body["system"] and saved.json()["speakers"] == 2

    ids = [p["id"] for p in client.get("/api/config").json()["profiles"]]
    assert "sales-call" in ids and ids[:2] == ["meeting", "interview"]  # built-in order intact

    # a built-in id is shadowed by the user file; deleting it restores the built-in
    client.post("/api/profiles", json={"id": "memo", "label": "내 메모", "system": "짧게"})
    assert client.get("/api/profiles/memo").json()["label"] == "내 메모"
    assert client.delete("/api/profiles/memo").status_code == 200
    assert client.get("/api/profiles/memo").json()["label"] != "내 메모"

    assert client.delete("/api/profiles/meeting").status_code == 400  # built-in cannot be deleted
    assert client.delete("/api/profiles/sales-call").status_code == 200
    assert not (tmp_path / "profiles" / "sales-call.toml").exists()


@pytest.mark.parametrize("bad", [{"id": "Bad Id", "system": "s"}, {"id": "ok", "system": " "},
                                 {"id": "ok", "system": "s", "speakers": 99}, {"id": "../x", "system": "s"}])
def test_profile_validation(client: TestClient, bad: dict[str, Any]) -> None:
    assert client.post("/api/profiles", json=bad).status_code == 400


def test_agent_status_and_install_use_the_ui_job_folder(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "Added", "")

    monkeypatch.setattr(manage.shutil, "which", lambda t: f"/usr/bin/{t}" if t == "claude" else None)
    monkeypatch.setattr(manage.subprocess, "run", fake_run)

    state = client.get("/api/agents").json()
    by = {a["target"]: a for a in state["agents"]}
    assert by["claude"]["installed"] and by["claude"]["mcp"] and not by["claude"]["guidance"]
    assert not by["codex"]["installed"] and not by["codex"]["mcp"]
    home = str((tmp_path / "jobs").resolve())
    assert state["mcp_config"]["mcpServers"]["voicekit"]["env"] == {"VOICEKIT_HOME": home}

    r = client.post("/api/agents/claude/install")
    assert r.status_code == 200 and r.json()["ok"]
    add = next(c for c in calls if c[:3] == ["claude", "mcp", "add"])
    assert f"VOICEKIT_HOME={home}" in add
    assert (tmp_path / "home" / ".claude" / "skills" / "voicekit" / "SKILL.md").exists()

    assert client.post("/api/agents/codex/install").status_code == 409  # CLI missing, nothing changed
    assert client.post("/api/agents/vim/install").status_code == 400
    assert not (tmp_path / "home" / ".codex").exists()


def test_mcp_config_snippet_is_valid_json() -> None:
    cfg = manage.mcp_config(Path("/data/jobs"))
    entry = json.loads(json.dumps(cfg))["mcpServers"]["voicekit"]
    assert entry["args"][-1] == "voicekit.mcp_server" and entry["env"]["VOICEKIT_HOME"] == "/data/jobs"
