"""Web UI job flow with STT and LLM stubbed out (no models, no network)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import meeting_minutes as mm
import meeting_minutes_web as web

SEGMENTS = [
    {
        "start": 0.0, "end": 4.0, "text": "안녕하세요 네 반갑습니다", "speaker": "SPEAKER_00",
        "words": [
            {"word": "안녕하세요", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"word": "네", "start": 1.5, "end": 2.0, "speaker": "SPEAKER_01"},
            {"word": "반갑습니다", "start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        ],
    }
]
ANALYSIS = {
    "speakers": [
        {"label": "SPEAKER_00", "role": "진행자", "candidates": [{"name": "김팀장", "prob": 1.0, "evidence": []}]},
        {"label": "SPEAKER_01", "role": None, "candidates": [{"name": "미상", "prob": 1.0, "evidence": []}]},
    ],
    "topics": [{"start_turn": 0, "end_turn": 1, "start": 0.0, "end": 4.0, "title": "인사", "summary": ""}],
    "key_turns": [],
}


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"minutes": [], "analysis": [], "corrections": [], "fail_analysis": False,
                             "fail_correction": False}

    def fake_stt(audio: Path, **kw: Any) -> list[dict[str, Any]]:
        assert audio.exists()
        kw["on_progress"]("ASR running")
        return SEGMENTS

    def fake_minutes(turns: list[mm.Turn], **kw: Any) -> str:
        state["minutes"].append({**kw, "texts": [t.text for t in turns]})
        if kw["llm"] == "openai:no-key":
            raise SystemExit("OPENAI_API_KEY 가 없습니다.")
        return f"# 회의록 ({kw['llm']}, {len(turns)} turns)"

    def fake_analysis(turns: list[mm.Turn], **kw: Any) -> dict[str, Any]:
        state["analysis"].append(kw)
        if state["fail_analysis"]:
            raise RuntimeError("analysis down")
        return {**ANALYSIS, "llm": kw["llm"]}

    def fake_correct(turns: list[mm.Turn], **kw: Any) -> dict[str, Any]:
        state["corrections"].append(kw)
        if state["fail_correction"]:
            raise RuntimeError("correction down")
        fixed = [mm.Turn(t.speaker, t.start, t.end, t.text.replace("반갑습니다", "반갑습니다!")) for t in turns]
        changes = [{"turn": i, "original": a.text, "corrected": b.text, "rejected": None}
                   for i, (a, b) in enumerate(zip(turns, fixed)) if a.text != b.text]
        return {"llm": kw["llm"], "turns": [mm.asdict(t) for t in fixed], "changes": changes}

    monkeypatch.setattr(mm, "correct_transcript", fake_correct)
    monkeypatch.setattr(mm, "transcribe_and_diarize", fake_stt)
    monkeypatch.setattr(mm, "write_minutes", fake_minutes)
    monkeypatch.setattr(mm, "analyze_transcript", fake_analysis)
    return state


@pytest.fixture
def client(tmp_path: Path, stubs: dict[str, Any]) -> TestClient:
    return TestClient(web.create_app(tmp_path, None))


def wait_done(c: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error") and "running" not in (job["analysis_status"], job["correction_status"]):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def upload(c: TestClient, **form: str) -> dict[str, Any]:
    r = c.post("/api/jobs", files={"file": ("회의.wav", b"RIFF" + b"\0" * 64, "audio/wav")}, data=form)
    assert r.status_code == 200, r.text
    return r.json()


def test_index_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "회의록 만들기" in r.text


def test_upload_runs_pipeline_and_serves_files(client: TestClient, stubs: dict[str, Any]) -> None:
    job = wait_done(client, upload(client, speakers="2", names=" 김팀장 ", llm="gpt")["id"])

    assert job["status"] == "done", job
    assert job["analysis_status"] == "done"
    assert job["names"] == "김팀장"  # returned so the UI can prefill the regenerate form
    assert job["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert stubs["minutes"][0]["llm"] == "gpt"
    assert stubs["analysis"][0]["llm"] == mm.DEFAULT_ANALYSIS_LLM
    assert client.get(f"/api/jobs/{job['id']}/files/minutes.md").text == "# 회의록 (gpt, 2 turns)"
    assert client.get(f"/api/jobs/{job['id']}/files/analysis.json").json()["topics"][0]["title"] == "인사"
    transcript = client.get(f"/api/jobs/{job['id']}/files/transcript.md").text
    assert "SPEAKER_01**: 네 반갑습니다" in transcript  # mid-segment speaker change split into its own turn


def test_analysis_failure_keeps_minutes(client: TestClient, stubs: dict[str, Any]) -> None:
    stubs["fail_analysis"] = True
    job = wait_done(client, upload(client)["id"])
    assert job["status"] == "done"
    assert job["analysis_status"] == "error"
    assert "analysis down" in job["analysis_error"]

    stubs["fail_analysis"] = False
    assert client.post(f"/api/jobs/{job['id']}/analysis", data={"names": "김팀장"}).status_code == 200
    job = wait_done(client, job["id"])
    assert job["analysis_status"] == "done"
    assert stubs["analysis"][-1]["names"] == "김팀장"


def test_confirmed_speaker_names_feed_minutes(client: TestClient, stubs: dict[str, Any]) -> None:
    job_id = wait_done(client, upload(client)["id"])["id"]
    r = client.put(f"/api/jobs/{job_id}/speakers", json={"SPEAKER_00": " 김팀장 ", "SPEAKER_01": ""})
    assert r.status_code == 200, r.text
    assert r.json()["speaker_map"] == {"SPEAKER_00": "김팀장"}

    client.post(f"/api/jobs/{job_id}/minutes", data={"llm": "claude"})
    job = wait_done(client, job_id)
    assert job["llm"] == "claude"
    assert stubs["minutes"][-1]["speaker_map"] == {"SPEAKER_00": "김팀장"}


def test_unknown_speaker_label_rejected(client: TestClient) -> None:
    job_id = wait_done(client, upload(client)["id"])["id"]
    r = client.put(f"/api/jobs/{job_id}/speakers", json={"SPEAKER_09": "누군가"})
    assert r.status_code == 400


def test_missing_key_surfaces_as_job_error(client: TestClient) -> None:
    job = wait_done(client, upload(client, llm="openai:no-key")["id"])
    assert job["status"] == "error"
    assert "OPENAI_API_KEY" in job["error"]
    assert job["has_transcript"]  # transcript survives, so the minutes can be regenerated with another model


def test_audio_supports_range_requests(client: TestClient) -> None:
    job_id = wait_done(client, upload(client)["id"])["id"]
    full = client.get(f"/api/jobs/{job_id}/audio")
    assert full.status_code == 200
    assert full.headers["content-type"] == "audio/wav"
    part = client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=0-3"})
    assert part.status_code == 206  # browsers need this to seek
    assert part.content == b"RIFF"


def test_jobs_survive_restart(tmp_path: Path, stubs: dict[str, Any]) -> None:
    first = TestClient(web.create_app(tmp_path, None))
    job_id = wait_done(first, upload(first)["id"])["id"]
    first.put(f"/api/jobs/{job_id}/speakers", json={"SPEAKER_00": "김팀장"})

    second = TestClient(web.create_app(tmp_path, None))
    job = second.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"
    assert job["speaker_map"] == {"SPEAKER_00": "김팀장"}


@pytest.mark.parametrize(("form", "status"), [({"llm": "nope"}, 400), ({"speakers": "0"}, 400)])
def test_rejects_bad_form(client: TestClient, form: dict[str, str], status: int) -> None:
    r = client.post("/api/jobs", files={"file": ("a.wav", b"x", "audio/wav")}, data=form)
    assert r.status_code == status


def test_only_known_output_files_are_served(client: TestClient) -> None:
    job_id = wait_done(client, upload(client)["id"])["id"]
    assert client.get(f"/api/jobs/{job_id}/files/input.wav").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/files/job.json").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/files/..%2F..%2Fpyproject.toml").status_code == 404
    assert client.get("/api/jobs/unknown/files/minutes.md").status_code == 404


def test_safe_suffix() -> None:
    assert web._safe_suffix("회의.M4A") == ".m4a"
    assert web._safe_suffix("../../x.sh;rm") == ".bin"
    assert web._safe_suffix("noext") == ".bin"


def test_legacy_output_dir_is_listed(tmp_path: Path, stubs: dict[str, Any]) -> None:
    """Runs from before job.json existed still show up, ready for analysis."""
    legacy = tmp_path / "abc123"
    turns = mm.segments_to_turns(SEGMENTS)
    mm.save_transcript_json(turns, legacy, source="/tmp/주간회의.wav")
    (legacy / "minutes.md").write_text("# old", encoding="utf-8")

    c = TestClient(web.create_app(tmp_path, None))
    job = c.get("/api/jobs/abc123").json()
    assert (job["filename"], job["status"], job["analysis_status"]) == ("주간회의.wav", "done", "none")

    assert c.post("/api/jobs/abc123/analysis").status_code == 200
    assert wait_done(c, "abc123")["analysis_status"] == "done"


def test_correction_feeds_minutes_and_keeps_original(client: TestClient, stubs: dict[str, Any]) -> None:
    job = wait_done(client, upload(client)["id"])
    assert job["correction_status"] == "done" and job["has_correction"]
    assert stubs["minutes"][-1]["texts"][-1] == "네 반갑습니다!"  # minutes are written from the corrected text
    corrected = client.get(f"/api/jobs/{job['id']}/files/transcript_corrected.json").json()
    assert corrected["changes"][0]["original"] == "네 반갑습니다"
    assert "반갑습니다!" not in client.get(f"/api/jobs/{job['id']}/files/transcript.json").text  # raw STT untouched


def test_correction_can_be_skipped(client: TestClient, stubs: dict[str, Any]) -> None:
    job = wait_done(client, upload(client, correct="false")["id"])
    assert job["correction_status"] == "none" and not stubs["corrections"]
    assert stubs["minutes"][-1]["texts"][-1] == "네 반갑습니다"


def test_correction_failure_falls_back_to_original(client: TestClient, stubs: dict[str, Any]) -> None:
    stubs["fail_correction"] = True
    job = wait_done(client, upload(client)["id"])
    assert job["status"] == "done"
    assert job["correction_status"] == "error" and "correction down" in job["correction_error"]
    assert stubs["minutes"][-1]["texts"][-1] == "네 반갑습니다"

    stubs["fail_correction"] = False
    assert client.post(f"/api/jobs/{job['id']}/correction").status_code == 200
    job = wait_done(client, job["id"])
    assert job["correction_status"] == "done" and job["analysis_status"] == "done"
    assert len(stubs["analysis"]) == 2  # analysis re-runs on the corrected text
