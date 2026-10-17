"""The per-profile term memory: what it learns from an edit, when a term starts to be used, and where it
is applied (Whisper prompt, new transcripts). No models, no network."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import meeting_minutes as mm
import meeting_minutes_web as web
from voicekit import manage, memory
from voicekit.jobs import Job
from voicekit.types import Turn


@pytest.fixture(autouse=True)
def mem_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VOICEKIT_MEMORY_DIR", str(tmp_path / "mem"))
    monkeypatch.delenv("VOICEKIT_MEMORY", raising=False)
    return tmp_path / "mem"


def test_extract_pairs_finds_word_swaps_and_ignores_rewrites() -> None:
    assert memory.extract_pairs("카 결과를 공유해 주세요", "QA 결과를 공유해 주세요") == [("카", "QA")]
    assert memory.extract_pairs("카를 먼저 봅시다", "QA를 먼저 봅시다") == [("카", "QA")]  # particle stays out
    assert memory.extract_pairs("큐 에이 결과", "QA 결과") == [("큐 에이", "QA")]
    assert memory.extract_pairs("오늘 일정은 이렇습니다", "내일 회의는 다르게 진행합니다 그리고 또") == []
    assert memory.extract_pairs("같은 문장", "같은 문장") == []


def test_learned_pair_is_used_only_after_it_is_confirmed_twice() -> None:
    first = memory.learn("meeting", [("쿠버네티", "쿠버네티스")])[0]
    assert (first.count, first.active, first.replace) == (1, False, False)
    assert memory.apply("쿠버네티 배포", "meeting")[0] == "쿠버네티 배포"
    assert memory.vocab_hint("meeting") == ""

    second = memory.learn("meeting", [("쿠버네티", "쿠버네티스")])[0]
    assert (second.count, second.active, second.replace) == (2, True, True)
    assert memory.apply("쿠버네티 배포", "meeting") == ("쿠버네티스 배포", [("쿠버네티", "쿠버네티스")])
    assert memory.vocab_hint("meeting") == "쿠버네티스"


def test_one_character_spellings_are_only_replaced_when_pinned() -> None:
    memory.learn("meeting", [("카", "QA")])
    second = memory.learn("meeting", [("카", "QA")])[0]
    assert (second.active, second.replace) == (True, False)  # QA is a prompt hint, but "카" is not rewritten
    assert memory.apply("카 결과", "meeting")[0] == "카 결과" and memory.vocab_hint("meeting") == "QA"
    memory.update("meeting", memory.load("meeting")[0]["id"], pinned=True)
    assert memory.apply("카 결과", "meeting")[0] == "QA 결과"


def test_apply_respects_word_boundaries_and_particles() -> None:
    memory.add("meeting", "QA", ["카", "큐에이"])
    assert memory.apply("카 결과", "meeting")[0] == "QA 결과"
    assert memory.apply("카를 봅시다", "meeting")[0] == "QA를 봅시다"
    assert memory.apply("카페에서 큐에이입니다", "meeting")[0] == "카페에서 QA입니다"
    assert memory.apply("자카르타 카드", "meeting")[0] == "자카르타 카드"  # inside other words: untouched
    assert memory.apply("QA 그대로", "meeting")[0] == "QA 그대로"


def test_scopes_are_separate_but_common_applies_everywhere() -> None:
    memory.add("common", "SKT", ["에스케이티"])
    memory.add("interview", "라포", ["라보"])
    assert memory.apply("에스케이티 라보", "meeting")[0] == "SKT 라보"
    assert memory.apply("에스케이티 라보", "interview")[0] == "SKT 라포"
    assert memory.vocab_hint("meeting") == "SKT"
    assert memory.vocab_hint("interview").split(", ") == ["SKT", "라포"]


def test_disabled_entries_and_the_global_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    e = memory.add("meeting", "QA", ["카"])
    memory.update("meeting", e["id"], enabled=False)
    assert memory.apply("카 결과", "meeting")[0] == "카 결과" and memory.vocab_hint("meeting") == ""
    memory.update("meeting", e["id"], enabled=True)
    monkeypatch.setenv("VOICEKIT_MEMORY", "off")
    assert memory.apply("카 결과", "meeting")[0] == "카 결과" and memory.merge_vocab("a", "meeting") == "a"
    monkeypatch.delenv("VOICEKIT_MEMORY")
    assert memory.apply("카 결과", "meeting")[0] == "QA 결과"
    memory.remove("meeting", e["id"])
    assert memory.view("meeting") == []


def test_undoing_a_mapping_forgets_it() -> None:
    memory.add("meeting", "QA", ["카"])
    memory.learn("meeting", [("QA", "카")])  # the user now writes it the other way round
    assert "카" not in memory.load("meeting")[0]["wrong"]


def test_hint_has_a_budget_and_merges_with_the_job_vocab() -> None:
    for i in range(60):
        memory.add("meeting", f"용어번호{i:02d}")
    hint = memory.vocab_hint("meeting")
    assert len(hint) <= memory.MAX_HINT_CHARS and hint.count(",") >= 10
    assert memory.merge_vocab("QA, 결제", "interview") == "QA, 결제"
    memory.add("meeting", "QA")
    merged = memory.merge_vocab("qa, 결제", "meeting") or ""
    assert merged.startswith("qa, 결제") and merged.lower().count("qa") == 1


def test_scope_names_cannot_leave_the_memory_folder(mem_dir: Path) -> None:
    for bad in ("../x", "a/b", "", ".hidden", "A" * 40):
        with pytest.raises(ValueError):
            memory.valid_scope(bad)
    with pytest.raises(ValueError):
        memory.add("../escape", "x")
    assert not (mem_dir.parent / "escape.json").exists()


def test_bad_terms_are_refused() -> None:
    with pytest.raises(ValueError):
        memory.add("meeting", "x" * 60)
    assert memory.learn("meeting", [("같음", "같음"), ("", "빈값")]) == []


def make_done_job(root: Path, with_correction: bool = False) -> Job:
    out = root / "jobedit00001"
    out.mkdir(parents=True)
    turns = [Turn("SPEAKER_00", 0.0, 3.0, "카 결과를 공유합니다"), Turn("SPEAKER_01", 3.0, 6.0, "네 알겠습니다")]
    mm.save_transcript_json(turns, out, source="a.m4a")
    mm.save_transcript_md(turns, out)
    if with_correction:
        mm.save_correction({"llm": "x", "turns": [t.__dict__ for t in turns], "changes": []}, out)
    job = Job(id=out.name, filename="a.m4a", out_dir=out, llm="claude", status="done", local_only=True)
    job.save_meta()
    return job


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(manage, "USER_PROFILE_DIR", tmp_path / "profiles")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    make_done_job(tmp_path / "jobs")
    return TestClient(web.create_app(tmp_path / "jobs", None))


def test_editing_a_turn_saves_it_and_teaches_the_memory(client: TestClient, tmp_path: Path) -> None:
    url = "/api/jobs/jobedit00001/turns/0"
    r = client.put(url, data={"text": "QA 결과를 공유합니다"})
    assert r.status_code == 200 and r.json()["learned"] == [{"wrong": "카", "right": "QA", "count": 1, "active": False, "replace": False}]
    saved = json.loads((tmp_path / "jobs/jobedit00001/transcript.json").read_text())
    assert saved["turns"][0]["text"] == "QA 결과를 공유합니다"
    assert "QA 결과를 공유합니다" in (tmp_path / "jobs/jobedit00001/transcript.md").read_text()
    assert json.loads((tmp_path / "jobs/jobedit00001/edits.json").read_text())[0]["before"] == "카 결과를 공유합니다"

    entries = client.get("/api/memory/meeting").json()["entries"]  # the job's recording type
    assert entries[0]["right"] == "QA" and entries[0]["active"] is False

    assert client.put(url, data={"text": "  "}).status_code == 400
    assert client.put("/api/jobs/jobedit00001/turns/9", data={"text": "x"}).status_code == 404
    assert client.put(url, data={"text": "x", "corrected": "true"}).status_code == 400  # no corrected file yet
    assert client.put(url, data={"text": "x", "scope": "../x"}).status_code == 400
    assert client.put(url, data={"text": "y"}, headers={"Origin": "http://evil.example"}).status_code == 403


def test_editing_the_corrected_text_and_common_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    make_done_job(tmp_path / "jobs", with_correction=True)
    c = TestClient(web.create_app(tmp_path / "jobs", None))
    r = c.put("/api/jobs/jobedit00001/turns/0", data={"text": "QA 결과를 공유합니다", "corrected": "true", "scope": "common"})
    assert r.status_code == 200
    corrected = json.loads((tmp_path / "jobs/jobedit00001/transcript_corrected.json").read_text())
    assert corrected["turns"][0]["text"] == "QA 결과를 공유합니다"
    assert corrected["changes"][0]["original"] == "카 결과를 공유합니다"
    original = json.loads((tmp_path / "jobs/jobedit00001/transcript.json").read_text())
    assert original["turns"][0]["text"] == "카 결과를 공유합니다"  # the recognizer's text is kept
    assert c.get("/api/memory/common").json()["entries"][0]["right"] == "QA"
    assert c.get("/api/memory/meeting").json()["entries"] == []


def test_memory_endpoints_manage_terms(client: TestClient) -> None:
    r = client.post("/api/memory/meeting", data={"right": "Kubernetes", "wrong": "쿠버네티스, 큐버네티스"})
    assert r.status_code == 200 and r.json()["active"] is True
    entry = client.get("/api/memory/meeting").json()["entries"][0]
    assert set(entry["wrong"]) == {"쿠버네티스", "큐버네티스"}
    assert client.put(f"/api/memory/meeting/{entry['id']}", data={"enabled": "false"}).status_code == 200
    assert client.get("/api/memory/meeting").json()["entries"][0]["enabled"] is False
    assert client.delete(f"/api/memory/meeting/{entry['id']}").status_code == 200
    assert client.delete(f"/api/memory/meeting/{entry['id']}").status_code == 404
    assert client.get("/api/memory/Bad.Scope").status_code == 400
    assert client.post("/api/memory/meeting", data={"right": "x" * 80}).status_code == 400
    assert client.get("/api/config").json()["memory_enabled"] is True


def test_pipeline_uses_the_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memory.add("meeting", "QA", ["카"])
    seen: dict[str, Any] = {}

    def fake_stt(audio: Path, **kw: Any) -> list[dict[str, Any]]:
        seen["vocab"] = kw["vocab"]
        return [{"start": 0.0, "end": 3.0, "text": "카 결과를 공유합니다", "speaker": "SPEAKER_00"}]

    monkeypatch.setattr(mm, "transcribe_and_diarize", fake_stt)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    c = TestClient(web.create_app(tmp_path / "jobs", None))

    def run(use_memory: str) -> dict[str, Any]:
        r = c.post("/api/jobs", files={"file": ("a.wav", b"RIFF")}, data={"local_only": "true", "vocab": "결제", "use_memory": use_memory})
        job_id = r.json()["id"]
        for _ in range(100):
            job = c.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("done", "error"):
                return job
            time.sleep(0.05)
        raise AssertionError("job did not finish")

    job = run("true")
    assert job["status"] == "done" and seen["vocab"] == "결제, QA"
    text = json.loads((tmp_path / "jobs" / job["id"] / "transcript.json").read_text())["turns"][0]["text"]
    assert text == "QA 결과를 공유합니다"
    assert json.loads((tmp_path / "jobs" / job["id"] / "memory_applied.json").read_text()) == [{"turn": 0, "wrong": "카", "right": "QA"}]

    job = run("false")
    assert seen["vocab"] == "결제"
    assert json.loads((tmp_path / "jobs" / job["id"] / "transcript.json").read_text())["turns"][0]["text"] == "카 결과를 공유합니다"
