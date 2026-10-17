"""Operations behind the CLI and the MCP server, as plain functions returning JSON-ready dicts.

Audio never leaves this machine here: ``start_transcription`` runs ASR, alignment and diarization
locally and stops. The document (minutes, interview notes...) is written either by the calling agent
from ``document_prompt`` and stored with ``save_document``, or by ``generate_document`` through a
configured LLM / local agent CLI.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import meeting_minutes as mm
from voicekit.jobs import Job, JobNotFound, JobStore
from voicekit.profiles import get_profile, load_profiles

DEFAULT_HOME = Path.home() / ".voicekit" / "jobs"
READ_CHARS = 20000


def home() -> Path:
    """Job root. ``VOICEKIT_HOME`` lets the web UI (``--out-dir``) and agents share one folder."""
    return Path(os.environ.get("VOICEKIT_HOME") or DEFAULT_HOME).expanduser()


class Service:
    def __init__(self, root: Optional[Path] = None, env_file: Optional[Path] = None) -> None:
        mm.configure_llms(None)
        env = os.environ.get("VOICEKIT_ENV_FILE")
        self.env_file = env_file or (Path(env) if env else None)
        self.store = JobStore(root or home(), self.env_file, mm.default_analysis_llm())

    def job(self, job_id: str) -> Job:
        if not job_id or "/" in job_id or job_id.startswith("."):
            raise JobNotFound(job_id)
        return self.store.get(job_id)

    @staticmethod
    def profiles() -> list[dict[str, Any]]:
        return [p.summary() for p in load_profiles().values()]

    def start_transcription(
        self, path: str, *, profile: str = "meeting", language: Optional[str] = "ko", speakers: Optional[int] = None,
        diarize: Optional[bool] = None, vocab: Optional[str] = None, names: Optional[str] = None,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Queue local transcription of an audio/video file; returns at once with the job id."""
        src = Path(path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"audio file not found: {src}")
        chosen = get_profile(profile)
        job_id = uuid.uuid4().hex[:12]
        out_dir = self.store.out_root / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = src.suffix.lower()
        audio = out_dir / f"input{suffix if 1 < len(suffix) <= 6 and suffix[1:].isalnum() else '.bin'}"
        shutil.copyfile(src, audio)
        job = Job(
            id=job_id, filename=src.name, out_dir=out_dir, llm=mm.default_llm(), analysis_llm=self.store.analysis_llm,
            names=names or None, context=context or None, vocab=vocab or None, correct=False,
            profile=chosen.id, language=None if language in (None, "", "auto") else language, diarize=diarize,
            local_only=True,
        )
        self.store.submit(job, audio, speakers)
        return job.summary()

    def status(self, job_id: str) -> dict[str, Any]:
        return self.job(job_id).summary()

    def wait(self, job_id: str, timeout: float = 50.0) -> dict[str, Any]:
        """Block until the job leaves queued/running or ``timeout`` passes (keeps MCP calls short)."""
        job = self.job(job_id)
        deadline = time.monotonic() + max(0.0, timeout)
        while job.status in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(1.0)
        return job.summary()

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        jobs = sorted(self.store.jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
        keys = ("id", "filename", "profile", "status", "stage", "speakers", "has_minutes")
        return [{k: s[k] for k in keys} for s in (j.summary() for j in jobs)]

    def read_transcript(
        self, job_id: str, *, fmt: str = "md", offset_turn: int = 0, max_chars: int = READ_CHARS, original: bool = False
    ) -> dict[str, Any]:
        """A page of the transcript (corrected text if there is one). Long recordings come back in pages."""
        job = self.job(job_id)
        if not (job.out_dir / "transcript.json").exists():
            raise RuntimeError(f"no transcript yet (status={job.status}, stage={job.stage})")
        turns = self.store._load_turns(job, original=original)
        end, size = offset_turn, 0
        while end < len(turns) and (end == offset_turn or size + len(turns[end].text) <= max_chars):
            size += len(turns[end].text)
            end += 1
        page = turns[offset_turn:end]
        text = (
            json.dumps([t.__dict__ for t in page], ensure_ascii=False)
            if fmt == "json"
            else "\n".join(f"[{mm._fmt_ts(t.start)}] {t.speaker}: {t.text}" for t in page)
        )
        return {
            "job_id": job_id, "total_turns": len(turns), "from_turn": offset_turn, "to_turn": end,
            "next_offset": end if end < len(turns) else None, "format": fmt, "text": text,
        }

    def document_prompt(
        self, job_id: str, *, profile: Optional[str] = None, names: Optional[str] = None,
        context: Optional[str] = None, vocab: Optional[str] = None,
    ) -> dict[str, Any]:
        """System + user prompt for the document, so the calling agent can write it with its own model."""
        job = self.job(job_id)
        chosen = get_profile(profile or job.profile)
        turns = self.store._load_turns(job)
        user = mm.build_user_prompt(
            turns, names=names or job.names, context=context or job.context, vocab=vocab or job.vocab,
            speaker_map=job.speaker_map, profile=chosen,
        )
        return {
            "job_id": job_id, "profile": chosen.id, "document": chosen.document, "system": chosen.system,
            "user": user, "save_with": "save_document(job_id, markdown)",
        }

    def save_document(self, job_id: str, markdown: str) -> dict[str, Any]:
        if not markdown.strip():
            raise ValueError("markdown is empty")
        job = self.job(job_id)
        self.store.save_document(job, markdown)
        return {"job_id": job_id, "saved": str(job.out_dir / "minutes.md"), "chars": len(markdown)}

    def generate_document(
        self, job_id: str, *, llm: str = "claude-code", profile: Optional[str] = None, names: Optional[str] = None,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Write the document through ``llm`` (an alias, provider:model, or claude-code / codex). Blocks."""
        job = self.job(job_id)
        mm.resolve_llm(llm)
        chosen = get_profile(profile or job.profile)
        text = mm.write_minutes(
            self.store._load_turns(job), llm=llm, names=names or job.names, context=context or job.context,
            vocab=job.vocab, env_file=self.env_file, speaker_map=job.speaker_map, profile=chosen,
        )
        self.store._update(job, llm=llm, profile=chosen.id)
        self.store.save_document(job, text)
        return {"job_id": job_id, "llm": llm, "saved": str(job.out_dir / "minutes.md"), "chars": len(text)}

    def set_speaker_names(self, job_id: str, mapping: dict[str, str]) -> dict[str, Any]:
        job = self.job(job_id)
        labels = {t.speaker for t in self.store._load_turns(job, original=True)}
        unknown = set(mapping) - labels
        if unknown:
            raise ValueError(f"unknown speaker labels: {sorted(unknown)} (known: {sorted(labels)})")
        self.store.set_speaker_map(job, {k: v.strip()[:40] for k, v in mapping.items() if v and v.strip()})
        return {"job_id": job_id, "speaker_map": job.speaker_map}
