"""Local web UI: upload a recording, watch progress, read the conversation and the document written from it.

Binds to 127.0.0.1 by default; audio and transcripts stay on this machine and only the
transcript text goes to the LLM, same as the CLI. STT jobs run one at a time because
ASR and diarization already saturate the CPU.

After STT, an optional correction pass fixes recognition errors from context (speakers and
timings untouched); then two LLM calls run side by side on that text: the minutes, and an
analysis (speaker-name probabilities, topic sections, key turns) that drives the conversation view. Speaker names
confirmed in the UI are saved per job and fed back into minutes regeneration.

Usage:
    uv run meeting-minutes-web --env-file .env
"""

from __future__ import annotations

import argparse
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any, Optional
from urllib.parse import urlparse

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import meeting_minutes as mm
from voicekit import manage
from voicekit.jobs import Job, JobNotFound, JobStore
from voicekit.profiles import get_profile, load_profiles

INDEX_HTML = Path(__file__).with_name("web") / "index.html"
OUTPUT_FILES = {
    "transcript.md", "transcript.json", "transcript_corrected.md", "transcript_corrected.json",
    "minutes.md", "analysis.json",
}
AUDIO_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "video/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".webm": "audio/webm", ".flac": "audio/flac",
}
def _clean(value: Optional[str]) -> Optional[str]:
    value = (value or "").strip()
    return value or None


def _safe_suffix(filename: str) -> str:
    """Keep a short alphanumeric extension only; ffmpeg sniffs the real format anyway."""
    suffix = Path(filename).suffix.lower()
    return suffix if 1 < len(suffix) <= 6 and suffix[1:].isalnum() else ".bin"


def create_app(out_root: Path, env_file: Optional[Path], analysis_llm: Optional[str] = None) -> FastAPI:
    """Build the FastAPI app around a job store writing under ``out_root``.

    Model aliases come from :func:`meeting_minutes.configure_llms`, which ``main`` calls first.
    """
    store = JobStore(out_root, env_file, analysis_llm or mm.default_analysis_llm())
    app = FastAPI(title="meeting-minutes", docs_url=None, redoc_url=None)
    app.state.store = store

    @app.middleware("http")
    async def same_origin_only(request: Request, call_next: Any) -> Any:
        """Changes (install an agent, write a profile, delete a job) must come from this page, not another site."""
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(JobNotFound)
    def job_not_found(_request: Any, _exc: JobNotFound) -> JSONResponse:
        return JSONResponse({"detail": "job not found"}, status_code=404)

    def check_llm(llm: str) -> None:
        try:
            mm.resolve_llm(llm)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    def require_transcript(job: Job) -> None:
        if not (job.out_dir / "transcript.json").exists():
            raise HTTPException(409, "no transcript yet")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML.read_text(encoding="utf-8")

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "llms": mm.llm_choices(), "default_llm": mm.default_llm(), "analysis_llm": store.analysis_llm,
            "profiles": [p.summary() for p in load_profiles().values()], "default_profile": "meeting",
            "home": str(store.out_root.resolve()), "user_profile_dir": str(manage.USER_PROFILE_DIR),
        }

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        store.rescan()
        jobs = sorted(store.jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.summary() for j in jobs]

    @app.post("/api/jobs")
    def create_job(
        file: Annotated[UploadFile, File()],
        speakers: Annotated[Optional[int], Form()] = None,
        names: Annotated[Optional[str], Form()] = None,
        context: Annotated[Optional[str], Form()] = None,
        vocab: Annotated[Optional[str], Form()] = None,
        llm: Annotated[Optional[str], Form()] = None,
        correct: Annotated[Optional[bool], Form()] = None,
        profile: Annotated[str, Form()] = "meeting",
        language: Annotated[Optional[str], Form()] = "ko",
        diarize: Annotated[Optional[bool], Form()] = None,
        local_only: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        llm = llm or mm.default_llm()
        check_llm(llm)
        try:
            chosen = get_profile(profile)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if speakers is not None and not 1 <= speakers <= 20:
            raise HTTPException(400, "speakers must be between 1 and 20")
        job_id = uuid.uuid4().hex[:12]
        out_dir = store.out_root / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = file.filename or "audio"
        audio = out_dir / f"input{_safe_suffix(filename)}"
        with audio.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        job = Job(
            id=job_id, filename=filename, out_dir=out_dir, llm=llm, analysis_llm=store.analysis_llm,
            names=_clean(names), context=_clean(context), vocab=_clean(vocab),
            correct=chosen.correct if correct is None else correct,
            profile=chosen.id, language=None if (language or "ko") == "auto" else _clean(language) or "ko",
            diarize=diarize, local_only=local_only,
        )
        store.submit(job, audio, speakers)
        return job.summary()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        store.rescan()
        return store.get(job_id).summary()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict[str, str]:
        try:
            store.delete(store.get(job_id))
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"deleted": job_id}

    @app.put("/api/jobs/{job_id}/document")
    def put_document(job_id: str, body: Annotated[dict[str, str], Body()]) -> dict[str, Any]:
        """Store a document written elsewhere (pasted from an agent); shown like one the app wrote."""
        job = store.get(job_id)
        text = (body.get("markdown") or "").strip()
        if not text:
            raise HTTPException(400, "markdown is empty")
        if job.status in ("queued", "running"):
            raise HTTPException(409, "job is still running")
        store.save_document(job, text + "\n")
        return job.summary()

    @app.post("/api/jobs/{job_id}/minutes")
    def regenerate_minutes(
        job_id: str,
        llm: Annotated[str, Form()],
        names: Annotated[Optional[str], Form()] = None,
        context: Annotated[Optional[str], Form()] = None,
    ) -> dict[str, Any]:
        job = store.get(job_id)
        check_llm(llm)
        if job.status in ("queued", "running"):
            raise HTTPException(409, "job is still running")
        require_transcript(job)
        store.resummarize(job, llm=llm, names=_clean(names), context=_clean(context))
        return job.summary()

    @app.post("/api/jobs/{job_id}/analysis")
    def regenerate_analysis(job_id: str, names: Annotated[Optional[str], Form()] = None) -> dict[str, Any]:
        job = store.get(job_id)
        if job.analysis_status == "running":
            raise HTTPException(409, "analysis is still running")
        require_transcript(job)
        store.reanalyze(job, names=_clean(names))
        return job.summary()

    @app.post("/api/jobs/{job_id}/correction")
    def regenerate_correction(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job.status in ("queued", "running") or "running" in (job.correction_status, job.analysis_status):
            raise HTTPException(409, "job is still running")
        require_transcript(job)
        store.recorrect(job)
        return job.summary()

    @app.put("/api/jobs/{job_id}/speakers")
    def set_speakers(job_id: str, mapping: Annotated[dict[str, str], Body()]) -> dict[str, Any]:
        """Confirm display names per speaker label; an empty name clears the confirmation."""
        job = store.get(job_id)
        require_transcript(job)
        labels = {t.speaker for t in store._load_turns(job, original=True)}
        unknown = set(mapping) - labels
        if unknown:
            raise HTTPException(400, f"unknown speaker labels: {sorted(unknown)}")
        cleaned = {k: v.strip()[:40] for k, v in mapping.items() if v and v.strip()}
        store.set_speaker_map(job, cleaned)
        return job.summary()

    @app.get("/api/jobs/{job_id}/files/{name}")
    def get_file(job_id: str, name: str, download: bool = False) -> FileResponse:
        if name not in OUTPUT_FILES:
            raise HTTPException(404, "unknown file")
        path = store.get(job_id).out_dir / name
        if not path.exists():
            raise HTTPException(404, "not ready")
        media = "application/json" if name.endswith(".json") else "text/markdown; charset=utf-8"
        return FileResponse(path, media_type=media, filename=name if download else None)

    @app.get("/api/jobs/{job_id}/prompt")
    def get_prompt(job_id: str) -> dict[str, Any]:
        """The prompt an external agent (Claude Code, Codex...) can follow to write the same document."""
        job = store.get(job_id)
        require_transcript(job)
        chosen = job._profile()
        user = mm.build_user_prompt(
            store._load_turns(job), names=job.names, context=job.context, vocab=job.vocab,
            speaker_map=job.speaker_map, profile=chosen,
        )
        return {"profile": chosen.id, "document": chosen.document, "system": chosen.system, "user": user}

    @app.get("/api/jobs/{job_id}/audio")
    def get_audio(job_id: str) -> FileResponse:
        path = store.get(job_id).audio_path()
        if path is None:
            raise HTTPException(404, "no audio")
        return FileResponse(path, media_type=AUDIO_TYPES.get(path.suffix, "application/octet-stream"))

    # ---------- agents and profiles ----------

    @app.get("/api/agents")
    def agents() -> dict[str, Any]:
        home = store.out_root.resolve()
        return {
            "agents": manage.agent_status(), "home": str(home), "mcp_config": manage.mcp_config(home),
            "steps": {t: manage.install_steps(t, home) for t in manage.TARGETS},
        }

    @app.post("/api/agents/{target}/install")
    def install_agent(target: str) -> dict[str, Any]:
        """Register the MCP server and skill with this PC's Claude Code / Codex, sharing this UI's job folder."""
        try:
            return manage.install_agent(target, store.out_root.resolve())
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except (RuntimeError, OSError) as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/profiles/{profile_id}")
    def profile_detail(profile_id: str) -> dict[str, Any]:
        try:
            return manage.profile_detail(profile_id)
        except ValueError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/profiles")
    def save_profile(body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        try:
            return manage.save_profile(body)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from e

    @app.delete("/api/profiles/{profile_id}")
    def delete_profile(profile_id: str) -> dict[str, str]:
        try:
            manage.delete_profile(profile_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"deleted": profile_id}

    return app


def main(argv: Optional[list[str]] = None) -> int:
    """Run the web UI with uvicorn."""
    import uvicorn

    p = argparse.ArgumentParser(prog="meeting-minutes-web", description="meeting-minutes 로컬 웹 화면")
    p.add_argument("--host", default="127.0.0.1", help="바인드 주소 (기본: 127.0.0.1, 이 PC에서만 접속)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--out-dir", type=Path, default=Path("out/web"), help="작업별 산출물 폴더")
    p.add_argument("--env-file", type=Path, help="API 키가 든 env 파일 (환경변수가 우선)")
    p.add_argument("--llm-config", type=Path, help=f"모델 별칭·기본값 TOML (기본: ./{mm.LLM_CONFIG_FILE})")
    p.add_argument(
        "--analysis-llm", help="보정·화자 이름 확률·주제 구간 분석 모델 (기본: llm.toml 의 analysis_default, 없으면 claude)"
    )
    args = p.parse_args(argv)
    try:
        mm.configure_llms(args.llm_config)
        if args.analysis_llm:
            mm.resolve_llm(args.analysis_llm)
    except ValueError as e:
        p.error(str(e))

    mm._log(f"web UI -> http://{args.host}:{args.port}")
    app = create_app(args.out_dir, args.env_file, args.analysis_llm)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
