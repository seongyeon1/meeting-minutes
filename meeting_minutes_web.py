"""Local web UI for meeting-minutes: upload audio, watch progress, read the conversation and minutes.

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
import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

import meeting_minutes as mm

INDEX_HTML = Path(__file__).with_name("web") / "index.html"
OUTPUT_FILES = {
    "transcript.md", "transcript.json", "transcript_corrected.md", "transcript_corrected.json",
    "minutes.md", "analysis.json",
}
AUDIO_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "video/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".webm": "audio/webm", ".flac": "audio/flac",
}
JOB_META = "job.json"

Status = Literal["queued", "running", "done", "error"]
StepStatus = Literal["none", "running", "done", "error"]


@dataclass
class Job:
    """One uploaded recording and the state of its pipeline run."""

    id: str
    filename: str
    out_dir: Path
    llm: str
    analysis_llm: str = field(default_factory=mm.default_analysis_llm)
    names: Optional[str] = None
    context: Optional[str] = None
    vocab: Optional[str] = None
    status: Status = "queued"
    stage: str = "대기 중"
    log: list[str] = field(default_factory=list)
    error: Optional[str] = None
    correct: bool = True
    correction_status: StepStatus = "none"
    correction_error: Optional[str] = None
    analysis_status: StepStatus = "none"
    analysis_error: Optional[str] = None
    speaker_map: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    speakers: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Job state without file contents, for the job list and polling."""
        return {
            "id": self.id,
            "filename": self.filename,
            "llm": self.llm,
            "analysis_llm": self.analysis_llm,
            "names": self.names,
            "context": self.context,
            "status": self.status,
            "stage": self.stage,
            "log": self.log,
            "error": self.error,
            "correct": self.correct,
            "correction_status": self.correction_status,
            "correction_error": self.correction_error,
            "analysis_status": self.analysis_status,
            "analysis_error": self.analysis_error,
            "speaker_map": self.speaker_map,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "speakers": self.speakers,
            "has_audio": self.audio_path() is not None,
            "has_transcript": (self.out_dir / "transcript.json").exists(),
            "has_minutes": (self.out_dir / "minutes.md").exists(),
            "has_analysis": (self.out_dir / "analysis.json").exists(),
            "has_correction": (self.out_dir / "transcript_corrected.json").exists(),
        }

    def audio_path(self) -> Optional[Path]:
        return next(iter(sorted(self.out_dir.glob("input.*"))), None)

    def save_meta(self) -> None:
        """Persist the job so it survives a server restart (outputs are already files)."""
        meta = {
            k: getattr(self, k)
            for k in (
                "id", "filename", "llm", "analysis_llm", "names", "context", "vocab", "status", "stage",
                "log", "error", "correct", "correction_status", "correction_error", "analysis_status",
                "analysis_error", "speaker_map", "created_at",
                "finished_at", "speakers",
            )
        }
        (self.out_dir / JOB_META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path) -> Job:
        if not (out_dir / JOB_META).exists():
            return cls._from_outputs(out_dir)
        meta = json.loads((out_dir / JOB_META).read_text(encoding="utf-8"))
        job = cls(out_dir=out_dir, **meta)
        if job.status in ("queued", "running"):
            job.status, job.stage, job.error = "error", "중단됨", "서버가 재시작되어 작업이 중단됐습니다."
        if job.analysis_status == "running":
            job.analysis_status, job.analysis_error = "error", "서버가 재시작되어 분석이 중단됐습니다."
        if job.correction_status == "running":
            job.correction_status, job.correction_error = "error", "서버가 재시작되어 보정이 중단됐습니다."
        return job


    @classmethod
    def _from_outputs(cls, out_dir: Path) -> Job:
        """Rebuild a job from its files alone (runs made before job.json existed)."""
        transcript = out_dir / "transcript.json"
        data = json.loads(transcript.read_text(encoding="utf-8"))
        has_minutes = (out_dir / "minutes.md").exists()
        return cls(
            id=out_dir.name,
            filename=Path(data.get("source") or out_dir.name).name,
            out_dir=out_dir,
            llm=mm.default_llm(),
            status="done" if has_minutes else "error",
            stage="완료" if has_minutes else "회의록 없음",
            analysis_status="done" if (out_dir / "analysis.json").exists() else "none",
            correct=False,
            correction_status="done" if (out_dir / "transcript_corrected.json").exists() else "none",
            created_at=transcript.stat().st_mtime,
            finished_at=transcript.stat().st_mtime,
            speakers=sorted({t["speaker"] for t in data["turns"]}),
        )


def _error_text(exc: BaseException) -> str:
    # SystemExit carries the user-facing hint (missing key, gated model) as its message.
    return str(exc) if isinstance(exc, SystemExit) else f"{type(exc).__name__}: {exc}"


class JobStore:
    """Job registry with a single STT worker and a separate LLM worker pool."""

    def __init__(self, out_root: Path, env_file: Optional[Path], analysis_llm: str) -> None:
        self.out_root = out_root
        self.env_file = env_file
        self.analysis_llm = analysis_llm
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.stt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        # LLM calls are network-bound; they shouldn't wait behind a long STT job.
        self.llm_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.out_root.exists():
            return
        for out_dir in self.out_root.iterdir():
            if not ((out_dir / JOB_META).exists() or (out_dir / "transcript.json").exists()):
                continue
            try:
                job = Job.load(out_dir)
            except (OSError, ValueError, TypeError, KeyError) as e:
                mm._log(f"skip {out_dir.name}: {e}")
                continue
            self.jobs[job.id] = job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return job

    def _update(self, job: Job, **changes: Any) -> None:
        with self.lock:
            for k, v in changes.items():
                setattr(job, k, v)
            job.save_meta()

    def _progress(self, job: Job, msg: str) -> None:
        mm._log(f"[{job.id}] {msg}")
        with self.lock:
            job.stage = msg
            job.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
            job.save_meta()

    def _fail(self, job: Job, exc: BaseException) -> None:
        traceback.print_exception(exc)
        self._update(job, status="error", error=_error_text(exc), stage="실패", finished_at=time.time())

    def submit(self, job: Job, audio: Path, speakers: Optional[int]) -> None:
        self.jobs[job.id] = job
        job.save_meta()
        self.stt_pool.submit(self._run_pipeline, job, audio, speakers)

    def resummarize(self, job: Job, *, llm: str, names: Optional[str], context: Optional[str]) -> None:
        self._update(
            job, status="queued", stage="회의록 재생성 대기", error=None, finished_at=None,
            llm=llm, names=names, context=context,
        )
        self.llm_pool.submit(self._run_minutes_only, job)

    def reanalyze(self, job: Job, *, names: Optional[str]) -> None:
        self._update(job, analysis_status="running", analysis_error=None, names=names)
        self.llm_pool.submit(self._guarded_analysis, job, self._load_turns(job))

    def recorrect(self, job: Job) -> None:
        """Correct the transcript, then refresh the analysis on the corrected text."""
        self._update(job, correction_status="running", correction_error=None, analysis_status="running",
                     analysis_error=None)
        self.llm_pool.submit(self._run_correction_then_analysis, job)

    def set_speaker_map(self, job: Job, mapping: dict[str, str]) -> None:
        self._update(job, speaker_map=mapping)

    @staticmethod
    def _load_turns(job: Job, *, original: bool = False) -> list[mm.Turn]:
        """Corrected text when there is one; speakers and timings are the same either way."""
        corrected = job.out_dir / "transcript_corrected.json"
        path = job.out_dir / "transcript.json" if original or not corrected.exists() else corrected
        data = json.loads(path.read_text(encoding="utf-8"))
        return [mm.Turn(**t) for t in data["turns"]]

    def _run_pipeline(self, job: Job, audio: Path, speakers: Optional[int]) -> None:
        try:
            self._update(job, status="running")
            segments = mm.transcribe_and_diarize(
                audio,
                asr_model=mm.DEFAULT_ASR_MODEL,
                language="ko",
                batch_size=8,
                compute_type="int8",
                num_speakers=speakers,
                min_speakers=None,
                max_speakers=None,
                diarize_device="cpu",
                vocab=job.vocab,
                on_progress=lambda m: self._progress(job, m),
            )
            turns = mm.segments_to_turns(segments)
            mm.save_transcript_json(turns, job.out_dir, source=job.filename)
            mm.save_transcript_md(turns, job.out_dir)
            self._update(job, speakers=sorted({t.speaker for t in turns}))
            if job.correct:
                turns = self._correct(job, turns)
            self._update(job, analysis_status="running")
            self._progress(job, f"대화 분석({job.analysis_llm})·회의록({job.llm}) 작성 중")
            analysis: Future[None] = self.llm_pool.submit(self._guarded_analysis, job, turns)
            self._write_minutes(job, turns)
            analysis.result()  # keep the STT slot until the analysis lands, so "완료" means both are done
            self._update(job, status="done", stage="완료", finished_at=time.time())
        except BaseException as e:  # noqa: BLE001 - surface every failure on the job, incl. SystemExit
            self._fail(job, e)

    def _run_minutes_only(self, job: Job) -> None:
        try:
            self._update(job, status="running")
            self._write_minutes(job, self._load_turns(job))
            self._update(job, status="done", stage="완료", finished_at=time.time())
        except BaseException as e:  # noqa: BLE001
            self._fail(job, e)

    def _write_minutes(self, job: Job, turns: list[mm.Turn]) -> None:
        t0 = time.monotonic()
        minutes = mm.write_minutes(
            turns, llm=job.llm, names=job.names, context=job.context, vocab=job.vocab,
            env_file=self.env_file, speaker_map=job.speaker_map,
        )
        (job.out_dir / "minutes.md").write_text(minutes, encoding="utf-8")
        self._progress(job, f"회의록 완료 ({job.llm}, {time.monotonic() - t0:.0f}s)")

    def _correct(self, job: Job, turns: list[mm.Turn]) -> list[mm.Turn]:
        """Returns corrected turns, or the originals if correction fails (shown on the job, not fatal)."""
        self._update(job, correction_status="running", correction_error=None)
        self._progress(job, f"녹취록 보정 중 ({job.analysis_llm})")
        try:
            t0 = time.monotonic()
            result = mm.correct_transcript(
                turns, llm=job.analysis_llm, names=job.names, context=job.context, vocab=job.vocab,
                env_file=self.env_file,
            )
            mm.save_correction(result, job.out_dir)
            self._progress(job, f"녹취록 보정 완료 ({len(result['changes'])}턴, {time.monotonic() - t0:.0f}s)")
            self._update(job, correction_status="done")
            return [mm.Turn(**t) for t in result["turns"]]
        except BaseException as e:  # noqa: BLE001
            traceback.print_exception(e)
            self._update(job, correction_status="error", correction_error=_error_text(e))
            return turns

    def _run_correction_then_analysis(self, job: Job) -> None:
        turns = self._correct(job, self._load_turns(job, original=True))
        self._guarded_analysis(job, turns)

    def _guarded_analysis(self, job: Job, turns: list[mm.Turn]) -> None:
        """Analysis is an add-on: its failure is shown on the job but never fails the minutes."""
        try:
            t0 = time.monotonic()
            analysis = mm.analyze_transcript(
                turns, llm=job.analysis_llm, names=job.names, context=job.context, env_file=self.env_file
            )
            (job.out_dir / "analysis.json").write_text(
                json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._progress(job, f"대화 분석 완료 ({job.analysis_llm}, {time.monotonic() - t0:.0f}s)")
            self._update(job, analysis_status="done")
        except BaseException as e:  # noqa: BLE001
            traceback.print_exception(e)
            self._update(job, analysis_status="error", analysis_error=_error_text(e))


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
        return {"llms": mm.llm_choices(), "default_llm": mm.default_llm(), "analysis_llm": store.analysis_llm}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
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
        correct: Annotated[bool, Form()] = True,
    ) -> dict[str, Any]:
        llm = llm or mm.default_llm()
        check_llm(llm)
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
            names=_clean(names), context=_clean(context), vocab=_clean(vocab), correct=correct,
        )
        store.submit(job, audio, speakers)
        return job.summary()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return store.get(job_id).summary()

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

    @app.get("/api/jobs/{job_id}/audio")
    def get_audio(job_id: str) -> FileResponse:
        path = store.get(job_id).audio_path()
        if path is None:
            raise HTTPException(404, "no audio")
        return FileResponse(path, media_type=AUDIO_TYPES.get(path.suffix, "application/octet-stream"))

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
