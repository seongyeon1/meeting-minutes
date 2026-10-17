"""Job registry shared by the web UI, the CLI and the MCP server.

A job is one recording and its output folder (``<root>/<id>/``): input audio, ``job.json`` state,
transcript, and the LLM-written document. The local stages (ASR, alignment, diarization) run on one
worker; LLM stages run on a separate pool so they never wait behind a long transcription. A job can
stop after the local stages (``local_only``) when an agent will write the document itself.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import meeting_minutes as mm
from voicekit.profiles import Profile, get_profile

JOB_META = "job.json"

Status = Literal["queued", "running", "done", "error"]
StepStatus = Literal["none", "running", "done", "error"]


class JobNotFound(KeyError):
    """Unknown job id; the web layer maps this to 404, the MCP layer to a tool error."""


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
    profile: str = "meeting"
    language: Optional[str] = "ko"  # None = auto-detect
    diarize: Optional[bool] = None  # None = the profile's default
    num_speakers: Optional[int] = None
    local_only: bool = False  # stop after the local stages; an agent writes the document

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
            "profile": self.profile,
            "document": self._profile().document,
            "language": self.language,
            "diarize": self.profile_default_diarize() if self.diarize is None else self.diarize,
            "analysis_enabled": self._profile().analysis,
            "local_only": self.local_only,
            "dir": str(self.out_dir.resolve()),
            "has_audio": self.audio_path() is not None,
            "has_transcript": (self.out_dir / "transcript.json").exists(),
            "has_minutes": (self.out_dir / "minutes.md").exists(),
            "has_analysis": (self.out_dir / "analysis.json").exists(),
            "has_correction": (self.out_dir / "transcript_corrected.json").exists(),
        }

    def profile_default_diarize(self) -> bool:
        return self._profile().diarize

    def _profile(self) -> Profile:
        try:
            return get_profile(self.profile)
        except ValueError:  # a profile file removed after the job ran must not break the job list
            return get_profile()

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
                "finished_at", "speakers", "profile", "language", "diarize", "num_speakers", "local_only",
            )
        }
        (self.out_dir / JOB_META).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, out_dir: Path, *, interrupted: bool = True) -> Job:
        """``interrupted=False`` keeps queued/running states, for jobs another process is still working on."""
        if not (out_dir / JOB_META).exists():
            return cls._from_outputs(out_dir)
        meta = json.loads((out_dir / JOB_META).read_text(encoding="utf-8"))
        job = cls(out_dir=out_dir, **meta)
        if not interrupted:
            return job
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
    """Job registry with a single local-processing worker and a separate LLM worker pool."""

    def __init__(self, out_root: Path, env_file: Optional[Path], analysis_llm: str) -> None:
        self.out_root = out_root
        self.env_file = env_file
        self.analysis_llm = analysis_llm
        self.jobs: dict[str, Job] = {}
        self.owned: set[str] = set()  # jobs this process runs; the rest belong to other processes (agents, the CLI)
        self._meta_mtime: dict[str, int] = {}  # job.json mtime when last read, to spot other processes' updates
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
            self._remember(job)

    def _remember(self, job: Job) -> None:
        meta = job.out_dir / JOB_META
        if meta.exists():
            self._meta_mtime[job.id] = meta.stat().st_mtime_ns

    def rescan(self) -> None:
        """Pick up jobs other processes created or advanced in the same folder (an agent over MCP, the CLI)."""
        if not self.out_root.exists():
            return
        for out_dir in self.out_root.iterdir():
            meta = out_dir / JOB_META
            if out_dir.name in self.owned or not meta.exists():
                continue
            if self._meta_mtime.get(out_dir.name) == meta.stat().st_mtime_ns:
                continue  # unchanged since we read it (also keeps the "interrupted" state of dead jobs)
            try:
                job = Job.load(out_dir, interrupted=False)
            except (OSError, ValueError, TypeError, KeyError):
                continue  # half-written by the other process; the next scan sees it whole
            self.jobs[job.id] = job
            self._remember(job)

    def delete(self, job: Job) -> None:
        """Remove the job and its folder. Refused while this process is still working on it."""
        if job.status in ("queued", "running") and job.id in self.owned:
            raise RuntimeError("job is still running")
        if "running" in (job.correction_status, job.analysis_status) and job.id in self.owned:
            raise RuntimeError("job is still running")
        shutil.rmtree(job.out_dir)
        self.jobs.pop(job.id, None)
        self.owned.discard(job.id)
        self._meta_mtime.pop(job.id, None)

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if not job:
            raise JobNotFound(job_id)
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

    def submit(self, job: Job, audio: Path, speakers: Optional[int] = None) -> None:
        job.num_speakers = speakers if speakers is not None else job.num_speakers
        self.jobs[job.id] = job
        self.owned.add(job.id)
        job.save_meta()
        self.stt_pool.submit(self._run_pipeline, job, audio)

    def save_document(self, job: Job, text: str) -> None:
        """Store a document written outside this process (an agent over MCP); the UI shows it as usual."""
        (job.out_dir / "minutes.md").write_text(text, encoding="utf-8")
        self._progress(job, "문서 저장됨 (외부 에이전트)")
        self._update(job, status="done", stage="완료", error=None, finished_at=time.time())

    def resummarize(self, job: Job, *, llm: str, names: Optional[str], context: Optional[str]) -> None:
        self.owned.add(job.id)
        self._update(
            job, status="queued", stage="회의록 재생성 대기", error=None, finished_at=None,
            llm=llm, names=names, context=context,
        )
        self.llm_pool.submit(self._run_minutes_only, job)

    def reanalyze(self, job: Job, *, names: Optional[str]) -> None:
        self.owned.add(job.id)
        self._update(job, analysis_status="running", analysis_error=None, names=names)
        self.llm_pool.submit(self._guarded_analysis, job, self._load_turns(job))

    def recorrect(self, job: Job) -> None:
        """Correct the transcript, then refresh the analysis on the corrected text."""
        self.owned.add(job.id)
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

    def _run_pipeline(self, job: Job, audio: Path) -> None:
        # Model libraries print to stdout; under an MCP stdio server that would corrupt the protocol stream.
        with contextlib.redirect_stdout(sys.stderr):
            self._run_pipeline_inner(job, audio)

    def _run_pipeline_inner(self, job: Job, audio: Path) -> None:
        try:
            profile = get_profile(job.profile)
            self._update(job, status="running")
            diarize = profile.diarize if job.diarize is None else job.diarize
            segments = mm.transcribe_and_diarize(
                audio,
                asr_model=mm.DEFAULT_ASR_MODEL,
                language=job.language,
                batch_size=8,
                compute_type="int8",
                num_speakers=job.num_speakers or profile.speakers,
                min_speakers=None,
                max_speakers=None,
                diarize_device="cpu",
                vocab=job.vocab,
                on_progress=lambda m: self._progress(job, m),
                diarize=diarize,
            )
            turns = mm.segments_to_turns(segments)
            mm.save_transcript_json(turns, job.out_dir, source=job.filename)
            mm.save_transcript_md(turns, job.out_dir)
            self._update(job, speakers=sorted({t.speaker for t in turns}))
            if job.local_only:
                self._update(job, status="done", stage="녹취록 완료 (문서는 에이전트가 작성)", finished_at=time.time())
                return
            if job.correct:
                turns = self._correct(job, turns)
            run_analysis = profile.analysis
            self._update(job, analysis_status="running" if run_analysis else "none")
            self._progress(job, f"{profile.document}({job.llm}) 작성 중" + (f", 대화 분석({job.analysis_llm})" if run_analysis else ""))
            analysis: Optional[Future[None]] = (
                self.llm_pool.submit(self._guarded_analysis, job, turns) if run_analysis else None
            )
            self._write_minutes(job, turns)
            if analysis:
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
            env_file=self.env_file, speaker_map=job.speaker_map, profile=job._profile(),
        )
        (job.out_dir / "minutes.md").write_text(minutes, encoding="utf-8")
        self._progress(job, f"{job._profile().document} 완료 ({job.llm}, {time.monotonic() - t0:.0f}s)")

    def _correct(self, job: Job, turns: list[mm.Turn]) -> list[mm.Turn]:
        """Returns corrected turns, or the originals if correction fails (shown on the job, not fatal)."""
        self._update(job, correction_status="running", correction_error=None)
        self._progress(job, f"녹취록 보정 중 ({job.analysis_llm})")
        try:
            t0 = time.monotonic()
            result = mm.correct_transcript(
                turns, llm=job.analysis_llm, names=job.names, context=job.context, vocab=job.vocab,
                env_file=self.env_file, profile=job._profile(),
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
                turns, llm=job.analysis_llm, names=job.names, context=job.context, env_file=self.env_file,
                profile=job._profile(),
            )
            (job.out_dir / "analysis.json").write_text(
                json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._progress(job, f"대화 분석 완료 ({job.analysis_llm}, {time.monotonic() - t0:.0f}s)")
            self._update(job, analysis_status="done")
        except BaseException as e:  # noqa: BLE001
            traceback.print_exception(e)
            self._update(job, analysis_status="error", analysis_error=_error_text(e))


