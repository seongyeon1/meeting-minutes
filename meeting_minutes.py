"""Speaker-diarized meeting minutes from an audio file.

Pipeline (audio never leaves the machine):
    WhisperX (faster-whisper ASR + wav2vec2 alignment + pyannote diarization)
    -> speaker turns -> transcript.md
    -> (optional) LLM transcript correction -> transcript_corrected.json
    -> LLM (Claude, OpenAI, Gemini, Ollama or any OpenAI-compatible server; transcript text only)
    -> minutes.md

Usage:
    uv run meeting-minutes meeting.m4a --speakers 3 --names "김팀장,이선임,박매니저"
    uv run meeting-minutes meeting.m4a --llm ollama:qwen3:14b      # fully local
    uv run meeting-minutes out/meeting/transcript.json   # re-summarize without re-running STT
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_ASR_MODEL = "large-v3-turbo"
CLAUDE_MODEL = "claude-opus-5-5"
# Overridable per install through `default` / `analysis_default` in llm.toml.
DEFAULT_LLM = "claude"
DEFAULT_ANALYSIS_LLM = "claude"
LLM_CONFIG_FILE = "llm.toml"
# OpenAI-compatible providers: name -> (default base URL, env var holding the key or None).
# `<NAME>_BASE_URL` overrides the URL, e.g. OPENAI_BASE_URL for Azure or a proxy.
OPENAI_COMPAT_PROVIDERS: dict[str, tuple[Optional[str], Optional[str]]] = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "compat": (None, "LLM_API_KEY"),  # any other OpenAI-compatible server; URL from LLM_BASE_URL
}
# Short names that work without any config file.
BUILTIN_LLMS = {"claude": f"anthropic:{CLAUDE_MODEL}", "gpt": "openai:gpt-5.2"}
UNKNOWN_SPEAKER = "SPEAKER_??"

# pyannote warns that torchcodec can't load FFmpeg; we hand it in-memory audio, so it is never used.
warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")
# pyannote's stats pooling warns on very short speech frames; the embedding is still produced.
warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0")


@dataclass
class Turn:
    """One continuous stretch of speech by a single speaker."""

    speaker: str
    start: float
    end: float
    text: str


def _log(msg: str) -> None:
    print(f"[meeting-minutes] {msg}", file=sys.stderr, flush=True)


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# Whisper's own hallucination test: a ~30s segment whose text compresses too well is a repetition loop.
COMPRESSION_RATIO_THRESHOLD = 2.4
UNREADABLE = "[인식 불가]"
# A decoder loop repeats a 1-5 word unit back to back. Single words need more repeats ("네 네 네 네" is speech).
_LOOP_MAX_UNIT = 5
_LOOP_MIN_REPEATS = {1: 5}
_LOOP_MIN_REPEATS_DEFAULT = 4
# Loops inside one word ("중독독독독독독독") show up as a syllable repeated many times.
_CHAR_LOOP_RE = re.compile(r"(\w)\1{5,}")


def compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def _find_loops(words: list[str]) -> list[tuple[int, int, int]]:
    """(start, unit, repeats) for each back-to-back repetition run, punctuation ignored."""
    keys = [w.strip(".,!?~") for w in words]
    runs, i = [], 0
    while i < len(words):
        best = None
        for n in range(1, _LOOP_MAX_UNIT + 1):
            unit = keys[i : i + n]
            if len(unit) < n or not all(unit):
                break
            k = 1
            while keys[i + k * n : i + (k + 1) * n] == unit:
                k += 1
            if k >= _LOOP_MIN_REPEATS.get(n, _LOOP_MIN_REPEATS_DEFAULT) and (not best or k * n > best[2] * best[1]):
                best = (i, n, k)
        if best:
            runs.append(best)
            i = best[0] + best[1] * best[2]
        else:
            i += 1
    return runs


def is_repetition_loop(text: str) -> bool:
    return bool(_find_loops(text.split()) or _CHAR_LOOP_RE.search(text))


def collapse_repeats(text: str) -> str:
    """Drop the repeats of a loop. A looping single token ("Q Q Q") is noise and becomes a marker;
    a looping phrase keeps its first occurrence, which is usually what was really said."""
    words = text.split()
    out, i = [], 0
    for start, n, k in _find_loops(words):
        out += words[i:start]
        out += [UNREADABLE] if n == 1 else words[start : start + n]
        i = start + n * k
    out += words[i:]
    return _CHAR_LOOP_RE.sub(lambda m: m.group(1) + UNREADABLE, " ".join(out))


# Batched decoding can also skip most of a window and emit only its last sentence(s); the aligner then
# pins that text to the start of the window, so the lost speech also shifts the speaker. Real speech
# runs 5-8 non-space chars/s in Korean (more in English), so a long segment far below that was cut short.
_SPARSE_MIN_SECONDS = 8.0
_SPARSE_MAX_CHARS_PER_SECOND = 2.5


def _chars(text: str) -> int:
    return len("".join(text.split()))


def is_sparse_segment(seg: dict[str, Any]) -> bool:
    duration = seg["end"] - seg["start"]
    return duration >= _SPARSE_MIN_SECONDS and _chars(seg["text"]) / duration < _SPARSE_MAX_CHARS_PER_SECOND


def _repair_segments(
    model: Any, audio: Any, segments: list[dict[str, Any]], *, language: str, vocab: Optional[str],
    log: Callable[[str], None],
) -> list[dict[str, Any]]:
    """Re-decode broken segments without batching, where faster-whisper's temperature fallback applies.

    WhisperX's batched decoding has no compression-ratio fallback, so a decoder that falls into a
    loop ("Q. Q. Q. ...") keeps it and swallows the real speech under it; whatever still loops after
    the retry is collapsed to a marker rather than shown as text. Segments with far too little text
    for their length (skipped speech) are retried too, and the retry wins only if it recovers more.
    """
    sr = 16000
    out = []
    for seg in segments:
        text = seg["text"]
        looping = is_repetition_loop(text) or len(text) > 40 and compression_ratio(text) > COMPRESSION_RATIO_THRESHOLD
        sparse = not looping and is_sparse_segment(seg)
        if not (looping or sparse):
            out.append(seg)
            continue
        clip = audio[int(seg["start"] * sr) : int(seg["end"] * sr)]
        pieces, _ = model.model.transcribe(
            clip, language=language, beam_size=5, condition_on_previous_text=False,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD, initial_prompt=vocab, vad_filter=False,
        )
        redone = " ".join(p.text.strip() for p in pieces).strip()
        if sparse:
            better = _chars(redone) > _chars(text) and not is_repetition_loop(redone)
            log(f"sparse segment at {_fmt_ts(seg['start'])} re-decoded ({_chars(text)} -> {_chars(redone)} chars"
                + ("" if better else ", kept original") + ")")
            out.append({**seg, "text": redone} if better else seg)
            continue
        fixed = collapse_repeats(redone) if redone else UNREADABLE
        log(f"repetition loop at {_fmt_ts(seg['start'])} re-decoded" + (" (still looping, collapsed)" if fixed != redone else ""))
        out.append({**seg, "text": fixed})
    return out


def transcribe_and_diarize(
    audio_path: Path,
    *,
    asr_model: str,
    language: Optional[str],
    batch_size: int,
    compute_type: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    diarize_device: str,
    vocab: Optional[str],
    on_progress: Optional[Callable[[str], None]] = None,
) -> list[dict[str, Any]]:
    """Run WhisperX ASR, word alignment and pyannote diarization.

    Returns WhisperX segments, each carrying a ``speaker`` key when one could be assigned.
    ``on_progress`` receives each stage's log line (the web UI shows it as job status).
    """
    log = on_progress or _log
    import whisperx
    from huggingface_hub.errors import GatedRepoError
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    # ctranslate2 has no MPS backend, so ASR and alignment run on CPU on Apple Silicon.
    device = "cpu"
    audio = whisperx.load_audio(str(audio_path))
    log(f"audio loaded: {_fmt_ts(len(audio) / 16000)}")

    log("ASR running")
    t0 = time.monotonic()
    # Domain terms in the initial prompt bias Whisper toward the right spelling (e.g. "QA" not "카").
    asr_options = {"initial_prompt": vocab} if vocab else None
    model = whisperx.load_model(
        asr_model, device, compute_type=compute_type, language=language, asr_options=asr_options
    )
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    lang = result["language"]
    result["segments"] = _repair_segments(
        model, audio, result["segments"], language=lang, vocab=vocab, log=log
    )
    log(f"ASR done ({lang}, {len(result['segments'])} segments, {time.monotonic() - t0:.0f}s)")
    del model

    log("alignment running")
    t0 = time.monotonic()
    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)
    log(f"alignment done ({time.monotonic() - t0:.0f}s)")
    del align_model

    log("diarization running")
    t0 = time.monotonic()
    try:
        diarizer = DiarizationPipeline(token=os.environ.get("HF_TOKEN"), device=diarize_device)
    except GatedRepoError as e:
        raise SystemExit(
            "pyannote 화자분리 모델은 사용 동의가 필요합니다.\n"
            "  1) https://huggingface.co/pyannote/speaker-diarization-community-1 에서 동의\n"
            "  2) `hf auth login` 하거나 HF_TOKEN 환경변수 설정"
        ) from e
    diarize_df = diarizer(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    result = assign_word_speakers(diarize_df, result)
    speakers = sorted(set(diarize_df["speaker"]))
    log(f"diarization done ({len(speakers)} speakers, {time.monotonic() - t0:.0f}s)")
    return list(result["segments"])


def segments_to_turns(segments: list[dict[str, Any]], max_gap: float = 2.0) -> list[Turn]:
    """Merge consecutive segments of the same speaker into turns.

    Segments are split per word speaker first, because a single Whisper segment
    often spans a speaker change.
    """
    pieces: list[Turn] = []
    for seg in segments:
        words = seg.get("words") or []
        timed = [w for w in words if "start" in w and "end" in w]
        if not timed:
            pieces.append(
                Turn(seg.get("speaker", UNKNOWN_SPEAKER), seg["start"], seg["end"], seg["text"].strip())
            )
            continue
        current: Optional[Turn] = None
        for w in words:
            # Untimed words (digits, symbols) inherit the running speaker.
            speaker = w.get("speaker") or (current.speaker if current else seg.get("speaker", UNKNOWN_SPEAKER))
            start = w.get("start", current.end if current else seg["start"])
            end = w.get("end", start)
            if current and current.speaker == speaker:
                current.text += " " + w["word"]
                current.end = end
            else:
                if current:
                    pieces.append(current)
                current = Turn(speaker, start, end, w["word"])
        if current:
            pieces.append(current)

    turns: list[Turn] = []
    for p in pieces:
        if turns and turns[-1].speaker == p.speaker and p.start - turns[-1].end <= max_gap:
            turns[-1].text += " " + p.text
            turns[-1].end = p.end
        else:
            turns.append(Turn(p.speaker, p.start, p.end, p.text.strip()))
    for t in turns:
        t.text = " ".join(t.text.split())
    return [t for t in turns if t.text]


def render_transcript(turns: list[Turn]) -> str:
    """Render turns as a Markdown transcript, one line per turn."""
    lines = ["# 화자별 녹취록", ""]
    for t in turns:
        lines.append(f"**[{_fmt_ts(t.start)}] {t.speaker}**: {t.text}")
        lines.append("")
    return "\n".join(lines)


MINUTES_SYSTEM = """\
당신은 한국 기업의 회의록을 정리하는 서기입니다. 화자분리된 녹취록을 받아 회의록을 작성합니다.

녹취록 특성:
- 화자 라벨(SPEAKER_00 등)은 음성 분석으로 자동으로 붙인 것이라 한두 문장 단위로 틀릴 수 있습니다.
- 음성 인식 오류(동음이의어, 고유명사·숫자 오인식)가 섞여 있습니다. 문맥상 명백한 것만 바로잡고, 애매하면 원문 표기를 유지하세요.

작성 규칙:
- 녹취록에 없는 내용을 지어내지 마세요. 결정되지 않은 것을 결정사항으로 쓰지 마세요.
- 화자 표에는 녹취록에 실제로 나오는 라벨만 씁니다. 라벨을 새로 만들지 마세요.
- 화자 이름은 그 화자가 스스로 이름을 밝히거나, 누군가 이름을 불러 질문한 직후 그 화자가 답한 경우에만 붙이고, 근거 발화를 화자 표에 적으세요.
  발화 속에서 언급된 사람(제3자)을 말하는 사람으로 착각하지 마세요. 근거가 없으면 이름 칸을 "미상"으로 두고 역할(진행자 등)만 적습니다.
- 담당자는 "제가 하겠다"고 말한 화자의 라벨(또는 확인된 이름)로 적으세요.
- 참석자 이름 후보가 주어지면 그 안에서 매칭하되, 근거가 약하면 매칭하지 마세요.
- 각 항목에 근거가 되는 타임스탬프를 [HH:MM:SS] 형식으로 붙이세요.

출력 형식 (Markdown, 이 순서 그대로):

# 회의록: <회의 주제 한 줄>

## 참석자 (화자 매핑)
| 화자 라벨 | 추정 이름/역할 | 근거 |

## 요약
3~5문장.

## 안건별 논의
### 1. <안건>
- 누가 어떤 입장이었는지 화자를 밝혀서 정리

## 결정사항
- 결정 내용 [타임스탬프]

## 액션 아이템
| 할 일 | 담당 | 기한 | 근거 |
(담당·기한이 언급되지 않았으면 "미정")

## 미결 사항 / 후속 논의 필요
"""


def build_user_prompt(
    turns: list[Turn],
    *,
    names: Optional[str],
    context: Optional[str],
    vocab: Optional[str],
    speaker_map: Optional[dict[str, str]] = None,
) -> str:
    """Assemble the user turn: speaker labels, optional hints, then the transcript."""
    labels = sorted({t.speaker for t in turns})
    hints = [f"녹취록의 화자 라벨 ({len(labels)}개): {', '.join(labels)}"]
    confirmed = {k: v for k, v in (speaker_map or {}).items() if k in labels and v}
    if confirmed:
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(confirmed.items()))
        hints.append(f"사용자가 확인한 화자 이름 (근거 없이 그대로 사용): {pairs}")
    if names:
        hints.append(f"참석자 이름 후보: {names}")
    if context:
        hints.append(f"회의 배경: {context}")
    if vocab:
        hints.append(
            f"용어 목록: {vocab}\n"
            "녹취록에서 이 용어와 발음이 비슷한데 문맥상 어색한 단어는 음성 인식 오류이니, 회의록에는 이 용어로 고쳐 적으세요."
        )
    return "\n".join(hints) + "\n\n<transcript>\n" + render_transcript(turns) + "\n</transcript>"


def _load_env_file(path: Optional[Path]) -> dict[str, str]:
    """Parse KEY=VALUE lines from an env file; missing file means no extra values."""
    out: dict[str, str] = {}
    if not path or not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            v = v.strip().strip("'\"")
            if v:
                out[k.strip()] = v
    return out


@dataclass(frozen=True)
class LLMSpec:
    """A resolved model: which API to call, with which model id and credentials."""

    name: str
    provider: str  # "anthropic" or a key of OPENAI_COMPAT_PROVIDERS
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    # Reasoning models (gpt-5.x, o-series) want max_completion_tokens and no temperature.
    reasoning: bool = False


_llm_config: dict[str, Any] = {}


def configure_llms(path: Optional[Path]) -> None:
    """Load model aliases and defaults from a TOML file (``llm.toml`` in the cwd if not given).

    The file keeps install-specific endpoints (a company gateway, a local server) out of the code::

        default = "local"
        [models.local]
        provider = "ollama"
        model = "qwen3:14b"
    """
    import tomllib

    global _llm_config
    path = path or Path(os.environ.get("MEETING_MINUTES_LLM_CONFIG") or LLM_CONFIG_FILE)
    _llm_config = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name in ("default", "analysis_default"):
        if name in _llm_config:
            resolve_llm(_llm_config[name])  # fail at startup, not on the first job


def default_llm() -> str:
    return str(_llm_config.get("default") or DEFAULT_LLM)


def default_analysis_llm() -> str:
    return str(_llm_config.get("analysis_default") or _llm_config.get("default") or DEFAULT_ANALYSIS_LLM)


def llm_choices() -> list[str]:
    """Names offered in the UI: configured aliases first, then the built-in ones."""
    configured = list(_llm_config.get("models") or {})
    return configured + [n for n in BUILTIN_LLMS if n not in configured]


def resolve_llm(name: str) -> LLMSpec:
    """Turn an alias (``claude``, a name from llm.toml) or ``provider:model`` into an LLMSpec."""
    models = _llm_config.get("models") or {}
    if name in models:
        m = models[name]
        provider = str(m.get("provider") or "compat")
        if provider != "anthropic" and provider not in OPENAI_COMPAT_PROVIDERS:
            raise ValueError(f"llm.toml models.{name}: unknown provider {provider!r}")
        if not m.get("model"):
            raise ValueError(f"llm.toml models.{name}: model is required")
        return LLMSpec(
            name=name, provider=provider, model=str(m["model"]), base_url=m.get("base_url"),
            api_key_env=m.get("api_key_env"), reasoning=bool(m.get("reasoning", provider == "openai")),
        )
    if name in BUILTIN_LLMS:
        return resolve_llm(BUILTIN_LLMS[name])
    provider, sep, model = name.partition(":")
    if sep and model and (provider == "anthropic" or provider in OPENAI_COMPAT_PROVIDERS):
        return LLMSpec(name=name, provider=provider, model=model, reasoning=provider == "openai")
    known = ", ".join([*models, *BUILTIN_LLMS])
    raise ValueError(
        f"unknown llm {name!r}: use one of [{known}] or provider:model "
        f"(providers: anthropic, {', '.join(OPENAI_COMPAT_PROVIDERS)})"
    )


def _secret(name: str, env_file: Optional[Path]) -> Optional[str]:
    return os.environ.get(name) or _load_env_file(env_file).get(name)


def _chat_url(spec: LLMSpec, env_file: Optional[Path]) -> str:
    default_url, _ = OPENAI_COMPAT_PROVIDERS[spec.provider]
    env_name = "LLM_BASE_URL" if spec.provider == "compat" else f"{spec.provider.upper()}_BASE_URL"
    url = spec.base_url or _secret(env_name, env_file) or default_url
    if not url:
        raise SystemExit(f"{spec.name}: base URL 이 없습니다. {env_name} 환경변수나 llm.toml 의 base_url 로 지정하세요.")
    url = url.rstrip("/")
    # Some gateways publish the full endpoint rather than an OpenAI-style /v1 base.
    return url if url.endswith("/chat/completions") else url + "/chat/completions"


def complete_openai_compat(user: str, *, spec: LLMSpec, env_file: Optional[Path], system: str) -> str:
    """Call any OpenAI-compatible Chat Completions endpoint, always streaming.

    Streaming keeps long minutes under proxy timeouts, and some gateways accept nothing else.
    An empty result is raised as an error: a few gateways answer unknown model ids with 200 and no text.
    """
    import httpx

    _, default_key = OPENAI_COMPAT_PROVIDERS[spec.provider]
    key_name = spec.api_key_env or default_key
    key = _secret(key_name, env_file) if key_name else None
    # Ollama and LLM_API_KEY-less compat servers run without auth; a named key is always required.
    if not key and (spec.api_key_env or spec.provider not in ("ollama", "compat")):
        raise SystemExit(f"{key_name} 가 없습니다. 환경변수로 넣거나 --env-file 로 키가 든 env 파일을 지정하세요.")
    url = _chat_url(spec, env_file)

    body: dict[str, Any] = {
        "model": spec.model,
        "stream": True,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if spec.reasoning:
        # Reasoning tokens count against this cap too, so leave headroom beyond the visible minutes.
        body["max_completion_tokens"] = 16384
    else:
        body.update(max_tokens=8192, temperature=0.2)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    parts: list[str] = []
    finish = None
    with httpx.stream("POST", url, json=body, headers=headers, timeout=httpx.Timeout(600, connect=20)) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{spec.name} HTTP {r.status_code}: {r.read().decode(errors='replace')[:300]}")
        for line in r.iter_lines():
            # Not every server puts a space after "data:".
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            if "choices" not in chunk and ("error" in chunk or "state" in chunk):
                # Errors can arrive mid-stream as a chunk without choices.
                detail = chunk.get("error") or chunk.get("res") or chunk
                raise RuntimeError(f"{spec.name} error: {json.dumps(detail, ensure_ascii=False)[:400]}")
            for ch in chunk.get("choices") or []:
                parts.append((ch.get("delta") or {}).get("content") or "")
                finish = ch.get("finish_reason") or finish
    text = "".join(parts)
    if not text:
        raise RuntimeError(f"{spec.name} returned an empty response (model={spec.model}, finish={finish})")
    if finish == "length":
        _log("warning: response hit max_tokens and may be truncated")
    return text


def minutes_via_claude(
    user: str, *, system: str = MINUTES_SYSTEM, model: str = CLAUDE_MODEL, env_file: Optional[Path] = None
) -> str:
    """Call Claude through the Anthropic API (ANTHROPIC_API_KEY from the env or --env-file)."""
    import anthropic

    client = anthropic.Anthropic(api_key=_secret("ANTHROPIC_API_KEY", env_file))
    with client.beta.messages.stream(
        model=model,
        max_tokens=64000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium"},
        # On a safety decline, the API re-runs the request on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined the request: {message.stop_details}")
    if message.stop_reason == "max_tokens":
        _log("warning: response hit max_tokens and may be truncated")
    return "".join(b.text for b in message.content if b.type == "text")


def save_transcript_json(turns: list[Turn], out_dir: Path, *, source: str) -> Path:
    """Write turns as transcript.json, the input for re-summarizing without STT."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "transcript.json"
    payload = {"source": source, "turns": [asdict(t) for t in turns]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_transcript_md(turns: list[Turn], out_dir: Path) -> Path:
    """Write turns as the human-readable transcript.md."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "transcript.md"
    path.write_text(render_transcript(turns), encoding="utf-8")
    return path


def save_correction(correction: dict[str, Any], out_dir: Path) -> None:
    """transcript_corrected.json keeps originals for the diff view; the .md is the readable version."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript_corrected.json").write_text(
        json.dumps(correction, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    turns = [Turn(**t) for t in correction["turns"]]
    (out_dir / "transcript_corrected.md").write_text(
        render_transcript(turns).replace("# 화자별 녹취록", "# 화자별 녹취록 (보정본)", 1), encoding="utf-8"
    )


def write_minutes(
    turns: list[Turn],
    *,
    llm: str,
    names: Optional[str],
    context: Optional[str],
    vocab: Optional[str],
    env_file: Optional[Path],
    speaker_map: Optional[dict[str, str]] = None,
) -> str:
    """Build the prompt from turns and return the minutes from the chosen LLM."""
    user = build_user_prompt(turns, names=names, context=context, vocab=vocab, speaker_map=speaker_map)
    return _complete(user, system=MINUTES_SYSTEM, llm=llm, env_file=env_file)


def _complete(user: str, *, system: str, llm: str, env_file: Optional[Path]) -> str:
    spec = resolve_llm(llm)
    if spec.provider == "anthropic":
        return minutes_via_claude(user, system=system, model=spec.model, env_file=env_file)
    return complete_openai_compat(user, spec=spec, env_file=env_file, system=system)


CORRECTION_SYSTEM = """\
당신은 음성 인식(STT) 녹취록 교정자입니다. 회의 맥락을 바탕으로 인식 오류만 바로잡습니다.

입력은 한 줄에 한 턴이며 형식은 `#턴번호 화자라벨: 발화` 입니다. <context> 는 앞뒤 흐름을 보라고 주는 참고용이고, 고칠 대상은 <target> 안의 턴뿐입니다.

고칠 것:
- 발음이 비슷해서 잘못 받아쓴 단어 (예: 회의 맥락상 "카 범위" → "QA 범위")
- 고유명사·용어·사람 이름 표기: 용어 목록과 참석자 이름 후보가 주어지면 그 표기로 맞춤. 같은 대상이 턴마다 다르게 적혔으면 하나로 통일
- 띄어쓰기, 명백한 오탈자
- 같은 단어가 비정상적으로 반복되는 인식 오류는 제거

지킬 것:
- 말한 내용을 요약하거나 다듬지 마세요. 구어체, 말버릇, 반복, 미완성 문장은 그대로 둡니다.
- 없는 내용을 더하지 마세요. 확신이 없으면 고치지 않습니다.
- 교체는 발음이 비슷한 말로만 합니다. 알아듣기 어려운 구간을 문맥으로 추측해서 다른 말로 채우지 마세요.
- [인식 불가] 표시는 그대로 둡니다.
- 화자 라벨과 턴 순서는 바꾸지 않습니다.

출력 (JSON만): 바뀐 턴만 담습니다. 바꿀 게 없으면 빈 배열.
{"turns": [{"turn": 12, "text": "교정된 발화 전체"}]}
"""

# Corrections may only fix words; a rewrite that shrinks or grows the turn this much is a summary or an addition.
CORRECTION_MIN_RATIO = 0.6
CORRECTION_MAX_RATIO = 1.5
CORRECTION_CHUNK_CHARS = 3000
CORRECTION_CONTEXT_TURNS = 2


def _correction_chunks(turns: list[Turn]) -> list[tuple[int, int]]:
    chunks, start, size = [], 0, 0
    for i, t in enumerate(turns):
        if size and size + len(t.text) > CORRECTION_CHUNK_CHARS:
            chunks.append((start, i - 1))
            start, size = i, 0
        size += len(t.text)
    if turns:
        chunks.append((start, len(turns) - 1))
    return chunks


def build_correction_prompt(
    turns: list[Turn], lo: int, hi: int, *, names: Optional[str], context: Optional[str], vocab: Optional[str]
) -> str:
    hints = []
    if context:
        hints.append(f"회의 배경: {context}")
    if names:
        hints.append(f"참석자 이름 후보: {names}")
    if vocab:
        hints.append(f"용어 목록: {vocab}")

    def lines(a: int, b: int) -> str:
        return "\n".join(f"#{i} {turns[i].speaker}: {turns[i].text}" for i in range(a, b + 1))

    parts = ["\n".join(hints)] if hints else []
    if lo > 0:
        parts.append(f"<context>\n{lines(max(0, lo - CORRECTION_CONTEXT_TURNS), lo - 1)}\n</context>")
    parts.append(f"<target>\n{lines(lo, hi)}\n</target>")
    if hi < len(turns) - 1:
        parts.append(f"<context>\n{lines(hi + 1, min(len(turns) - 1, hi + CORRECTION_CONTEXT_TURNS))}\n</context>")
    return "\n\n".join(parts)


def _accept_correction(original: str, corrected: str) -> bool:
    if not corrected or corrected == original:
        return False
    if is_repetition_loop(original):
        return True  # dropping a loop is supposed to shrink the turn
    ratio = len(corrected) / max(1, len(original))
    return CORRECTION_MIN_RATIO <= ratio <= CORRECTION_MAX_RATIO


def correct_transcript(
    turns: list[Turn],
    *,
    llm: str,
    names: Optional[str],
    context: Optional[str],
    vocab: Optional[str],
    env_file: Optional[Path],
) -> dict[str, Any]:
    """Context-aware STT error correction; speakers and timings never change, only turn text.

    Returns ``{"llm", "turns", "changes"}`` where ``changes`` keeps each original next to its
    correction so the UI can show what moved. Rejected rewrites (too short/long) are listed too.
    """
    # Loops are removed deterministically first so the model never spends tokens echoing them.
    texts = [collapse_repeats(t.text) if is_repetition_loop(t.text) else t.text for t in turns]
    base = [Turn(t.speaker, t.start, t.end, txt) for t, txt in zip(turns, texts, strict=True)]
    chunks = _correction_chunks(base)

    def run(chunk: tuple[int, int]) -> dict[int, str]:
        lo, hi = chunk
        reply = _complete(
            build_correction_prompt(base, lo, hi, names=names, context=context, vocab=vocab),
            system=CORRECTION_SYSTEM, llm=llm, env_file=env_file,
        )
        out: dict[int, str] = {}
        for item in _extract_json(reply).get("turns") or []:
            i = _as_turn(item.get("turn"), len(base))
            if i is not None and lo <= i <= hi:  # edits to context turns belong to another chunk
                out[i] = " ".join(str(item.get("text") or "").split())
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        edits: dict[int, str] = {}
        for result in pool.map(run, chunks):
            edits.update(result)

    corrected, changes = [], []
    for i, (t, b) in enumerate(zip(turns, base, strict=True)):
        proposed = edits.get(i)
        accepted = proposed is not None and _accept_correction(b.text, proposed)
        final = proposed if accepted else b.text
        rejected = proposed if proposed is not None and not accepted and proposed != b.text else None
        if final != t.text or rejected:
            changes.append({"turn": i, "original": t.text, "corrected": final, "rejected": rejected})
        corrected.append(Turn(t.speaker, t.start, t.end, final))
    return {"llm": llm, "turns": [asdict(t) for t in corrected], "changes": changes}


ANALYSIS_SYSTEM = """\
당신은 회의 녹취록 분석기입니다. 화면에 대화 흐름을 보여주기 위한 구조 정보를 JSON 하나로만 출력합니다.

입력 녹취록은 한 줄에 한 턴이며 형식은 `#턴번호 [HH:MM:SS] 화자라벨: 발화` 입니다.
화자 라벨은 음성 분석으로 자동으로 붙인 것이라 턴 경계에서 한두 단어씩 틀릴 수 있고, 음성 인식 오류도 섞여 있습니다.

1. speakers: 녹취록에 나오는 화자 라벨마다 "이 화자가 누구인지" 이름 후보별 확률을 냅니다.
   - 후보 이름은 참석자 이름 후보가 주어지면 그 안에서, 없으면 녹취록에서 사람 이름으로 불린 것 중에서 고릅니다. 항상 "미상" 후보를 포함합니다.
   - 강한 근거: 그 화자가 스스로 이름을 밝힘, 누군가 이름을 불러 질문한 직후 그 화자가 답함.
   - 약한 근거: 역할·담당 업무 언급이 맞아떨어짐.
   - 발화 속에서 언급된 제3자는 그 화자의 근거가 아닙니다. 근거가 약하면 "미상"에 확률을 크게 줍니다.
   - 한 사람이 여러 라벨에서 동시에 높은 확률을 갖지 않도록 라벨끼리 배타적으로 따집니다.
   - 라벨마다 prob 합은 1. evidence 에는 근거가 된 턴번호와 짧은 이유를 적습니다.
   - role: 진행자, 발표자, 보고자 같은 대화 속 역할. 모르면 null.
2. topics: 대화를 시간 순서대로 주제 구간 2~8개로 나눕니다. 첫 구간의 start_turn 은 0.
   title 은 20자 이내, summary 는 한 문장, 누가 무엇을 말했는지가 드러나게.
3. key_turns: 흐름에서 중요한 턴. type 은 decision(결정), action(할 일·담당), question(답을 기다리는 질문), issue(문제·리스크) 중 하나. note 는 15자 이내.

출력 (JSON만, 코드블록·설명 없이):
{"speakers": [{"label": "SPEAKER_00", "role": "진행자", "candidates": [{"name": "김팀장", "prob": 0.8, "evidence": [{"turn": 0, "reason": "스스로 이름을 밝힘"}]}, {"name": "미상", "prob": 0.2, "evidence": []}]}],
 "topics": [{"start_turn": 0, "title": "...", "summary": "..."}],
 "key_turns": [{"turn": 5, "type": "decision", "note": "..."}]}
"""

KEY_TURN_TYPES = {"decision", "action", "question", "issue"}
UNKNOWN_NAME = "미상"


def build_analysis_prompt(turns: list[Turn], *, names: Optional[str], context: Optional[str]) -> str:
    """Numbered transcript so the model cites turn indexes, which we can validate, not timestamps."""
    labels = sorted({t.speaker for t in turns})
    hints = [f"화자 라벨 ({len(labels)}개): {', '.join(labels)}", f"턴 수: {len(turns)} (#0 ~ #{len(turns) - 1})"]
    if names:
        hints.append(f"참석자 이름 후보: {names}")
    if context:
        hints.append(f"회의 배경: {context}")
    lines = [f"#{i} [{_fmt_ts(t.start)}] {t.speaker}: {t.text}" for i, t in enumerate(turns)]
    return "\n".join(hints) + "\n\n<transcript>\n" + "\n".join(lines) + "\n</transcript>"


def _extract_json(text: str) -> Any:
    """Parse the first JSON object in a reply, tolerating code fences or stray prose around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in analysis reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _as_turn(value: Any, n: int) -> Optional[int]:
    try:
        i = int(value)
    except (TypeError, ValueError):
        return None
    return i if 0 <= i < n else None


def normalize_analysis(raw: dict[str, Any], turns: list[Turn]) -> dict[str, Any]:
    """Keep only what the transcript can back: known labels, real turn indexes, probabilities summing to 1."""
    n = len(turns)
    labels = sorted({t.speaker for t in turns})
    speakers: dict[str, dict[str, Any]] = {}
    for sp in raw.get("speakers") or []:
        label = str(sp.get("label", ""))
        if label not in labels or label in speakers:
            continue  # invented or duplicate labels are dropped
        merged: dict[str, dict[str, Any]] = {}
        for c in sp.get("candidates") or []:
            name = str(c.get("name") or "").strip() or UNKNOWN_NAME
            try:
                prob = max(0.0, float(c.get("prob", 0)))
            except (TypeError, ValueError):
                prob = 0.0
            evidence = []
            for e in c.get("evidence") or []:
                i = _as_turn(e.get("turn") if isinstance(e, dict) else e, n)
                if i is not None:
                    evidence.append({"turn": i, "reason": str(e.get("reason", "")) if isinstance(e, dict) else ""})
            slot = merged.setdefault(name, {"name": name, "prob": 0.0, "evidence": []})
            slot["prob"] += prob
            slot["evidence"] += evidence
        total = sum(c["prob"] for c in merged.values())
        if total <= 0:
            merged = {UNKNOWN_NAME: {"name": UNKNOWN_NAME, "prob": 1.0, "evidence": []}}
            total = 1.0
        candidates = sorted(
            ({**c, "prob": round(c["prob"] / total, 3)} for c in merged.values()), key=lambda c: -c["prob"]
        )
        role = sp.get("role")
        speakers[label] = {"label": label, "role": str(role) if role else None, "candidates": candidates}
    for label in labels:
        speakers.setdefault(
            label,
            {"label": label, "role": None, "candidates": [{"name": UNKNOWN_NAME, "prob": 1.0, "evidence": []}]},
        )

    starts: dict[int, dict[str, str]] = {}
    for tp in raw.get("topics") or []:
        i = _as_turn(tp.get("start_turn"), n)
        if i is not None and i not in starts:
            starts[i] = {"title": str(tp.get("title") or "").strip(), "summary": str(tp.get("summary") or "").strip()}
    if n and 0 not in starts:
        # The first topic must cover the opening turns; stretch the earliest one back to #0.
        first = min(starts) if starts else None
        starts[0] = starts.pop(first) if first is not None else {"title": "전체 대화", "summary": ""}
    topics = []
    order = sorted(starts)
    for k, i in enumerate(order):
        last = (order[k + 1] - 1) if k + 1 < len(order) else n - 1
        topics.append({
            "start_turn": i, "end_turn": last, "start": turns[i].start, "end": turns[last].end, **starts[i],
        })

    key_turns = []
    seen: set[int] = set()
    for kt in raw.get("key_turns") or []:
        i = _as_turn(kt.get("turn"), n)
        kind = str(kt.get("type") or "")
        if i is None or kind not in KEY_TURN_TYPES or i in seen:
            continue
        seen.add(i)
        key_turns.append({"turn": i, "type": kind, "note": str(kt.get("note") or "").strip()})
    key_turns.sort(key=lambda k: k["turn"])
    return {"speakers": [speakers[label] for label in labels], "topics": topics, "key_turns": key_turns}


def analyze_transcript(
    turns: list[Turn],
    *,
    llm: str,
    names: Optional[str],
    context: Optional[str],
    env_file: Optional[Path],
) -> dict[str, Any]:
    """Speaker-name probabilities, topic sections and key turns for the conversation view."""
    reply = _complete(
        build_analysis_prompt(turns, names=names, context=context),
        system=ANALYSIS_SYSTEM, llm=llm, env_file=env_file,
    )
    result = normalize_analysis(_extract_json(reply), turns)
    result["llm"] = llm
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="meeting-minutes",
        description="로컬 WhisperX로 화자분리 녹취록을 만들고 LLM으로 회의록을 작성합니다.",
    )
    p.add_argument("input", type=Path, help="오디오/비디오 파일, 또는 이전 실행의 transcript.json")
    p.add_argument("-o", "--out-dir", type=Path, help="출력 폴더 (기본: ./out/<파일명>)")
    p.add_argument("--language", default="ko", help="음성 언어 코드, 'auto'면 자동 감지 (기본: ko)")
    p.add_argument("--asr-model", default=DEFAULT_ASR_MODEL, help=f"Whisper 모델 (기본: {DEFAULT_ASR_MODEL})")
    p.add_argument("--compute-type", default="int8", help="ctranslate2 연산 타입 (CPU: int8 권장)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--speakers", type=int, help="화자 수를 알면 지정 (정확도가 가장 크게 오름)")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--diarize-device", default="cpu", help="pyannote 장치: cpu | mps (기본: cpu)")
    p.add_argument("--vocab", help='자주 나오는 용어·고유명사, 쉼표 구분 (예: "QA,결제 모듈")')
    p.add_argument("--names", help='참석자 이름 후보, 쉼표 구분 (예: "김팀장,이선임")')
    p.add_argument("--context", help="회의 배경 한두 줄 (회의록 품질에 도움)")
    p.add_argument(
        "--llm",
        help="회의록 모델: claude, gpt, llm.toml 의 별칭, 또는 provider:model "
        "(예: openai:gpt-5.2, gemini:gemini-2.5-pro, ollama:qwen3:14b). 기본: llm.toml 의 default, 없으면 claude",
    )
    p.add_argument("--llm-config", type=Path, help=f"모델 별칭·기본값 TOML (기본: ./{LLM_CONFIG_FILE})")
    p.add_argument("--env-file", type=Path, help="API 키가 든 env 파일 (환경변수가 우선)")
    p.add_argument("--no-summary", action="store_true", help="녹취록만 만들고 LLM 호출은 건너뜀")
    p.add_argument(
        "--correct", action="store_true",
        help="회의록 전에 녹취록 인식 오류를 맥락 기반으로 보정하고 그 보정본으로 회의록 작성 (--analysis-llm 사용)",
    )
    p.add_argument(
        "--analyze", action="store_true", help="화자 이름 확률·주제 구간·핵심 턴을 analysis.json 으로 저장"
    )
    p.add_argument(
        "--analysis-llm", help="--correct, --analyze 에 쓸 모델 (기본: llm.toml 의 analysis_default, 없으면 --llm 과 같은 기본값)"
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configure_llms(args.llm_config)
        args.llm = args.llm or default_llm()
        args.analysis_llm = args.analysis_llm or default_analysis_llm()
        resolve_llm(args.llm)
        resolve_llm(args.analysis_llm)
    except ValueError as e:
        parser.error(str(e))
    src: Path = args.input
    if not src.exists():
        _log(f"file not found: {src}")
        return 2

    if src.suffix == ".json":
        out_dir = args.out_dir or src.parent
        turns = [Turn(**t) for t in json.loads(src.read_text(encoding="utf-8"))["turns"]]
        _log(f"loaded {len(turns)} turns from {src}")
    else:
        out_dir = args.out_dir or Path("out") / src.stem
        segments = transcribe_and_diarize(
            src,
            asr_model=args.asr_model,
            language=None if args.language == "auto" else args.language,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
            num_speakers=args.speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            diarize_device=args.diarize_device,
            vocab=args.vocab,
        )
        turns = segments_to_turns(segments)
        save_transcript_json(turns, out_dir, source=str(src))

    save_transcript_md(turns, out_dir)
    _log(f"transcript -> {out_dir / 'transcript.md'}")

    if args.no_summary:
        return 0
    if args.correct:
        correction = correct_transcript(
            turns, llm=args.analysis_llm, names=args.names, context=args.context, vocab=args.vocab,
            env_file=args.env_file,
        )
        save_correction(correction, out_dir)
        turns = [Turn(**t) for t in correction["turns"]]
        _log(f"corrected {len(correction['changes'])} turns -> {out_dir / 'transcript_corrected.md'}")
    _log(f"writing minutes with {args.llm} (transcript text only is sent)")
    minutes = write_minutes(
        turns, llm=args.llm, names=args.names, context=args.context, vocab=args.vocab, env_file=args.env_file
    )
    (out_dir / "minutes.md").write_text(minutes, encoding="utf-8")
    _log(f"minutes -> {out_dir / 'minutes.md'}")
    if args.analyze:
        analysis = analyze_transcript(
            turns, llm=args.analysis_llm, names=args.names, context=args.context, env_file=args.env_file
        )
        (out_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"analysis -> {out_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
