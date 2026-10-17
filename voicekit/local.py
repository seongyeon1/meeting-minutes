"""Local speech pipeline: ASR, word alignment and speaker diarization. No LLM, no network at run time.

Everything here runs on this machine. The stages (``transcribe`` -> ``align`` -> ``diarize``) are
separate functions so an agent or script can run only the ones it needs.
"""

from __future__ import annotations

import os
import re
import time
import warnings
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from voicekit.types import Turn, fmt_ts, log

DEFAULT_ASR_MODEL = "large-v3-turbo"
UNKNOWN_SPEAKER = "SPEAKER_??"
SINGLE_SPEAKER = "SPEAKER_00"

# pyannote warns that torchcodec can't load FFmpeg; we hand it in-memory audio, so it is never used.
warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")
# pyannote's stats pooling warns on very short speech frames; the embedding is still produced.
warnings.filterwarnings("ignore", message=r"std\(\): degrees of freedom is <= 0")

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
            log(f"sparse segment at {fmt_ts(seg['start'])} re-decoded ({_chars(text)} -> {_chars(redone)} chars"
                + ("" if better else ", kept original") + ")")
            out.append({**seg, "text": redone} if better else seg)
            continue
        fixed = collapse_repeats(redone) if redone else UNREADABLE
        log(f"repetition loop at {fmt_ts(seg['start'])} re-decoded" + (" (still looping, collapsed)" if fixed != redone else ""))
        out.append({**seg, "text": fixed})
    return out


def _asr(
    audio: Any, *, asr_model: str, language: Optional[str], batch_size: int, compute_type: str,
    vocab: Optional[str], log: Callable[[str], None],
) -> dict[str, Any]:
    import whisperx

    log("ASR running")
    t0 = time.monotonic()
    # Domain terms in the initial prompt bias Whisper toward the right spelling (e.g. "QA" not "카").
    asr_options = {"initial_prompt": vocab} if vocab else None
    # ctranslate2 has no MPS backend, so ASR and alignment run on CPU on Apple Silicon.
    model = whisperx.load_model(asr_model, "cpu", compute_type=compute_type, language=language, asr_options=asr_options)
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    result["segments"] = _repair_segments(
        model, audio, result["segments"], language=result["language"], vocab=vocab, log=log
    )
    log(f"ASR done ({result['language']}, {len(result['segments'])} segments, {time.monotonic() - t0:.0f}s)")
    return result


def _align(audio: Any, result: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    """Word-level timestamps via wav2vec2. Languages without an alignment model keep segment timing."""
    import whisperx

    log("alignment running")
    t0 = time.monotonic()
    try:
        model, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
    except (ValueError, KeyError) as e:
        log(f"alignment skipped for {result['language']!r} (no wav2vec2 model: {e}); speakers use segment timing")
        return result
    aligned = whisperx.align(result["segments"], model, metadata, audio, "cpu")
    log(f"alignment done ({time.monotonic() - t0:.0f}s)")
    return {**result, "segments": aligned["segments"]}


def _diarize(
    audio: Any, result: dict[str, Any], *, num_speakers: Optional[int], min_speakers: Optional[int],
    max_speakers: Optional[int], device: str, log: Callable[[str], None],
) -> dict[str, Any]:
    from huggingface_hub.errors import GatedRepoError
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    log("diarization running")
    t0 = time.monotonic()
    try:
        diarizer = DiarizationPipeline(token=os.environ.get("HF_TOKEN"), device=device)
    except GatedRepoError as e:
        raise SystemExit(
            "pyannote 화자분리 모델은 사용 동의가 필요합니다.\n"
            "  1) https://huggingface.co/pyannote/speaker-diarization-community-1 에서 동의\n"
            "  2) `hf auth login` 하거나 HF_TOKEN 환경변수 설정"
        ) from e
    diarize_df = diarizer(audio, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
    out = assign_word_speakers(diarize_df, result)
    log(f"diarization done ({len(set(diarize_df['speaker']))} speakers, {time.monotonic() - t0:.0f}s)")
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
    diarize: bool = True,
    align: bool = True,
) -> list[dict[str, Any]]:
    """Run WhisperX ASR, word alignment and pyannote diarization, all on this machine.

    Returns WhisperX segments, each carrying a ``speaker`` key. ``diarize=False`` (lectures, voice memos:
    one voice) skips the slowest stage and labels everything ``SPEAKER_00``; ``align=False`` skips wav2vec2.
    ``on_progress`` receives each stage's log line (the web UI shows it as job status).
    """
    emit = on_progress or log
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    emit(f"audio loaded: {fmt_ts(len(audio) / 16000)}")
    result = _asr(
        audio, asr_model=asr_model, language=language, batch_size=batch_size, compute_type=compute_type,
        vocab=vocab, log=emit,
    )
    if align:
        result = _align(audio, result, emit)
    if not diarize:
        return [{**seg, "speaker": SINGLE_SPEAKER} for seg in result["segments"]]
    result = _diarize(
        audio, result, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers,
        device=diarize_device, log=emit,
    )
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
