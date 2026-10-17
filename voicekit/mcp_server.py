"""MCP server: lets any MCP-capable agent (Claude Code, Codex, ...) use the local speech pipeline.

Transcription, word alignment and speaker diarization run on this machine. Nothing is sent to a model
unless the agent itself reads the transcript, which is the point: the agent writes the document with
its own model (``get_document_prompt`` -> ``save_document``), so no API key is configured here.

Run it over stdio::

    claude mcp add voicekit -- voicekit mcp
    codex mcp add voicekit -- voicekit mcp
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Optional

try:  # mcp 2.x renamed FastMCP to MCPServer; the decorators are the same
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError

from voicekit.service import Service

INSTRUCTIONS = """\
Local speech toolkit. Typical flow for a recording:
1. list_profiles: pick the recording type (meeting, interview, call, lecture, podcast, memo, general).
2. transcribe_audio(path, profile=...): starts local ASR + word alignment + speaker diarization, returns a job_id at once.
   A 10 minute recording takes about 10 minutes on CPU (diarization is most of it; lecture/memo profiles skip it).
3. wait_for_job(job_id) repeatedly until status is done (each call blocks up to ~50s).
4. read_transcript(job_id): read it in pages (next_offset).
5. get_document_prompt(job_id): returns the system and user prompt; write the document yourself from them,
   then save_document(job_id, markdown) so it shows up in the web UI. Or call generate_document to use another model.
Speaker labels (SPEAKER_00...) are automatic and can be wrong at turn edges. Ask the user for names rather than guessing,
then set_speaker_names.
"""

mcp = _Server("voicekit", instructions=INSTRUCTIONS)
_service: Optional[Service] = None
_EXPECTED = (KeyError, ValueError, FileNotFoundError, RuntimeError, SystemExit)


def tool() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a tool whose expected failures reach the agent as readable messages, not a generic crash."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def run(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except _EXPECTED as e:
                msg = f"job not found: {e.args[0]}" if isinstance(e, KeyError) else str(e) or type(e).__name__
                raise ToolError(msg) from e

        return mcp.tool()(run)

    return wrap


def svc() -> Service:
    global _service
    if _service is None:
        _service = Service()
    return _service


@tool()
def list_profiles() -> list[dict[str, Any]]:
    """Recording types and what each one produces (document name, whether speakers are separated)."""
    return svc().profiles()


@tool()
def transcribe_audio(
    path: str, profile: str = "meeting", language: str = "ko", speakers: Optional[int] = None,
    diarize: Optional[bool] = None, vocab: Optional[str] = None,
) -> dict[str, Any]:
    """Start local transcription (ASR + word alignment + speaker diarization) of an audio/video file.

    Returns immediately with a job_id; poll with wait_for_job. language is a code like ko, en, ja, or "auto".
    speakers fixes the speaker count when known (best accuracy gain). diarize=false skips speaker separation.
    vocab is a comma-separated list of names and terms that Whisper should spell correctly.
    """
    return svc().start_transcription(path, profile=profile, language=language, speakers=speakers, diarize=diarize, vocab=vocab)


@tool()
def wait_for_job(job_id: str, timeout_seconds: float = 50.0) -> dict[str, Any]:
    """Wait up to timeout_seconds for the job to finish; returns its state (status, stage, log)."""
    return svc().wait(job_id, min(timeout_seconds, 55.0))


@tool()
def job_status(job_id: str) -> dict[str, Any]:
    """Current state of one job without waiting."""
    return svc().status(job_id)


@tool()
def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """Recent jobs, newest first."""
    return svc().list_jobs(limit)


@tool()
def read_transcript(job_id: str, offset_turn: int = 0, max_chars: int = 20000, format: str = "md") -> dict[str, Any]:
    """Read the speaker-labelled transcript in pages. Continue with next_offset until it is null."""
    return svc().read_transcript(job_id, fmt=format, offset_turn=offset_turn, max_chars=max_chars)


@tool()
def get_document_prompt(
    job_id: str, profile: Optional[str] = None, names: Optional[str] = None, context: Optional[str] = None
) -> dict[str, Any]:
    """The prompt that turns this transcript into its document (minutes, interview notes, lecture notes...).

    Follow `system` as your instructions and `user` as the input, write the Markdown, then call save_document.
    names = candidate names for the speakers, context = one or two lines of background.
    """
    return svc().document_prompt(job_id, profile=profile, names=names, context=context)


@tool()
def save_document(job_id: str, markdown: str) -> dict[str, Any]:
    """Store the document you wrote so the web UI and later tool calls can show it."""
    return svc().save_document(job_id, markdown)


@tool()
def generate_document(
    job_id: str, llm: str = "claude-code", profile: Optional[str] = None, names: Optional[str] = None,
    context: Optional[str] = None,
) -> dict[str, Any]:
    """Write the document with another model instead: claude-code, codex, an llm.toml alias, or provider:model.

    Blocks until done (tens of seconds). Prefer get_document_prompt + save_document when you can write it yourself.
    """
    return svc().generate_document(job_id, llm=llm, profile=profile, names=names, context=context)


@tool()
def set_speaker_names(job_id: str, mapping: dict[str, str]) -> dict[str, Any]:
    """Confirm who each speaker label is, e.g. {"SPEAKER_00": "김팀장"}. Used by later get_document_prompt calls."""
    return svc().set_speaker_names(job_id, mapping)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
