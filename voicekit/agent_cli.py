"""Hand text generation to a local coding-agent CLI (Claude Code or Codex) instead of calling an API.

The agent runs headless in an empty temp directory with its tools switched off, so it only reads the
prompt and answers; it never sees the audio, your project files or your settings. The transcript text
goes to whichever service that CLI is logged in to, exactly as with an API key, but no key is needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

AGENTS = ("claude", "codex")
DEFAULT_TIMEOUT = 900


def agent_binary(agent: str) -> str:
    """Path to the agent CLI; ``VOICEKIT_CLAUDE_BIN`` / ``VOICEKIT_CODEX_BIN`` override PATH lookup."""
    if agent not in AGENTS:
        raise ValueError(f"unknown agent {agent!r}: use one of {', '.join(AGENTS)}")
    found = os.environ.get(f"VOICEKIT_{agent.upper()}_BIN") or shutil.which(agent)
    if not found:
        raise SystemExit(f"`{agent}` CLI 를 찾을 수 없습니다. 설치하고 로그인했는지 확인하세요 (PATH 또는 VOICEKIT_{agent.upper()}_BIN).")
    return found


def available_agents() -> list[str]:
    return [a for a in AGENTS if os.environ.get(f"VOICEKIT_{a.upper()}_BIN") or shutil.which(a)]


def build_command(agent: str, *, system: str, model: Optional[str], out_file: Path) -> list[str]:
    """argv for one headless, tool-less run; the user prompt is piped through stdin."""
    binary = agent_binary(agent)
    if agent == "claude":
        cmd = [binary, "-p", "--tools", "", "--no-session-persistence", "--setting-sources", "",
               "--system-prompt", system]
        return cmd + (["--model", model] if model else [])
    cmd = [binary, "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral", "--ignore-rules",
           "-o", str(out_file)]
    return cmd + (["-m", model] if model else []) + ["-"]


def complete_via_agent(
    user: str, *, system: str, agent: str, model: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT
) -> str:
    """Run ``claude -p`` or ``codex exec`` on the prompt and return its final answer."""
    with tempfile.TemporaryDirectory(prefix="voicekit-agent-") as tmp:
        out_file = Path(tmp) / "answer.txt"
        cmd = build_command(agent, system=system, model=model, out_file=out_file)
        # Codex has no system-prompt flag, so its instructions ride in front of the user text.
        prompt = user if agent == "claude" else f"{system}\n\n---\n\n{user}"
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, cwd=tmp, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"{agent} CLI did not answer within {timeout}s") from e
        if proc.returncode != 0:
            raise RuntimeError(f"{agent} CLI exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-400:]}")
        text = out_file.read_text(encoding="utf-8") if agent == "codex" and out_file.exists() else proc.stdout
    text = text.strip()
    if not text:
        raise RuntimeError(f"{agent} CLI returned an empty answer")
    return text
