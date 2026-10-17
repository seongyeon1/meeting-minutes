"""Shared types and small helpers for the local audio pipeline (no heavy imports)."""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class Turn:
    """One continuous stretch of speech by a single speaker."""

    speaker: str
    start: float
    end: float
    text: str


def log(msg: str) -> None:
    print(f"[meeting-minutes] {msg}", file=sys.stderr, flush=True)


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def render_transcript(turns: list[Turn]) -> str:
    """Render turns as a Markdown transcript, one line per turn."""
    lines = ["# 화자별 녹취록", ""]
    for t in turns:
        lines.append(f"**[{fmt_ts(t.start)}] {t.speaker}**: {t.text}")
        lines.append("")
    return "\n".join(lines)
