"""Recording-type profiles: what kind of audio this is and how to turn its transcript into a document.

A profile is a TOML file. Built-in ones live in ``voicekit/profiles/``; yours are picked up from
``$VOICEKIT_PROFILES`` (a path list), ``./profiles/`` and ``~/.config/voicekit/profiles/``, later
sources overriding earlier ones by ``id``. Nothing here touches audio or calls an LLM::

    id = "sales-call"
    label = "영업 통화"
    document = "영업 통화 요약"
    diarize = true
    system = "..."            # or: system_file = "sales-call.md" (relative to the TOML)
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from voicekit.types import log

BUILTIN_DIR = Path(__file__).with_name("profiles")
DEFAULT_PROFILE = "meeting"
# Display order for the built-ins (UI dropdown, listings); anything else follows alphabetically.
BUILTIN_ORDER = ("meeting", "interview", "call", "lecture", "podcast", "memo", "general")


@dataclass(frozen=True)
class Profile:
    """One recording type: pipeline defaults plus the prompt that writes its document."""

    id: str
    label: str  # short name shown in the UI ("회의")
    document: str  # what the LLM writes ("회의록")
    system: str  # system prompt for the document
    description: str = ""
    diarize: bool = True  # False for one-voice recordings: skips the slowest local stage
    speakers: Optional[int] = None  # default speaker count when the user gives none
    analysis: bool = True  # speaker-name probabilities and topic sections are meaningful
    correct: bool = True  # context-aware STT error correction by default
    context_label: str = "녹음 배경"
    names_label: str = "등장인물 이름 후보"
    source: str = "builtin"

    def summary(self) -> dict[str, Any]:
        """Prompt-free view for listings (UI dropdowns, MCP ``list_profiles``)."""
        return {
            "id": self.id, "label": self.label, "document": self.document, "description": self.description,
            "diarize": self.diarize, "speakers": self.speakers, "analysis": self.analysis,
            "correct": self.correct, "context_label": self.context_label, "names_label": self.names_label,
            "source": self.source,
        }


def _parse(path: Path, source: str) -> Profile:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    system = data.get("system")
    if not system and data.get("system_file"):
        system = (path.parent / str(data["system_file"])).read_text(encoding="utf-8")
    if not system:
        raise ValueError("system (or system_file) is required")
    pid = str(data.get("id") or path.stem)
    return Profile(
        id=pid,
        label=str(data.get("label") or pid),
        document=str(data.get("document") or data.get("label") or pid),
        system=str(system),
        description=str(data.get("description") or ""),
        diarize=bool(data.get("diarize", True)),
        speakers=int(data["speakers"]) if data.get("speakers") else None,
        analysis=bool(data.get("analysis", True)),
        correct=bool(data.get("correct", True)),
        context_label=str(data.get("context_label") or "녹음 배경"),
        names_label=str(data.get("names_label") or "등장인물 이름 후보"),
        source=source,
    )


def profile_dirs() -> list[tuple[Path, str]]:
    """Search order, lowest priority first."""
    dirs = [(BUILTIN_DIR, "builtin"), (Path.home() / ".config" / "voicekit" / "profiles", "user")]
    dirs.append((Path("profiles"), "project"))
    for extra in filter(None, os.environ.get("VOICEKIT_PROFILES", "").split(os.pathsep)):
        dirs.append((Path(extra).expanduser(), "env"))
    return dirs


def load_profiles() -> dict[str, Profile]:
    """All profiles by id. A broken user file is logged and skipped, never fatal for the rest."""
    found: dict[str, Profile] = {}
    for directory, source in profile_dirs():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.toml")):
            try:
                profile = _parse(path, source)
            except (OSError, ValueError, tomllib.TOMLDecodeError) as e:
                if source == "builtin":
                    raise
                log(f"profile {path} skipped: {e}")
                continue
            found[profile.id] = profile
    rank = {pid: i for i, pid in enumerate(BUILTIN_ORDER)}
    return dict(sorted(found.items(), key=lambda kv: (rank.get(kv[0], len(rank)), kv[0])))


def get_profile(profile_id: Optional[str] = None) -> Profile:
    profiles = load_profiles()
    pid = profile_id or DEFAULT_PROFILE
    if pid not in profiles:
        raise ValueError(f"unknown profile {pid!r}: use one of [{', '.join(profiles)}]")
    return profiles[pid]


def with_profile(system: str, profile: Optional[Profile]) -> str:
    """Prefix the meeting-flavoured correction/analysis prompts with the recording type.

    The base prompts say "회의"; for other types a one-line preamble re-reads that word, so one
    tested prompt serves every profile instead of N copies drifting apart.
    """
    if profile is None or profile.id == DEFAULT_PROFILE:
        return system
    note = f"이 녹음의 유형은 「{profile.label}」입니다"
    if profile.description:
        note += f" ({profile.description})"
    return f"{note}. 아래 지시문의 '회의'는 이 녹음을 뜻하고, 참석자는 이 녹음의 화자를 뜻합니다.\n\n{system}"
