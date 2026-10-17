"""Setup actions shared by the CLI and the web UI: connect agents (MCP + skill) and manage profiles.

Nothing here touches audio or calls an LLM. Agent registration edits the user's global Claude Code /
Codex configuration, so callers (the CLI ``--apply`` flag, a confirmed click in the UI) must make that
an explicit choice.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from voicekit.profiles import BUILTIN_DIR, load_profiles

SERVER_NAME = "voicekit"
SKILL_FILE = Path(__file__).with_name("skill") / "SKILL.md"
AGENTS_MARK = ("<!-- voicekit:begin -->", "<!-- voicekit:end -->")
TARGETS = ("claude", "codex")
USER_PROFILE_DIR = Path.home() / ".config" / "voicekit" / "profiles"
PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def mcp_command() -> list[str]:
    return [sys.executable, "-m", "voicekit.mcp_server"]


def mcp_config(home: Optional[Path] = None) -> dict[str, Any]:
    """The ``mcpServers`` entry for any MCP-capable agent; ``home`` pins the job folder it shares with the UI."""
    cmd = mcp_command()
    entry: dict[str, Any] = {"command": cmd[0], "args": cmd[1:]}
    if home:
        entry["env"] = {"VOICEKIT_HOME": str(home)}
    return {"mcpServers": {SERVER_NAME: entry}}


def _skill_dir() -> Path:
    return Path.home() / ".claude" / "skills" / SERVER_NAME


def _agents_md() -> Path:
    return Path.home() / ".codex" / "AGENTS.md"


def _registered(target: str) -> bool:
    """Whether ``<target> mcp get voicekit`` finds the server (short timeout: `claude mcp get` health-checks)."""
    try:
        r = subprocess.run(
            [target, "mcp", "get", SERVER_NAME], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and "No MCP server named" not in (r.stdout + r.stderr)


def agent_status() -> list[dict[str, Any]]:
    """Per agent CLI: installed on PATH, MCP server registered, skill / instructions in place."""
    out = []
    for target in TARGETS:
        path = shutil.which(target)
        if target == "claude":
            guidance = (_skill_dir() / "SKILL.md").exists()
        else:
            md = _agents_md()
            guidance = md.exists() and AGENTS_MARK[0] in md.read_text(encoding="utf-8")
        out.append({
            "target": target, "cli": path, "installed": path is not None,
            "mcp": bool(path) and _registered(target), "guidance": guidance,
            "guidance_label": "스킬" if target == "claude" else "AGENTS.md 안내",
        })
    return out


def install_steps(target: str, home: Optional[Path] = None) -> list[str]:
    """What ``install_agent`` will do, as readable lines (the CLI dry run, the UI confirmation)."""
    if target not in TARGETS:
        raise ValueError(f"unknown agent {target!r}: use one of {list(TARGETS)}")
    if target == "claude":
        return [f"스킬 복사: {_skill_dir() / 'SKILL.md'}", " ".join(_register_cmd(target, home))]
    return [f"안내 블록 추가: {_agents_md()}", " ".join(_register_cmd(target, home))]


def _register_cmd(target: str, home: Optional[Path]) -> list[str]:
    env = [f"{'-e' if target == 'claude' else '--env'}", f"VOICEKIT_HOME={home}"] if home else []
    if target == "claude":
        return ["claude", "mcp", "add", "--scope", "user", *env, SERVER_NAME, "--", *mcp_command()]
    return ["codex", "mcp", "add", *env, SERVER_NAME, "--", *mcp_command()]


def install_agent(target: str, home: Optional[Path] = None) -> dict[str, Any]:
    """Put the guidance in place and register the MCP server. Replaces an existing registration."""
    steps = install_steps(target, home)
    if shutil.which(target) is None:
        raise RuntimeError(f"`{target}` 이(가) PATH 에 없습니다")
    done: list[str] = []
    if target == "claude":
        _skill_dir().mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SKILL_FILE, _skill_dir() / "SKILL.md")
        done.append(f"skill -> {_skill_dir() / 'SKILL.md'}")
    else:
        write_agents_block(_agents_md())
        done.append(f"instructions -> {_agents_md()}")
    # `mcp add` fails on an existing name; remove first so re-installing moves the job folder too.
    subprocess.run([target, "mcp", "remove", *(["--scope", "user"] if target == "claude" else []), SERVER_NAME],
                   capture_output=True, text=True, timeout=60, check=False)
    r = subprocess.run(_register_cmd(target, home), capture_output=True, text=True, timeout=60, check=False)
    done.append((r.stdout + r.stderr).strip() or f"mcp add exit {r.returncode}")
    return {"target": target, "ok": r.returncode == 0, "steps": steps, "output": done}


def write_agents_block(path: Path) -> None:
    """Idempotent marked block so Codex knows when to reach for the tools."""
    body = SKILL_FILE.read_text(encoding="utf-8").split("---", 2)[-1].strip()
    block = f"{AGENTS_MARK[0]}\n## voicekit (local speech pipeline)\n\n{body}\n{AGENTS_MARK[1]}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if AGENTS_MARK[0] in text:
        head, _, rest = text.partition(AGENTS_MARK[0])
        text = head.rstrip("\n") + ("\n\n" if head.strip() else "") + block + rest.partition(AGENTS_MARK[1])[2].lstrip("\n")
    else:
        text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + block
    path.write_text(text, encoding="utf-8")


# ---------- profiles ----------

def profile_detail(profile_id: str) -> dict[str, Any]:
    """A profile with its system prompt, for the editor form."""
    profiles = load_profiles()
    if profile_id not in profiles:
        raise ValueError(f"unknown profile {profile_id!r}")
    p = profiles[profile_id]
    return {**p.summary(), "system": p.system}


def _toml_str(value: str) -> str:
    """A TOML basic string. JSON escaping is a valid subset (\\n, \\", \\uXXXX), and keeps Korean readable."""
    return json.dumps(value, ensure_ascii=False)


def save_profile(data: dict[str, Any], directory: Optional[Path] = None) -> dict[str, Any]:
    """Write ``<directory>/<id>.toml``; a built-in id is overridden by the user file, not edited in place."""
    directory = directory or USER_PROFILE_DIR
    pid = str(data.get("id") or "").strip()
    if not PROFILE_ID.match(pid):
        raise ValueError("id 는 영문 소문자·숫자·-·_ 로 1~32자 (첫 글자는 영문/숫자)")
    system = str(data.get("system") or "").strip()
    if not system:
        raise ValueError("문서 작성 지시문(system)이 필요합니다")
    label = str(data.get("label") or pid).strip()
    lines = [
        f"id = {_toml_str(pid)}",
        f"label = {_toml_str(label)}",
        f"document = {_toml_str(str(data.get('document') or label).strip())}",
        f"description = {_toml_str(str(data.get('description') or '').strip())}",
        f"diarize = {'true' if data.get('diarize', True) else 'false'}",
        f"analysis = {'true' if data.get('analysis', True) else 'false'}",
        f"correct = {'true' if data.get('correct', True) else 'false'}",
    ]
    speakers = data.get("speakers")
    if speakers not in (None, ""):
        n = int(speakers)
        if not 1 <= n <= 20:
            raise ValueError("speakers 는 1~20")
        lines.append(f"speakers = {n}")
    for key in ("context_label", "names_label"):
        if data.get(key):
            lines.append(f"{key} = {_toml_str(str(data[key]).strip())}")
    lines.append(f"system = {_toml_str(system)}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pid}.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profile_detail(pid)


def delete_profile(profile_id: str, directory: Optional[Path] = None) -> None:
    """Remove a user profile file. Built-ins cannot be deleted (a user file with the same id only shadows one)."""
    directory = directory or USER_PROFILE_DIR
    if not PROFILE_ID.match(profile_id or ""):
        raise ValueError("invalid profile id")
    path = directory / f"{profile_id}.toml"
    if not path.exists():
        kind = "내장 프로필은 삭제할 수 없습니다" if (BUILTIN_DIR / f"{profile_id}.toml").exists() else "사용자 프로필이 아닙니다"
        raise ValueError(f"{profile_id}: {kind}")
    path.unlink()
