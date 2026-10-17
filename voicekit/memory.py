"""Per-profile memory of terms you corrected, so the same mistake is fixed the next time. No LLM, all local.

Every time you fix a word in a transcript, the (wrong, right) pair is remembered for that recording type
(``common`` applies to all). A pair stays a candidate until it has been confirmed ``MIN_COUNT`` times or
you pin it; active terms then

- go into Whisper's ``initial_prompt`` (the right spelling), and into the correction and document prompts;
- replace the wrong spelling in new transcripts, as whole words (a trailing Korean particle is allowed).

Files live in ``$VOICEKIT_MEMORY_DIR`` (default ``~/.config/voicekit/memory/``), one JSON per scope. Set
``VOICEKIT_MEMORY=off`` to ignore the memory entirely. Nothing here touches audio or calls a model.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

COMMON = "common"
MIN_COUNT = 2  # confirmations before a learned term is used (pinned terms are used at once)
MAX_HINT_CHARS = 300  # Whisper's prompt window is ~224 tokens; leave room for the job's own vocab
MAX_TERM_CHARS = 40
MAX_ENTRIES = 500
_SCOPE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_WORD = "0-9A-Za-z가-힣"
_PUNCT = " \t\n.,?!;:\"'`()[]{}<>~…·"
# Korean particles that may follow a replaced word ("카를" -> "QA를"). Longest first for the regex.
PARTICLES = tuple(sorted((
    "은", "는", "이", "가", "을", "를", "의", "에", "에서", "에는", "에게", "와", "과", "도", "로", "으로", "랑", "이랑",
    "만", "까지", "부터", "처럼", "보다", "입니다", "이에요", "예요", "이고", "이며", "라고", "이라고", "인", "이란", "란",
    "님", "씨",
), key=len, reverse=True))
_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get("VOICEKIT_MEMORY", "").lower() not in ("off", "0", "false")


def memory_dir() -> Path:
    """Resolved at call time so tests (and users) can point it elsewhere."""
    return Path(os.environ.get("VOICEKIT_MEMORY_DIR") or Path.home() / ".config" / "voicekit" / "memory")


def valid_scope(scope: str) -> str:
    if not _SCOPE.fullmatch(scope):
        raise ValueError(f"invalid scope {scope!r}: use {COMMON!r} or a profile id")
    return scope


def _path(scope: str) -> Path:
    return memory_dir() / f"{valid_scope(scope)}.json"


def _clean(text: str) -> str:
    return " ".join(str(text).split()).strip(_PUNCT)


def _entry_active(e: dict[str, Any]) -> bool:
    return bool(e.get("enabled", True)) and (bool(e.get("pinned")) or int(e.get("count", 0)) >= MIN_COUNT)


def _wrong_active(e: dict[str, Any], wrong: str, n: int) -> bool:
    return bool(e.get("enabled", True)) and (bool(e.get("pinned")) or n >= MIN_COUNT) and (
        len(wrong) >= 2 or bool(e.get("pinned"))
    )


# ---------- storage ----------


def load(scope: str) -> list[dict[str, Any]]:
    path = _path(scope)  # a bad scope name is an error, not an empty memory
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("entries") if isinstance(data, dict) else None
    return [e for e in items or [] if isinstance(e, dict) and e.get("right")]


def _save(scope: str, entries: list[dict[str, Any]]) -> None:
    path = _path(scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "entries": entries}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def view(scope: str) -> list[dict[str, Any]]:
    """Entries for the UI, with whether each is in use yet."""
    out = []
    for e in load(scope):
        out.append({**e, "active": _entry_active(e), "min_count": MIN_COUNT})
    return sorted(out, key=lambda e: (not e.get("pinned"), -int(e.get("count", 0)), e["right"]))


# ---------- changing entries ----------


@dataclass(frozen=True)
class Learned:
    wrong: Optional[str]
    right: str
    count: int
    active: bool  # the right spelling now goes into the recognizer's prompt
    replace: bool  # the wrong spelling is now replaced in new transcripts (not for one-character spellings, unless pinned)

    def as_dict(self) -> dict[str, Any]:
        return {"wrong": self.wrong, "right": self.right, "count": self.count, "active": self.active, "replace": self.replace}


def add(scope: str, right: str, wrong: Optional[list[str]] = None, *, pinned: bool = True) -> dict[str, Any]:
    """A term you typed in by hand: pinned, so it is used at once."""
    right = _clean(right)
    wrongs = [w for w in (_clean(w) for w in wrong or []) if w and w != right]
    if not right or len(right) > MAX_TERM_CHARS or any(len(w) > MAX_TERM_CHARS for w in wrongs):
        raise ValueError(f"terms must be 1-{MAX_TERM_CHARS} characters")
    with _LOCK:
        entries = load(scope)
        e = next((x for x in entries if x["right"] == right), None)
        if e is None:
            if len(entries) >= MAX_ENTRIES:
                raise ValueError(f"at most {MAX_ENTRIES} terms per scope")
            e = {"id": uuid.uuid4().hex[:8], "right": right, "wrong": {}, "count": 0, "pinned": pinned,
                 "enabled": True, "source": "manual"}
            entries.append(e)
        e["pinned"] = bool(e.get("pinned")) or pinned
        e["count"] = max(int(e.get("count", 0)), 1)
        for w in wrongs:
            e.setdefault("wrong", {}).setdefault(w, 1)
        e["updated"] = time.time()
        _save(scope, entries)
        return {**e, "active": _entry_active(e), "min_count": MIN_COUNT}


def update(scope: str, entry_id: str, *, enabled: Optional[bool] = None, pinned: Optional[bool] = None) -> None:
    with _LOCK:
        entries = load(scope)
        e = next((x for x in entries if x.get("id") == entry_id), None)
        if e is None:
            raise KeyError(entry_id)
        if enabled is not None:
            e["enabled"] = enabled
        if pinned is not None:
            e["pinned"] = pinned
        e["updated"] = time.time()
        _save(scope, entries)


def remove(scope: str, entry_id: str) -> None:
    with _LOCK:
        entries = load(scope)
        kept = [x for x in entries if x.get("id") != entry_id]
        if len(kept) == len(entries):
            raise KeyError(entry_id)
        _save(scope, kept)


def learn(scope: str, pairs: list[tuple[str, str]]) -> list[Learned]:
    """Remember corrections the user made. Returns what each one now stands at."""
    result: list[Learned] = []
    if not pairs:
        return result
    with _LOCK:
        entries = load(scope)
        for wrong, right in pairs:
            wrong, right = _clean(wrong), _clean(right)
            if not wrong or not right or wrong == right or max(len(wrong), len(right)) > MAX_TERM_CHARS:
                continue
            for other in entries:  # the user now undoes an earlier "right -> wrong" mapping: forget that one
                if other["right"] == wrong:
                    other.get("wrong", {}).pop(right, None)
            e = next((x for x in entries if x["right"] == right), None)
            if e is None:
                if len(entries) >= MAX_ENTRIES:
                    continue
                e = {"id": uuid.uuid4().hex[:8], "right": right, "wrong": {}, "count": 0, "pinned": False,
                     "enabled": True, "source": "edit"}
                entries.append(e)
            e["count"] = int(e.get("count", 0)) + 1
            e.setdefault("wrong", {})
            e["wrong"][wrong] = int(e["wrong"].get(wrong, 0)) + 1
            e["updated"] = time.time()
            result.append(Learned(
                wrong, right, e["wrong"][wrong], _entry_active(e), _wrong_active(e, wrong, e["wrong"][wrong])
            ))
        _save(scope, entries)
    return result


# ---------- finding the pairs in an edit ----------


def _strip_particle(a: str, b: str) -> tuple[str, str]:
    """"카를" -> "QA를": drop a particle both sides share so the pair is the word itself."""
    for p in PARTICLES:
        if a.endswith(p) and b.endswith(p) and len(a) > len(p) and len(b) > len(p):
            return a[: -len(p)], b[: -len(p)]
    return a, b


def extract_pairs(before: str, after: str) -> list[tuple[str, str]]:
    """Word replacements between two versions of a turn. A rewrite (many changes) teaches nothing."""
    x, y = before.split(), after.split()
    ops = [o for o in difflib.SequenceMatcher(a=x, b=y, autojunk=False).get_opcodes() if o[0] != "equal"]
    if not ops or len(ops) > 6:
        return []
    pairs = []
    for tag, i1, i2, j1, j2 in ops:
        if tag != "replace" or i2 - i1 > 3 or j2 - j1 > 3:
            continue
        wrong, right = _strip_particle(" ".join(x[i1:i2]), " ".join(y[j1:j2]))
        wrong, right = _clean(wrong), _clean(right)
        if wrong and right and wrong != right:
            pairs.append((wrong, right))
    return pairs


# ---------- using the memory ----------


def _scopes(profile: Optional[str]) -> list[str]:
    return [COMMON] + ([profile] if profile and profile != COMMON and _SCOPE.fullmatch(profile) else [])


def vocab_hint(profile: Optional[str], *, limit: int = MAX_HINT_CHARS) -> str:
    """Active right spellings for Whisper's prompt: pinned first, then most confirmed, within the budget."""
    if not enabled():
        return ""
    entries = [e for s in _scopes(profile) for e in load(s) if _entry_active(e)]
    entries.sort(key=lambda e: (not e.get("pinned"), -int(e.get("count", 0))))
    picked: list[str] = []
    size = 0
    for e in entries:
        word = e["right"]
        if word in picked:
            continue
        if size + len(word) + 2 > limit:
            break
        picked.append(word)
        size += len(word) + 2
    return ", ".join(picked)


def merge_vocab(job_vocab: Optional[str], profile: Optional[str]) -> Optional[str]:
    """The job's own vocab plus the remembered terms (without repeating any)."""
    hint = vocab_hint(profile)
    if not hint:
        return job_vocab
    have = (job_vocab or "").lower()
    extra = [w for w in hint.split(", ") if w.lower() not in have]
    return ", ".join(p for p in (job_vocab, ", ".join(extra)) if p) or None


def _rules(profile: Optional[str]) -> dict[str, tuple[str, int]]:
    rules: dict[str, tuple[str, int]] = {}
    for scope in _scopes(profile):  # profile entries come last and win ties
        for e in load(scope):
            for wrong, n in (e.get("wrong") or {}).items():
                usable = wrong != e["right"] and _wrong_active(e, wrong, int(n))
                if usable and (wrong not in rules or int(n) >= rules[wrong][1]):
                    rules[wrong] = (e["right"], int(n))
    return rules


def apply(text: str, profile: Optional[str], rules: Optional[dict[str, tuple[str, int]]] = None) -> tuple[str, list[tuple[str, str]]]:
    """Replace remembered wrong spellings. Returns the new text and the (wrong, right) pairs used."""
    if not enabled():
        return text, []
    rules = _rules(profile) if rules is None else rules
    if not rules:
        return text, []
    words = "|".join(re.escape(w) for w in sorted(rules, key=len, reverse=True))
    particles = "|".join(re.escape(p) for p in PARTICLES)
    pattern = re.compile(rf"(?<![{_WORD}])({words})(?=(?:{particles})?(?![{_WORD}]))")
    used: list[tuple[str, str]] = []

    def sub(m: re.Match[str]) -> str:
        used.append((m.group(1), rules[m.group(1)][0]))
        return rules[m.group(1)][0]

    return pattern.sub(sub, text), used


def apply_to_turns(turns: list[Any], profile: Optional[str]) -> list[dict[str, Any]]:
    """Fix ``turns`` in place (objects with ``.text``); returns what changed, for a record the user can read."""
    rules = _rules(profile) if enabled() else {}
    applied: list[dict[str, Any]] = []
    if not rules:
        return applied
    for i, t in enumerate(turns):
        new, used = apply(t.text, profile, rules)
        if used:
            t.text = new
            applied.extend({"turn": i, "wrong": w, "right": r} for w, r in used)
    return applied
