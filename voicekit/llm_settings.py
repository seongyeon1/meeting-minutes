"""LLM settings you change from the web UI instead of editing ``llm.toml`` and exporting environment variables.

Two small files in ``$VOICEKIT_CONFIG_DIR`` (default ``~/.config/voicekit/``):

- ``llm.toml``: your model aliases and the two defaults, same format as the project's ``llm.toml``. It is
  layered over that file; a model with the same name replaces the project's.
- ``secrets.env``: API keys as ``NAME=value`` lines, readable by you only (mode 0600). A key set in the
  environment or in ``--env-file`` still wins over this file, so nothing you exported stops working.

Keys are only ever written here, never sent back to the browser. Nothing in this module calls a model.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import tomllib
from pathlib import Path
from typing import Any, Optional

from voicekit.types import log

MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
KEY_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
MAX_VALUE = 4096
_LOCK = threading.Lock()


def config_dir() -> Path:
    """Resolved at call time so tests (and users) can point it elsewhere."""
    return Path(os.environ.get("VOICEKIT_CONFIG_DIR") or Path.home() / ".config" / "voicekit")


def config_path() -> Path:
    return config_dir() / "llm.toml"


def secrets_path() -> Path:
    return config_dir() / "secrets.env"


def _atomic_write(path: Path, text: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")  # mkstemp creates it 0600
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------- models and defaults ----------


def load_config() -> dict[str, Any]:
    """The user's ``llm.toml``; a broken file is logged and ignored so it never stops the server."""
    path = config_path()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log(f"{path} ignored: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def _q(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def dump_config(cfg: dict[str, Any]) -> str:
    lines = ["# Written by the voicekit web UI (LLM 설정). Safe to edit by hand."]
    for key in ("default", "analysis_default"):
        if cfg.get(key):
            lines.append(f"{key} = {_q(str(cfg[key]))}")
    for name, model in (cfg.get("models") or {}).items():
        lines.append(f"\n[models.{_q(name)}]")
        for key in ("provider", "model", "base_url", "api_key_env"):
            if model.get(key):
                lines.append(f"{key} = {_q(str(model[key]))}")
        if "reasoning" in model:
            lines.append(f"reasoning = {'true' if model['reasoning'] else 'false'}")
    return "\n".join(lines) + "\n"


def _clean_model(model: dict[str, Any], providers: set[str]) -> dict[str, Any]:
    provider = str(model.get("provider") or "").strip()
    name_of_model = str(model.get("model") or "").strip()
    if provider not in providers:
        raise ValueError(f"provider must be one of {sorted(providers)}")
    if not name_of_model or len(name_of_model) > 200:
        raise ValueError("model id is required (at most 200 characters)")
    out: dict[str, Any] = {"provider": provider, "model": name_of_model}
    base_url = str(model.get("base_url") or "").strip()
    if base_url:
        if not re.fullmatch(r"https?://\S{1,300}", base_url):
            raise ValueError("base_url must be an http(s) URL")
        out["base_url"] = base_url
    key_env = str(model.get("api_key_env") or "").strip()
    if key_env:
        if not KEY_NAME.fullmatch(key_env):
            raise ValueError("api_key_env must look like MY_API_KEY (capital letters, digits, underscores)")
        out["api_key_env"] = key_env
    if model.get("reasoning") is not None:
        out["reasoning"] = bool(model["reasoning"])
    return out


def upsert_model(name: str, model: dict[str, Any], providers: set[str]) -> None:
    if not MODEL_NAME.fullmatch(name):
        raise ValueError("name: letters, digits, . _ - (at most 40 characters, no colon)")
    cleaned = _clean_model(model, providers)
    with _LOCK:
        cfg = load_config()
        cfg.setdefault("models", {})[name] = cleaned
        _atomic_write(config_path(), dump_config(cfg), 0o600)


def delete_model(name: str) -> None:
    with _LOCK:
        cfg = load_config()
        models = cfg.get("models") or {}
        if name not in models:
            raise KeyError(name)
        del models[name]
        for key in ("default", "analysis_default"):
            if cfg.get(key) == name:
                del cfg[key]  # a default that points nowhere would fall back to the project's anyway
        _atomic_write(config_path(), dump_config(cfg), 0o600)


def set_defaults(default: Optional[str], analysis_default: Optional[str]) -> None:
    with _LOCK:
        cfg = load_config()
        for key, value in (("default", default), ("analysis_default", analysis_default)):
            if value:
                cfg[key] = value
            elif value is not None:
                cfg.pop(key, None)
        _atomic_write(config_path(), dump_config(cfg), 0o600)


# ---------- keys ----------


def saved_keys() -> dict[str, str]:
    try:
        text = secrets_path().read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if v.strip():
                out[k.strip()] = v.strip()
    return out


def _write_keys(keys: dict[str, str]) -> None:
    body = "# API keys saved from the voicekit web UI. Keep this file private.\n"
    _atomic_write(secrets_path(), body + "".join(f"{k}={v}\n" for k, v in sorted(keys.items())), 0o600)


def set_key(name: str, value: str) -> None:
    value = value.strip()
    if not KEY_NAME.fullmatch(name):
        raise ValueError("key name must look like MY_API_KEY (capital letters, digits, underscores)")
    if not value or len(value) > MAX_VALUE or any(c in value for c in "\r\n\x00"):
        raise ValueError("the value must be one line, at most 4096 characters")
    with _LOCK:
        keys = saved_keys()
        keys[name] = value
        _write_keys(keys)


def delete_key(name: str) -> None:
    with _LOCK:
        keys = saved_keys()
        if name not in keys:
            raise KeyError(name)
        del keys[name]
        _write_keys(keys)


def hint(value: str) -> str:
    """Enough of a key to recognise it, never enough to use it."""
    return "…" + value[-4:] if len(value) >= 12 else "저장됨"
