"""``voicekit``: run the local speech pipeline from a shell, hand the document to an agent, or serve MCP.

    voicekit profiles
    voicekit transcribe meeting.m4a --profile interview --speakers 2
    voicekit prompt <job>                      # the prompt an agent should follow to write the document
    voicekit save-document <job> < doc.md
    voicekit handoff <job> --agent codex       # let a local Claude Code / Codex CLI write it
    voicekit install claude --apply            # register the MCP server and the skill
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _emit(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _service(job: Optional[str] = None, root: Optional[Path] = None) -> tuple[Any, Optional[str]]:
    """A Service rooted where ``job`` lives. ``job`` may be an id or the job directory itself."""
    from voicekit.service import Service

    if job and Path(job).is_dir():
        path = Path(job).resolve()
        return Service(path.parent), path.name
    return Service(root), job


def cmd_profiles(args: argparse.Namespace) -> int:
    from voicekit.service import Service

    profiles = Service.profiles()
    if args.json:
        _emit(profiles)
        return 0
    for p in profiles:
        print(f"{p['id']:<10} {p['label']}: {p['description']} -> {p['document']}" + ("" if p["diarize"] else " [단일 화자]"))
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    svc, _ = _service(root=args.out_dir)
    job = svc.start_transcription(
        args.audio, profile=args.profile, language=args.language, speakers=args.speakers,
        diarize=False if args.no_diarize else None, vocab=args.vocab, names=args.names, context=args.context,
    )
    while True:  # stage lines already go to stderr as they happen
        state = svc.status(job["id"])
        if state["status"] in ("done", "error"):
            break
        time.sleep(1.0)
    if state["status"] == "error":
        print(f"failed: {state['error']}", file=sys.stderr)
        return 1
    out = svc.job(job["id"]).out_dir
    _emit({
        "job_id": job["id"], "job_dir": str(out), "profile": state["profile"], "speakers": state["speakers"],
        "transcript_md": str(out / "transcript.md"), "transcript_json": str(out / "transcript.json"),
    })
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    svc, job_id = _service(args.job)
    data = svc.document_prompt(job_id, profile=args.profile, names=args.names, context=args.context)
    if args.json:
        _emit(data)
    else:
        print(f"# SYSTEM\n\n{data['system']}\n\n# USER\n\n{data['user']}")
    return 0


def cmd_save_document(args: argparse.Namespace) -> int:
    svc, job_id = _service(args.job)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    _emit(svc.save_document(job_id, text))
    return 0


def cmd_handoff(args: argparse.Namespace) -> int:
    svc, job_id = _service(args.job)
    _emit(svc.generate_document(
        job_id, llm=f"cli:{args.agent}", profile=args.profile, names=args.names, context=args.context
    ))
    return 0


def cmd_mcp(_args: argparse.Namespace) -> int:
    from voicekit.mcp_server import main as serve

    serve()
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    """Register the MCP server and the skill / instructions with Claude Code or Codex."""
    from voicekit import manage

    home = Path(args.home).expanduser().resolve() if args.home else None
    if not args.apply:
        print("dry run; add --apply to do this:\n  " + "\n  ".join(manage.install_steps(args.target, home)))
        return 0
    result = manage.install_agent(args.target, home)
    print("\n".join(result["output"]))
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voicekit", description="로컬 음성 인식·단어 정렬·화자분리 도구 (에이전트 연동 지원)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("profiles", help="녹음 유형 목록")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_profiles)

    s = sub.add_parser("transcribe", help="로컬에서 녹취록 만들기 (LLM 호출 없음)")
    s.add_argument("audio")
    s.add_argument("--profile", default="meeting")
    s.add_argument("--language", default="ko", help="언어 코드, auto 면 자동 감지")
    s.add_argument("--speakers", type=int)
    s.add_argument("--no-diarize", action="store_true", help="화자분리 생략 (한 명이 말하는 녹음)")
    s.add_argument("--vocab")
    s.add_argument("--names")
    s.add_argument("--context")
    s.add_argument("-o", "--out-dir", type=Path, help="작업 폴더 루트 (기본: $VOICEKIT_HOME 또는 ~/.voicekit/jobs)")
    s.set_defaults(fn=cmd_transcribe)

    for name, fn, help_ in (
        ("prompt", cmd_prompt, "문서 작성용 프롬프트 출력 (에이전트가 직접 작성할 때)"),
        ("handoff", cmd_handoff, "로컬 Claude Code / Codex CLI 에 넘겨 문서 작성"),
    ):
        s = sub.add_parser(name, help=help_)
        s.add_argument("job", help="작업 id 또는 작업 폴더")
        s.add_argument("--profile")
        s.add_argument("--names")
        s.add_argument("--context")
        if name == "prompt":
            s.add_argument("--json", action="store_true")
        else:
            s.add_argument("--agent", choices=["claude", "codex"], default="claude")
        s.set_defaults(fn=fn)

    s = sub.add_parser("save-document", help="작성한 문서를 작업에 저장 (stdin 또는 --file)")
    s.add_argument("job")
    s.add_argument("--file")
    s.set_defaults(fn=cmd_save_document)

    s = sub.add_parser("mcp", help="MCP 서버 실행 (stdio)")
    s.set_defaults(fn=cmd_mcp)

    s = sub.add_parser("install", help="Claude Code / Codex 에 MCP 서버와 스킬 등록")
    s.add_argument("target", choices=["claude", "codex"])
    s.add_argument("--apply", action="store_true", help="실제로 실행 (기본은 할 일만 출력)")
    s.add_argument("--home", help="에이전트가 쓸 작업 폴더 (웹 화면의 --out-dir 와 같게 하면 작업을 공유)")
    s.set_defaults(fn=cmd_install)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
