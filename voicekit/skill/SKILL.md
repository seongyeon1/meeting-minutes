---
name: voicekit
description: Turn recordings (meetings, interviews, support calls, lectures, podcasts, voice memos) into speaker-labelled transcripts and structured documents. Speech recognition, word alignment and speaker diarization run locally; use when the user gives an audio or video file, or asks for minutes, notes or a summary of a recording.
---

# voicekit: local speech pipeline

Audio never leaves this machine. Only transcript text reaches a model, and when you write the document yourself
that model is you.

## Pick the path

- **MCP tools available** (`transcribe_audio`, `get_document_prompt`, ...): use them. See "With MCP".
- **No MCP**: use the `voicekit` CLI. See "With the CLI".

## Choose a profile first

Run `list_profiles` (or `voicekit profiles`). Match the recording, do not default to `meeting`:

| id | use for | speakers separated |
|---|---|---|
| meeting | several people deciding things | yes |
| interview | interviewer and interviewee | yes |
| call | support or sales call | yes |
| podcast | host and guests | yes |
| lecture | one speaker presenting | no (faster) |
| memo | someone talking to themselves | no (faster) |
| general | type unknown | yes |

If the type is genuinely unclear from the file name and the user's words, ask once. Also ask for what raises accuracy:
speaker count (`speakers`), names to look for (`names`), and domain terms (`vocab`, e.g. product names, acronyms).

## With MCP

1. `transcribe_audio(path, profile, language, speakers, vocab)` returns a `job_id` at once.
   Local ASR is slow on CPU: roughly the length of the recording for a meeting, much less without diarization.
   Tell the user the expected wait.
2. `wait_for_job(job_id)` repeatedly until `status` is `done` (or `error`: show `error` verbatim).
3. `read_transcript(job_id)`, following `next_offset` until it is null.
4. `get_document_prompt(job_id, names=..., context=...)` returns `system` and `user`.
   Write the document from them yourself, following `system` exactly, then `save_document(job_id, markdown)`.
5. Show the user the document and mention the speaker labels are automatic.

Alternative to step 4: `generate_document(job_id, llm="claude-code" | "codex" | "<alias>")` has another model write it.

## With the CLI

```bash
voicekit transcribe meeting.m4a --profile meeting --speakers 3 --vocab "QA,결제 모듈"   # prints JSON with job dir
voicekit prompt <job>  --profile meeting --names "김팀장,이선임"                          # prints system + user prompt
voicekit save-document <job> < document.md
voicekit handoff <job> --agent claude        # or codex: that CLI writes the document
```

`<job>` is the id or the job directory. Add `--json` to `prompt` for machine-readable output.

## Rules for the document

- Speaker labels (`SPEAKER_00`...) come from audio analysis and can be wrong for a word or two at turn edges.
  Name a speaker only when the transcript supports it (they say their name, or are addressed and answer). Otherwise
  keep the label or write 미상, and ask the user. Confirm with `set_speaker_names`.
- Speech recognition errors exist, especially names, numbers and jargon. Fix only what the context makes obvious.
- Never invent decisions, owners or dates the recording does not contain.
- Calls may contain personal data (phone, ID, card numbers). Do not copy them into the document.

## Troubleshooting

- `pyannote ... 사용 동의가 필요합니다`: the user must accept the model terms at
  huggingface.co/pyannote/speaker-diarization-community-1 and set `HF_TOKEN` (or `hf auth login`). Or retry with `diarize=false`.
- `alignment skipped`: that language has no word-alignment model; speaker boundaries fall back to sentence timing.
- Repeated nonsense in the transcript ("Q. Q. Q."): the pipeline already re-decodes such segments; remaining
  `[인식 불가]` marks are unreadable audio.
