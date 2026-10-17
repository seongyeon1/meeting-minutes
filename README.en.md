# meeting-minutes

Turn any recording into a **speaker-attributed transcript** and a **structured document** (meeting minutes, interview notes, call summary, lecture notes...). [한국어](README.md)

![Conversation view](docs/images/ui-conversation.png)

## Highlights

- **Audio never leaves your machine.** Speech recognition, word alignment and speaker diarization run locally with no LLM (WhisperX + pyannote).
- **An LLM only writes the document.** Minutes, transcript correction and conversation analysis use Claude, OpenAI, Gemini, OpenRouter, or any OpenAI-compatible server (Ollama, vLLM, a company gateway). With Ollama nothing leaves your machine.
- **Models and API keys are set in the UI.** No environment variables or file editing: add a model in "LLM 설정" and test the connection.
- **Many kinds of recordings.** Profiles (meeting, interview, call, lecture, podcast, voice memo, general) each have their own document template; add your own.
- **It improves as you use it.** Fix a misrecognized word and it is remembered per recording type, then used as a recognition hint and an automatic fix next time (local, no LLM).
- **Works with agents.** The local part is a skill and an MCP server: attach it to Claude Code or Codex and the agent transcribes and writes the document itself. You can also chat with an agent from the UI to refine the document.

Prompts and the UI are in Korean and the default language is `ko`; other languages work with `--language`, but the minutes template will still be Korean.

```
audio ─▶ faster-whisper (ASR, loop and skipped-speech repair) ─▶ wav2vec2 alignment ─▶ pyannote diarization
      ─▶ speaker turns transcript.json
      ─▶ LLM: transcript correction ─▶ minutes.md + analysis.json
```

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
brew install ffmpeg            # Ubuntu: sudo apt install ffmpeg
git clone https://github.com/seongyeon1/meeting-minutes.git
cd meeting-minutes
uv sync
uv run meeting-minutes-web     # http://127.0.0.1:8765
```

Accept the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), then `hf auth login` or set `HF_TOKEN`. Model weights are downloaded on first run.

Then open **LLM 설정** in the top bar, paste an API key (or pick `claude-code` / `codex`, which use the CLI you are already logged in to) and press the connection test. Upload or record audio with **+ 새 녹음**; a 7-minute meeting takes about 8 minutes, mostly diarization.

## Tour of the UI

> The `ui-*` screenshots show a made-up example meeting. "Conversation", "correction" and "minutes" are from a made-up three-person meeting (1:40) synthesized with macOS TTS and run through the real pipeline (2:31 end to end on an M4 Pro CPU).

**New recording**: pick a profile, drop a file or record from the microphone (level meter, preview), and give speaker count, name candidates, terms and context. "Transcript only" never calls an LLM.

<img src="docs/images/ui-new-recording.png" alt="New recording" width="380">

**Conversation view**: per-topic conversation, a speaker timeline, key turns, and speaker cards with name probabilities and evidence. Click a turn to play the audio from there.

![Conversation](docs/images/conversation.png)

**Fix a turn, remember a term**: "수정" on a turn teaches the memory a (wrong → right) pair for this recording type. After the same fix twice, or when pinned, the right spelling goes into Whisper's initial prompt and the correction and document prompts, and the wrong spelling is replaced as a whole word (a trailing Korean particle is kept) in new transcripts. One-character spellings are replaced only when pinned; a big rewrite teaches nothing. Local only; opt out per recording ("기억한 용어 사용"); files in `~/.config/voicekit/memory/<type>.json` (`VOICEKIT_MEMORY_DIR`); `VOICEKIT_MEMORY=off` ignores it all. Edits are logged in the job's `edits.json`, automatic replacements in `memory_applied.json`.

![Edit a turn](docs/images/ui-turn-edit.png)
<img src="docs/images/ui-memory.png" alt="Remembered terms" width="560">

**Transcript correction** and **minutes**:

![Correction](docs/images/correction.png)
![Minutes](docs/images/minutes.png)

**Chat with an agent** (the "에이전트" tab): talk to Claude Code or Codex on this PC about the job and keep asking for changes; earlier turns are kept in the job folder until "새 대화". It may use the voicekit tools only: no shell, files or web, one job only (no other recordings, no new transcription, no other models), one run at a time, cancellable. No global connection is needed.

![Agent chat](docs/images/ui-agent-chat.png)

**LLM settings**: choose the default models (document, and correction/analysis), save API keys, add models (gateway, vLLM, Ollama) and test each one. Keys go to `~/.config/voicekit/secrets.env` (mode 0600), are never sent back to the browser, and an exported variable or `--env-file` still wins. Models go to `~/.config/voicekit/llm.toml`, layered over the project's `llm.toml` (`VOICEKIT_CONFIG_DIR`).

<img src="docs/images/ui-llm-settings.png" alt="LLM settings" width="560">

**Connect agents** shows whether the Claude Code / Codex CLI, the MCP registration and the skill are in place and connects with a button after showing exactly what will change in your global config. The agent then shares this UI's job folder (`--out-dir`).

<img src="docs/images/ui-agents-connect.png" alt="Connect agents" width="560">

**Recording types**: browse, add, clone, edit and delete profiles (saved to `~/.config/voicekit/profiles/<id>.toml`).

<img src="docs/images/ui-profiles.png" alt="Recording types" width="560">

> [!WARNING]
> The web server has no authentication. Keep it on `127.0.0.1`. Changing requests (agent install, profile or key save, job delete) from another origin are refused.

## LLMs

Easiest is "LLM 설정" in the UI. From the CLI pass an alias or `provider:model` to `--llm`:

| Value | Target | Key |
|---|---|---|
| `claude` (default) | `anthropic:claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| `gpt` | `openai:gpt-5.2` | `OPENAI_API_KEY` |
| `claude-code`, `codex` | the CLI you are logged in to | none |
| `openai:<model>` | OpenAI (`OPENAI_BASE_URL` for Azure or a proxy) | `OPENAI_API_KEY` |
| `gemini:<model>` | Gemini's OpenAI-compatible endpoint | `GEMINI_API_KEY` |
| `openrouter:<model>` | OpenRouter | `OPENROUTER_API_KEY` |
| `ollama:<model>` | local Ollama (`OLLAMA_BASE_URL`) | none |
| `compat:<model>` | any OpenAI-compatible server (`LLM_BASE_URL`) | `LLM_API_KEY` (optional) |

Keys are read from the environment, then `--env-file .env` (see [.env.example](.env.example)), then the keys saved in the UI. To name your own models or change the defaults by file, create `llm.toml` from [llm.example.toml](llm.example.toml).

Correction and speaker-name inference need a model that follows "don't guess" rules; large reasoning models (Claude, GPT-5 class) did best. A split such as a small model for minutes and a large one for analysis (`default` / `analysis_default`) works well.

## CLI

```bash
uv run meeting-minutes meeting.m4a --speakers 3 --names "Alice,Bob" --correct --analyze
uv run meeting-minutes meeting.m4a --llm ollama:qwen3:14b --analysis-llm ollama:qwen3:14b
uv run meeting-minutes talk.m4a --profile lecture --no-diarize     # one speaker: skips the slowest stage
uv run meeting-minutes call.wav --profile call --llm claude-code   # a local Claude Code CLI writes the summary
uv run meeting-minutes meeting.m4a --no-summary                    # transcript only, no LLM
```

## Agents and MCP

```bash
uv sync --extra mcp
uv run voicekit install claude --apply     # skill + MCP server (dry run without --apply); also: codex
uv run voicekit transcribe meeting.m4a --profile interview         # CLI route, prints job dir (no LLM)
uv run voicekit prompt <job>               # prompt for an agent to follow; then: voicekit save-document <job> < doc.md
```

Profiles live in `voicekit/profiles/*.toml`; drop your own in `./profiles/`, `~/.config/voicekit/profiles/` or `$VOICEKIT_PROFILES`.
MCP tools: `list_profiles`, `transcribe_audio`, `wait_for_job`, `job_status`, `list_jobs`, `read_transcript`, `get_document_prompt`,
`save_document`, `generate_document`, `set_speaker_names`. The agent writes the document from `get_document_prompt` with its own model.
Agent CLIs run headless in an empty temp dir with tools disabled. Share jobs with the web UI via `--out-dir ~/.voicekit/jobs`.

## Privacy

Audio never leaves the machine. The transcript **text** is sent to the LLM provider you pick (with `claude-code` / `codex`, to whatever that CLI is logged in to; with MCP, to the attached agent's model). Use `ollama:` or `--no-summary` for sensitive meetings. Saved API keys stay in `~/.config/voicekit/secrets.env`; remembered terms and agent chats are plain files under `~/.config/voicekit/memory/` and the job folder.

## License

[MIT](LICENSE). Models keep their own licenses (WhisperX BSD-2-Clause, faster-whisper MIT, pyannote community-1 CC-BY-4.0 with gated access, wav2vec2-large-xlsr-korean Apache-2.0).
