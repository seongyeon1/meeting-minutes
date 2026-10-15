# meeting-minutes

Turn a meeting recording into a **speaker-attributed transcript** and **meeting minutes**. [한국어](README.md)

- Speech recognition, word alignment and speaker diarization run on your machine (WhisperX + pyannote).
- Minutes, transcript correction and conversation analysis use an LLM of your choice: Claude, OpenAI, Gemini, OpenRouter, or any OpenAI-compatible server (Ollama, vLLM, a company gateway). With Ollama nothing leaves your machine.
- A local web UI shows the conversation by topic, a per-speaker timeline, key turns and the minutes. Click a turn to play the original audio from there.

Prompts and the UI are in Korean and the default language is `ko`; other languages work with `--language`, but the minutes template will still be Korean.

```
audio ─▶ faster-whisper (ASR, loop repair) ─▶ wav2vec2 alignment ─▶ pyannote diarization
      ─▶ speaker turns transcript.json
      ─▶ LLM: transcript correction ─▶ minutes.md + analysis.json
```

## Install

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
brew install ffmpeg            # Ubuntu: sudo apt install ffmpeg
git clone https://github.com/seongyeon1/meeting-minutes.git
cd meeting-minutes
uv sync
```

Accept the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), then `hf auth login` or set `HF_TOKEN`.

## LLMs

Pass an alias or `provider:model` to `--llm`:

| Value | Target | Key |
|---|---|---|
| `claude` (default) | `anthropic:claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| `gpt` | `openai:gpt-5.2` | `OPENAI_API_KEY` |
| `openai:<model>` | OpenAI (`OPENAI_BASE_URL` for Azure or a proxy) | `OPENAI_API_KEY` |
| `gemini:<model>` | Gemini's OpenAI-compatible endpoint | `GEMINI_API_KEY` |
| `openrouter:<model>` | OpenRouter | `OPENROUTER_API_KEY` |
| `ollama:<model>` | local Ollama (`OLLAMA_BASE_URL`) | none |
| `compat:<model>` | any OpenAI-compatible server (`LLM_BASE_URL`) | `LLM_API_KEY` (optional) |

Keys are read from the environment or `--env-file .env` (see [.env.example](.env.example)). To name your own models or change the defaults, create `llm.toml` from [llm.example.toml](llm.example.toml).

Correction and speaker-name inference need a model that follows "don't guess" rules; large reasoning models (Claude, GPT-5 class) did best. A split such as a small model for minutes and a large one for analysis (`default` / `analysis_default`) works well.

## Use

```bash
uv run meeting-minutes-web --env-file .env      # http://127.0.0.1:8765
uv run meeting-minutes meeting.m4a --speakers 3 --names "Alice,Bob" --correct --analyze
uv run meeting-minutes meeting.m4a --llm ollama:qwen3:14b --analysis-llm ollama:qwen3:14b
```

> [!WARNING]
> The web server has no authentication. Keep it on `127.0.0.1`.

## Privacy

Audio never leaves the machine. The transcript **text** is sent to the LLM provider you pick. Use `ollama:` or `--no-summary` for sensitive meetings.

## License

[MIT](LICENSE). Models keep their own licenses (WhisperX BSD-2-Clause, faster-whisper MIT, pyannote community-1 CC-BY-4.0 with gated access, wav2vec2-large-xlsr-korean Apache-2.0).
