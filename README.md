# meeting-minutes

녹음 파일 하나로 **화자별 녹취록**과 **회의록**을 만드는 로컬 도구입니다. [English](README.en.md)

- 음성 인식, 단어 정렬, 화자 분리는 전부 내 PC에서 돕니다 (WhisperX + pyannote).
- 회의록, 녹취록 보정, 대화 분석에는 LLM을 씁니다. Claude, OpenAI, Gemini, OpenRouter, 그리고 Ollama·vLLM·사내 게이트웨이 같은 OpenAI 호환 서버면 무엇이든 연결할 수 있습니다. Ollama를 쓰면 처음부터 끝까지 외부로 나가는 데이터가 없습니다.
- 브라우저 화면에서 업로드하고, 대화 흐름(주제 구간, 화자 타임라인, 핵심 발화)과 회의록을 봅니다. 발화를 누르면 그 시점부터 원본 음성이 재생됩니다.

```
audio ─▶ faster-whisper (ASR, 반복 인식 복구) ─▶ wav2vec2 단어 정렬 ─▶ pyannote 화자 분리
      ─▶ 화자 턴 transcript.json
      ─▶ LLM: 녹취록 보정 ─▶ 회의록 minutes.md + 대화 분석 analysis.json
```

구조와 설계 이유는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)에 정리했습니다.

## 설치

필요한 것: Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg

```bash
brew install ffmpeg            # Ubuntu: sudo apt install ffmpeg
git clone https://github.com/seongyeon1/meeting-minutes.git
cd meeting-minutes
uv sync
```

화자 분리 모델은 Hugging Face에서 사용 동의가 필요합니다.

1. https://huggingface.co/pyannote/speaker-diarization-community-1 에서 동의
2. `hf auth login` 하거나 `HF_TOKEN` 환경변수 설정

모델 가중치(Whisper, wav2vec2, pyannote)는 처음 실행할 때 한 번 내려받습니다.

## LLM 연결

`--llm` 에 별칭이나 `제공자:모델` 을 넘깁니다.

| 값 | 연결 대상 | 필요한 키 |
|---|---|---|
| `claude` (기본) | `anthropic:claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| `gpt` | `openai:gpt-5.2` | `OPENAI_API_KEY` |
| `anthropic:<모델>` | Anthropic API | `ANTHROPIC_API_KEY` |
| `openai:<모델>` | OpenAI (`OPENAI_BASE_URL` 로 Azure·프록시) | `OPENAI_API_KEY` |
| `gemini:<모델>` | Gemini OpenAI 호환 엔드포인트 | `GEMINI_API_KEY` |
| `openrouter:<모델>` | OpenRouter | `OPENROUTER_API_KEY` |
| `ollama:<모델>` | 로컬 Ollama (`OLLAMA_BASE_URL`, 기본 `http://localhost:11434/v1`) | 없음 |
| `compat:<모델>` | 그 밖의 OpenAI 호환 서버 (`LLM_BASE_URL`) | `LLM_API_KEY` (선택) |

키는 환경변수나 `--env-file .env` 로만 읽습니다. [.env.example](.env.example)을 복사해서 쓰세요.

자주 쓰는 모델에 이름을 붙이거나 기본값을 바꾸려면 `llm.toml` 을 만듭니다 ([llm.example.toml](llm.example.toml)). 사내 게이트웨이처럼 주소와 키 이름이 정해진 엔드포인트를 코드 수정 없이 붙일 때 씁니다. `llm.toml` 은 gitignore 되어 있습니다.

```toml
default = "local"
analysis_default = "smart"

[models.local]
provider = "ollama"
model = "qwen3:14b"

[models.smart]
provider = "openai"
model = "gpt-5.2"
```

**모델 고르기**: 녹취록 보정과 화자 이름 추정은 "근거 없으면 고치지 말 것", "발화 속 제3자를 화자로 착각하지 말 것" 같은 규칙을 지켜야 해서, 추론형 대형 모델(Claude, GPT-5 급)에서 결과가 가장 좋았습니다. 작은 모델은 없는 화자 라벨을 만들거나 담당자를 잘못 붙이는 경우가 있었습니다. 회의록만 작은 모델로 쓰고 분석은 큰 모델로 나누는 구성(`default` / `analysis_default`)을 권장합니다.

## 웹 화면

```bash
uv run meeting-minutes-web --env-file .env
# http://127.0.0.1:8765
```

녹음 파일을 올리고 화자 수, 참석자 이름 후보, 용어, 모델을 고르면 진행 단계(음성 인식 → 단어 정렬 → 화자 분리 → 녹취록 보정 → 회의록 · 대화 분석)가 실시간으로 보입니다.

- **녹취록 보정**: 회의록을 쓰기 전에 LLM이 앞뒤 맥락, 이름 후보, 용어로 인식 오류만 고칩니다. 화자와 시각은 그대로 두고, 요약이나 추가로 보이는 큰 수정은 코드에서 거절합니다. 턴마다 원문과 단어 단위로 비교할 수 있습니다.
- **대화 흐름**: 주제 구간별 대화, 화자별 타임라인, 화자·검색·핵심 발화 필터, 결정/할 일/질문/이슈 표시.
- **화자 카드**: 발화 비중, 이름 후보별 확률과 근거 발화. 이름을 확정하면 화면 전체와 이후 회의록에 반영됩니다.
- **회의록**: 모델이나 이름 후보를 바꿔 다시 만들 수 있습니다.

> [!WARNING]
> 웹 서버에는 인증이 없습니다. 기본값처럼 `127.0.0.1` 에만 열어 두고, `--host 0.0.0.0` 으로 다른 사람에게 열지 마세요.

산출물은 `out/web/<작업 id>/` 에 쌓이고, 서버를 다시 켜면 이 폴더에서 작업 목록을 다시 읽습니다.

## CLI

```bash
# 기본
uv run meeting-minutes meeting.m4a --env-file .env

# 화자 수, 참석자, 용어를 알려주면 정확도가 오릅니다
uv run meeting-minutes meeting.m4a --speakers 3 \
  --names "김팀장,이선임,박매니저" \
  --vocab "QA,결제 모듈" \
  --context "11월 출시 일정 점검 회의" \
  --correct --analyze

# 완전 로컬
uv run meeting-minutes meeting.m4a --llm ollama:qwen3:14b --analysis-llm ollama:qwen3:14b

# 녹취록만 (LLM 호출 없음)
uv run meeting-minutes meeting.m4a --no-summary

# STT 없이 회의록만 다시 생성 (transcript.json 을 손으로 고친 뒤에도)
uv run meeting-minutes out/meeting/transcript.json --llm gpt
```

| 옵션 | 설명 |
|---|---|
| `--speakers N` | 화자 수를 알면 지정. 화자 분리 정확도에 가장 큰 영향 |
| `--min-speakers` / `--max-speakers` | 범위만 아는 경우 |
| `--vocab` | 고유명사·약어. Whisper initial prompt와 LLM 단계 둘 다에 들어감 |
| `--names` | 화자 라벨을 이름으로 매핑할 때 쓰는 후보. 근거가 약하면 매핑하지 않음 |
| `--context` | 회의 배경 한두 줄 |
| `--llm` | 회의록 모델 (위 표) |
| `--analysis-llm` | `--correct`, `--analyze` 에 쓸 모델 |
| `--llm-config` | `llm.toml` 경로 (기본: 현재 폴더) |
| `--env-file` | API 키가 든 env 파일. 환경변수가 우선 |
| `--correct` | 회의록 전에 녹취록 보정, `transcript_corrected.{json,md}` 저장 |
| `--analyze` | 화자 이름 확률, 주제 구간, 핵심 발화를 `analysis.json` 으로 저장 |
| `--asr-model` | 기본 `large-v3-turbo`. 정확도 우선이면 `large-v3` (약 2배 느림) |
| `--language` | 기본 `ko`. `auto` 면 자동 감지 |
| `--diarize-device mps` | pyannote를 Apple GPU로 (미측정, 불안정하면 기본값 `cpu`) |

## 데이터가 어디로 가나요

- 음성 파일은 PC 밖으로 나가지 않습니다. 모델 가중치만 처음에 Hugging Face에서 받습니다.
- 고른 LLM 제공자에게는 **녹취록 텍스트**가 전송됩니다. 민감한 회의라면 `ollama:` 나 사내 서버를 쓰거나 `--no-summary` 로 녹취록만 만드세요.
- 웹 화면은 marked, DOMPurify를 cdnjs에서 불러옵니다 (회의 데이터는 보내지 않음).

## 성능 (Apple M4 Pro, CPU)

| 입력 | 음성 인식 | 단어 정렬 | 화자 분리 |
|---|---|---|---|
| 7분 회의, 3명 | 108s | 24s | 315s |
| 24분 회의, 5명 | 493s | 68s | 1063s |

화자 분리가 병목이고 음성 길이의 약 0.7배가 걸립니다. LLM 단계(보정, 분석, 회의록)는 모델에 따라 수십 초입니다. CTranslate2에 MPS 백엔드가 없어서 음성 인식은 Apple Silicon에서도 CPU로 돕니다.

## 개발

```bash
uv run --group dev pytest       # 모델·네트워크 없이 도는 테스트
uvx ruff check .
```

## 라이선스

[MIT](LICENSE). 사용하는 모델은 각자의 라이선스를 따릅니다: [WhisperX](https://github.com/m-bain/whisperX) (BSD-2-Clause), [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT), [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) (CC-BY-4.0, 사용 동의 필요), [wav2vec2-large-xlsr-korean](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean) (Apache-2.0).
