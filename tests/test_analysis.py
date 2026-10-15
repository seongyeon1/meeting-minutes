"""LLM post-processing: analysis normalisation, transcript correction and repetition-loop handling."""

from __future__ import annotations

import json
from typing import Any

import pytest

import meeting_minutes as mm

TURNS = [
    mm.Turn("SPEAKER_00", 0.0, 5.0, "진행 맡은 김팀장입니다"),
    mm.Turn("SPEAKER_01", 5.5, 9.0, "이선임입니다 결제 모듈 보고드립니다"),
    mm.Turn("SPEAKER_00", 9.5, 15.0, "출시일은 20일로 확정하죠"),
    mm.Turn("SPEAKER_01", 15.5, 18.0, "네 제가 공유하겠습니다"),
]


def test_drops_invented_labels_and_bad_turns() -> None:
    raw = {
        "speakers": [
            {"label": "SPEAKER_00", "role": "진행자", "candidates": [
                {"name": "김팀장", "prob": 0.6, "evidence": [{"turn": 0, "reason": "자기소개"}, {"turn": 99}]},
                {"name": "미상", "prob": 0.2},
            ]},
            {"label": "SPEAKER_03", "candidates": [{"name": "박매니저", "prob": 1.0}]},
        ],
        "topics": [],
        "key_turns": [{"turn": 2, "type": "decision", "note": "출시일"}, {"turn": 42, "type": "action"},
                      {"turn": 3, "type": "gossip"}],
    }
    out = mm.normalize_analysis(raw, TURNS)

    assert [s["label"] for s in out["speakers"]] == ["SPEAKER_00", "SPEAKER_01"]
    sp0 = out["speakers"][0]
    assert sp0["candidates"][0] == {"name": "김팀장", "prob": 0.75, "evidence": [{"turn": 0, "reason": "자기소개"}]}
    assert sp0["candidates"][1]["prob"] == 0.25  # renormalised to sum to 1
    assert out["speakers"][1]["candidates"] == [{"name": "미상", "prob": 1.0, "evidence": []}]  # missing label filled
    assert out["key_turns"] == [{"turn": 2, "type": "decision", "note": "출시일"}]
    assert out["topics"] == [{"start_turn": 0, "end_turn": 3, "start": 0.0, "end": 18.0, "title": "전체 대화", "summary": ""}]


def test_topics_cover_all_turns_in_order() -> None:
    raw = {"topics": [
        {"start_turn": 2, "title": "결정", "summary": "s2"},
        {"start_turn": 1, "title": "보고", "summary": "s1"},
        {"start_turn": 1, "title": "중복", "summary": ""},
        {"start_turn": "x", "title": "깨짐"},
    ]}
    topics = mm.normalize_analysis(raw, TURNS)["topics"]

    # The earliest topic is stretched back to #0 so the opening turns are not orphaned.
    assert [(t["start_turn"], t["end_turn"], t["title"]) for t in topics] == [(0, 1, "보고"), (2, 3, "결정")]
    assert topics[1]["start"] == 9.5 and topics[1]["end"] == 18.0


def test_extract_json_tolerates_fences() -> None:
    assert mm._extract_json('```json\n{"speakers": []}\n```') == {"speakers": []}


def test_confirmed_names_reach_minutes_prompt() -> None:
    prompt = mm.build_user_prompt(
        TURNS, names=None, context=None, vocab=None, speaker_map={"SPEAKER_01": "이선임", "SPEAKER_09": "유령"}
    )
    assert "사용자가 확인한 화자 이름 (근거 없이 그대로 사용): SPEAKER_01=이선임" in prompt
    assert "유령" not in prompt  # labels not in the transcript are ignored


def test_analysis_prompt_numbers_turns() -> None:
    prompt = mm.build_analysis_prompt(TURNS, names="김팀장,이선임", context=None)
    assert "#0 [00:00:00] SPEAKER_00: 진행 맡은 김팀장입니다" in prompt
    assert "#3 [00:00:15] SPEAKER_01:" in prompt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("이번 주 진행할 내용은 Q. Q. Q. Q. Q. Q Q Q 그 부분을", "이번 주 진행할 내용은 [인식 불가] 그 부분을"),
        ("마무리할게요 지금 보고 있나요? 지금 보고 있나요? 지금 보고 있나요? 지금 보고 있나요?", "마무리할게요 지금 보고 있나요?"),
        ("네 네 네 네 알겠습니다", "네 네 네 네 알겠습니다"),  # four short replies is real speech
        ("그래서 그래서 이제 그거를", "그래서 그래서 이제 그거를"),
        ("적을 때는 중독독독독독독독독독", "적을 때는 중독[인식 불가]"),
    ],
)
def test_collapse_repeats(text: str, expected: str) -> None:
    assert mm.collapse_repeats(text) == expected
    assert mm.is_repetition_loop(text) == (text != expected)


def test_correction_keeps_speakers_and_rejects_rewrites(monkeypatch: pytest.MonkeyPatch) -> None:
    turns = [*TURNS, mm.Turn("SPEAKER_00", 18.5, 30.0, "다음 안건은 카 범위 Q Q Q Q Q Q 입니다")]
    replies = {
        "#0": {"turns": [{"turn": 1, "text": "이선임입니다. 결제 모듈 보고드립니다"},
                          {"turn": 2, "text": "확정"},                       # summary-sized rewrite: rejected
                          {"turn": 4, "text": "다음 안건은 QA 범위 입니다"},
                          {"turn": 9, "text": "범위 밖"}]},                  # not a target turn: ignored
    }
    prompts: list[str] = []

    def fake_complete(user: str, **kw: Any) -> str:
        prompts.append(user)
        return json.dumps(replies["#0"], ensure_ascii=False)

    monkeypatch.setattr(mm, "_complete", fake_complete)
    out = mm.correct_transcript(turns, llm="gpt", names=None, context=None, vocab="QA", env_file=None)

    assert "Q Q Q" not in prompts[0]  # loops are collapsed before the model sees them
    assert "용어 목록: QA" in prompts[0]
    assert [t["speaker"] for t in out["turns"]] == [t.speaker for t in turns]
    assert [t["start"] for t in out["turns"]] == [t.start for t in turns]
    texts = [t["text"] for t in out["turns"]]
    assert texts[1] == "이선임입니다. 결제 모듈 보고드립니다"
    assert texts[2] == TURNS[2].text
    assert texts[4] == "다음 안건은 QA 범위 입니다"
    by_turn = {c["turn"]: c for c in out["changes"]}
    assert by_turn[2]["rejected"] == "확정" and by_turn[2]["corrected"] == TURNS[2].text
    assert by_turn[4]["original"] == "다음 안건은 카 범위 Q Q Q Q Q Q 입니다"
    assert set(by_turn) == {1, 2, 4}


def test_correction_chunks_cover_every_turn() -> None:
    turns = [mm.Turn("S", float(i), float(i) + 1, "가" * 900) for i in range(10)]
    chunks = mm._correction_chunks(turns)
    covered = [i for lo, hi in chunks for i in range(lo, hi + 1)]
    assert covered == list(range(10)) and len(chunks) > 1
    prompt = mm.build_correction_prompt(turns, *chunks[1], names=None, context=None, vocab=None)
    assert prompt.count("<context>") == 2  # neighbours on both sides for a middle chunk


class _Piece:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisper:
    """Stands in for whisperx's pipeline; ``model.model.transcribe`` is the non-batched decoder."""

    def __init__(self, text: str) -> None:
        self.model, self.text, self.calls = self, text, 0

    def transcribe(self, clip: Any, **kw: Any) -> tuple[list[_Piece], None]:
        self.calls += 1
        return [_Piece(t) for t in self.text.split("|")], None


def _repair(segments: list[dict[str, Any]], redo: str) -> tuple[list[dict[str, Any]], _FakeWhisper, list[str]]:
    model, logs = _FakeWhisper(redo), []
    out = mm._repair_segments(model, [0.0] * 16000 * 60, segments, language="ko", vocab=None, log=logs.append)
    return out, model, logs


def test_skipped_speech_is_redecoded() -> None:
    # 18 s window that came back with only its last sentence (~1.7 chars/s).
    seg = {"start": 36.8, "end": 55.0, "text": "그럼 출시일은 어떻게 할까요? 원래 계획은 다음 달 20일이었는데요."}
    full = "박지훈입니다.|QA 범위는 대여, 반납, 연장 세 가지로 잡았습니다.|" + seg["text"]
    out, model, logs = _repair([seg], full)
    assert model.calls == 1 and out[0]["text"].startswith("박지훈입니다.")
    assert "sparse segment at 00:00:36" in logs[0]


def test_sparse_retry_that_recovers_nothing_keeps_original() -> None:
    seg = {"start": 0.0, "end": 20.0, "text": "네 그럼 다음으로 넘어가죠"}
    out, _, logs = _repair([seg], "네")
    assert out == [seg] and "kept original" in logs[0]


def test_normal_and_short_segments_are_left_alone() -> None:
    segs = [
        {"start": 0.0, "end": 10.0, "text": "대여하기와 반납하기 화면은 지난주에 개발을 마쳤고 지금은 알림 기능을 붙이고 있습니다"},
        {"start": 10.0, "end": 14.0, "text": "네"},  # short reply in a short window
    ]
    out, model, _ = _repair(segs, "unused")
    assert out == segs and model.calls == 0
