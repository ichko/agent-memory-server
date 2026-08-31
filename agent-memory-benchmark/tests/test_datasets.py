from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import pytest

from agent_memory_benchmark.datasets.longmemeval import LongMemEvalAdapter
from agent_memory_benchmark.prompts.judge import BY_TYPE


def test_v1_judge_covers_the_six_published_task_types() -> None:
    assert set(BY_TYPE) == {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "single-session-preference",
    }


def test_longmemeval_v1_converts_cached_fixture_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_file = tmp_path / "longmemeval-v1" / "longmemeval_oracle.json"
    cache_file.parent.mkdir()
    cache_file.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-1",
                    "question_type": "temporal-reasoning",
                    "question": "When?",
                    "answer": "Tuesday",
                    "question_date": "2025/01/03",
                    "answer_session_ids": [1],
                    "haystack_dates": ["2025/01/02 (Thu) 09:30"],
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "Meeting Tuesday 😀"},
                            {"role": "assistant", "content": "Noted"},
                            {"ignored": "malformed"},
                        ]
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network must not be used for cached data")

    monkeypatch.setattr(
        "agent_memory_benchmark.datasets.longmemeval.httpx.get", fail_network
    )
    adapter = LongMemEvalAdapter(cache_dir=tmp_path)
    examples = adapter.load()

    assert adapter.load() is examples
    assert len(examples) == 1
    example = examples[0]
    assert example.qa_pairs[0].question_id == "q-1"
    assert example.qa_pairs[0].question_type == "temporal-reasoning"
    assert example.metadata["answer_session_ids"] == [1]
    assert [message.text for message in example.sessions[0].messages] == [
        "Meeting Tuesday ",
        "Noted",
    ]
    assert example.sessions[0].date is not None
    assert example.sessions[0].date.tzinfo == timezone.utc
    assert example.sessions[0].date.hour == 9


def test_longmemeval_v1_download_follows_huggingface_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        content = (
            b'[{"question_id":"q","question":"Q","answer":"A",'
            b'"haystack_sessions":[],"haystack_dates":[]}]'
        )

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(
        "agent_memory_benchmark.datasets.longmemeval.httpx.get", fake_get
    )
    adapter = LongMemEvalAdapter(split="oracle", cache_dir=tmp_path)
    rows = adapter.load_raw()

    assert captured["follow_redirects"] is True
    assert "longmemeval_oracle.json" in str(captured["url"])
    assert len(rows) == 1
