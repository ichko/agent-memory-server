from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

import pytest

from agent_memory_benchmark.datasets.longmemeval import LongMemEvalAdapter
from agent_memory_benchmark.datasets.longmemeval_v2 import LongMemEvalV2Adapter
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


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_longmemeval_v2_groups_shared_haystacks_and_converts_local_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "haystacks").mkdir()
    _write_jsonl(
        tmp_path / "questions.jsonl",
        [
            {
                "id": "q2",
                "domain": "web",
                "question_type": "dynamic-environment-abs",
                "question": "Missing?",
                "answer": "",
                "eval_function": "judge",
            },
            {
                "id": "q1",
                "domain": "web",
                "question_type": "procedure",
                "question": "How?",
                "answer": "Click save",
            },
            {
                "id": "enterprise-only",
                "domain": "enterprise",
                "question": "Ignored",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "trajectories.jsonl",
        [
            {
                "id": "t1",
                "goal": "Save a record",
                "start_url": "https://example.test",
                "states": [
                    {
                        "step": 1,
                        "url": "https://example.test/edit",
                        "thought": "Need save",
                        "action": "click(7)",
                        "accessibility_tree": "first line\nsecond line",
                    }
                ],
            },
            {"id": "unused", "goal": "Not loaded", "states": []},
        ],
    )
    (tmp_path / "haystacks" / "lme_v2_small.json").write_text(
        json.dumps({"q1": ["t1"], "q2": ["t1"], "enterprise-only": ["unused"]}),
        encoding="utf-8",
    )

    adapter = LongMemEvalV2Adapter(
        root=tmp_path,
        axtree_part_chars=12,
        max_axtree_chars=20,
    )
    groups = adapter.haystack_groups()
    examples = adapter.load()

    assert len(groups) == 1
    assert groups[0].trajectory_ids == ("t1",)
    assert {row["id"] for row in groups[0].questions} == {"q1", "q2"}
    assert len(examples) == 1
    assert examples[0].metadata["trajectory_ids"] == ["t1"]
    assert {qa.question_id for qa in examples[0].qa_pairs} == {"q1", "q2"}
    abstention = next(qa for qa in examples[0].qa_pairs if qa.question_id == "q2")
    assert abstention.metadata["category"] == "dynamic-abs"
    assert abstention.metadata["is_abstention"] is True

    session = examples[0].sessions[0]
    assert session.label.startswith("Trajectory t1")
    assert session.messages[0].speaker == "user"
    assert "Task goal: Save a record" in session.messages[0].text
    assert any(message.speaker == "environment" for message in session.messages)
    assert adapter.sessions_for(("t1",))[0] is session


def test_longmemeval_v2_reports_missing_local_files(tmp_path: Path) -> None:
    adapter = LongMemEvalV2Adapter(root=tmp_path)
    with pytest.raises(FileNotFoundError, match="LME_V2_DATA_ROOT"):
        adapter.load()
