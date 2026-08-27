from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from agent_memory_benchmark.benchmark import runner
from agent_memory_benchmark.benchmark.models import (
    AnswerRecord,
    ExperimentMetadata,
    append_jsonl,
    read_jsonl,
)
from agent_memory_benchmark.datasets.models import (
    ContextMessage,
    DatasetExample,
    QAPair,
    Session,
)
from agent_memory_benchmark.memory import QueryResult, TokenUsage


def _answer(**overrides: Any) -> AnswerRecord:
    values = {
        "example_idx": 2,
        "question_id": "q-2",
        "dataset": "LongMemEval",
        "split": "oracle",
        "provider": "fake",
        "question": "Where?",
        "ground_truth": "Home",
        "predicted_answer": "At home",
        "prompt": [{"role": "user", "content": "Where?"}],
        "question_type": "single-session-user",
        "metadata": {"source": "fixture"},
        "token_usage": {"total_tokens": 3},
    }
    values.update(overrides)
    return AnswerRecord(**values)


def test_result_jsonl_roundtrip_preserves_nested_metadata(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    record = _answer(metadata={"nested": {"values": [1, "two"]}})
    second = _answer(question_id="q-3", example_idx=3)

    append_jsonl(path, record)
    append_jsonl(path, second)
    loaded = read_jsonl(path, AnswerRecord)

    assert loaded == [record, second]
    assert json.loads(path.read_text().splitlines()[0])["metadata"]["nested"][
        "values"
    ] == [1, "two"]


@pytest.mark.parametrize(
    "contents",
    [
        '{"not": "closed"\n',
        '{"example_idx": 1, "unexpected": true}\n',
        "\n" + json.dumps({"example_idx": 1}) + "\n",
    ],
)
def test_result_jsonl_rejects_invalid_rows_with_line_number(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=r"answers\.jsonl:[12]"):
        read_jsonl(path, AnswerRecord)


def test_experiment_metadata_save_load_is_atomic_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    metadata = ExperimentMetadata(
        run_name="local",
        benchmark="longmemeval-v1",
        provider="fake",
        benchmark_config={"split": "oracle"},
        provider_params={"threshold": 0.5},
        provider_metadata={"version": "test"},
    )

    metadata.save(path)
    loaded = ExperimentMetadata.load(path)

    assert loaded == metadata
    assert not path.with_suffix(".json.tmp").exists()
    assert json.loads(path.read_text())["provider_metadata"] == {"version": "test"}


class FakeStore:
    constructed: ClassVar[list[str]] = []
    queried: ClassVar[list[str]] = []
    ingested: ClassVar[list[list[Session]]] = []
    resets: ClassVar[int] = 0

    def __init__(self, *, user_id: str, **kwargs: Any) -> None:
        self.user_id = user_id
        self.kwargs = kwargs
        self.usage = TokenUsage()
        self.constructed.append(user_id)

    async def __aenter__(self) -> FakeStore:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def reset(self) -> None:
        type(self).resets += 1
        self.usage = TokenUsage()

    async def ingest(self, sessions: list[Session]) -> None:
        self.ingested.append(sessions)

    async def wait_for_extraction(self) -> list[str]:
        return ["fixture memory"]

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        self.queried.append(question)
        self.usage.query_llm_prompt_tokens += 2
        return QueryResult(
            answer=f"answer for {question}",
            prompt=[{"role": "user", "content": question}],
            retrieval_latency_ms=1.5,
            llm_prompt_tokens=2,
            llm_completion_tokens=1,
        )

    def get_token_usage(self) -> TokenUsage:
        return self.usage

    def get_store_metadata(self) -> dict[str, str]:
        return {"kind": "fake"}


def _examples() -> list[DatasetExample]:
    session = Session(
        label="local",
        messages=[ContextMessage("user", "Remember this")],
    )
    return [
        DatasetExample(
            sessions=[session],
            qa_pairs=[
                QAPair(
                    question=f"question {index}",
                    answer=f"truth {index}",
                    question_id=f"q-{index}",
                )
            ],
            metadata={"question_date": "2026-01-01"},
        )
        for index in range(2)
    ]


@pytest.mark.asyncio
async def test_v1_runner_resumes_and_skips_completed_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeAdapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def load(self) -> list[DatasetExample]:
            return _examples()

    FakeStore.constructed.clear()
    FakeStore.queried.clear()
    FakeStore.ingested.clear()
    FakeStore.resets = 0
    monkeypatch.setitem(runner.STORES, "fake", FakeStore)
    monkeypatch.setattr(runner, "LongMemEvalAdapter", FakeAdapter)

    kwargs = {
        "provider": "fake",
        "split": "oracle",
        "results_root": tmp_path,
        "run_name": "resume-me",
        "provider_params": {"mode": "local"},
        "retries": 1,
    }
    out_dir = await runner.run_longmemeval_v1(**kwargs)
    first_records = read_jsonl(out_dir / "answers.jsonl", AnswerRecord)
    await runner.run_longmemeval_v1(**kwargs)
    resumed_records = read_jsonl(out_dir / "answers.jsonl", AnswerRecord)

    assert len(first_records) == 2
    assert resumed_records == first_records
    assert sorted(FakeStore.queried) == ["question 0", "question 1"]
    assert len(FakeStore.constructed) == 2
    assert all(record.memories == ["fixture memory"] for record in first_records)
    assert all(record.token_usage["total_tokens"] == 2 for record in first_records)
    metadata = ExperimentMetadata.load(out_dir / "metadata.json")
    assert metadata.num_answers == 2
    assert metadata.completed_at is not None
    assert metadata.provider_metadata == {"kind": "fake"}


@pytest.mark.asyncio
async def test_v1_runner_rejects_resume_with_different_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyAdapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def load(self) -> list[DatasetExample]:
            return []

    monkeypatch.setitem(runner.STORES, "fake", FakeStore)
    monkeypatch.setattr(runner, "LongMemEvalAdapter", EmptyAdapter)
    await runner.run_longmemeval_v1(
        provider="fake",
        split="oracle",
        results_root=tmp_path,
        run_name="fixed",
        retries=1,
    )

    with pytest.raises(ValueError, match="different configuration"):
        await runner.run_longmemeval_v1(
            provider="fake",
            split="oracle",
            results_root=tmp_path,
            run_name="fixed",
            limit=1,
            retries=1,
        )


@pytest.mark.asyncio
async def test_v2_limit_is_global_across_haystack_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = Session(
        label="trajectory",
        messages=[ContextMessage("environment", "Observed page")],
    )

    class Group:
        def __init__(self, prefix: str) -> None:
            self.trajectory_ids = (f"{prefix}-trajectory",)
            self.sessions = [session]
            self.questions = [
                {
                    "id": f"{prefix}-{index}",
                    "question": f"{prefix} question {index}",
                    "answer": "answer",
                }
                for index in range(2)
            ]

    class FakeV2Adapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def haystack_groups(self) -> list[Group]:
            return [Group("first"), Group("second")]

    FakeStore.constructed.clear()
    FakeStore.queried.clear()
    FakeStore.ingested.clear()
    FakeStore.resets = 0
    monkeypatch.setitem(runner.STORES, "fake", FakeStore)
    monkeypatch.setattr(runner, "LongMemEvalV2Adapter", FakeV2Adapter)

    out_dir = await runner.run_longmemeval_v2(
        provider="fake",
        tier="small",
        domains=["web"],
        results_root=tmp_path,
        run_name="v2-limit",
        limit=2,
        retries=1,
    )

    records = read_jsonl(out_dir / "answers.jsonl", AnswerRecord)
    assert {record.question_id for record in records} == {"first-0", "first-1"}
    assert len(FakeStore.ingested) == 1
