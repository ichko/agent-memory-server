from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_memory_benchmark.benchmark.judge import (
    _score,
    compute_metrics,
    judge_experiment,
)
from agent_memory_benchmark.benchmark.models import (
    AnswerRecord,
    ExperimentMetadata,
    JudgmentRecord,
    append_jsonl,
)
from agent_memory_benchmark.prompts import build_judge_prompt


def _answer(
    question_id: str | None,
    question_type: str | None,
    *,
    example_idx: int = 0,
) -> AnswerRecord:
    return AnswerRecord(
        example_idx=example_idx,
        question_id=question_id,
        dataset="LongMemEval",
        split="oracle",
        provider="fake",
        question=f"question {example_idx}",
        ground_truth="truth",
        predicted_answer="prediction",
        question_type=question_type,
    )


def _judgment(answer: AnswerRecord, score: int) -> JudgmentRecord:
    values: dict[str, Any] = {
        name: getattr(answer, name)
        for name in (
            "example_idx",
            "question_id",
            "dataset",
            "split",
            "provider",
            "question",
            "ground_truth",
            "predicted_answer",
        )
    }
    return JudgmentRecord(
        **values,
        score=score,
        reasoning="local fixture",
        judge_model="fake-judge",
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("yes", 1),
        (" YES. ", 1),
        ("The answer is no.", 0),
        ("yesterday", -1),
        ("correct", -1),
        ("", -1),
    ],
)
def test_score_parser_uses_standalone_yes_or_no(response: str, expected: int) -> None:
    assert _score(response) == expected


def test_compute_metrics_excludes_unparseable_and_macro_averages_types() -> None:
    answers = [
        _answer("q1", "type-a", example_idx=1),
        _answer("q2", "type-a", example_idx=2),
        _answer("q3_abs", "type-b", example_idx=3),
        _answer(None, None, example_idx=4),
    ]
    judgments = [
        _judgment(answers[0], 1),
        _judgment(answers[1], 0),
        _judgment(answers[2], 1),
        _judgment(answers[3], -1),
    ]

    metrics = compute_metrics(judgments, answers)

    assert metrics["overall"] == {"count": 3, "accuracy": 0.6667}
    assert metrics["task_averaged_accuracy"] == 0.75
    assert metrics["abstention"] == {"count": 1, "accuracy": 1.0}
    assert metrics["unparseable"] == 1
    assert metrics["per_question_type"] == {
        "type-a": {"count": 2, "accuracy": 0.5},
        "type-b": {"count": 1, "accuracy": 1.0},
    }


def test_compute_metrics_handles_no_valid_scores() -> None:
    answer = _answer("q1", "type-a")
    metrics = compute_metrics([_judgment(answer, -1)], [answer])

    assert metrics["overall"] == {"count": 0, "accuracy": None}
    assert metrics["task_averaged_accuracy"] is None
    assert metrics["abstention"] == {"count": 0, "accuracy": None}
    assert metrics["unparseable"] == 1
    assert metrics["per_question_type"] == {}


def test_judge_prompt_selects_abstention_and_temporal_rubrics() -> None:
    abstention = build_judge_prompt(
        question="Unknown?",
        answer="No evidence",
        response="I cannot tell",
        question_type="single-session-user",
        abstain=True,
    )
    temporal = build_judge_prompt(
        question="How long?",
        answer="Two weeks",
        response="14 days",
        question_type="temporal-reasoning",
        abstain=False,
    )

    assert "question is unanswerable" in abstention
    assert "off-by-one error" in temporal


@pytest.mark.asyncio
async def test_judge_rejects_v2_experiment_before_api_call(tmp_path: Path) -> None:
    run_dir = tmp_path / "v2"
    run_dir.mkdir()
    answer = _answer("q1", "static")
    answer.dataset = "LongMemEval-V2"
    append_jsonl(run_dir / "answers.jsonl", answer)
    ExperimentMetadata(
        run_name="v2",
        benchmark="longmemeval-v2",
        provider="fake",
        benchmark_config={},
        provider_params={},
    ).save(run_dir / "metadata.json")

    with pytest.raises(ValueError, match="supports LongMemEval v1 only"):
        await judge_experiment(experiment=str(run_dir), results_root=tmp_path)
