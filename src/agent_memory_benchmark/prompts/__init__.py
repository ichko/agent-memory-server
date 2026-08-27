"""Benchmark protocol prompts."""

from agent_memory_benchmark.prompts.answer import V2_ANSWER_INSTRUCTIONS
from agent_memory_benchmark.prompts.judge import build_judge_prompt

__all__ = ["V2_ANSWER_INSTRUCTIONS", "build_judge_prompt"]
