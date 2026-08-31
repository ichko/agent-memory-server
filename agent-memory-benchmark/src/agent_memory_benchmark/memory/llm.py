from __future__ import annotations

from typing import Any

from agent_memory_benchmark.memory.base import QueryResult

ANSWER_SYSTEM_PROMPT = """Answer the question using only the supplied memory context.
Be accurate, concise, and direct. Use dates in the context to resolve temporal
questions. Follow output-format and unknown-answer instructions in the question
exactly. If no format is requested and the context does not contain the answer,
say that the available memories are insufficient."""

_client: Any = None


def get_openai_client() -> Any:
    global _client
    if _client is None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI answer generation requires the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc
        _client = AsyncOpenAI()
    return _client


def build_prompt(
    context: str, question: str, *, question_date: str | None = None
) -> list[dict[str, str]]:
    parts = [ANSWER_SYSTEM_PROMPT]
    if question_date:
        parts.append(f"\nQuestion date: {question_date}")
    parts.append(f"\n\nMemory context:\n{context or '(no memories found)'}")
    return [
        {"role": "system", "content": "".join(parts)},
        {"role": "user", "content": question},
    ]


async def generate_answer(
    context: str,
    question: str,
    *,
    model: str = "gpt-4o",
    question_date: str | None = None,
) -> tuple[QueryResult, dict[str, int]]:
    messages = build_prompt(context, question, question_date=question_date)
    response = await get_openai_client().chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    result = QueryResult(
        answer=response.choices[0].message.content or "",
        prompt=messages,
        llm_prompt_tokens=prompt_tokens,
        llm_completion_tokens=completion_tokens,
    )
    return result, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
