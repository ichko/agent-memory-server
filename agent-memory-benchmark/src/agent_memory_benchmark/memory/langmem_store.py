from __future__ import annotations

from typing import Any

from agent_memory_benchmark.memory.base import (
    QueryResult,
    SessionLike,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.common import AnsweringStore, missing_dependency


class LangMemStore(AnsweringStore):
    """LangMem manager plus LangGraph's provider-supported indexed store."""

    def __init__(
        self,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
    ) -> None:
        super().__init__(model=model)
        try:
            from langgraph.store.memory import InMemoryStore
            from langmem import create_memory_manager
        except ImportError as exc:
            raise missing_dependency("langmem", "langmem", "LangMem", exc) from exc
        self._manager = create_memory_manager(f"openai:{model}")
        self._store = InMemoryStore(
            index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
        )
        self._namespace = ("memories", user_id)
        self._search_limit = search_limit
        self._existing: list[Any] = []

    async def ingest(self, sessions: list[SessionLike]) -> None:
        for session in sessions:
            messages = [
                {"role": normalize_role(item.speaker), "content": item.text}
                for item in session.messages
                if item.text.strip()
            ]
            self._existing = await self._manager.ainvoke(
                {"messages": messages, "existing": self._existing}
            )
        for index, memory in enumerate(self._existing):
            value = memory[1] if isinstance(memory, tuple) else memory
            text = getattr(value, "content", value)
            self._store.put(self._namespace, str(index), {"text": str(text)})

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        results = self._store.search(
            self._namespace, query=question, limit=self._search_limit
        )
        context = "\n".join(str(item.value.get("text", "")) for item in results)
        return await self._answer(context, question, question_date)

    async def list_memories(self) -> list[str]:
        return [
            str(item.value.get("text", ""))
            for item in self._store.search(self._namespace, limit=100)
        ]

    async def reset(self) -> None:
        for item in self._store.search(self._namespace, limit=100):
            self._store.delete(self._namespace, item.key)
        self._existing.clear()
        self._token_usage = TokenUsage()
