from __future__ import annotations

from typing import Any

from agent_memory_benchmark.memory.base import MemoryStore, QueryResult, SessionLike


class MastraOMStore(MemoryStore):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "Mastra Observational Memory has no public Python client. Use "
            "Mastra's public TypeScript package and contribute an adapter that "
            "calls it over a documented transport."
        )

    async def ingest(self, sessions: list[SessionLike]) -> None:
        raise NotImplementedError

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        raise NotImplementedError

    async def list_memories(self) -> list[str]:
        raise NotImplementedError

    async def reset(self) -> None:
        raise NotImplementedError
