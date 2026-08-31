from __future__ import annotations

import os
import uuid

from agent_memory_benchmark.memory.base import (
    QueryResult,
    SessionLike,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.common import AnsweringStore, missing_dependency


class ZepMemoryStore(AnsweringStore):
    def __init__(
        self,
        *,
        user_id: str | None = None,
        model: str = "gpt-4o",
        search_limit: int = 10,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model=model)
        try:
            from zep_cloud.client import AsyncZep
        except ImportError as exc:
            raise missing_dependency("zep-cloud", "zep", "Zep", exc) from exc
        key = api_key or os.environ.get("ZEP_API_KEY")
        if not key:
            raise RuntimeError("Zep requires ZEP_API_KEY or api_key=")
        self._client = AsyncZep(api_key=key)
        self._user_id = user_id or f"memory-{uuid.uuid4().hex[:12]}"
        self._search_limit = search_limit
        self._threads: list[str] = []

    async def close(self) -> None:
        if hasattr(self._client, "close"):
            await self._client.close()

    async def ingest(self, sessions: list[SessionLike]) -> None:
        from zep_cloud.types import Message

        await self._client.user.add(user_id=self._user_id)
        for index, session in enumerate(sessions):
            thread_id = f"{self._user_id}-{index}"
            self._threads.append(thread_id)
            await self._client.thread.create(thread_id=thread_id, user_id=self._user_id)
            messages = [
                Message(
                    role=normalize_role(item.speaker),
                    name=normalize_role(item.speaker),
                    content=item.text,
                )
                for item in session.messages
                if item.text.strip()
            ]
            for start in range(0, len(messages), 30):
                await self._client.thread.add_messages(
                    thread_id, messages=messages[start : start + 30]
                )

    async def _facts(self, query: str, limit: int) -> list[str]:
        result = await self._client.graph.search(
            user_id=self._user_id, query=query, scope="edges", limit=limit
        )
        return [getattr(edge, "fact", str(edge)) for edge in (result.edges or [])]

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        return await self._answer(
            "\n".join(await self._facts(question, self._search_limit)),
            question,
            question_date,
        )

    async def list_memories(self) -> list[str]:
        return await self._facts("*", 50)

    async def reset(self) -> None:
        await self._client.user.delete(self._user_id)
        self._threads.clear()
        self._token_usage = TokenUsage()
