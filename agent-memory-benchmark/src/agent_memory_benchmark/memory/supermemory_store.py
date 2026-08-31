from __future__ import annotations

import asyncio
import os
import time

from agent_memory_benchmark.memory.base import QueryResult, SessionLike, TokenUsage
from agent_memory_benchmark.memory.common import (
    AnsweringStore,
    missing_dependency,
    session_text,
)


class SupermemoryStore(AnsweringStore):
    def __init__(
        self,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
        api_key: str | None = None,
        wait_timeout: float = 600,
        wait_poll_interval: float = 2,
    ) -> None:
        super().__init__(model=model)
        try:
            from supermemory import AsyncSupermemory
        except ImportError as exc:
            raise missing_dependency(
                "supermemory", "supermemory", "Supermemory", exc
            ) from exc
        key = api_key or os.environ.get("SUPERMEMORY_API_KEY")
        if not key:
            raise RuntimeError("Supermemory requires SUPERMEMORY_API_KEY or api_key=")
        self._client = AsyncSupermemory(api_key=key)
        self._user_id = user_id
        self._search_limit = search_limit
        self._wait_timeout = wait_timeout
        self._wait_poll_interval = wait_poll_interval
        self._documents: list[str] = []
        self._document_ids: set[str] = set()

    async def close(self) -> None:
        await self._client.close()

    async def ingest(self, sessions: list[SessionLike]) -> None:
        for index, session in enumerate(sessions):
            content = session_text(session)
            response = await self._client.add(
                content=content,
                container_tags=[self._user_id],
                custom_id=f"{self._user_id}-{index}",
            )
            self._documents.append(content)
            document_id = getattr(response, "id", None)
            if document_id:
                self._document_ids.add(str(document_id))

    async def wait_for_extraction(
        self, *, timeout: float = 120, poll_interval: float = 2
    ) -> list[str]:
        if not self._document_ids:
            return list(self._documents)
        effective_timeout = timeout if timeout != 120 else self._wait_timeout
        effective_poll = (
            poll_interval if poll_interval != 2 else self._wait_poll_interval
        )
        deadline = time.monotonic() + effective_timeout
        while time.monotonic() < deadline:
            response = await self._client.documents.list(
                container_tags=[self._user_id],
                limit=max(50, len(self._document_ids)),
            )
            documents = getattr(response, "memories", None) or []
            terminal: set[str] = set()
            failed: set[str] = set()
            for document in documents:
                document_id = getattr(document, "id", None)
                status = str(getattr(document, "status", "")).lower()
                if document_id and status in {"done", "failed", "error"}:
                    terminal.add(str(document_id))
                if document_id and status in {"failed", "error"}:
                    failed.add(str(document_id))
            if failed & self._document_ids:
                raise RuntimeError(
                    "Supermemory failed to process document(s): "
                    + ", ".join(sorted(failed & self._document_ids))
                )
            if self._document_ids.issubset(terminal):
                return list(self._documents)
            await asyncio.sleep(effective_poll)
        raise TimeoutError(
            f"Supermemory did not process all documents within {effective_timeout}s"
        )

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        response = await self._client.search.execute(
            q=question,
            container_tags=[self._user_id],
            limit=self._search_limit,
        )
        results = getattr(response, "results", None) or []
        context = "\n".join(
            str(getattr(item, "content", None) or getattr(item, "text", item))
            for item in results
        )
        return await self._answer(context, question, question_date)

    async def list_memories(self) -> list[str]:
        return list(self._documents)

    async def reset(self) -> None:
        await self._client.documents.delete_bulk(container_tags=[self._user_id])
        self._documents.clear()
        self._document_ids.clear()
        self._token_usage = TokenUsage()
