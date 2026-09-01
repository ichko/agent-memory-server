from __future__ import annotations

import asyncio
import os

from agent_memory_benchmark.memory.base import (
    QueryResult,
    SessionLike,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.common import AnsweringStore, missing_dependency


class BedrockAgentCoreStore(AnsweringStore):
    def __init__(
        self,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        region: str | None = None,
        memory_id: str,
        namespace_path: str,
        top_k: int = 10,
    ) -> None:
        super().__init__(model=model)
        try:
            from bedrock_agentcore.memory import MemoryClient
        except ImportError as exc:
            raise missing_dependency(
                "bedrock-agentcore", "aws", "AWS Bedrock AgentCore", exc
            ) from exc
        self._client = MemoryClient(
            region_name=region
            or os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
            or "us-east-1"
        )
        self._memory_id = memory_id
        self._namespace_path = namespace_path.format(user_id=user_id)
        self._user_id = user_id
        self._top_k = top_k
        self._sessions: list[str] = []

    async def ingest(self, sessions: list[SessionLike]) -> None:
        for index, session in enumerate(sessions):
            session_id = f"{self._user_id}-{index}"
            self._sessions.append(session_id)
            messages = [
                (
                    item.text,
                    "USER" if normalize_role(item.speaker) == "user" else "ASSISTANT",
                )
                for item in session.messages
                if item.text.strip()
            ]
            await asyncio.to_thread(
                self._client.create_event,
                memory_id=self._memory_id,
                actor_id=self._user_id,
                session_id=session_id,
                messages=messages,
                event_timestamp=session.date,
            )

    async def _retrieve(self, query: str, limit: int) -> list[str]:
        records = await asyncio.to_thread(
            self._client.retrieve_memories,
            memory_id=self._memory_id,
            namespace_path=self._namespace_path,
            query=query,
            top_k=limit,
        )
        return [
            str(record.get("content", {}).get("text", record))
            for record in records or []
        ]

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        return await self._answer(
            "\n".join(await self._retrieve(question, self._top_k)),
            question,
            question_date,
        )

    async def list_memories(self) -> list[str]:
        return await self._retrieve("*", 50)

    async def reset(self) -> None:
        await self._delete_remote()
        self._sessions.clear()
        self._token_usage = TokenUsage()

    async def _delete_remote(self) -> None:
        list_events = getattr(self._client, "list_events", None)
        delete_event = getattr(self._client, "delete_event", None)
        if list_events and delete_event:
            for session_id in list(self._sessions):
                payload = await asyncio.to_thread(
                    list_events,
                    memory_id=self._memory_id,
                    actor_id=self._user_id,
                    session_id=session_id,
                )
                for event in _items(payload, "events"):
                    event_id = _field(event, "eventId", "event_id")
                    if not event_id:
                        continue
                    await asyncio.to_thread(
                        delete_event,
                        memory_id=self._memory_id,
                        actor_id=self._user_id,
                        session_id=session_id,
                        event_id=event_id,
                    )
        delete_record = getattr(self._client, "delete_memory_record", None)
        if not delete_record:
            return
        records = await asyncio.to_thread(
            self._client.retrieve_memories,
            memory_id=self._memory_id,
            namespace_path=self._namespace_path,
            query="*",
            top_k=50,
        )
        for record in _items(records, "memoryRecords"):
            record_id = _field(record, "memoryRecordId", "id")
            if record_id:
                await asyncio.to_thread(
                    delete_record,
                    memory_id=self._memory_id,
                    memory_record_id=record_id,
                )


def _items(payload: object, key: str) -> list[object]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        value = payload.get(key) or payload.get("items")
        return value if isinstance(value, list) else []
    return []


def _field(item: object, *names: str) -> str | None:
    for name in names:
        value = item.get(name) if isinstance(item, dict) else getattr(item, name, None)
        if value:
            return str(value)
    return None
