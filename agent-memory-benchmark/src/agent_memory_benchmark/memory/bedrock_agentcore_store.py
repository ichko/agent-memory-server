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
        self._sessions.clear()
        self._token_usage = TokenUsage()
