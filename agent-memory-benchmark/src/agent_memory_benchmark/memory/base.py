from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Protocol


class MessageLike(Protocol):
    speaker: str
    text: str


class SessionLike(Protocol):
    label: str
    date: Any
    messages: list[MessageLike]


@dataclass
class TokenUsage:
    ingest_llm_prompt_tokens: int = 0
    ingest_llm_completion_tokens: int = 0
    ingest_embed_tokens: int = 0
    query_llm_prompt_tokens: int = 0
    query_llm_completion_tokens: int = 0
    query_embed_tokens: int = 0

    @property
    def ingest_llm_total(self) -> int:
        return self.ingest_llm_prompt_tokens + self.ingest_llm_completion_tokens

    @property
    def ingest_total(self) -> int:
        return self.ingest_llm_total + self.ingest_embed_tokens

    @property
    def query_llm_total(self) -> int:
        return self.query_llm_prompt_tokens + self.query_llm_completion_tokens

    @property
    def query_total(self) -> int:
        return self.query_llm_total + self.query_embed_tokens

    @property
    def total_llm_tokens(self) -> int:
        return self.ingest_llm_total + self.query_llm_total

    @property
    def total_embed_tokens(self) -> int:
        return self.ingest_embed_tokens + self.query_embed_tokens

    @property
    def total(self) -> int:
        return self.ingest_total + self.query_total

    def to_dict(self) -> dict[str, int]:
        result = asdict(self)
        result.update(
            ingest_llm_total_tokens=self.ingest_llm_total,
            ingest_total_tokens=self.ingest_total,
            query_llm_total_tokens=self.query_llm_total,
            query_total_tokens=self.query_total,
            total_llm_tokens=self.total_llm_tokens,
            total_embed_tokens=self.total_embed_tokens,
            total_tokens=self.total,
        )
        return result


@dataclass
class QueryResult:
    answer: str
    prompt: list[dict[str, str]]
    retrieval_latency_ms: float | None = None
    llm_prompt_tokens: int | None = None
    llm_completion_tokens: int | None = None


def normalize_role(speaker: str) -> str:
    return (
        "user"
        if speaker.lower().strip() in {"user", "human", "environment", "tool"}
        else "assistant"
    )


class MemoryStore(ABC):
    async def __aenter__(self) -> MemoryStore:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        return None

    @abstractmethod
    async def ingest(self, sessions: list[SessionLike]) -> None:
        pass

    @abstractmethod
    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        pass

    @abstractmethod
    async def list_memories(self) -> list[str]:
        pass

    async def wait_for_extraction(
        self, *, timeout: float = 120, poll_interval: float = 2
    ) -> list[str]:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            memories = await self.list_memories()
            if memories:
                return memories
            await asyncio.sleep(poll_interval)
        return await self.list_memories()

    @abstractmethod
    async def reset(self) -> None:
        pass

    def get_token_usage(self) -> TokenUsage:
        return TokenUsage()

    def get_store_metadata(self) -> dict[str, Any]:
        return {}
