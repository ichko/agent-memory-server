from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from typing import Any

from agent_memory_benchmark.memory.base import (
    MemoryStore,
    QueryResult,
    SessionLike,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.llm import generate_answer


def _missing(
    distribution: str, extra: str, provider: str, exc: Exception
) -> ImportError:
    return ImportError(
        f"{provider} requires the optional '{distribution}' dependency. "
        f"Install it with: pip install 'agent-memory-benchmark[{extra}]'"
    ).with_traceback(exc.__traceback__)


def _session_text(session: SessionLike) -> str:
    lines = [f"Conversation date: {session.label}"] if session.label else []
    lines.extend(
        f"{normalize_role(message.speaker)}: {message.text}"
        for message in session.messages
        if message.text.strip()
    )
    return "\n".join(lines)


class _AnsweringStore(MemoryStore):
    def __init__(self, *, model: str = "gpt-4o") -> None:
        self._model = model
        self._token_usage = TokenUsage()

    async def _answer(
        self, context: str, question: str, question_date: str | None
    ) -> QueryResult:
        result, usage = await generate_answer(
            context, question, model=self._model, question_date=question_date
        )
        self._token_usage.query_llm_prompt_tokens += usage["prompt_tokens"]
        self._token_usage.query_llm_completion_tokens += usage["completion_tokens"]
        return result

    def get_token_usage(self) -> TokenUsage:
        return self._token_usage

    def get_store_metadata(self) -> dict[str, Any]:
        return {"answer_model": self._model}


class RedisAMSMemoryStore(_AnsweringStore):
    """Redis Agent Memory Server REST adapter using server-default extraction."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        namespace: str = "benchmark",
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
    ) -> None:
        super().__init__(model=model)
        try:
            from agent_memory_client import MemoryAPIClient, MemoryClientConfig
        except ImportError as exc:
            raise _missing(
                "agent-memory-client", "redis-ams", "Redis Agent Memory Server", exc
            ) from exc
        self._client = MemoryAPIClient(
            MemoryClientConfig(
                base_url=base_url
                or os.environ.get("AMS_BASE_URL", "http://localhost:8000"),
                default_namespace=namespace,
            )
        )
        self._namespace = namespace
        self._user_id = user_id
        self._search_limit = search_limit
        self._session_ids: list[str] = []

    async def close(self) -> None:
        await self._client.close()

    async def ingest(self, sessions: list[SessionLike]) -> None:
        from agent_memory_client.models import MemoryMessage, WorkingMemory

        for index, session in enumerate(sessions):
            session_id = f"{self._namespace}-{self._user_id}-{index}"
            self._session_ids.append(session_id)
            timestamp = session.date or datetime.now(timezone.utc)
            messages = [
                MemoryMessage(
                    role=normalize_role(message.speaker),
                    content=message.text,
                    created_at=timestamp,
                )
                for message in session.messages
                if message.text.strip()
            ]
            await self._client.put_working_memory(
                session_id,
                WorkingMemory(
                    session_id=session_id,
                    namespace=self._namespace,
                    user_id=self._user_id,
                    messages=messages,
                ),
            )

    async def _search(self, text: str, limit: int) -> list[Any]:
        from agent_memory_client.filters import UserId

        response = await self._client.search_long_term_memory(
            text=text, user_id=UserId(eq=self._user_id), limit=limit
        )
        return list(response.memories or [])

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        memories = await self._search(question, self._search_limit)
        return await self._answer(
            "\n".join(memory.text for memory in memories), question, question_date
        )

    async def list_memories(self) -> list[str]:
        return [memory.text for memory in await self._search("*", 100)]

    async def reset(self) -> None:
        memories = await self._search("*", 100)
        ids = [memory.id for memory in memories if memory.id]
        if ids:
            await self._client.delete_long_term_memories(ids)
        for session_id in self._session_ids:
            try:
                await self._client.delete_working_memory(session_id=session_id)
            except Exception:
                pass
        self._session_ids.clear()
        self._token_usage = TokenUsage()


class RedisAMSMCPStore(_AnsweringStore):
    """Redis Agent Memory Server MCP adapter without LLM-authored tool prompts."""

    def __init__(
        self,
        mcp_url: str | None = None,
        *,
        namespace: str = "benchmark",
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
    ) -> None:
        super().__init__(model=model)
        self._mcp_url = mcp_url or os.environ.get(
            "AMS_MCP_URL", "http://localhost:9050/sse"
        )
        self._namespace = namespace
        self._user_id = user_id
        self._search_limit = search_limit
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    async def __aenter__(self) -> RedisAMSMCPStore:
        try:
            from mcp.client.session import ClientSession
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise _missing("mcp", "redis-ams-mcp", "Redis AMS MCP", exc) from exc
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        read, write = await self._stack.enter_async_context(sse_client(self._mcp_url))
        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        return self

    async def close(self) -> None:
        if self._stack:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError("Use RedisAMSMCPStore as an async context manager")
        raw = await self._session.call_tool(name, arguments)
        payload = raw.model_dump(by_alias=True) if hasattr(raw, "model_dump") else raw
        structured = (
            payload.get("structuredContent") if isinstance(payload, dict) else None
        )
        if structured is not None:
            return structured
        for item in payload.get("content", []) if isinstance(payload, dict) else []:
            text = (
                item.get("text")
                if isinstance(item, dict)
                else getattr(item, "text", None)
            )
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
        return payload

    async def ingest(self, sessions: list[SessionLike]) -> None:
        memories = []
        for session in sessions:
            for message in session.messages:
                if message.text.strip():
                    text = (
                        f"[{session.label}] {normalize_role(message.speaker)}: "
                        f"{message.text}"
                    )
                    memories.append(
                        {
                            "text": text,
                            "user_id": self._user_id,
                            "namespace": self._namespace,
                        }
                    )
        if memories:
            await self._call("create_long_term_memories", {"memories": memories})

    async def _search(self, text: str, limit: int) -> list[dict[str, Any]]:
        result = await self._call(
            "search_long_term_memory",
            {"text": text, "user_id": {"eq": self._user_id}, "limit": limit},
        )
        return result.get("memories", []) if isinstance(result, dict) else []

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        memories = await self._search(question, self._search_limit)
        return await self._answer(
            "\n".join(str(memory.get("text", "")) for memory in memories),
            question,
            question_date,
        )

    async def list_memories(self) -> list[str]:
        return [str(item.get("text", "")) for item in await self._search("*", 100)]

    async def reset(self) -> None:
        memories = await self._search("*", 100)
        ids = [item.get("id") for item in memories if item.get("id")]
        if ids:
            await self._call("delete_long_term_memories", {"memory_ids": ids})
        self._token_usage = TokenUsage()


class Mem0MemoryStore(_AnsweringStore):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
    ) -> None:
        super().__init__(model=model)
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise _missing("mem0ai", "mem0", "Mem0", exc) from exc
        self._memory = Memory.from_config(config or {})
        self._user_id = user_id
        self._search_limit = search_limit

    async def ingest(self, sessions: list[SessionLike]) -> None:
        for session in sessions:
            messages = [
                {"role": normalize_role(item.speaker), "content": item.text}
                for item in session.messages
                if item.text.strip()
            ]
            await asyncio.to_thread(
                self._memory.add,
                messages,
                user_id=self._user_id,
                metadata={"date": session.label},
            )

    @staticmethod
    def _items(raw: Any) -> list[Any]:
        if isinstance(raw, dict):
            return raw.get("results", raw.get("memories", []))
        return raw

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        raw = await asyncio.to_thread(
            self._memory.search,
            question,
            user_id=self._user_id,
            limit=self._search_limit,
        )
        context = "\n".join(
            item.get("memory", str(item)) if isinstance(item, dict) else str(item)
            for item in self._items(raw)
        )
        return await self._answer(context, question, question_date)

    async def list_memories(self) -> list[str]:
        raw = await asyncio.to_thread(self._memory.get_all, user_id=self._user_id)
        return [
            item.get("memory", str(item)) if isinstance(item, dict) else str(item)
            for item in self._items(raw)
        ]

    async def reset(self) -> None:
        await asyncio.to_thread(self._memory.delete_all, user_id=self._user_id)
        self._token_usage = TokenUsage()


class LangMemStore(_AnsweringStore):
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
            raise _missing("langmem", "langmem", "LangMem", exc) from exc
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


class ZepMemoryStore(_AnsweringStore):
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
            raise _missing("zep-cloud", "zep", "Zep", exc) from exc
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


class GraphitiStore(_AnsweringStore):
    def __init__(
        self,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
    ) -> None:
        super().__init__(model=model)
        try:
            from graphiti_core import Graphiti
        except ImportError as exc:
            raise _missing("graphiti-core", "graphiti", "Graphiti", exc) from exc
        password = neo4j_password or os.environ.get("NEO4J_PASSWORD")
        if not password:
            raise RuntimeError("Graphiti requires NEO4J_PASSWORD or neo4j_password=")
        self._graph = Graphiti(
            uri=neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=neo4j_user or os.environ.get("NEO4J_USER", "neo4j"),
            password=password,
        )
        self._group_id = user_id
        self._search_limit = search_limit

    async def __aenter__(self) -> GraphitiStore:
        await self._graph.build_indices_and_constraints()
        return self

    async def close(self) -> None:
        await self._graph.close()

    async def ingest(self, sessions: list[SessionLike]) -> None:
        from graphiti_core.nodes import EpisodeType

        for index, session in enumerate(sessions):
            reference_time = session.date or datetime.now(timezone.utc)
            if reference_time.tzinfo is None:
                reference_time = reference_time.replace(tzinfo=timezone.utc)
            await self._graph.add_episode(
                name=f"session-{index}",
                episode_body=_session_text(session),
                source=EpisodeType.message,
                source_description="conversation session",
                reference_time=reference_time,
                group_id=self._group_id,
            )

    async def _facts(self, query: str, limit: int) -> list[str]:
        edges = await self._graph.search(
            query=query,
            group_ids=[self._group_id],
            num_results=limit,
        )
        return [getattr(edge, "fact", str(edge)) for edge in edges or []]

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
        await self._graph.driver.graph_ops.clear_data(
            self._graph.driver,
            group_ids=[self._group_id],
        )
        self._token_usage = TokenUsage()


class SupermemoryStore(_AnsweringStore):
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
            raise _missing("supermemory", "supermemory", "Supermemory", exc) from exc
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
            content = _session_text(session)
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


class VertexMemoryBankStore(_AnsweringStore):
    """Vertex Memory Bank adapter for an existing Agent Engine resource."""

    def __init__(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
        agent_engine_name: str,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        search_limit: int = 10,
    ) -> None:
        super().__init__(model=model)
        try:
            import vertexai
        except ImportError as exc:
            raise _missing(
                "google-cloud-aiplatform", "google", "Vertex Memory Bank", exc
            ) from exc
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "Vertex Memory Bank requires project= or GOOGLE_CLOUD_PROJECT"
            )
        self._client = vertexai.Client(
            project=project,
            location=location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
        self._engine = agent_engine_name
        self._scope = {"user_id": user_id}
        self._search_limit = search_limit

    async def ingest(self, sessions: list[SessionLike]) -> None:
        for session in sessions:
            await asyncio.to_thread(
                self._client.agent_engines.generate_memories,
                name=self._engine,
                direct_contents_source={
                    "events": [
                        {
                            "content": {
                                "role": normalize_role(item.speaker),
                                "parts": [{"text": item.text}],
                            }
                        }
                        for item in session.messages
                        if item.text.strip()
                    ]
                },
                scope=self._scope,
                config={"wait_for_completion": True},
            )

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        results = await asyncio.to_thread(
            self._client.agent_engines.retrieve_memories,
            name=self._engine,
            scope=self._scope,
            similarity_search_params={
                "search_query": question,
                "top_k": self._search_limit,
            },
        )
        context = "\n".join(
            str(getattr(getattr(item, "memory", item), "fact", "")) for item in results
        )
        return await self._answer(context, question, question_date)

    async def list_memories(self) -> list[str]:
        memories = await asyncio.to_thread(
            lambda: list(self._client.agent_engines.list_memories(name=self._engine))
        )
        return [
            str(getattr(item, "fact", ""))
            for item in memories
            if getattr(item, "scope", None) == self._scope
        ]

    async def reset(self) -> None:
        memories = await asyncio.to_thread(
            lambda: list(self._client.agent_engines.list_memories(name=self._engine))
        )
        for memory in memories:
            if getattr(memory, "scope", None) == self._scope:
                await asyncio.to_thread(
                    self._client.agent_engines.delete_memory, name=memory.name
                )
        self._token_usage = TokenUsage()


class BedrockAgentCoreStore(_AnsweringStore):
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
            raise _missing(
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


class OracleAgentMemoryStore(_AnsweringStore):
    def __init__(
        self,
        *,
        user_id: str = "benchmark",
        model: str = "gpt-4o",
        max_search_results: int = 10,
        connection: Any = None,
    ) -> None:
        super().__init__(model=model)
        try:
            import oracledb
            from oracleagentmemory.core import OracleAgentMemory
        except ImportError as exc:
            raise _missing(
                "oracleagentmemory", "oracle", "Oracle Agent Memory", exc
            ) from exc
        if connection is None:
            password = os.environ.get("ORACLE_MEMORY_DB_PASSWORD")
            if not password:
                raise RuntimeError(
                    "Oracle Agent Memory requires connection= or "
                    "ORACLE_MEMORY_DB_PASSWORD"
                )
            connection = oracledb.create_pool(
                user=os.environ.get("ORACLE_MEMORY_DB_USER"),
                password=password,
                dsn=os.environ.get("ORACLE_MEMORY_DB_CONNECT_STRING"),
            )
        self._connection = connection
        self._memory = OracleAgentMemory(connection=connection)
        self._user_id = user_id
        self._max_search_results = max_search_results
        self._thread: Any = None

    async def close(self) -> None:
        close = getattr(self._connection, "close", None)
        if close:
            await asyncio.to_thread(close)

    async def ingest(self, sessions: list[SessionLike]) -> None:
        from oracleagentmemory.apis.thread import Message

        self._thread = await asyncio.to_thread(
            self._memory.create_thread, user_id=self._user_id
        )
        for session in sessions:
            messages = [
                Message(
                    role=normalize_role(item.speaker),
                    content=item.text,
                    timestamp=session.date.isoformat() if session.date else None,
                )
                for item in session.messages
                if item.text.strip()
            ]
            await asyncio.to_thread(self._thread.add_messages, messages)

    async def _search(self, query: str, limit: int) -> list[str]:
        from oracleagentmemory.apis.searchscope import SearchScope

        results = await asyncio.to_thread(
            self._memory.search,
            query,
            scope=SearchScope(user_id=self._user_id),
            max_results=limit,
        )
        return [str(getattr(item, "content", item)) for item in results]

    async def query(
        self, question: str, *, question_date: str | None = None
    ) -> QueryResult:
        return await self._answer(
            "\n".join(await self._search(question, self._max_search_results)),
            question,
            question_date,
        )

    async def list_memories(self) -> list[str]:
        return await self._search("memories", 100)

    async def reset(self) -> None:
        delete_user = getattr(self._memory, "delete_user", None)
        if delete_user:
            await asyncio.to_thread(delete_user, self._user_id)
        self._thread = None
        self._token_usage = TokenUsage()


class MastraOMStore(MemoryStore):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "Mastra Observational Memory has no public Python provider adapter. "
            "The source implementation was intentionally not copied because it "
            "depends on private prompt logic. Use Mastra's public TypeScript "
            "package directly and contribute a transport-backed adapter."
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


class EmergenceFastStore(MemoryStore):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "The available Emergence source is a local RAG reimplementation, "
            "not a public provider API adapter, so it was intentionally omitted."
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
