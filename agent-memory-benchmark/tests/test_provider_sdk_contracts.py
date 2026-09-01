from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

import agent_memory_benchmark.memory.common as common
from agent_memory_benchmark.datasets.models import ContextMessage, Session
from agent_memory_benchmark.memory import QueryResult, TokenUsage
from agent_memory_benchmark.memory.graphiti_store import GraphitiStore
from agent_memory_benchmark.memory.mem0_store import Mem0MemoryStore
from agent_memory_benchmark.memory.oracle_agent_memory_store import (
    OracleAgentMemoryStore,
)
from agent_memory_benchmark.memory.vertex_memory_bank import VertexMemoryBankStore
from agent_memory_benchmark.memory.zep_store import ZepMemoryStore


def _install_module(monkeypatch: pytest.MonkeyPatch, name: str, module: Any) -> None:
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        dotted = ".".join(parts[:index])
        if dotted not in sys.modules:
            monkeypatch.setitem(sys.modules, dotted, types.ModuleType(dotted))
    monkeypatch.setitem(sys.modules, name, module)


async def _fake_answer(
    context: str,
    question: str,
    *,
    model: str,
    question_date: str | None,
) -> tuple[QueryResult, dict[str, int]]:
    return (
        QueryResult(answer=context or "none", prompt=[]),
        {"prompt_tokens": 1, "completion_tokens": 1},
    )


class _Mem0:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> _Mem0:
        return cls()

    def search(self, query: str, **kwargs: Any) -> dict[str, list[dict[str, str]]]:
        self.calls.append(("search", (query,), kwargs))
        return {"results": [{"memory": "likes coffee"}]}

    def get_all(self, **kwargs: Any) -> dict[str, list[dict[str, str]]]:
        self.calls.append(("get_all", (), kwargs))
        return {"results": [{"memory": "likes coffee"}]}

    def add(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def delete_all(self, **kwargs: Any) -> None:
        self.calls.append(("delete_all", (), kwargs))


@pytest.mark.asyncio
async def test_mem0_search_and_list_use_v2_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_module(monkeypatch, "mem0", SimpleNamespace(Memory=_Mem0))
    monkeypatch.setattr(common, "generate_answer", _fake_answer)
    store = Mem0MemoryStore(user_id="u1", search_limit=7)
    memories = await store.list_memories()
    await store.query("drink?")
    assert memories == ["likes coffee"]
    assert store._memory.calls[0] == ("get_all", (), {"filters": {"user_id": "u1"}})
    assert store._memory.calls[1][0] == "search"
    assert store._memory.calls[1][2] == {"filters": {"user_id": "u1"}, "top_k": 7}


@pytest.mark.asyncio
async def test_vertex_uses_memories_subclient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Memories:
        def generate(self, **_kwargs: Any) -> None:
            calls.append("generate")

        def retrieve(self, **_kwargs: Any) -> list[Any]:
            calls.append("retrieve")
            return [SimpleNamespace(memory=SimpleNamespace(fact="likes coffee"))]

        def list(self, **_kwargs: Any) -> list[Any]:
            calls.append("list")
            return [
                SimpleNamespace(
                    fact="likes coffee",
                    scope={"user_id": "u1"},
                    name="projects/p/memories/m1",
                )
            ]

        def delete(self, **_kwargs: Any) -> None:
            calls.append("delete")

    class Client:
        def __init__(self, **_kwargs: Any) -> None:
            self.agent_engines = SimpleNamespace(memories=Memories())

    _install_module(monkeypatch, "vertexai", SimpleNamespace(Client=Client))
    monkeypatch.setattr(common, "generate_answer", _fake_answer)
    store = VertexMemoryBankStore(
        project="p", agent_engine_name="engines/e", user_id="u1"
    )
    await store.ingest(
        [Session(label="now", messages=[ContextMessage("user", "I like coffee")])]
    )
    await store.query("drink?")
    await store.list_memories()
    await store.reset()
    assert calls == ["generate", "retrieve", "list", "list", "delete"]


@pytest.mark.asyncio
async def test_zep_reset_ignores_missing_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NotFoundError(Exception):
        status_code = 404

    class Users:
        async def delete(self, _user_id: str) -> None:
            raise NotFoundError("missing")

    class Client:
        def __init__(self) -> None:
            self.user = Users()

        async def close(self) -> None:
            return None

    _install_module(
        monkeypatch,
        "zep_cloud.client",
        SimpleNamespace(AsyncZep=lambda api_key: Client()),
    )
    store = ZepMemoryStore(api_key="k", user_id="missing-user")
    await store.reset()
    assert store._threads == []


@pytest.mark.asyncio
async def test_graphiti_reset_uses_clear_data_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    async def clear_data(driver: Any, group_ids: list[str] | None = None) -> None:
        calls.append((driver, group_ids))

    _install_module(
        monkeypatch,
        "graphiti_core.utils.maintenance.graph_data_operations",
        SimpleNamespace(clear_data=clear_data),
    )
    store = GraphitiStore.__new__(GraphitiStore)
    store._graph = SimpleNamespace(driver="driver")
    store._group_id = "group-1"
    store._token_usage = TokenUsage()
    await store.reset()
    assert calls == [("driver", ["group-1"])]


def test_oracle_store_passes_llm_and_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: dict[str, Any] = {}

    class OracleAgentMemory:
        def __init__(self, **kwargs: Any) -> None:
            constructed.update(kwargs)

    _install_module(monkeypatch, "oracledb", SimpleNamespace())
    _install_module(
        monkeypatch,
        "oracleagentmemory.core.llms.llm",
        SimpleNamespace(Llm=lambda model: f"llm:{model}"),
    )
    _install_module(
        monkeypatch,
        "oracleagentmemory.core.embedders.embedder",
        SimpleNamespace(Embedder=lambda model: f"embed:{model}"),
    )
    sys.modules["oracleagentmemory.core"].OracleAgentMemory = OracleAgentMemory
    OracleAgentMemoryStore(connection=object(), model="gpt-4o", embedder_model="emb")
    assert constructed["llm"] == "llm:gpt-4o"
    assert constructed["embedder"] == "embed:emb"
