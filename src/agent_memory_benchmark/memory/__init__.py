"""Provider-neutral memory adapters.

Third-party SDKs are imported only when their adapter is instantiated, so
importing this package never requires provider extras or credentials.
"""

from agent_memory_benchmark.memory.base import (
    MemoryStore,
    QueryResult,
    TokenUsage,
    normalize_role,
)
from agent_memory_benchmark.memory.providers import (
    BedrockAgentCoreStore,
    EmergenceFastStore,
    GraphitiStore,
    LangMemStore,
    MastraOMStore,
    Mem0MemoryStore,
    OracleAgentMemoryStore,
    SupermemoryStore,
    VertexMemoryBankStore,
    ZepMemoryStore,
)

STORES: dict[str, type[MemoryStore]] = {
    "mem0": Mem0MemoryStore,
    "langmem": LangMemStore,
    "zep": ZepMemoryStore,
    "graphiti": GraphitiStore,
    "supermemory": SupermemoryStore,
    "vertex-memory-bank": VertexMemoryBankStore,
    "bedrock-agentcore": BedrockAgentCoreStore,
    "oracle-agent-memory": OracleAgentMemoryStore,
}


def get_store_class(name: str) -> type[MemoryStore]:
    """Resolve an adapter by public registry name."""
    try:
        return STORES[name]
    except KeyError as exc:
        available = ", ".join(sorted(STORES))
        raise KeyError(
            f"Unknown memory store {name!r}. Available: {available}"
        ) from exc


__all__ = [
    "BedrockAgentCoreStore",
    "EmergenceFastStore",
    "GraphitiStore",
    "LangMemStore",
    "MastraOMStore",
    "Mem0MemoryStore",
    "MemoryStore",
    "OracleAgentMemoryStore",
    "QueryResult",
    "STORES",
    "SupermemoryStore",
    "TokenUsage",
    "VertexMemoryBankStore",
    "ZepMemoryStore",
    "get_store_class",
    "normalize_role",
]
