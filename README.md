<div align=center>

# Redis Agent Memory

[Redis Agent Memory in Redis Iris](https://redis.io/agent-memory/) is Redis's
official memory service for agents. This repository preserves the open research
that helped make that product possible: the pivotal [`V0/`](./V0/) foundation
and an auditable [LongMemEval benchmark harness](./agent-memory-benchmark/).

</div>

## Redis Agent Memory in Redis Iris

[Redis Agent Memory in Redis Iris](https://redis.io/agent-memory/) is Redis’s official managed path for teams that want agent memory as a service, not another subsystem to build and operate themselves. [Redis Iris](https://redis.io/iris/) is the real-time context engine for agents, designed to deliver fresh, relevant context at runtime, and Redis Agent Memory is the part of Iris that makes context compound across turns, sessions, channels, and agents.

Ready to build with the supported product?

1. [Create a Redis Cloud database](https://redis.io/docs/latest/operate/rc/databases/create-database/).
2. [Create an Agent Memory service](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/create-service/).
3. Choose the [Python SDK](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/python-sdk-quickstart/), [TypeScript SDK](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/typescript-sdk-quickstart/), or [REST API](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/rest-api-quickstart/) quickstart.
4. Use the [API reference](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/api-reference/) as you integrate.

See the [Redis Agent Memory docs](https://redis.io/docs/latest/develop/ai/context-engine/agent-memory/) for the full developer guide and [Redis Agent Memory on Redis Cloud](https://redis.io/docs/latest/operate/rc/context-engine/agent-memory/) for service operations.

Redis Agent Memory in Iris gives you the Redis-managed experience: a persistent, structured memory layer for AI agents exposed through a REST API and client libraries, with dedicated endpoints, secure API key management, configurable memory schemas, and automatic TTL-based lifecycle management. The point is not just storage. It is to remove the custom memory infrastructure teams otherwise end up building around session handling, extraction, retrieval, and lifecycle management.

Redis Agent Memory uses a two-tier model. Session memory keeps the active conversation state, session history, and session-specific metadata close at hand, with configurable TTL control for retention. Long-term memory stores extracted facts and learned patterns from past interactions as text plus vector embeddings for semantic retrieval. As new events are written to working memory, Redis Agent Memory automatically extracts important information and promotes it to long-term memory in the background, so memory accumulates without slowing down the live agent loop.

That matters because Redis Iris is not just a memory feature in isolation. It is a broader context engine built to address the production problems agents actually hit: fragmented data, stale operational state, slow retrieval, and interactions that do not improve over time. Within that story, Redis Agent Memory is the compounding memory layer; [Redis Context Retriever](https://redis.io/context-retriever/) makes business data navigable; [Redis Data Integration](https://redis.io/data-integration/) keeps operational state fresh; and [Redis LangCache](https://redis.io/langcache/) helps repeated work stay inside the latency budget.

## V0 — the open-source research foundation

[**`V0/`**](./V0/) contains the original Redis Agent Memory Server: an
open-source reference implementation with REST and MCP interfaces, working and
long-term memory, configurable extraction strategies, and Redis-backed semantic
search.

This work was pivotal to Redis's agent-memory productization. It established and
tested the two-tier architecture and memory lifecycle that informed the official
Redis Agent Memory product. We intentionally preserve it under `V0/` as an open
research artifact: useful for study, experimentation, and community development,
but not presented as the current supported production distribution.

- **Start here:** [`V0/README.md`](./V0/README.md)
- **Documentation:** [`V0/docs/`](./V0/docs/index.md)
- Build, test, and run everything from inside `V0/` (e.g. `cd V0 && make test`).

## Agent Memory Benchmark — an open audit surface

[**`agent-memory-benchmark/`**](./agent-memory-benchmark/) makes the evaluation
loop behind our memory research inspectable. For each LongMemEval question it
ingests prior chat sessions, retrieves context, generates an answer with a shared
answering step, and records a task-specific LLM judgment.

We include the harness to give the community a place to audit methods, reproduce
results, challenge assumptions, and add implementations—not to turn one run into
a timeless claim that Redis is "the best." It follows the open-evaluation pattern
established by [LongMemEval](https://github.com/xiaowu0162/LongMemEval) and
provider-published implementations such as [Zep's LongMemEval harness](https://github.com/getzep/zep/tree/main/benchmarks/longmemeval).

The harness has adapters for Redis Agent Memory and eight other open-source or
managed memory systems. It standardizes the outer evaluation loop while leaving
each system's extraction, indexing, and retrieval behavior visible. Per-question
answers, prompts, retrieved context, judgments, configuration, and errors are
written as reviewable artifacts.

- **Start here:** [`agent-memory-benchmark/README.md`](./agent-memory-benchmark/README.md)
- **Provider recipes:** [`agent-memory-benchmark/docs/providers/`](./agent-memory-benchmark/docs/providers/README.md)
- Run everything from inside `agent-memory-benchmark/` (e.g. `cd agent-memory-benchmark && uv run memory-bench providers`).

The harness is not itself a vendor ranking. A result is evidence about a specific
provider version, service configuration, dataset split, retrieval budget, answer
model, judge model, and point in time. Those conditions must match before scores
can support a comparison.

## License

Both projects are licensed under the **Apache License 2.0** (Redis, Inc.). See
[`LICENSE`](./LICENSE) at the repository root.
