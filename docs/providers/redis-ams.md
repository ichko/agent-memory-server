# Redis Agent Memory Server (REST and MCP)

Public Redis Agent Memory Server (AMS) wrappers. REST uses `agent-memory-client`. MCP calls public AMS tools over SSE. Both pass retrieved memory into the shared answer step.

Docs:

- Open-source server: [redis.github.io/agent-memory-server](https://redis.github.io/agent-memory-server/)
- Source: [github.com/redis/agent-memory-server](https://github.com/redis/agent-memory-server)
- MCP: [MCP Server](https://redis.github.io/agent-memory-server/mcp/)

CLI ids: `redis-ams`, `redis-ams-mcp`.

## Install

```bash
uv sync --extra redis-ams
# MCP wrapper
uv sync --extra redis-ams-mcp
```

## Local REST (this repository)

```bash
cp .env.example .env   # set OPENAI_API_KEY; AMS uses it for extraction
docker compose up -d
```

Compose starts Redis 8 and `redislabs/agent-memory-server:latest` on `http://localhost:8000`. Long-term extraction is enabled. This Compose file does **not** start MCP SSE. Compose also loads `.env` if present, so leftover AMS variables (generation model, auth flags) apply to the container.

```bash
OPENAI_API_KEY=replace-me
AMS_BASE_URL=http://localhost:8000
```

## Auth

Local AMS in this Compose file sets `DISABLE_AUTH=true`. If you enable token or OAuth auth, set the values AMS documents in `.env`.

## Run (REST)

```bash
uv run memory-bench run \
  --provider redis-ams \
  --dataset longmemeval \
  --split oracle \
  --limit 1 \
  --run-name smoke-redis-ams \
  --provider-param base_url=http://localhost:8000 \
  --provider-param model=gpt-4o

uv run memory-bench judge --experiment smoke-redis-ams --judge-model gpt-4o
```

REST ingest writes working memory and relies on server-side long-term extraction. Query searches long-term memory.

## MCP

MCP needs an AMS SSE endpoint. This repo’s Compose file does not expose one. Follow the [AMS MCP docs](https://redis.github.io/agent-memory-server/mcp/). Hosted Compose in that project often maps SSE to `9050`. A CLI SSE server uses `--mode sse` (default port `9000`).

```bash
AMS_MCP_URL=http://localhost:9050/sse

uv run memory-bench run \
  --provider redis-ams-mcp \
  --dataset longmemeval \
  --split oracle \
  --limit 1 \
  --run-name smoke-redis-ams-mcp \
  --provider-param mcp_url=http://localhost:9050/sse \
  --provider-param model=gpt-4o
```

MCP ingest calls `create_long_term_memories` directly. It does not use an LLM tool loop.

## Cleanup

```bash
docker compose down
```

Isolation uses `namespace` and a unique per-example `user_id`. Flush Redis if you reuse the instance across unrelated runs.

Do not point these wrappers at unpublished hosted research endpoints.
