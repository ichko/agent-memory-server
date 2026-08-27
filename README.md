# Agent Memory Benchmark

This repository is a shared **LongMemEval** harness. You ingest sessions into a memory product, retrieve for a test question, generate an answer, and (for v1) score that answer with an LLM judge.

It is not a vendor ranking and not a SOTA claim. The public wrappers talk to vendor APIs and default extraction. They do **not** include Redis research strategies used in internal experiments.

## What you can and cannot reproduce

The Redis write-up is here: [Building and evaluating long-term conversational memory](https://redis.github.io/redis-ai-research-public/longmemeval-agent-memory/).

From that page, these facts are public:

| Item | Value |
|------|--------|
| Dataset | LongMemEval **Small** v1 |
| Size | **500** questions |
| Coverage | **six** task types |
| Models | **gpt-4o** for answers and judging |
| Snapshot | **86.1%** task-averaged accuracy for an **internal combined research configuration** |

You **cannot** reproduce 86.1% from this checkout. That score used a private combined configuration that is not here. Scores from these wrappers are a different experiment.

You **can** reproduce the protocol pieces: the public dataset, the six task types, gpt-4o answer+judge defaults, and the task-averaged metric.

## Quick start

You need Python 3.10–3.12, [uv](https://docs.astral.sh/uv/), and an OpenAI key.

```bash
cp .env.example .env
# Put OPENAI_API_KEY in .env. Do not commit that file.

uv sync --extra langmem --group dev
uv run memory-bench providers
```

The cheapest smoke run uses LongMemEval **oracle** (short haystacks) and one question. LangMem needs no extra server.

```bash
uv run memory-bench run \
  --provider langmem \
  --dataset longmemeval \
  --split oracle \
  --limit 1 \
  --run-name smoke-langmem \
  --provider-param model=gpt-4o-mini
```

Judge that run (use `gpt-4o` if you want the published judge model):

```bash
uv run memory-bench judge \
  --experiment smoke-langmem \
  --judge-model gpt-4o-mini
```

`memory-bench` reads `.env` from the project root. The first `run` downloads the split into `~/.cache/memory-bench/`. Later runs reuse the cache.

Put `-v` before the subcommand if you want debug logs (`memory-bench -v run ...`). Debug logs can include conversation text.

To match the **published protocol** (not the 86.1% score): use `--split small`, no `--limit` (500 questions), and `model=gpt-4o` for both `run` and `judge`. That is expensive.

## Local Redis Agent Memory Server

This repo includes Compose for Redis 8 and the public AMS image.

```bash
cp .env.example .env   # OPENAI_API_KEY is required; AMS uses it for extraction
uv sync --extra redis-ams --group dev
docker compose up -d
```

Wait until `http://localhost:8000` is healthy, then:

```bash
uv run memory-bench run \
  --provider redis-ams \
  --dataset longmemeval \
  --split oracle \
  --limit 1 \
  --run-name smoke-redis-ams \
  --provider-param model=gpt-4o-mini
```

REST defaults to `http://localhost:8000`. MCP is a separate AMS mode; this Compose file only starts the HTTP API. See [docs/providers/redis-ams.md](docs/providers/redis-ams.md).

## How a run works

For each example:

1. Ingest prior sessions into the selected provider.
2. Wait until memories are listed (async extractors poll for up to 120 seconds).
3. Retrieve memory for the question and generate an answer.
4. For LongMemEval v1, run `memory-bench judge` (yes/no LLM judge, task-averaged accuracy).

Re-run the same command after a crash. Completed question ids are skipped.

Default answer and judge model is `gpt-4o`. Override with `--provider-param model=...` and `--judge-model`.

## Supported providers

Install extras from `pyproject.toml`. Pass `--provider` to `memory-bench run`.

| CLI id | Extra | Recipe |
|--------|-------|--------|
| `redis-ams` | `redis-ams` | [Redis AMS REST](docs/providers/redis-ams.md) |
| `redis-ams-mcp` | `redis-ams-mcp` | [Redis AMS MCP](docs/providers/redis-ams.md) |
| `mem0` | `mem0` | [Mem0](docs/providers/mem0.md) |
| `langmem` | `langmem` | [LangMem](docs/providers/langmem.md) |
| `zep` | `zep` | [Zep](docs/providers/zep.md) |
| `graphiti` | `graphiti` | [Graphiti](docs/providers/graphiti.md) |
| `supermemory` | `supermemory` | [Supermemory](docs/providers/supermemory.md) |
| `vertex-memory-bank` | `google` | [Google Vertex Memory Bank](docs/providers/google-vertex-memory-bank.md) |
| `bedrock-agentcore` | `aws` | [AWS Bedrock AgentCore](docs/providers/aws-bedrock-agentcore.md) |
| `oracle-agent-memory` | `oracle` | [Oracle Agent Memory](docs/providers/oracle-agent-memory.md) |

Index: [docs/providers/README.md](docs/providers/README.md).

Mastra OM and Emergence Fast are documented gaps, not CLI ids. Their Python samples reimplement strategy logic rather than wrap a public API, so that code is not in this tree.

This checkout does not claim that every cloud wrapper was live-tested.

## Datasets

### LongMemEval v1

Public files: [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). Upstream: [LongMemEval](https://github.com/xiaowu0162/LongMemEval).

Splits: `oracle`, `small`, `medium`. Default CLI split is `small`.

### LongMemEval v2

Public files: [`xiaowu0162/longmemeval-v2`](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2). The adapter is text-only. Screenshots are ignored.

Download once (set `LME_V2_DATA_ROOT` if you store files elsewhere):

```bash
mkdir -p data/longmemeval-v2/haystacks
base=https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/main
curl -L "$base/questions.jsonl" -o data/longmemeval-v2/questions.jsonl
curl -L "$base/trajectories.jsonl" -o data/longmemeval-v2/trajectories.jsonl
curl -L "$base/haystacks/lme_v2_small.json" -o data/longmemeval-v2/haystacks/lme_v2_small.json
curl -L "$base/haystacks/lme_v2_medium.json" -o data/longmemeval-v2/haystacks/lme_v2_medium.json
export LME_V2_DATA_ROOT="$PWD/data/longmemeval-v2"
```

```bash
uv run memory-bench run \
  --provider langmem \
  --dataset longmemeval-v2 \
  --tier small \
  --domain web \
  --limit 2 \
  --run-name smoke-lme-v2
```

This release does not score v2 with the v1 LLM judge. Official v2 evaluation uses different functions.

Repeat `--provider-param KEY=VALUE` for wrapper options. See each provider page.

## Outputs

Each run writes `experiment_results/<run-name>/`:

| File | Contents |
|------|----------|
| `answers.jsonl` | One JSON object per question |
| `judgments.jsonl` | Judge score and rationale |
| `metadata.json` | Provider, dataset, split, models, flags |
| `errors.jsonl` | Failed examples |
| `metrics.json` | Aggregates after a v1 judge run |

Do not commit `experiment_results/` or `*.jsonl`. Parameter names that look like keys, tokens, secrets, or passwords are redacted in metadata.

## Adapter extension

1. Implement `MemoryStore`: `ingest`, `query`, `list_memories`, `reset`.
2. Register the CLI id.
3. Add an optional extra in `pyproject.toml` if you need a vendor SDK.
4. Add `docs/providers/<id>.md`.

Do not copy private strategy modules into this tree.

## Costs and credentials

- A full Small v1 run (500 questions) plus ingest can be a large OpenAI bill. Start with `--limit` and `--split oracle`.
- Hosted memory APIs bill separately. Extraction tokens may not appear in usage fields.
- Keep keys in `.env`. Tear down cloud resources after a run.

## License

Apache License 2.0.
