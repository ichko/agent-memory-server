# Agent Memory Benchmark

This directory is a shared **LongMemEval** v1 harness. You ingest sessions into a memory product, retrieve for a test question, generate an answer, and score that answer with an LLM judge.

It is one of the two projects in this repository. The other is [`V0/`](../V0/), the open-source Agent Memory Server. See the [repository README](../README.md) for how they fit together.

Each wrapper talks to a vendor API and uses that vendor's default extraction, so a run measures the product as shipped.

## The protocol

This harness follows the LongMemEval setup described in [Building and evaluating long-term conversational memory](https://redis.github.io/redis-ai-research-public/longmemeval-agent-memory/):

| Item | Value |
|------|--------|
| Dataset | LongMemEval **Small** |
| Size | **500** questions |
| Coverage | **six** task types |
| Models | **gpt-4o** for answers and judging |
| Metric | Task-averaged accuracy |

Your score depends on the provider, the models, and the split you choose. Compare runs only when those match.

## Quick start

You need Python 3.10–3.12, [uv](https://docs.astral.sh/uv/), and an OpenAI key.

```bash
cp .env.example .env    # set OPENAI_API_KEY
uv sync --extra langmem --group dev
uv run memory-bench providers
```

The cheapest smoke run uses LongMemEval **oracle** (short haystacks) and one question. LangMem needs no extra server.

```bash
uv run memory-bench run \
  --provider langmem \
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

`memory-bench` reads `.env` from the working directory or a parent, so keep yours in `agent-memory-benchmark/`. The first `run` downloads the split into `~/.cache/memory-bench/`. Later runs reuse the cache.

Put `-v` before the subcommand if you want debug logs (`memory-bench -v run ...`). Debug logs can include conversation text.

For the full protocol, use `--split small`, no `--limit` (500 questions), and `model=gpt-4o` for both `run` and `judge`. That is expensive.

## How a run works

For each example:

1. Ingest prior sessions into the selected provider.
2. Wait until memories are listed (async extractors poll for up to 120 seconds).
3. Retrieve memory for the question and generate an answer.
4. Run `memory-bench judge` (yes/no LLM judge, task-averaged accuracy).

Re-run the same command after a crash. Completed question ids are skipped.

Default answer and judge model is `gpt-4o`. Override with `--provider-param model=...` and `--judge-model`.

## Supported providers

Install extras from `pyproject.toml`. Pass `--provider` to `memory-bench run`.

| CLI id | Extra | Recipe |
|--------|-------|--------|
| `mem0` | `mem0` | [Mem0](docs/providers/mem0.md) |
| `langmem` | `langmem` | [LangMem](docs/providers/langmem.md) |
| `zep` | `zep` | [Zep](docs/providers/zep.md) |
| `graphiti` | `graphiti` | [Graphiti](docs/providers/graphiti.md) |
| `supermemory` | `supermemory` | [Supermemory](docs/providers/supermemory.md) |
| `vertex-memory-bank` | `google` | [Google Vertex Memory Bank](docs/providers/google-vertex-memory-bank.md) |
| `bedrock-agentcore` | `aws` | [AWS Bedrock AgentCore](docs/providers/aws-bedrock-agentcore.md) |
| `oracle-agent-memory` | `oracle` | [Oracle Agent Memory](docs/providers/oracle-agent-memory.md) |

Index: [docs/providers/README.md](docs/providers/README.md).

Mastra OM and Emergence Fast have no CLI id. Neither ships a public Python provider API to wrap. See [Mastra OM](docs/providers/mastra-om.md) and [Emergence](docs/providers/emergence.md).

## Dataset

Public files: [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). Upstream: [LongMemEval](https://github.com/xiaowu0162/LongMemEval).

Splits: `oracle`, `small`, `medium`. Default CLI split is `small`.

Repeat `--provider-param KEY=VALUE` for wrapper options. See each provider page.

## Outputs

Each run writes `experiment_results/<run-name>/`:

| File | Contents |
|------|----------|
| `answers.jsonl` | One JSON object per question |
| `judgments.jsonl` | Judge score and rationale |
| `metadata.json` | Provider, dataset, split, models, flags |
| `errors.jsonl` | Failed examples |
| `metrics.json` | Aggregates after a judge run |

Provider parameters whose names look like keys, tokens, secrets, or passwords are redacted in `metadata.json`.

## Adapter extension

1. Implement `MemoryStore` in `src/agent_memory_benchmark/memory/<id>_store.py`.
2. Register the CLI id in `memory/__init__.py`.
3. Add an optional extra in `pyproject.toml` if you need a vendor SDK.
4. Add `docs/providers/<id>.md`.

## Costs

Prices below use OpenAI list rates as of August 2026: gpt-4o at $2.50 / $10 per million input / output tokens, gpt-5.6-luna at $0.20 / $1.20. They cover LLM calls from this harness, not hosted-memory invoices.

The Small split is 500 questions and about 61 million tokens of haystack chat. Oracle is the same 500 questions with about 3 million tokens of haystack.

| Split | What you pay for | gpt-4o | gpt-5.6-luna |
|-------|------------------|--------|--------------|
| Small | Answer + judge only | ~$5 | ~$0.50 |
| Small | Extract every session, then answer + judge | ~$200 | ~$20 |
| Oracle | Extract every session, then answer + judge | ~$15 | ~$1.50 |

The middle row is the LangMem path: each session goes through the same chat model you pass as `model=`. Providers that extract on their own side sit closer to the first row here plus their own bill.

`--limit 1 --split oracle` is cents. Medium is an order of magnitude above Small. Managed cloud resources keep billing until you delete them; each provider page has the teardown steps.

## License

Apache License 2.0.
