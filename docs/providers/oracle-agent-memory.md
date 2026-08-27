# Oracle Agent Memory

Enterprise memory SDK (`oracleagentmemory`) on Oracle AI Database. Hybrid search plus LLM extraction.

Docs:

- [Oracle Help Center](https://docs.oracle.com/en/database/oracle/agent-memory/)
- [Get started](https://docs.oracle.com/en/database/oracle/agent-memory/26.6/guide/get-started.html)
- Product page: [oracle.com/database/ai-agent-memory](https://www.oracle.com/database/ai-agent-memory/)

CLI id: `oracle-agent-memory`. Extra: `oracle`.

## Install

```bash
uv sync --extra oracle
```

You need an Oracle AI Database (or Oracle Database 23ai Free) with the privileges the SDK documents. Local Free edition is enough for a smoke test if you accept disk and tablespace limits.

## Auth

```bash
OPENAI_API_KEY=replace-me
ORACLE_MEMORY_DB_USER=dmuser
ORACLE_MEMORY_DB_PASSWORD=REPLACE_ME
ORACLE_MEMORY_DB_CONNECT_STRING=localhost:1521/FREEPDB1
```

## Provision

1. Create a PDB and user as in Oracle Agent Memory get-started.
2. Grant the SDK the create-table and vector privileges listed there.
3. Confirm `sqlplus` or `oracledb` can connect with the DSN above.

Do not use a production PDB. The SDK manages its schema according to its
documented defaults.

## Run

```bash
uv run memory-bench run \
  --provider oracle-agent-memory \
  --dataset longmemeval \
  --split oracle \
  --limit 1 \
  --run-name smoke-oracle \
  --provider-param model=gpt-4o \
  --provider-param max_search_results=20

uv run memory-bench judge --experiment smoke-oracle --judge-model gpt-4o
```

## Cleanup

The wrapper calls the SDK's user deletion operation during reset. Inspect the
database after interrupted runs and remove benchmark users or drop the
dedicated PDB when finished.

Stop the database container if you used one.

## Notes

- The SDK is synchronous. The wrapper runs it in worker threads.
- Oracle Free has a small tablespace cap. Full Small-split ingest may need a larger database.
- This recipe is from public Oracle docs and SDK env vars. It is not a claim that Oracle Agent Memory was live-tested in this checkout.
