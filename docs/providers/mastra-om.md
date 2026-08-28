# Mastra Observational Memory compatibility status

Mastra's public Observational Memory implementation is TypeScript:

- [Observational Memory documentation](https://mastra.ai/docs/memory/observational-memory)
- [API reference](https://mastra.ai/reference/memory/observational-memory)
- [Mastra source](https://github.com/mastra-ai/mastra)

There is no public Python client, so this Python harness has no API to wrap.

The package keeps a compatibility class that raises an actionable error, but it
is not registered as a runnable benchmark provider. An adapter should call the
public Mastra TypeScript package over a documented transport, such as a small
HTTP service you run beside the benchmark.
