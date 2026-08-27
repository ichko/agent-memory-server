# Mastra Observational Memory compatibility status

Mastra's public Observational Memory implementation is TypeScript:

- [Observational Memory documentation](https://mastra.ai/docs/memory/observational-memory)
- [API reference](https://mastra.ai/reference/memory/observational-memory)
- [Mastra source](https://github.com/mastra-ai/mastra)

The benchmark source used during research contained a Python reimplementation
of the strategy, not a wrapper around a public provider API. Publishing that
implementation would violate this repository's provider-wrapper boundary.

The package keeps a compatibility class that raises an actionable error, but it
is not registered as a runnable benchmark provider. A future adapter should
invoke the public Mastra TypeScript package through a documented transport and
must not reproduce its strategy inside this Python package.
