# Emergence Simple Fast compatibility status

The public reference implementation is available at
[EmergenceAI/emergence_simple_fast](https://github.com/EmergenceAI/emergence_simple_fast).

The benchmark source used during research contained a local Python
reimplementation of its retrieval and prompting strategy, not a wrapper around
a public provider API. That strategy implementation is outside this
repository's publication boundary.

The package keeps a compatibility class that raises an actionable error, but it
is not registered as a runnable benchmark provider. A future adapter should
call a supported public service or package interface without copying strategy
logic into this repository.
