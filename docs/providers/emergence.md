# Emergence Simple Fast compatibility status

The public reference implementation is available at
[EmergenceAI/emergence_simple_fast](https://github.com/EmergenceAI/emergence_simple_fast).

It is a reference retrieval pipeline, not a hosted service or a client SDK, so
there is nothing for a provider wrapper to call.

The package keeps a compatibility class that raises an actionable error, but it
is not registered as a runnable benchmark provider. An adapter becomes possible
once there is a supported service or package interface to call.
