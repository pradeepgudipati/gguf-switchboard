# ADR 001: Positioning vs llama-swap

## Status

Accepted

## Context

[llama-swap](https://github.com/mostlygeek/llama-swap) is a mature Go proxy that swaps OpenAI-compatible backends on demand. gguf-switchboard solves the same user problem for home-lab setups that standardize on **llama.cpp / GGUF**.

## Decision

Build a **Rust, llama.cpp-only** swap proxy focused on:

- Single-slot model swapping on constrained single-GPU machines
- System memory-pressure unloading (not live GPU VRAM telemetry in v0.1.x)
- OOM-class context reduction on failed loads
- Idle priority-model return
- SQLite `/v1/usage` history

Deliberately **not** competing on: multi-backend matrices, web dashboard, API keys, request rewriting, or Anthropic/SDAPI surfaces.

## Consequences

- Credibility requires honest docs (experimental, LAN-trusted, compatibility matrix).
- Scheduler correctness (rollback, drain, OOM classification) matters more than feature breadth.
- Users needing multi-backend or concurrent models should prefer llama-swap.
