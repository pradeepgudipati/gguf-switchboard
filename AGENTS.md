# GGUF Switchboard

## Project Identity

Repository: `gguf-switchboard`
Context Harbor Project: `GGUF-Switchboard`

This repository maps exclusively to the Context Harbor project `GGUF-Switchboard`.

Never retrieve or write Context Harbor knowledge under another project unless explicitly requested.

When using Context Harbor MCP tools, always scope operations to:

`GGUF-Switchboard`

## Context Sources

### Source Code — jbcontext

Use jbcontext for current gguf-switchboard implementation, including:

- source-code discovery
- repository structure (`src/`, `tests/`, `scripts/`)
- modules, structs, traits, and functions
- dependencies and call relationships
- locating existing behavior (model swap, eviction, routing, backend spawn)
- understanding affected implementation before a change

Prefer targeted jbcontext retrieval over manually scanning large portions of the repository.

### Documentation — Context Harbor MCP

When retrieving product documentation, architecture decisions, specifications,
historical decisions, or durable memory, ALWAYS scope requests to:

`GGUF-Switchboard`

**Product:** gguf-switchboard — a Rust, OpenAI/Anthropic-compatible GGUF model-swap proxy for `llama.cpp`'s `llama-server` (a `llama-swap` alternative).

Deeper architecture and behavior docs:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/COMPARISON.md`](docs/COMPARISON.md)
- [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md)
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- [`docs/USAGE.md`](docs/USAGE.md)
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)
- [`docs/adr/001-positioning-vs-llama-swap.md`](docs/adr/001-positioning-vs-llama-swap.md)

## Source of Truth

### Current Implementation

The repository source code is authoritative for what the system currently does.

Use jbcontext to discover and understand current implementation before making substantial changes.

### Intended Product Behavior

For intended behavior, consult in order:

1. Current approved specs and docs under `docs/`
2. Context Harbor project `GGUF-Switchboard`
3. Current source implementation

Context Harbor should be used for historical decisions, requirements, architecture rationale, and durable project memory (model lifecycle, VRAM/GPU-layer heuristics, multi-GPU direction, deployment/operational decisions).

### Conflicts

If source code, repository docs, and Context Harbor knowledge conflict:

1. Do not silently choose one.
2. Identify the conflicting sources.
3. Determine whether implementation or documentation is stale.
4. Ask for clarification when intended behavior cannot be established.
5. Save the resulting confirmed durable decision to Context Harbor project `GGUF-Switchboard`.

Never change implementation solely because an older document or memory contradicts current code.

## Engineering Workflow

### Routine Implementation

For small, well-defined changes:

1. Identify the affected subsystem (`src/backend`, `src/routes`/API, model registry, eviction, config).
2. Use jbcontext to locate and understand relevant implementation.
3. Inspect directly related tests under `tests/`.
4. Implement the smallest correct change.
5. Run targeted tests while developing (`cargo test <name>`).
6. Run the mandatory completion gate before declaring completion.

Do not query Context Harbor when current code and requirements are already unambiguous.

### Requirement-Driven Implementation

When implementation depends on product behavior, architecture, previous decisions, or specs:

1. Use jbcontext to understand current implementation.
2. Retrieve targeted knowledge from Context Harbor project `GGUF-Switchboard`.
3. Read relevant repository docs under `docs/` where applicable.
4. Compare intended behavior against current implementation.
5. Identify contradictions or ambiguity before implementing.
6. Implement the smallest coherent change.
7. Run targeted tests.
8. Run the mandatory completion gate.

### Bug Fixes

1. Start from observable failure evidence (logs, `tracing` output, failing test, repro request).
2. Use jbcontext to locate the execution path.
3. Establish root cause before editing.
4. Consult Context Harbor only when expected behavior is unclear.
5. Implement the smallest verified fix.
6. Add or update regression tests under `tests/` where appropriate.
7. Run the mandatory completion gate.

Avoid unrelated refactoring during bug fixes.

### Architecture Changes

Before significant architectural changes (e.g. multi-GPU support, new backend, storage changes):

1. Use jbcontext to understand current boundaries and dependencies.
2. Review relevant docs (`docs/ARCHITECTURE.md`, `docs/adr/`).
3. Search Context Harbor project `GGUF-Switchboard` for previous decisions and constraints.
4. Identify alternatives and trade-offs.
5. Do not implement a major architectural deviation without explicit approval.

After an architectural decision is accepted, save the durable decision to Context Harbor and, where warranted, add an ADR under `docs/adr/`.

## Commands

### Development

```bash
cargo build                       # debug build
cargo run                         # run the server (reads config.toml / models.toml)
cargo watch -x run                # optional hot-reload if cargo-watch is installed
```

### Build & Lint

```bash
cargo build --release             # optimized release build
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all                   # format
cargo fmt --all -- --check        # format check (CI)
```

### Tests

```bash
cargo test                        # unit + integration tests
cargo test <test_name>            # run a single test
cargo test --test integration     # run tests/integration.rs only
cargo test --test scheduler_switch
cargo test --test responses
```

### Scripts

```bash
./scripts/update-llama-cpp.sh          # build/install llama.cpp llama-server (Linux/NVIDIA)
./scripts/test-deploy-models.sh        # deploy smoke test against real models
./scripts/bench-vs-llama-swap.sh       # benchmark vs llama-swap
./scripts/install-hooks.sh             # install git hooks
./precommit.sh                         # local pre-commit checks
```

### Deployment

```bash
./deploy.sh                       # systemd install/upgrade (Linux)
docker compose up -d --build      # containerized deployment
```

## Mandatory Completion Gate

A gguf-switchboard implementation is NOT complete until the applicable verification checks pass.

For normal implementation work, the authoritative final verification sequence is:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

Individual commands may be used during development to shorten feedback loops, but the full sequence above remains the final completion gate. Run `./precommit.sh` if present/updated to mirror CI.

If the gate fails:

1. Identify the failing check.
2. Determine whether the failure was introduced by the current change.
3. Fix implementation-caused failures.
4. Re-run the relevant targeted check when useful.
5. Re-run the full gate.
6. Repeat until it passes.

Do not weaken tests, lints, or CI checks merely to make verification pass.
Do not report implementation as complete while an implementation-caused mandatory check is failing.

## Change-Specific Verification

### Model Swap / Scheduler Changes

For changes to model load/unload, single-slot swap, drain, or rollback logic:

- run `cargo test --test scheduler_switch`
- run `cargo test --test integration`
- manually verify a swap against a real `llama-server` when behavior around draining/rollback changes

### API Route Changes (OpenAI / Anthropic compatibility)

For changes to `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/messages`, `/v1/models*`:

- run `cargo test --test integration` and `cargo test --test responses`
- verify request/response contracts against `docs/USAGE.md` and Swagger UI (`/swagger-ui/`)
- verify streaming (SSE) behavior when touched

### Memory-Pressure / Eviction / Auto-GPU-Layer Changes

- verify behavior against `docs/CONFIGURATION.md` and `docs/COMPATIBILITY.md`
- add/extend targeted unit tests for the heuristic
- note any change to `vram_gb` / `auto_ngl` heuristics as a durable decision candidate in Context Harbor

### Deployment / Docker / systemd Changes

- run `docker compose build` where Docker is affected
- validate `deploy.sh` / `gguf-switchboard.service` changes on a Linux target when feasible
- run `./scripts/test-deploy-models.sh` when model discovery/deploy behavior is affected

## Architecture

### Core Model

gguf-switchboard is a **swap proxy**, not an inference engine. It spawns and manages a single `llama-server` child process at a time, hot-swapping between GGUF models on demand:

```
Client (OpenAI/Anthropic SDK)
   → gguf-switchboard (axum HTTP server)
       → routes: /v1/chat/completions, /v1/completions, /v1/embeddings,
                 /v1/responses, /v1/messages, /v1/models*, /v1/audio/*
       → scheduler: drain in-flight → unload current model → load requested model
       → backend: spawns/manages llama-server child process (localhost)
   ← forwarded/streamed response (SSE where applicable)
```

Full detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Key Subsystems

- **Registry / discovery** — scans GGUF directories (filename → header → metadata validation ladder), enriches from Hugging Face, persists `models.json`.
- **Scheduler** — single-slot hot-swap: drains, unloads, loads, rolls back on failed switch.
- **Backend** — spawns/supervises `llama-server` processes; OOM context fallback.
- **Eviction** — system memory-pressure monitor unloads the resident model past a critical threshold; idle-timeout priority-model warm-up.
- **API layer** — axum routes translating OpenAI Chat/Completions/Responses and Anthropic Messages APIs onto the loaded `llama-server`'s OpenAI-compatible backend; tool-calling translation for the Responses API.
- **Observability** — Prometheus `/metrics`, usage history `/v1/usage`, `tracing` structured logs.
- **Persistence** — `rusqlite` (bundled SQLite) for usage tracking; `models.toml` / `models.json` for registry.

### Config

- `config.toml` — server/runtime config (see `config.example.toml`, `config.docker.toml`).
- `models.toml` — model registry defaults (see `models.example.toml`, `models.docker.toml`).
- Full reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### Platform Notes

- Linux + NVIDIA/CUDA is the primary target (`deploy.sh`, systemd unit `gguf-switchboard.service`).
- macOS: build from source only, no systemd, Apple Metal via `llama.cpp`.
- Single GPU, single resident model — see `README.md` Status note before proposing multi-model-concurrent designs; treat that as an architecture change requiring approval.

## Memory Architecture — Mem0

gguf-switchboard integrates Mem0 (via Context Harbor's Mem0 MCP integration) as the durable memory layer for agent and project knowledge.

Memory is separate from document RAG (Context Harbor MCP docs) and from source code (jbcontext).

### MCP — Explicit Tools Only

Use **only** Context Harbor's memory MCP tools:

- `search_memory(project_name="GGUF-Switchboard", query="...")`
- `add_memory(project_name="GGUF-Switchboard", messages=[...])`
- `forget_memory(project_name="GGUF-Switchboard", memory_id="...")`

**Never** use another memory MCP or plugin (e.g. `plugin-mem0-mem0`, `mem0.ai`) for GGUF-Switchboard knowledge.

### Responsibilities

- **jbcontext** → current source-code understanding
- **Context Harbor MCP (docs/RAG)** → project documents, specs, ADRs, transcripts, and other indexed material under `docs/`
- **Mem0** → durable project memory: decisions, constraints, historical rationale, and knowledge that should survive across agent sessions

Do not use Mem0 as a replacement for document retrieval or current source-code inspection.

### Memory Scope

Memories MUST be scoped to the active Context Harbor project.

For this repository:

`GGUF-Switchboard`

Never write GGUF-Switchboard memories into another project namespace. When project identity is uncertain, resolve the project before reading or writing memory.

### Memory Retrieval

Use memory retrieval when previous decisions or durable project knowledge may affect the current task. Typical examples:

- VRAM allocation / `auto_ngl` heuristic decisions
- multi-GPU direction and constraints
- model lifecycle and eviction policy decisions
- API compatibility decisions (OpenAI vs Anthropic translation choices)
- deployment/operational constraints (systemd, Docker, `deploy.sh`)
- security/trust-boundary decisions (LAN-only assumption, auth)
- previously resolved ambiguities

Do not retrieve memory automatically for every routine coding operation.

For implementation tasks, prefer this order:

1. jbcontext for current source implementation
2. `docs/` for approved specs/ADRs
3. Context Harbor document RAG for other requirements/source material
4. Mem0 when historical decisions or durable knowledge are relevant

### Memory Writes — Mandatory

Memory capture is **mandatory**. A task is NOT complete until Mem0 has been reviewed for durable knowledge.

**Fixed end-of-task sequence:**

1. `search_memory(project_name="GGUF-Switchboard", query="<related decision>")` — check for existing memory.
2. If nothing durable already covers it: `add_memory(project_name="GGUF-Switchboard", messages=[...])` — record the decision. On 429 (rate limit), retry once after a brief pause.
3. If durable memory is explicitly not required: state `Durable memory: NOT REQUIRED`.
4. Always report durable memory status in the completion report.

### Parent Only — Subagent Restriction

Only the top-level/parent session writes memory. A delegated subagent must **not** call `add_memory` or `forget_memory`. If a subagent surfaces a durable decision, the parent must handle the memory write.

### Good Candidates for Memory

- accepted architecture decisions (e.g. positioning vs llama-swap, single-slot vs multi-slot)
- confirmed product requirements
- API contract decisions (OpenAI/Anthropic compatibility choices)
- VRAM/GPU-layer heuristic decisions
- eviction/memory-pressure policy decisions
- security/trust-boundary decisions
- deployment constraints
- decisions resolving conflicting requirements or specs

A durable decision should capture: decision, rationale, significant alternatives, consequences/constraints, and relevant subsystem.

Do not store: temporary debugging observations, command output, generated code, test results, speculative ideas, routine implementation details, or anything already clearly represented by current source code or `docs/`.

### Memory Deduplication

Before storing a significant decision:

1. Search existing memory for related decisions.
2. Avoid creating substantially duplicate memories.
3. Update or supersede existing knowledge when supported.
4. Preserve meaningful historical rationale when a previous decision is replaced.

### Memory Conflicts

When Mem0 memory conflicts with current source code or approved docs:

- current source code remains authoritative for current implementation
- approved docs represent intended documented behavior
- Mem0 represents historical project knowledge and rationale

Do not silently treat an older memory as current truth. Identify the conflict, establish which source is stale, and once resolved, store the confirmed durable outcome in memory.

## Code Style

- Rust 2024 edition, `rustfmt` defaults.
- `clippy::all = "warn"` is enforced (see `Cargo.toml`); treat warnings as must-fix before completion.
- Prefer `thiserror` for error types, `tracing` for structured logging (already project convention).
- Keep changes scoped to the affected subsystem; avoid drive-by refactors outside the task.

## Completion Report

When completing implementation work, report:

- Change implemented: concise summary
- Targeted tests: PASS / FAIL / NOT APPLICABLE
- Full completion gate (`fmt` + `clippy` + `test`): PASS / FAIL
- Scheduler/swap verification: PASS / NOT APPLICABLE
- API compatibility verification: PASS / NOT APPLICABLE
- Deployment verification: PASS / NOT APPLICABLE
- Durable decision recorded (Mem0, project `GGUF-Switchboard`): YES / NO / NOT REQUIRED
- Remaining issues: none or list them

Never claim completion when an implementation-caused mandatory verification step is failing.

**Durable memory lines are mandatory** — cannot be omitted after meaningful work. Always state whether durable memory was reviewed and what was stored (or that it wasn't required).
