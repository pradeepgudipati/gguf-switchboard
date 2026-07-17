# Comparison

> [← Back to README](../README.md)

How gguf-switchboard fits next to Ollama, llama-swap, and related tools.

## Why existing tools fall short

When running local LLMs you usually manage models by hand: start `llama-server` for one model, stop it, start another. Multiple tools (Cursor, Continue, embeddings) means juggling processes, ports, and GPU memory yourself.

### Landscape comparison

**[llama-swap](https://github.com/mostlygeek/llama-swap)** is the closest mature alternative for on-demand model swapping. gguf-switchboard targets the same problem — single endpoint, swap on request — and adds memory-pressure eviction, context reduction on OOM, and usage history. Trade-off: llama.cpp only (no multi-backend swap matrix or web dashboard). See [vs llama-swap](#vs-llama-swap) below for a feature-by-feature table.

| Tool | OpenAI API | Loads by model name | Auto unload/load | GGUF support | Memory-pressure scheduling | Worth using? |
|------|:----------:|:-------------------:|:----------------:|:------------:|:---------------------:|--------------|
| **Ollama** | Yes | Yes | Yes — `keep_alive` unloads idle models; not memory-pressure aware | Yes | No | Easy drop-in; auto-unload is TTL-based, not scheduler-driven |
| **llama.cpp** (`llama-server`) | Yes | Manual — one process per model | No — you manage start/stop | Yes | No | Low-level; you own process and port management |
| **[llama-swap](https://github.com/mostlygeek/llama-swap)** | Yes | Yes | Yes — swaps on request (TTL-based) | Yes | No | **Mature swap proxy** — multi-backend, swap matrix, web dashboard; no memory-pressure eviction, context reduction on OOM, or `/v1/usage` history |
| **vLLM** | Yes | Yes (pre-loaded) | No — serves loaded models; not on-demand swap | No (HuggingFace weights) | Partial — memory-efficient serving | Datacenter / multi-tenant throughput; not GGUF or per-request model lifecycle |
| **LocalAI** | Yes | Yes | Partial — not memory-pressure aware | Yes | No | Full-stack alternative; not designed for proactive eviction under memory pressure |
| **Open WebUI** | Yes (proxy) | Via backend | Depends on backend | Via backend | Via backend | UI layer — not a model scheduler |
| **LiteLLM** | Yes | Routes only | No — does not load models | Via providers | No | API router, not a model loader |
| **gguf-switchboard** (this project) | Yes | Yes | Yes — single-slot swap + idle priority model | Yes | Yes (system RAM) | **llama-swap alternative (Rust)** — memory eviction, OOM context fallback, Swagger UI, usage tracking; llama.cpp-only |

### The gaps in existing tools

- **llama-swap** is the closest single-binary alternative — download, configure, run, done. It swaps llama-server processes on request. What it lacks: no system memory pressure monitoring, no context-size reduction on OOM, no idle timeout / priority model, no usage tracking.
- **Ollama** unloads idle models via `keep_alive`, but you do not get explicit swap-on-request scheduling or memory-pressure eviction — models can still sit resident longer than you expect on tight GPUs.
- **LocalAI** is a full-stack alternative supporting many backends and formats. It is not designed for proactive eviction when GPU memory is under pressure; you still manage capacity yourself.
- **LiteLLM** is an excellent API gateway for routing, fallbacks, and retries across cloud and local providers. It does not spawn backends, load GGUF weights, or manage GPU memory — that is not its job.

**The core gap:** a tool that treats **model loading as a scheduling problem** on constrained local GPU hardware, not just an API compatibility layer.

### What makes gguf-switchboard different

gguf-switchboard is a **llama-swap-style swap proxy in Rust**, extended for constrained GPUs with:
- **Memory-pressure eviction** — monitors system RAM and unloads when thresholds are crossed
- **Automatic context-size reduction** — on OOM, reduces context and retries (OOM-only fallback)
- **Idle priority model** — keeps your preferred model warm automatically
- **Built-in usage tracking** — `/v1/usage` history and Prometheus metrics
- **Single OpenAI endpoint** — no port juggling, no process management

Your tools never manage processes or ports — they just point at `http://localhost:9090/v1` and pick a model name from the config.

### Separation of concerns

This project does one job well: **local GPU scheduling and model lifecycle**.

| Layer | Responsibility |
|-------|----------------|
| **[OmniRoute](https://github.com/diegosouzapw/OmniRoute)** / **[LiteLLM](https://www.litellm.ai/)** | Provider routing, fallbacks, retries across cloud and local endpoints |
| **gguf-switchboard** (this project) | Local GPU scheduling, process lifecycle, model loading/unloading |
| **Client tools** (Cursor, Cline, Codex, Open WebUI, etc.) | Talk to gguf-switchboard as a normal OpenAI server — no special integration needed |

Point your IDE or agent at `http://localhost:9090/v1`, set a model name from your config, and requests flow through the scheduler automatically.

## vs llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) (Go) is the closest comparable project — a single-binary proxy that swaps `llama-server` processes on demand. gguf-switchboard (Rust) solves the same core problem but the two have diverged on feature scope:

| Feature | llama-swap | gguf-switchboard |
|---|---|---|
| Backends supported | Any OpenAI-compatible server (llama.cpp, vllm, tabbyAPI, stable-diffusion.cpp, ...) | llama.cpp only |
| Concurrent models | Yes — custom "swap matrix" DSL / groups run multiple models at once | No — one model loaded at a time |
| System memory-pressure eviction | No — unload is TTL-based only | Yes — monitors system RAM and unloads at critical threshold |
| context-size reduction on OOM | No | Yes — auto-retries only on OOM-class failures at a lower `-c` |
| Persistent usage tracking | Live dashboard metrics (not a queryable history API) | Yes — `/v1/usage` backed by SQLite, queryable per-model |
| Web dashboard | Yes — playground, live token metrics, request/response inspection, live log streaming | No — Swagger UI only (API docs, not a monitoring dashboard) |
| API key / auth | Yes — `apiKeys` config restricts endpoint access | No — no auth on any endpoint |
| Model aliasing | Yes — `aliases` map friendly names to real model ids | No |
| Request filtering | Yes — `filters`/`stripParams`/`setParams` rewrite requests per model | No |
| Startup preload | Yes — explicit `hooks` | Only incidental, via the idle-timeout priority-model watcher (~30s after boot, not deterministic) |
| Custom stop command | Yes — `cmdStop` (e.g. graceful Docker/Podman stop) | No |
| Other API surfaces | Anthropic `/messages`, image gen (SDAPI), reranking, infilling | OpenAI-only: chat, completions, embeddings, responses, audio |
| Remote CLI log streaming | Yes | No — stdout JSON logs only |

Both are thin proxies in front of `llama-server`. To measure proxy overhead on your hardware, see [Benchmarks](BENCHMARKS.md).
