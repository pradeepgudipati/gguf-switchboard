# Architecture

> [← Back to README](../README.md)

Scheduler, llama.cpp backend, ModelFitPlanner, and repository layout.

## Overview

```
Client Request
     │
     ▼
┌──────────┐    ┌───────────────┐    ┌──────────────┐
│  Axum    │───▶│   Scheduler   │───▶│   Backend    │
│  Router  │    │               │    │  (llama.cpp) │
│          │    │ • swap slot   │    │              │
│ /v1/...  │    │ • Load lock   │    │ • child proc │
│ /health  │    │ • Priority    │    │ • health ck  │
│ /metrics │    │   watcher     │    │ • HTTP proxy │
└──────────┘    └───────┬───────┘    └──────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ ModelFitPlan  │
                │   ner         │
                │               │
                │ • VRAM probe  │
                │ • Context/nGL │
                │   planning    │
                │ • Bounded     │
                │   fallback    │
                │   ladder      │
                └───────────────┘
```

**Scheduler** is the core component:
1. Request arrives for model `X`
2. If `X` is loaded → forward immediately
3. If model `Y` is loaded → drain `Y` → unload `Y` (frees VRAM) → load `X` → wait for health → forward; if `X` fails, `Y` is re-loaded (`switch_strategy = "load_first"` instead loads `X` next to `Y` and needs VRAM for both)
4. After `idle_timeout` seconds with no requests, the priority model auto-loads

**ModelFitPlanner** (opt-in, `[fit]` section in `config.toml`):
- Before every model load, inspects GPU topology / free VRAM, model metadata, and requested context
- Produces a safe launch profile (context size, nGL, split mode, KV cache type)
- On OOM, advances through a bounded degradation sequence instead of blindly retrying:
  1. Requested context + default KV + auto-fit GPU
  2. Requested context + Q8 KV + auto-fit GPU
  3. 75% context + Q8 KV + auto-fit GPU
  4. 50% context + Q8 KV + auto-fit GPU
  5. 25% context + Q8 KV + reduced GPU offload
- Caches known-good profiles to `model-profiles.json` so subsequent loads skip the fallback ladder entirely

**Backend** (llama.cpp implementation):
- Spawns `llama-server` as a child process using the configured `command` + `args`
- Polls the health endpoint until healthy or timeout
- Proxies all OpenAI-compatible HTTP requests to the backend URL
- Parses SSE streams and re-emits them with proper framing
- Probes tool-call capability at load time (`tools_verified` on model info)
- Auto-injects `--chat-template llama3` for Llama 3.1 models unless overridden

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirects to Swagger UI |
| `/health` | GET | Liveness probe |
| `/status` | GET | Detailed status with `last_switch` timing |
| `/metrics` | GET | Prometheus metrics |
| `/v1/models` | GET | List configured models (enabled only) |
| `/v1/models/{model_id}` | GET | Individual model info with `tools_verified` |
| `/v1/models/{model_id}/runtime` | GET | Runtime profile (context, nGL, split, cache types) |
| `/v1/models/registry.json` | GET | Portable registry JSON export |
| `/v1/models/refresh` | POST | Hot-reload model registry from disk |
| `/v1/chat/completions` | POST | OpenAI Chat Completions (stream + non-stream) |
| `/v1/completions` | POST | OpenAI Text Completions (stream + non-stream) |
| `/v1/embeddings` | POST | OpenAI Embeddings |
| `/v1/responses` | POST | OpenAI Responses API (function tools, streaming) |
| `/v1/messages` | POST | Anthropic Messages API (stream + non-stream, tool calling) |
| `/v1/audio/transcriptions` | POST | Audio transcription (proxied to backend) |
| `/v1/audio/speech` | POST | Text-to-speech (proxied to backend) |
| `/v1/usage` | GET | Token usage history (SQLite) |
| `/v1/usage/recent` | GET | Recent usage summary |
| `/api-docs/openapi.json` | GET | OpenAPI spec (live model dropdown injection) |
| `/swagger-ui/` | GET | Swagger UI with model picker |

## Project Structure

```
.
├── Cargo.toml              # Dependencies and build config
├── config.example.toml     # Tracked default for runtime config.toml
├── config.docker.toml      # Docker server configuration
├── models.example.toml     # Tracked default for runtime models.toml
├── models.docker.toml      # Docker model registry
├── deploy.sh               # Build, install, discover models
├── banner.png              # README hero banner
├── docs/                   # Configuration, usage, architecture, comparison
├── CHANGELOG.md            # Version index (details in releases/)
├── releases/               # Per-tag release notes (published to GitHub Releases by CI)
├── gguf-switchboard.service  # Systemd unit file
├── swagger-ui-overrides/   # Swagger UI customizations (model picker, editable payloads)
├── .github/workflows/
│   └── ci.yml              # CI: check, clippy, build, test; publishes releases/ notes on version tags
└── src/
    ├── main.rs             # Entry point; discover-models / export-registry / sync-hf-metadata / models CLI
    ├── lib.rs              # Library root
    ├── config/
    │   ├── mod.rs          # Config loading (vram_gb, models_file, fit, switch_strategy, etc.)
    │   ├── models_registry.rs  # models.toml/json registry, context sizing heuristic, rescan/merge
    │   ├── models_cmd.rs   # CLI handlers for models search/files/pull
    │   ├── hf_download.rs  # Hugging Face download with aria2c/native fallback
    │   └── hf_sync.rs      # HF metadata enrichment (sync-hf-metadata)
    ├── errors/mod.rs       # OpenAI-compatible error responses
    ├── types/              # Request/response type definitions
    │   ├── mod.rs          # Shared types (ModelInfo, Usage, RuntimeProfileInfo, etc.)
    │   ├── chat.rs         # Chat completion types (tools, content parts)
    │   ├── completions.rs  # Text completion types
    │   ├── embeddings.rs   # Embedding types
    │   ├── models.rs       # Model permission types
    │   ├── responses.rs    # Responses API types (function tools, streaming events)
    │   ├── audio.rs        # TranscriptionRequest, SpeechRequest
    │   └── anthropic.rs    # Anthropic Messages API types + OpenAI conversion
    ├── backend/
    │   ├── mod.rs          # Backend trait definition
    │   ├── llama_cpp.rs    # llama.cpp backend implementation
    │   └── tool_probe.rs   # Tool-call capability verification at load time
    ├── scheduler/mod.rs    # Single-slot swapping, priority model, memory watcher, prewarm
    ├── fit.rs              # ModelFitPlanner — hardware-aware load planning with bounded fallback ladder
    ├── fit_profile.rs      # Known-good profile persistence (model-profiles.json)
    ├── gpu.rs              # nvidia-smi VRAM probing (GpuDeviceInfo, probe_all_gpus)
    ├── ngl.rs              # GPU layer count helpers, auto_ngl, split-mode
    ├── context.rs          # Context size helpers (context_for_attempt, next_lower_context)
    ├── batch.rs            # Batch/ubatch size helpers for embedding models
    ├── kind_guard.rs       # Reject API calls when model kind doesn't match endpoint
    ├── load_failure.rs     # Classify load failures (OOM-weights, OOM-kv, port-conflict, missing-file)
    ├── sanitize.rs         # Strip Swagger UI placeholder values before forwarding
    ├── quant_profile.rs    # FIT/SPEED/PRECISION scoring for quants
    ├── state/mod.rs        # Shared application state (AppState, rescan watcher)
    ├── memory/mod.rs       # System memory pressure monitoring
    ├── db/mod.rs           # Token usage tracking (SQLite)
    ├── proxy/mod.rs        # SSE proxy helpers (GuardedStream, proxy_sse_response)
    ├── metrics/mod.rs      # Prometheus metric collectors
    ├── openapi_models.rs   # Live model enum injection into OpenAPI spec
    └── api/
        ├── mod.rs          # Router setup, OpenAPI doc generation
        ├── chat.rs         # POST /v1/chat/completions
        ├── completions.rs  # POST /v1/completions
        ├── embeddings.rs   # POST /v1/embeddings
        ├── models.rs       # GET /v1/models, /v1/models/{id}, /v1/models/{id}/runtime, /v1/models/refresh, /v1/models/registry.json
        ├── responses.rs    # POST /v1/responses
        ├── audio.rs        # POST /v1/audio/transcriptions, /v1/audio/speech
        ├── anthropic.rs    # POST /v1/messages (Anthropic Messages API)
        ├── health.rs       # GET /health, /status
        ├── metrics.rs      # GET /metrics
        └── usage.rs        # GET /v1/usage, /v1/usage/recent
```

## Kind Guard

Every inbound request is checked against the model's `kind` field (`kind_guard.rs`):

| Endpoint family | Allowed kinds |
|-----------------|---------------|
| `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` | `chat`, `coder`, `vision` |
| `/v1/embeddings` | `embedding` |

Mismatched requests return a `400` with a clear error message indicating the model's actual kind.

## Load Failure Classification

`load_failure.rs` classifies `llama-server` startup failures from stderr to drive the fallback ladder:

| Class | Detection | Response |
|-------|-----------|----------|
| OOM-weights | `out of memory` / `cannot allocate` during weight loading | Reduce context size, retry |
| OOM-kv-cache | `kv_cache` / `alloc` failure after weights loaded | Reduce context, try Q8 KV cache |
| Port conflict | `address already in use` | Retry with next port |
| Missing file | `No such file` / `failed to open` | Fail immediately (no retry) |
| Other | Unrecognized stderr pattern | Fail immediately |
