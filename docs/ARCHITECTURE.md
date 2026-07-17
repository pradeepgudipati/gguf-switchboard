# Architecture

> [← Back to README](../README.md)

Scheduler, llama.cpp backend, and repository layout.

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
└──────────┘    └───────────────┘    └──────────────┘
```

**Scheduler** is the core component:
1. Request arrives for model `X`
2. If `X` is loaded → forward immediately
3. If model `Y` is loaded → unload `Y` → load `X` → wait for health → forward
4. After `idle_timeout` seconds with no requests, the priority model auto-loads

**Backend** (llama.cpp implementation):
- Spawns `llama-server` as a child process using the configured `command` + `args`
- Polls the health endpoint until healthy or timeout
- Proxies all OpenAI-compatible HTTP requests to the backend URL
- Parses SSE streams and re-emits them with proper framing

## Project Structure

```
.
├── Cargo.toml              # Dependencies and build config
├── config.toml             # Server configuration (bind, vram_gb, models_file)
├── config.docker.toml      # Docker server configuration
├── models.toml             # Model registry template (aliases → GGUF files)
├── models.docker.toml      # Docker model registry
├── models.local.toml       # Deploy-synced registry copy (gitignored)
├── models.local.json       # Portable registry export (gitignored)
├── deploy.sh               # Build, install, discover models, sync registry
├── banner.png              # README hero banner
├── docs/                   # Configuration, usage, architecture, comparison
├── CHANGELOG.md            # Version index (details in releases/)
├── releases/               # Per-tag release notes (published to GitHub Releases by CI)
├── gguf-switchboard.service  # Systemd unit file
├── swagger-ui-overrides/   # Swagger UI customizations (model picker, editable payloads)
├── .github/workflows/
│   └── ci.yml              # CI: check, clippy, build, test; publishes releases/ notes on version tags
└── src/
    ├── main.rs             # Entry point; discover-models / export-registry CLI
    ├── config/
    │   ├── mod.rs          # config.toml loading (vram_gb, models_file)
    │   └── models_registry.rs  # models.toml/json registry, context sizing heuristic
    ├── errors/mod.rs       # OpenAI-compatible error responses
    ├── types/              # Request/response type definitions
    │   ├── mod.rs          # Shared types (ModelInfo, Usage, etc.)
    │   ├── chat.rs         # Chat completion types
    │   ├── completions.rs  # Text completion types
    │   ├── embeddings.rs   # Embedding types
    │   ├── models.rs       # Model permission types
    │   └── responses.rs    # Responses API types
    ├── backend/
    │   ├── mod.rs          # Backend trait definition
    │   └── llama_cpp.rs    # llama.cpp backend implementation
    ├── scheduler/mod.rs    # Single-slot swapping, priority model, memory watcher
    ├── state/mod.rs        # Shared application state
    ├── memory/mod.rs       # System memory pressure monitoring
    ├── db/mod.rs           # Token usage tracking (SQLite)
    ├── proxy/mod.rs        # SSE proxy helpers
    ├── metrics/mod.rs      # Prometheus metric collectors
    └── api/
        ├── mod.rs          # Router setup
        ├── chat.rs         # POST /v1/chat/completions
        ├── completions.rs  # POST /v1/completions
        ├── embeddings.rs   # POST /v1/embeddings
        ├── models.rs       # GET /v1/models, /v1/models/registry.json
        ├── responses.rs    # POST /v1/responses
        ├── audio.rs        # POST /v1/audio/*
        ├── health.rs       # GET /health, /status
        ├── metrics.rs      # GET /metrics
        └── usage.rs        # GET /v1/usage
```
