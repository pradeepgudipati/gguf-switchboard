# Changelog

Release notes for each version live in [`releases/`](releases/) and on [GitHub Releases](https://github.com/pradeepgudipati/gguf-switchboard/releases). Edit the matching `releases/vX.Y.Z.md` before tagging; CI publishes that file to the release page.

## Unreleased


## [v0.1.8](https://github.com/pradeepgudipati/gguf-switchboard/releases/tag/v0.1.8) — 2026-09-03

- **Adoption-ready documentation** — reorganized the documentation into focused getting-started, runtime, model, API, deployment, troubleshooting, and client-integration guides, with `llms.txt` and `llms-full.txt` for AI-assisted discovery.
- **Branded API consoles** — added project branding, correctly embedded logo assets, icons, and clearer navigation across Swagger UI and the Conformance Console.
- **Community workflows** — added contribution guidance, structured issue templates, and a pull-request checklist.
- **Safer deployment updates** — `deploy.sh` now merges stray `models.toml` registries into the canonical deployed registry and skips unnecessary binary installation and service restarts after no-change pulls.
- **Clearer supported scope** — refreshed product positioning around GGUF through llama.cpp and SafeTensors through vLLM, added dedicated integration guides, and removed the unsupported Docker deployment files.
- **CLI discoverability** — added `ggs version`, `--version`, and `-V`, plus reorganized help and copyable examples.


## [v0.1.6](https://github.com/pradeepgudipati/gguf-switchboard/releases/tag/v0.1.6) — 2026-08-31

- **Dual vLLM and llama.cpp backends** — registry-driven routing supports safetensors models through vLLM and GGUF models through llama.cpp under the same alias, with hardware-fit planning, model-source reporting, and safer search/pull behavior.
- **Default dual-backend installation** — `deploy.sh` installs llama.cpp and vLLM by default, validates that vLLM models have a usable source, and preserves aligned service and registry configuration.
- **Reranker support** — reranker discovery, metadata, lifecycle handling, and the OpenAI-compatible `POST /v1/rerank` endpoint work across supported backends.
- **Tool/template conformance console** — new Swagger UI diagnostics and `/v1/conformance/*` endpoints inspect structured tool calls, template rendering, fixed conformance batteries, and side-by-side model behavior.
- **Model-specific runtime controls** — `chat_template_file`, `reasoning_format`, and a minimum `ctx` floor can be configured per model.
- **Operations and observability** — `ggs stop` / `ggs restart`, model-aware NVIDIA monitoring, improved embedding admission and VRAM profiles, clearer CLI help, and refreshed deployment smoke tests.
- **CI and release hardening** — pinned Rust release toolchain, cross-compilation fixes, Woodpecker Linux release support, PR-only hosted checks, and robust release checksum generation.

## [v0.1.5](https://github.com/pradeepgudipati/gguf-switchboard/releases/tag/v0.1.5) — 2026-08-20

- **Tool-call capability verification** — `tools`-tagged models are now probed with a real tool call at load time instead of trusting the static HF-derived tag; a failed probe is tracked per model and tool-bearing requests to it get a clear `400` instead of silently forwarding malformed output. Verdict exposed as `tools_verified` on `/v1/models`. Fixes a known llama.cpp defect where `tool_calls[].function.arguments` is returned as a JSON object instead of a string.
- **`models search` FIT/SPEED/PRECISION scoring** — replaces the binary Supported Yes/No column with continuous VRAM-fit, memory-bandwidth speed, and quality-loss precision scores, plus a BALANCED quant recommendation at the size midpoint. See `docs/QUANT_SCORING.md`.
- **Faster model switches** — new `switch_strategy` (default `unload_first`) stops the resident `llama-server` *before* starting the requested one. Previously the new model was started while the old one still held the GPU, so `auto_ngl` / the fit planner planned against leftover VRAM and `llama-server` OOM'd into the context/`-ngl` fallback ladder (several spawns per switch, then a partially CPU-resident model). `load_first` restores the old behaviour for multi-GPU boxes. Failed switches still roll back by re-loading the previous model.
- Hardened `unload_first`: unload failures are now logged/retried instead of swallowed, and a drain wait confirms VRAM is actually reclaimed before the next model loads.
- Optional `prewarm_recent_models` keeps recently used GGUFs in the page cache so switching back is served from RAM.
- `llama-server` stdout is no longer piped-but-unread (could stall a noisy server mid-load); `--version` is cached per binary instead of re-run on every load.
- **Switch/load observability** — per-model `gguf_switchboard_model_switch_seconds`, `..._switch_phase_seconds` (drain / unload_previous / plan / spawn_to_healthy / rollback), `..._model_load_seconds{result}`, `..._model_load_attempts_total`, `..._model_last_load_seconds`, `..._request_model_wait_seconds`, `..._request_model_hit_total`, `..._model_switches_total{from,to,trigger}`, `..._model_unloads_total{reason}`, `..._loaded_model_info{model}`; `/status` exposes `last_switch`. Load/inference histograms now have buckets beyond 10 s.
- `gguf_switchboard_inference_latency_seconds` no longer includes model load time and, for streams, is observed when the stream ends rather than when headers are sent. `active_requests` / `streaming_requests` gauges no longer leak when a request fails before the backend call.
- **Embeddings fixes** — per-model `batch_size`/`ubatch_size` auto-configure `llama-server -b`/`-ub`; oversized requests split into multiple batches instead of failing; input cardinality no longer flattened on multi-input requests.
- `deploy.sh` shallow-clones llama.cpp on first install.

## [v0.1.3](https://github.com/pradeepgudipati/gguf-switchboard/releases/tag/v0.1.3) — 2026-08-01

- **Model management CLI** — `models search`, `models files`, `models pull` for one-command HF GGUF discovery, download, validation, and registry
- **Anthropic Messages API** — `POST /v1/messages` (stream + non-stream) translated onto llama-server
- **Responses API function tools** — strict streaming events, function tool/call translation, fragmented argument reassembly
- **Auto GPU layers** — opt-in `auto_ngl` picks `-ngl` from free VRAM + GGUF size
- **Swagger UI** — hides incompatible endpoints by model kind; rich model cards from HF metadata
- **HF metadata enrichment** — automatic sync on launch/refresh; `sync-hf-metadata` CLI
- **Embeddings fixes** — `--embeddings` flag, null-safe encoding_format, omitted null optionals
- **Scheduler reliability** — load-then-unload rollback, OOM context fallback, request draining
- Scheduler integration tests, CI hardening, `docs/COMPATIBILITY.md`

## [v0.1.2](https://github.com/pradeepgudipati/gguf-switchboard/releases/tag/v0.1.2) — 2026-07-10

- Portable `models.toml` / `models.json` registry with `discover-models` and context sizing heuristic (`vram_gb`)
- README repositioned as a llama-swap alternative; prebuilt Linux install from `main`
- Safer default context (`16384`); Swagger UI payload fixes

## [v0.1.1](https://github.com/pradeepgudipati/gguf-switchboard/releases/tag/v0.1.1) — 2026-07-09

- First tagged release as **gguf-switchboard** with prebuilt Linux/macOS binaries
- OpenAI-compatible GGUF swap proxy, usage tracking, Swagger UI, `deploy.sh` systemd install
