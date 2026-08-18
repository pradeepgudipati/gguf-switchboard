# Changelog

Release notes for each version live in [`releases/`](releases/) and on [GitHub Releases](https://github.com/pradeepgudipati/gguf-switchboard/releases). Edit the matching `releases/vX.Y.Z.md` before tagging; CI publishes that file to the release page.

## Unreleased

- **Faster model switches** — new `switch_strategy` (default `unload_first`) stops the resident `llama-server` *before* starting the requested one. Previously the new model was started while the old one still held the GPU, so `auto_ngl` / the fit planner planned against leftover VRAM and `llama-server` OOM'd into the context/`-ngl` fallback ladder (several spawns per switch, then a partially CPU-resident model). `load_first` restores the old behaviour for multi-GPU boxes. Failed switches still roll back by re-loading the previous model.
- Optional `prewarm_recent_models` keeps recently used GGUFs in the page cache so switching back is served from RAM.
- `llama-server` stdout is no longer piped-but-unread (could stall a noisy server mid-load); `--version` is cached per binary instead of re-run on every load.
- **Switch/load observability** — per-model `gguf_switchboard_model_switch_seconds`, `..._switch_phase_seconds` (drain / unload_previous / plan / spawn_to_healthy / rollback), `..._model_load_seconds{result}`, `..._model_load_attempts_total`, `..._model_last_load_seconds`, `..._request_model_wait_seconds`, `..._request_model_hit_total`, `..._model_switches_total{from,to,trigger}`, `..._model_unloads_total{reason}`, `..._loaded_model_info{model}`; `/status` exposes `last_switch`. Load/inference histograms now have buckets beyond 10 s.
- `gguf_switchboard_inference_latency_seconds` no longer includes model load time and, for streams, is observed when the stream ends rather than when headers are sent. `active_requests` / `streaming_requests` gauges no longer leak when a request fails before the backend call.

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
