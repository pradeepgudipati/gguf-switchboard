# Changelog

Release notes for each version live in [`releases/`](releases/) and on [GitHub Releases](https://github.com/pradeepgudipati/gguf-switchboard/releases). Edit the matching `releases/vX.Y.Z.md` before tagging; CI publishes that file to the release page.

## Unreleased

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
