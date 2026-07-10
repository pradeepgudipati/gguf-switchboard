# Changelog

Release notes for each version live in [`releases/`](releases/) and on [GitHub Releases](https://github.com/pradeepgudipati/gguf-switchboard/releases). Edit the matching `releases/vX.Y.Z.md` before tagging; CI publishes that file to the release page.

## Unreleased

- Honest docs: experimental home-lab positioning, memory-pressure (system RAM) terminology, single-slot swapping
- Scheduler: load-then-unload with rollback on failed switches
- OOM-only context fallback via stderr classification
- Per-model request draining before model switches (`switch_drain_timeout_secs`)
- Priority watcher respects active requests, user switches, and failure cooldown
- Cancellable background watchers on shutdown
- `llama_server_version` on `/health` and `/status`
- Scheduler integration tests with fake llama-server fixture
- CI: least-privilege permissions, SHA-pinned actions
- `docs/COMPATIBILITY.md` and `docs/adr/001-positioning-vs-llama-swap.md`

## [v0.1.2](https://github.com/pradeepgudipati/gguf-switchboard/releases/tag/v0.1.2) — 2026-07-10

- Portable `models.toml` / `models.json` registry with `discover-models` and context sizing heuristic (`vram_gb`)
- README repositioned as a llama-swap alternative; prebuilt Linux install from `main`
- Safer default context (`16384`); Swagger UI payload fixes

## [v0.1.1](https://github.com/pradeepgudipati/gguf-switchboard/releases/tag/v0.1.1) — 2026-07-09

- First tagged release as **gguf-switchboard** with prebuilt Linux/macOS binaries
- OpenAI-compatible GGUF swap proxy, usage tracking, Swagger UI, `deploy.sh` systemd install
