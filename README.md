# GGUF Switchboard

![GGUF Switchboard](banner.png)

**One API. Any GGUF Model. Seamless local LLM switching.**

A lightweight OpenAI-compatible API server that loads, manages, and switches between GGUF models on a single GPU. Point any OpenAI SDK or tool at it (Python, Node, Cursor, Cline, Continue) — no manual process or port juggling.

A **[llama-swap](https://github.com/mostlygeek/llama-swap) alternative in Rust** with system memory-pressure eviction, OOM-only context fallback, Swagger UI, and built-in usage tracking.

**Requires** [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` and GGUF model files — see [Prerequisites](#prerequisites).

> **Status:** Experimental — single-GPU home labs and development machines on a **trusted LAN**. One model loaded at a time. System RAM is monitored for pressure eviction; `vram_gb` sizes context heuristically. Opt-in `auto_ngl` can pick GPU layers from free VRAM (nvidia-smi or `vram_gb` fallback) — still a heuristic, not live layer telemetry. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

![gguf-switchboard demo](gguf-switchboard-demo.gif)

<sub>[▶ Watch with audio](demo.mp4)</sub>

## Features

- **Fast & Lightweight** — Minimal overhead, maximum performance
- **Hot-Swap Models** — Switch between GGUF models on the fly
- **Open & Extensible** — Modular, easy to extend, community-driven
- **100% Local** — Your models. Your data. Your machine.

Also included:

- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/models`, `/v1/models/registry.json`, `/v1/audio/*`
- **Tool calling** — `tools` / `tool_choice` / `tool_calls` are accepted and forwarded to `llama-server`; actual behavior depends on the model and llama.cpp build (see [COMPATIBILITY](docs/COMPATIBILITY.md))
- **Swagger UI** — Try-it-out at `http://localhost:9090/swagger-ui/` (live model dropdown from the registry)
- **Auto-discovery** — Scans GGUF dirs with a cheap validation ladder (filename → header → metadata); sidecars skipped
- **Single-slot hot-swap** — One resident model; switches drain in-flight requests; failed switches roll back
- **Memory-pressure eviction** — Unloads when system RAM crosses the critical threshold
- **Auto GPU layers (`auto_ngl`)** — Opt-in: at load, pick `-ngl` from free VRAM + GGUF size (manual `ngl` / `extra_args` still win)
- **Idle priority model** — Preferred model auto-loads after a configurable idle timeout
- **llama.cpp backend** — Spawns and manages `llama-server` child processes
- **SSE streaming**, **Prometheus** (`/metrics`), **usage history** (`/v1/usage`), **portable `models.json`**

### How it works

```
Models (GGUF)                    OpenAI endpoints
 Mistral / Llama / Phi / …   →   /v1/chat/completions
         ↓                       /v1/completions
   gguf-switchboard  ─────────▶  /v1/embeddings
   (single-slot swap)            /v1/responses, /v1/audio/*
```

Request for model `B` while `A` is loaded → drain → unload `A` → load `B` → forward. After `idle_timeout`, the priority model warms back up. Details in [Architecture](docs/ARCHITECTURE.md).

## Why gguf-switchboard

When running local LLMs you usually juggle `llama-server` processes, ports, and GPU memory by hand. gguf-switchboard is a **llama-swap-style swap proxy in Rust** for constrained GPUs: memory-pressure eviction, OOM context fallback, idle priority model, and usage tracking — one OpenAI endpoint, llama.cpp only.

Full landscape table and vs llama-swap feature matrix: **[docs/COMPARISON.md](docs/COMPARISON.md)**.

## Quick Start

### Prerequisites

gguf-switchboard is a **swap proxy** — it does not run inference itself. You must install **[llama.cpp](https://github.com/ggerganov/llama.cpp)** (`llama-server`) and have GGUF models on disk before models will load.

| Requirement | Notes |
|-------------|--------|
| **`llama-server` (required)** | From [llama.cpp](https://github.com/ggerganov/llama.cpp). Must be on `PATH` or set via `defaults.llama_server` in `models.toml`. |
| **GGUF model files** | Directory of `.gguf` weights (default scan: `~/models`, or set `MODELS_DIR`). |
| **Linux** (recommended) | Ubuntu/Debian for `deploy.sh` (`apt`). Other distros: install build deps yourself. |
| **macOS** | Build from source only — no systemd. See [macOS](#macos). |
| **Rust** | Installed automatically by `deploy.sh` if missing; otherwise [rustup](https://rustup.rs/). |
| **GPU stack** | NVIDIA + CUDA toolkit on Linux, or Apple Metal on macOS (CPU-only llama.cpp works but is slow). |

#### Install llama.cpp (`llama-server`)

Build from [llama.cpp](https://github.com/ggerganov/llama.cpp) and put `llama-server` on your `PATH` (commonly `/usr/local/bin`).

**Linux (NVIDIA / CUDA):**

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j"$(nproc)"
sudo cp build/bin/llama-server /usr/local/bin/
llama-server --version
```

**macOS (Metal):**

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"
sudo cp build/bin/llama-server /usr/local/bin/
llama-server --version
```

If the binary is not on `PATH`, point the registry at it:

```toml
# models.toml
[defaults]
llama_server = "/path/to/llama-server"
```

`deploy.sh` does **not** install llama.cpp — only gguf-switchboard, its config, and the systemd unit. Without `llama-server`, the proxy starts but model loads fail.

#### Install GGUF models

```bash
mkdir -p ~/models
# Download any GGUF (Hugging Face, etc.) into ~/models
# Example layout:
#   ~/models/Qwen3.5-9B-Q4_K_M.gguf
#   ~/models/gemma-4-E4B-it-Q4_K_M.gguf
```

Then run `./deploy.sh` (first install) or `./deploy.sh --refresh-models` so discovery registers them.

### Fresh machine (Linux + systemd)

Clone, review `deploy.sh`, then run it. The script builds from source and installs a systemd service — no remote pipe-to-bash.

```bash
# Before first deploy (required):
#   1. Install llama-server  (see Prerequisites above)
#   2. Put GGUFs in ~/models  (or export MODELS_DIR=/path/to/ggufs)

git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
./deploy.sh
```

What `deploy.sh` does:

1. Pulls latest `main` (stashes dirty working tree first — see [Updating](#updating))
2. Installs build deps + Rust if needed
3. Builds the release binary → `/usr/local/bin/gguf-switchboard`
4. Uses **`config.toml` / `models.toml` in the repo checkout** (override with `GGUF_SWITCHBOARD_CONFIG_DIR`)
5. Auto-generates `models.toml` on first install; syncs gitignored `models.local.toml` / `models.local.json`
6. Enables and starts the systemd service on `0.0.0.0:9090`

```bash
# Custom GGUF directory on first install
MODELS_DIR=/path/to/gguf-files ./deploy.sh

# Config outside the repo (optional)
GGUF_SWITCHBOARD_CONFIG_DIR=/etc/gguf-switchboard ./deploy.sh
```

Then open **http://localhost:9090/swagger-ui/**.

### Updating

From the existing checkout, re-run deploy. That is the supported upgrade path:

```bash
cd ~/gguf-switchboard   # or wherever you cloned
./deploy.sh
```

| Goal | Command |
|------|---------|
| Pull + rebuild + restart | `./deploy.sh` |
| Rebuild only (no `git pull`) | `./deploy.sh --skip-pull` |
| Pick up new GGUF files (merge registry) | `./deploy.sh --refresh-models` |
| Restart without rebuild | `sudo systemctl restart gguf-switchboard` |

**Important:**

- Deploy **stashes uncommitted changes** (including untracked files) before `git pull`. Recover with `git stash list` / `git stash pop`.
- Your live registry stays in `models.toml` (or `GGUF_SWITCHBOARD_CONFIG_DIR`). Gitignored `models.local.*` copies avoid `git pull` conflicts — edit `models.toml`, not the template in a way that fights upstream.
- After editing aliases / `priority` / `extra_args`, restart: `sudo systemctl restart gguf-switchboard`.

```bash
# Logs
sudo systemctl status gguf-switchboard
sudo journalctl -u gguf-switchboard -f
```

### Prebuilt binary (Linux)

```bash
# amd64 (see Releases for arm64 + checksums)
curl -fsSL -o gguf-switchboard \
  https://github.com/pradeepgudipati/gguf-switchboard/releases/latest/download/gguf-switchboard-linux-amd64
chmod +x gguf-switchboard
sudo mv gguf-switchboard /usr/local/bin/

# Still need config templates from the repo
git clone --branch main --depth 1 https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
gguf-switchboard discover-models ~/models -o models.toml
gguf-switchboard config.toml
```

### Build without systemd

Local binary only — no `sudo`, no systemd (Linux or macOS):

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
cargo build --release

./target/release/gguf-switchboard discover-models ~/models -o models.toml
./target/release/gguf-switchboard config.toml
```

### macOS

`deploy.sh` is **Linux-only** (systemd). On a Mac, use [Build without systemd](#build-without-systemd).

| Step | Linux (`deploy.sh`) | macOS |
|------|---------------------|-------|
| Clone + `cargo build` | Yes | Yes |
| Model discovery | Yes | Yes |
| systemd auto-start | Yes | No — terminal or your own `launchd` plist |
| Auto-install build deps | Yes (`apt`) | Xcode CLI tools; `jq` via Homebrew if needed |

Use a **Metal** build of `llama-server`. Keep `config.toml` / `models.toml` in the checkout.

### Verify

```bash
curl -s http://localhost:9090/health
curl -s http://localhost:9090/status | jq .
curl -s http://localhost:9090/v1/models | jq '.data[].id'
```

### Troubleshooting first install

| Symptom | Likely fix |
|---------|------------|
| Service unhealthy / no models | Put GGUFs in `~/models` (or set `MODELS_DIR`) and run `./deploy.sh --refresh-models` |
| `llama-server: not found` when loading | Install llama.cpp server; put it on `PATH` or set `defaults.llama_server` in `models.toml` |
| Empty `/v1/models` | Check `models_dir` in `models.toml`; enable `auto_discover = true`; restart |
| Deploy "lost" my edits | `git stash list` — deploy stashes dirty trees before pull |
| Port 9090 in use | Change `bind` in `config.toml` and restart |

## Further documentation

| Doc | Contents |
|-----|----------|
| **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** | `config.toml`, `models.toml`, discovery, context sizing, CLI |
| **[docs/USAGE.md](docs/USAGE.md)** | API examples, SDKs, IDE setup, monitoring, local run |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Scheduler/backend overview and project layout |
| **[docs/COMPARISON.md](docs/COMPARISON.md)** | Landscape vs Ollama / llama-swap / others |
| **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** | Throughput, swap latency, bench script |
| **[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)** | OpenAI endpoint coverage |

### Configuration (short)

Two files: **`config.toml`** (bind, idle timeout, `vram_gb`) and **`models.toml`** (aliases → GGUF paths). Defaults live in the repo checkout after `deploy.sh`. Full reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

```bash
# After install, tweak models then restart
sudo systemctl restart gguf-switchboard
```

### Try the API

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"YOUR_ALIAS","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

Swagger UI: **http://localhost:9090/swagger-ui/** — more examples in [docs/USAGE.md](docs/USAGE.md).

## License

MIT
