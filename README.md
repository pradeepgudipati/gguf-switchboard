# GGUF Switchboard

![GGUF Switchboard](banner.png)

**One API. Any GGUF Model. Seamless local LLM switching.**

A lightweight OpenAI and Anthropic-compatible API server that loads, manages, and switches between GGUF models on a single GPU. Point any OpenAI or Anthropic SDK or tool at it (Python, Node, Cursor, Cline, Continue) — no manual process or port juggling.

A **[llama-swap](https://github.com/mostlygeek/llama-swap) alternative in Rust** with system memory-pressure eviction, OOM-only context fallback, Swagger UI, Hugging Face metadata enrichment, and built-in usage tracking.

**Requires** [llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server` and GGUF model files — on Linux NVIDIA hosts install with `./scripts/update-llama-cpp.sh` (see [Quick Start](#quick-start)).

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
- **Anthropic Messages API** — `POST /v1/messages` (stream + non-stream); translated onto the loaded `llama-server` OpenAI backend
- **Tool calling** — Chat Completions forwards `tools` / `tool_choice` / `tool_calls`; the Responses API translates function tools, function calls, and strict streaming events to/from `llama-server`. Actual model behavior depends on the model and llama.cpp build (see [COMPATIBILITY](docs/COMPATIBILITY.md))
- **Swagger UI** — Try-it-out at `http://localhost:9090/swagger-ui/` (live model dropdown, Rescan Models, hides chat vs embedding endpoints by selected model kind)
- **Auto-discovery** — Scans GGUF dirs with a cheap validation ladder (filename → header → metadata); sidecars skipped
- **Live model rescan** — `POST /v1/models/refresh` plus a configurable daily watcher (`models_rescan_interval_secs`); merges new GGUFs without a full redeploy
- **HF metadata enrichment** — fills empty `description` / context / VRAM / `capabilities` / `hf_repo` from Hugging Face on launch and rescan (`sync-hf-metadata` CLI also available)
- **Model management** — `models search`, `models files`, and `models pull` for one-command GGUF discovery, download, validation, and registry from Hugging Face
- **Kind-aware routing** — chat / completions / messages / responses require chat-like kinds; embeddings require `embedding` (and pass `--embeddings` to `llama-server`)
- **Single-slot hot-swap** — One resident model; switches drain in-flight requests; failed switches roll back
- **Memory-pressure eviction** — Unloads when system RAM crosses the critical threshold
- **Auto GPU layers (`auto_ngl`)** — Opt-in: at load, pick `-ngl` from free VRAM + GGUF size (manual `ngl` / `extra_args` still win)
- **Idle priority model** — Preferred model auto-loads after a configurable idle timeout
- **llama.cpp backend** — Spawns and manages `llama-server` child processes
- **SSE streaming**, **Prometheus** (`/metrics`), **usage history** (`/v1/usage`), **portable `models.json`**

### How it works

```
Models (GGUF)                    API endpoints
 Mistral / Llama / Phi / …   →   /v1/chat/completions
         ↓                       /v1/completions
   gguf-switchboard  ─────────▶  /v1/embeddings
   (single-slot swap)            /v1/responses, /v1/audio/*
                                 /v1/messages (Anthropic)
```

Request for model `B` while `A` is loaded → drain → unload `A` → load `B` → forward. After `idle_timeout`, the priority model warms back up. Details in [Architecture](docs/ARCHITECTURE.md).

## Why gguf-switchboard

When running local LLMs you usually juggle `llama-server` processes, ports, and GPU memory by hand. gguf-switchboard is a **llama-swap-style swap proxy in Rust** for constrained GPUs: memory-pressure eviction, OOM context fallback, idle priority model, HF-enriched registry metadata, and usage tracking — one OpenAI/Anthropic endpoint, llama.cpp only.

Full landscape table and vs llama-swap feature matrix: **[docs/COMPARISON.md](docs/COMPARISON.md)**.

## Quick Start

### Prerequisites

gguf-switchboard is a **swap proxy** — it does not run inference itself. You need a working **[llama.cpp](https://github.com/ggerganov/llama.cpp)** `llama-server` and GGUF models on disk before the systemd service will stay enabled.

| Requirement | Notes |
|-------------|--------|
| **`llama-server` (required)** | From [llama.cpp](https://github.com/ggerganov/llama.cpp). On Linux NVIDIA hosts prefer `./scripts/update-llama-cpp.sh` → `/usr/local/bin/llama-server`. Otherwise put it on `PATH` or set `defaults.llama_server` in `models.toml`. |
| **GGUF model files** | Directory of `.gguf` weights (default scan: `~/models`, or set `MODELS_DIR`). |
| **Linux** (recommended) | Ubuntu/Debian for `deploy.sh` (`apt`). Other distros: install build deps yourself. |
| **macOS** | Build from source only — no systemd. See [macOS](#macos). |
| **Rust** | Installed automatically by `deploy.sh` if missing; otherwise [rustup](https://rustup.rs/). |
| **GPU stack** | NVIDIA + CUDA toolkit on Linux, or Apple Metal on macOS (CPU-only llama.cpp works but is slow). |

### Install (Linux + NVIDIA / CUDA)

Clone the repo, install a CUDA `llama-server` into `/usr/local`, then deploy the switchboard service:

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard

# 1) Build + install llama.cpp with CUDA (idempotent upgrade path)
#    Run as your user; the script sudo's only for install / service steps.
./scripts/update-llama-cpp.sh

# 2) Build + install gguf-switchboard and enable the systemd unit
./deploy.sh
```

`./scripts/update-llama-cpp.sh` does: check CUDA → clone/pull `~/llama.cpp` → CUDA Release build → verify GPU → stop `gguf-switchboard` if present → `cmake --install` to `/usr/local` → strip stale RUNPATH → `ldconfig` → assert libs are not resolved from the source build tree → restart the unit only if it exists.

Overrides: `LLAMA_DIR` (default `~/llama.cpp`), `PREFIX` (default `/usr/local`), `SERVICE` (default `gguf-switchboard`), `SKIP_PULL=1`, `SKIP_SERVICE=1`.

`deploy.sh` does **not** install llama.cpp. Before starting the unit it checks for a working `llama-server` (`--version`). If missing or broken, it still installs the switchboard binary, leaves the service disabled, and prints the `./scripts/update-llama-cpp.sh` help. The same gate applies when no GGUF models are registered yet.

What `deploy.sh` does when ready:

1. Pulls latest `main` (stashes dirty working tree first — see [Updating](#updating))
2. Installs build deps + Rust if needed
3. Builds the release binary → `/usr/local/bin/gguf-switchboard`
4. Creates **user-owned, gitignored `config.toml` / `models.toml`** in the repo checkout (override with `GGUF_SWITCHBOARD_CONFIG_DIR`)
5. Creates `~/models` by default, or the single directory supplied through `MODELS_DIR`
6. Detects `llama-server` from `PATH` / common install locations and records it in `models.toml`
7. Generates `models.toml` on first install
8. Verifies `llama-server` and model candidates, then enables and starts the systemd service on `0.0.0.0:9090`

```bash
# Custom GGUF directory on first install
MODELS_DIR=/path/to/gguf-files ./deploy.sh

# Config outside the repo (optional)
GGUF_SWITCHBOARD_CONFIG_DIR=/etc/gguf-switchboard ./deploy.sh
```

Then open **http://localhost:9090/swagger-ui/**.

If the binary is not on `PATH`, point the registry at it:

```toml
# models.toml
[defaults]
llama_server = "/path/to/llama-server"
```

#### Manual llama.cpp (optional)

Prefer `./scripts/update-llama-cpp.sh` on CUDA hosts — it installs shared libs correctly and strips RUNPATH. Manual one-shot (binary copy only):

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j"$(nproc)"
sudo cp build/bin/llama-server /usr/local/bin/
llama-server --version
```

**macOS (Metal)** — no systemd; build `llama-server` yourself, then use [Build without systemd](#build-without-systemd):

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"
sudo cp build/bin/llama-server /usr/local/bin/
llama-server --version
```

### Prebuilt binary (Linux)

Still needs a working `llama-server` first (`./scripts/update-llama-cpp.sh` on CUDA hosts).

```bash
# amd64 (see Releases for arm64 + checksums)
curl -fsSL -o gguf-switchboard \
  https://github.com/pradeepgudipati/gguf-switchboard/releases/latest/download/gguf-switchboard-linux-amd64
chmod +x gguf-switchboard
sudo mv gguf-switchboard /usr/local/bin/

# Copy the tracked examples to user-owned runtime files
git clone --branch main --depth 1 https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
cp config.example.toml config.toml
cp models.example.toml models.toml
gguf-switchboard discover-models ~/models -o models.toml
gguf-switchboard config.toml
```

### Build without systemd

Local binary only — no `sudo`, no systemd (Linux or macOS):

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
cargo build --release

cp config.example.toml config.toml
cp models.example.toml models.toml
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

Use a **Metal** build of `llama-server`. Create runtime files from `config.example.toml` / `models.example.toml`; keep user-owned `config.toml` / `models.toml` in the checkout.

### Install GGUF models

With gguf-switchboard installed, search, browse, and download GGUF models from Hugging Face:

```bash
mkdir -p ~/models

# Search for models
gguf-switchboard models search "Qwen3.5 9B"

# Browse available files in a repo
gguf-switchboard models files lmstudio-community/Qwen3.5-9B-GGUF

# Download, validate, and register a model
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --dir ~/models

# Optional: tune parallel aria2 connections (default 8, maximum 16)
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --connections 8
```

Public downloads automatically use `aria2c` when available, then verify the expected size, Hugging Face LFS checksum, and GGUF metadata before registration. If Hugging Face rejects parallel range requests, the native downloader resumes the partial file. Authenticated downloads using `HF_TOKEN`, or systems without `aria2c`, use the native downloader directly. A successful pull refreshes a running gguf-switchboard server automatically.

Or download manually — any `.gguf` file in `~/models` works:

```bash
# Example layout after manual download:
#   ~/models/Qwen3.5-9B-Q4_K_M.gguf
#   ~/models/gemma-4-E4B-it-Q4_K_M.gguf
```

If you downloaded models manually, run `./deploy.sh --refresh-models` so discovery registers them.

### Verify

```bash
curl -s http://localhost:9090/health
curl -s http://localhost:9090/status | jq .
curl -s http://localhost:9090/v1/models | jq '.data[].id'
```

### Updating

Supported upgrade path from an existing checkout:

```bash
cd ~/gguf-switchboard   # or wherever you cloned

# Refresh CUDA llama-server first when the backend changed upstream
./scripts/update-llama-cpp.sh

# Then rebuild / reinstall the switchboard service
./deploy.sh
```

| Goal | Command |
|------|---------|
| Pull + rebuild + restart switchboard | `./deploy.sh` |
| Rebuild only (no `git pull`) | `./deploy.sh --skip-pull` |
| Install / refresh CUDA `llama-server` in `/usr/local` | `./scripts/update-llama-cpp.sh` |
| Pick up new GGUF files (merge registry) | `./deploy.sh --refresh-models` |
| Live rescan while running | `curl -X POST http://localhost:9090/v1/models/refresh` (or Swagger **Rescan Models**) |
| Restart without rebuild | `sudo systemctl restart gguf-switchboard` |

**Important:**

- Deploy **stashes uncommitted changes** (including untracked files) before `git pull`. Recover with `git stash list` / `git stash pop`.
- Your live `config.toml`, `models.toml`, and `models.json` are gitignored and preserved across deploys. Tracked defaults live in `config.example.toml` and `models.example.toml`.
- After editing aliases / `priority` / `extra_args`, restart: `sudo systemctl restart gguf-switchboard`.
- `deploy.sh` will not start the unit if `llama-server` is missing/broken or no GGUF models are registered — fix with `./scripts/update-llama-cpp.sh` and/or model pull, then re-run deploy.

```bash
# Logs
sudo systemctl status gguf-switchboard
sudo journalctl -u gguf-switchboard -f
```

### Shell alias (optional)

Add a short `ggs` alias so you can type `ggs` instead of `gguf-switchboard`. `deploy.sh` offers to add this automatically on Linux without conflicting with the common `gs='git status'` alias.

**Linux (bash):**

```bash
echo "alias ggs='gguf-switchboard'" >> ~/.bashrc
source ~/.bashrc
```

**Linux / macOS (zsh):**

```bash
echo "alias ggs='gguf-switchboard'" >> ~/.zshrc
source ~/.zshrc
```

**Windows (PowerShell):**

```powershell
# Add to your PowerShell profile
if (!(Test-Path $PROFILE)) { New-Item -Path $PROFILE -Force }
Add-Content $PROFILE "Set-Alias -Name ggs -Value gguf-switchboard"
. $PROFILE
```

After that:

```bash
ggs models search "Qwen3.5"
ggs models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M
ggs config.toml
```

### Troubleshooting first install

| Symptom | Likely fix |
|---------|------------|
| Deploy leaves service disabled; prints `./scripts/update-llama-cpp.sh` | Install or refresh CUDA `llama-server`: `./scripts/update-llama-cpp.sh`, then `./deploy.sh` |
| `llama-server: not found` / models fail to load | Same as above, or set `defaults.llama_server` in `models.toml` to a working binary |
| Service unhealthy / no models | Put GGUFs in `~/models` (or set `MODELS_DIR`) and run `./deploy.sh --refresh-models` |
| First install has no GGUF models | Not fatal. Installer leaves the service stopped and prints `ggs models search`, `ggs models pull`, and `./deploy.sh --refresh-models` |
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

Two runtime files: **`config.toml`** (bind, idle timeout, `vram_gb`) and **`models.toml`** (aliases → GGUF paths). `deploy.sh` creates these gitignored files from the tracked `.example.toml` defaults when required. Full reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

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
