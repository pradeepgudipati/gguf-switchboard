# GGUF Switchboard

![GGUF Switchboard](banner.png)

**Run and switch between local GGUF and Safetensors models through one OpenAI-compatible API.**

GGUF Switchboard uses [llama.cpp](https://github.com/ggml-org/llama.cpp) for GGUF models and [vLLM](https://github.com/vllm-project/vllm) for Safetensors models. It downloads models from Hugging Face, evaluates what fits the available hardware, and safely swaps one model at a time without manual process, port, or VRAM management.

> **The local model router for llama.cpp and vLLM.**

> **Status:** Experimental. Built for single-GPU development machines and trusted home-lab deployments, not internet-facing multi-tenant serving.

## Top features

- **Two model formats, one API:** GGUF through llama.cpp and Hugging Face Safetensors through vLLM.
- **Automatic model lifecycle:** request-driven single-slot switching, in-flight request draining, failed-switch rollback, and idle priority warm-up.
- **Hardware-aware model management:** Hugging Face search and pull, GGUF quant scoring, vLLM quantization detection, VRAM fit checks, and bounded llama.cpp OOM fallback.
- **Broad client compatibility:** OpenAI Chat Completions, Completions, Embeddings, Responses, Rerank and Audio APIs, plus Anthropic Messages.
- **Built for operation:** Swagger UI with live GPU/CPU status, Prometheus metrics, usage history, a tool-calling conformance console with persisted run history, memory-pressure eviction, model rescans, and capability probing.

## Installation

Linux with NVIDIA/CUDA is the primary deployment target. The default installer sets up CUDA llama.cpp, an isolated vLLM environment, gguf-switchboard, and its systemd service:

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
./deploy.sh
```

Download and register either model format from Hugging Face:

```bash
# GGUF served by llama.cpp
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF \
  --quant Q4_K_M \
  --dir /var/lib/gguf-switchboard/models \
  --registry /opt/gguf-switchboard/models.toml

# Safetensors served by vLLM
gguf-switchboard models pull vllm Qwen/Qwen2.5-7B-Instruct \
  --dir /var/lib/gguf-switchboard/vllm-models \
  --registry /opt/gguf-switchboard/models.toml
```

List the registered model IDs, then send a request through the same endpoint regardless of backend:

```bash
curl -s http://localhost:9090/v1/models | jq -r '.data[].id'

MODEL_ID="$(curl -s http://localhost:9090/v1/models | jq -r '[.data[] | select(.kind == "chat" or .kind == "coder" or .kind == "vision")][0].id')"
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg model "$MODEL_ID" '{
    model: $model,
    messages: [{role: "user", content: "Hello"}]
  }')"
```

Name the model you want. GGUF Switchboard selects its registered backend, unloads the resident model when necessary, starts the requested model, and forwards the request through the same API.

## Details

![gguf-switchboard demo](gguf-switchboard-demo.gif)

<sub>[▶ Watch with audio](demo.mp4)</sub>

### Where it fits

| Tool | Best fit | Model lifecycle | Formats and backends |
|------|----------|-----------------|----------------------|
| **GGUF Switchboard** | Development machines and trusted home labs with more models than available GPU memory | Single-slot, request-driven switching with drain and rollback | GGUF via llama.cpp; Safetensors via vLLM |
| **Ollama** | Simple local model use with its own model library and CLI | Loads and unloads models with `keep_alive` | Ollama-managed models, including converted GGUF |
| **llama.cpp** | Direct, low-level GGUF inference | You manage each `llama-server` process and port | GGUF |
| **vLLM** | High-throughput serving of models that fit the available GPU resources | Serves configured models; no Switchboard lifecycle | Primarily Safetensors; [upstream GGUF support is experimental](https://docs.vllm.ai/en/latest/features/quantization/gguf/) |

See [the detailed comparison](docs/COMPARISON.md) for llama-swap, LocalAI, LiteLLM, and feature-level trade-offs.

### Supported scope

- **Primary target:** Linux, NVIDIA GPUs, and CUDA. See the separate [macOS](docs/INSTALL-MACOS.md) and [Windows](docs/INSTALL-WINDOWS.md) installation guides for platform-specific support.
- **One resident model:** the scheduler runs one model at a time across llama.cpp and vLLM. This is model switching, not concurrent multi-model serving.
- **Two explicit paths:** GGUF runs through llama.cpp; Safetensors runs through vLLM. GGUF Switchboard does not currently route GGUF through vLLM.
- **Trusted networks:** there is no built-in authentication. Do not expose the service directly to the public internet.
- **Hardware fit is estimated:** model size, context, KV cache, quantization, and runtime allocations can still cause a load to fail. Failed switches roll back to the previous model when possible.

### Why this exists

Local model experimentation usually means managing backend processes, ports, downloads, model formats, and limited GPU memory by hand. GGUF Switchboard turns model loading into one controlled lifecycle behind a stable API:

- searches Hugging Face and downloads either GGUF or Safetensors models;
- scores GGUF quantizations against detected hardware and plans backend memory use;
- drains in-flight requests before swaps and rolls back failed switches;
- exposes OpenAI-compatible APIs, Anthropic Messages, Swagger UI, Prometheus metrics, and persistent usage history.

### Complete feature list

- **Fast & Lightweight** — Minimal overhead, maximum performance
- **Hot-Swap Models** — Switch between GGUF and Safetensors models on demand
- **Open & Extensible** — Modular, easy to extend, community-driven
- **100% Local** — Your models. Your data. Your machine.

Also included:

- **OpenAI-compatible API** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/rerank`, `/v1/models`, `/v1/models/registry.json`, `/v1/audio/*`
- **Anthropic Messages API** — `POST /v1/messages` (stream + non-stream); translated onto the selected backend's OpenAI-compatible API; tool calling and content blocks supported
- **Tool calling** — Chat Completions forwards `tools` / `tool_choice` / `tool_calls`; the Responses API translates function tools, function calls, and strict streaming events to/from the selected backend; Anthropic Messages translates tool definitions and calls bidirectionally. Actual behavior depends on the model and backend build (see [COMPATIBILITY](docs/COMPATIBILITY.md))
- **Tool-call capability probe** — `tools`-tagged models are probed with a real tool call at load time; verdict exposed as `tools_verified` on `/v1/models`
- **Swagger UI** — Try-it-out at `http://localhost:9090/swagger-ui/` (model dropdown grouped by kind, Rescan Models, hides chat vs embedding endpoints by selected model kind, live status badge, CPU/GPU utilization pills)
- **Live status telemetry** — `/status` reports `active_requests` plus per-GPU load / VRAM / temperature (`gpus[]`, via `nvidia-smi`) and host `cpu`/`memory` usage; surfaced on the Swagger topbar
- **Conformance console** — `/swagger-ui/conformance.html`: Inspect (where did the tool call land), Resolved Template, Battery (4 fixed tool-calling cases, pass/fail), and A/B Compare. Every run is persisted to a self-contained `conformance.db` (SQLite) with a History tab and per-tab recent-run tables; each surface can target a **custom OpenAI-compatible endpoint** (base URL + model + API key, key held in-browser only) to diagnose models outside the switchboard
- **Auto-discovery** — Scans GGUF dirs with a cheap validation ladder (filename → header → metadata); sidecars skipped
- **Live model rescan** — `POST /v1/models/refresh` plus a configurable daily watcher (`models_rescan_interval_secs`); merges new GGUFs without a full redeploy
- **HF metadata enrichment** — fills empty `description` / context / VRAM / `capabilities` / `hf_repo` from Hugging Face on launch and rescan (`sync-hf-metadata` CLI also available)
- **Model management** — `models search`, `models files`, and `models pull` for GGUF/llama.cpp; `models search vllm` and `models pull vllm` for Safetensors/vLLM; `models list` for an on-disk inventory and `models delete <name|#>` to remove a GGUF/Safetensors model plus its registry entry
- **Quant scoring** — `models search`/`models files` score every discovered quant against your detected hardware: a 0–100 FIT score, an estimated tok/s from a memory-bandwidth model, and a precision-retention % from published per-quant perplexity data, then recommend the fastest, most balanced, and least-lossy quant separately with `Try:` pull commands (see [docs/QUANT_SCORING.md](docs/QUANT_SCORING.md))
- **Kind-aware routing** — chat / completions / messages / responses require chat-like kinds; embeddings require `embedding` (and pass `--embeddings` to `llama-server`)
- **Single-slot hot-swap** — One resident model; switches drain in-flight requests; failed switches roll back
- **Memory-pressure eviction** — Unloads when system RAM crosses the critical threshold
- **Auto GPU layers (`auto_ngl`)** — Opt-in: at load, pick `-ngl` from free VRAM + GGUF size (manual `ngl` / `extra_args` still win)
- **ModelFitPlanner** — Opt-in `[fit]` section: inspects GPU topology and free VRAM before every load to produce a safe launch profile; bounded fallback ladder on OOM; caches known-good profiles to skip the ladder on subsequent loads
- **Idle priority model** — Preferred model auto-loads after a configurable idle timeout
- **llama.cpp and vLLM backends** — Spawns the registered engine while preserving one public API and one scheduler slot
- **SSE streaming**, **Prometheus** (`/metrics`), **usage history** (`/v1/usage`), **conformance run history** (`/v1/conformance/history`), **portable `models.json`**

#### How it works

```
Models                              API endpoints
 GGUF → llama.cpp               →   /v1/chat/completions
 Safetensors → vLLM             →   /v1/completions
          ↓                         /v1/embeddings, /v1/rerank
    gguf-switchboard  ──────────▶   /v1/responses, /v1/audio/*
    (single-slot swap)               /v1/messages (Anthropic)
```

Request for model `B` while `A` is loaded → drain → unload `A` → load `B` → forward. After `idle_timeout`, the priority model warms back up. With `[fit]` enabled, each load is preceded by a hardware-aware planning step that picks safe context/nGL/KV parameters. Details in [Architecture](docs/ARCHITECTURE.md).

### Why gguf-switchboard

When running local models you usually juggle backend processes, ports, formats, and GPU memory by hand. GGUF Switchboard is a **single-slot model router in Rust** for constrained GPUs: GGUF through llama.cpp, Safetensors through vLLM, memory-pressure eviction, backend-specific fit planning, rollback, Hugging Face model management, and usage tracking behind one OpenAI/Anthropic endpoint.

Full landscape table and vs llama-swap feature matrix: **[docs/COMPARISON.md](docs/COMPARISON.md)**.

### Detailed setup

#### Prerequisites

gguf-switchboard is a **model router**; llama.cpp and vLLM perform inference. The default systemd installer provisions both engines. Model weights remain opt-in, and an empty first install leaves the service installed but stopped until either a GGUF or Safetensors model is registered.

| Requirement | Notes |
|-------------|--------|
| **llama.cpp** | Installed or updated by `./deploy.sh` into `/usr/local`; serves GGUF models. |
| **vLLM** | Installed or updated by `./deploy.sh` in `/opt/gguf-switchboard/vllm-runtime`; serves Safetensors models. Verify it with `/usr/local/bin/uv run --project /opt/gguf-switchboard/vllm-runtime vllm --version`. See the [official GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) for non-default environments. |
| **Model weights** | Pulled separately after installation. GGUF defaults to `/var/lib/gguf-switchboard/models`; vLLM weights use the adjacent managed Safetensors directory. |
| **Linux** (recommended) | Ubuntu/Debian for `deploy.sh` (`apt`). Other distros: install build deps yourself. |
| **macOS** | Build from source with Metal. See [Install on macOS](docs/INSTALL-MACOS.md). |
| **Windows** | Use WSL2; native Windows deployment is not supported. See [Install on Windows](docs/INSTALL-WINDOWS.md). |
| **Rust** | Installed automatically by `deploy.sh` if missing; otherwise [rustup](https://rustup.rs/). |
| **GPU stack** | NVIDIA + CUDA toolkit on Linux, or Apple Metal on macOS (CPU-only llama.cpp works but is slow). |

#### Install (Linux + NVIDIA / CUDA)

Clone the repo and run the default dual-backend deployment:

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
./deploy.sh
```

`deploy.sh` calls `scripts/update-llama-cpp.sh` with service management disabled. The helper defaults to the latest manually cut upstream `vMAJOR.MINOR.PATCH` release and rebuilds only when that release differs from the last successfully installed marker, or when `llama-server` is missing or broken. Automated `bNNNNN` snapshots are available only through the explicit nightly channel. A required rebuild checks CUDA, builds a CUDA Release, verifies GPU discovery, installs under `/usr/local`, fixes runtime library paths, and refreshes the linker cache. If the release check is unavailable while the installed runtime remains healthy, deploy retains it and prints a warning.

Overrides: `LLAMA_DIR` (default `~/llama.cpp`), `PREFIX` (default `/usr/local`), `SERVICE` (default `gguf-switchboard`), `LLAMA_RELEASE_CHANNEL=stable|nightly` (default `stable`), `SKIP_PULL=1`, `SKIP_SERVICE=1`, `FORCE_REBUILD=1`.

The same deployment installs `uv` when needed, checks PyPI for the latest stable vLLM release allowed by `vllm-runtime/pyproject.toml`, and runs `uv sync` only when that version changed or the environment is missing or broken. Use `--skip-llama-cpp` or `--skip-vllm` to bypass even the corresponding update check.

What `deploy.sh` does when ready:

1. Pulls latest `main` (stashes dirty working tree first — see [Updating](#updating))
2. Creates system user `ggs` and directories under `/opt/gguf-switchboard` + `/var/lib/gguf-switchboard`
3. Installs build dependencies and Rust if needed
4. Checks the configured llama.cpp release channel and rebuilds only when required
5. Builds the release switchboard binary → `/usr/local/bin/gguf-switchboard`
6. Syncs the project into `/opt/gguf-switchboard`
7. Checks the latest compatible vLLM release and synchronizes only when required
8. Writes runtime configuration without replacing user-owned values
9. Installs the systemd unit, validates the required engines as `ggs`, and checks `/health`

```bash
# Optional: copy legacy ~/models into the system models dir (never deletes the source)
./deploy.sh --migrate-models

# Discover from an alternate directory while still registering the canonical models_dir
MODELS_DIR=/path/to/gguf-files ./deploy.sh --refresh-models
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

#### Prebuilt binary (Linux)

The prebuilt switchboard binary does not bundle llama.cpp or vLLM. Use the default `./deploy.sh` path when you want both engines installed automatically.

```bash
# amd64 (see Releases for arm64 + checksums)
curl -fsSL -o gguf-switchboard \
  https://github.com/pradeepgudipati/gguf-switchboard/releases/latest/download/gguf-switchboard-linux-amd64
chmod +x gguf-switchboard
sudo mv gguf-switchboard /usr/local/bin/

# Copy the tracked examples to system runtime paths (or run ./deploy.sh)
git clone --branch main --depth 1 https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
sudo mkdir -p /opt/gguf-switchboard /var/lib/gguf-switchboard/models
sudo cp config.example.toml /opt/gguf-switchboard/config.toml
gguf-switchboard discover-models /var/lib/gguf-switchboard/models -o /opt/gguf-switchboard/models.toml
gguf-switchboard /opt/gguf-switchboard/config.toml
```

#### Build without systemd

Local binary only — no `sudo` or systemd:

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
cargo build --release

cp config.example.toml config.toml
cp models.example.toml models.toml
./target/release/gguf-switchboard discover-models ~/models -o models.toml
./target/release/gguf-switchboard config.toml
```

#### Install models

With gguf-switchboard installed, search, browse, and download GGUF models from Hugging Face:

```bash
sudo mkdir -p /var/lib/gguf-switchboard/models

# Search for models
gguf-switchboard models search "Qwen3.5 9B"

# Browse available files in a repo
gguf-switchboard models files lmstudio-community/Qwen3.5-9B-GGUF

# Download, validate, and register a model (runs a quick speed test if the server is up)
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --dir /var/lib/gguf-switchboard/models

# Optional: tune parallel aria2 connections (default 8, maximum 16)
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --connections 8

# Skip the post-pull speed test
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --no-bench
```

`models search` scores every discovered quant against your detected RAM/VRAM and prints the fastest, most balanced, and least-lossy one separately, instead of a flat "supported: yes/no":

```
Hardware: System RAM 32.0 GiB | NVIDIA VRAM 24.0 GiB | Total 56.0 GiB
Speed model inputs: GPU bandwidth 1008 GB/s (NVIDIA GeForce RTX 4090) | RAM bandwidth 40 GB/s (assumed) | GPU efficiency 0.55 | CPU efficiency 0.35

REPO                          | FILES | SIZE    | FIT | CONTEXT   | ARCH  | SPEED           | BALANCED               | PRECISION   | QUANT
bartowski/Qwen3.5-9B-GGUF     |    24 | 9421 MB | 100 | 32768 tok | qwen3 | Q4_K_M ~127tok/s | Q5_K_M ~91tok/s/~98.9% | Q6_K ~99.6% | Q2_K,Q3_K_M,Q4_K_M,Q5_K_M,Q6_K,Q8_0
...
Try: ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q4_K_M   (fastest, ~127 tok/s est.)
     ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q5_K_M   (balanced, ~91 tok/s / ~98.9% quality est.)
     ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q6_K   (least precision loss, ~99.6% quality est.)
```

FIT is a continuous 0–100 memory-fit score (replaces the old binary "Supported: Yes/No"). SPEED/PRECISION show whichever quant maximizes each dimension for your machine; BALANCED is the one quant that gives up the least on both instead of maxing out either extreme — see [docs/QUANT_SCORING.md](docs/QUANT_SCORING.md) for the exact formulas, sources, and how to override RAM bandwidth (`--ram-bandwidth-gbps`) with a measured value.

Public downloads automatically use `aria2c` when available, then verify the expected size, Hugging Face LFS checksum, and GGUF metadata before registration. If Hugging Face rejects parallel range requests, the native downloader resumes the partial file. Authenticated downloads using `HF_TOKEN`, or systems without `aria2c`, use the native downloader directly. A successful pull refreshes a running gguf-switchboard server automatically and (unless `--no-bench`) runs a short chat completion to print prompt and generation tok/s.

Or download manually — any `.gguf` file in `/var/lib/gguf-switchboard/models` works:

```bash
# Example layout after manual download:
#   /var/lib/gguf-switchboard/models/Qwen3.5-9B-Q4_K_M.gguf
#   /var/lib/gguf-switchboard/models/gemma-4-E4B-it-Q4_K_M.gguf
```

If you downloaded models manually, run `./deploy.sh --refresh-models` so discovery registers them.

List what is on disk and remove models you no longer want:

```bash
# Numbered inventory of every GGUF file / Safetensors dir under the model dirs,
# with the registered alias (if any). Add --json for machine output.
gguf-switchboard models list

# Delete by alias/name or by the number from `models list` — prompts for
# confirmation (add --yes to skip), removes the file/dir and the models.toml entry
gguf-switchboard models delete qwen3-embedding-4b
gguf-switchboard models delete 2 --yes
```

For Safetensors models, use the vLLM search and pull lane. The pull validates that the repository contains `config.json` and Safetensors weights, downloads the tokenizer/configuration files, detects supported quantization metadata, and registers the isolated vLLM backend:

```bash
gguf-switchboard models search vllm "Qwen 7B Instruct"
gguf-switchboard models pull vllm Qwen/Qwen2.5-7B-Instruct \
  --registry /opt/gguf-switchboard/models.toml
./deploy.sh --skip-llama-cpp --skip-vllm
```

One alias may contain both a GGUF and a Safetensors source. Unless the registry pins `backend`, gguf-switchboard prefers vLLM when its weights fit available VRAM and falls back to llama.cpp when they do not.

#### Verify

```bash
curl -s http://localhost:9090/health
curl -s http://localhost:9090/status | jq .   # includes gpus[] and host CPU/RAM usage
curl -s http://localhost:9090/v1/models | jq '.data[].id'

# NVIDIA processes with the loaded GGUF model name
./scripts/nvidia-smi-models.sh
./scripts/nvidia-smi-models.sh --watch 2
```

The model-aware NVIDIA view prints the standard `nvidia-smi` dashboard, then
adds a process table that joins GPU usage with each process's `-m` or `--model`
argument from `/proc`. Run it as the same user as `llama-server`, or with
sufficient permission to read that process's command line. Processes whose
command line is inaccessible show `-` for the model.

#### Updating

Supported upgrade path from an existing checkout:

```bash
cd ~/gguf-switchboard   # or wherever you cloned
./deploy.sh
```

| Goal | Command |
|------|---------|
| Update both engines + rebuild + restart switchboard | `./deploy.sh` |
| Rebuild only (no `git pull`) | `./deploy.sh --skip-pull` |
| Keep existing llama.cpp during an update | `./deploy.sh --skip-llama-cpp` |
| Keep existing vLLM during an update | `./deploy.sh --skip-vllm` |
| Pick up new GGUF files (merge registry) | `./deploy.sh --refresh-models` |
| Copy legacy `~/models` into system models dir | `./deploy.sh --migrate-models` |
| Live rescan while running | `curl -X POST http://localhost:9090/v1/models/refresh` (or Swagger **Rescan Models**) |
| List models on disk | `ggs models list` |
| Delete a model (files + registry entry) | `ggs models delete <name\|#>` |
| Check whether the background service is running | `ggs status` |
| Restart without rebuild | `ggs restart` |

**Important:**

- Deploy **stashes uncommitted changes** (including untracked files) before `git pull`. Recover with `git stash list` / `git stash pop`.
- Live config lives under `/opt/gguf-switchboard/` (`config.toml`, `models.toml`); models, `usage.db`, and `conformance.db` (conformance-console run history) under `/var/lib/gguf-switchboard/`. Tracked defaults live in `config.example.toml` and `models.example.toml`.
- After editing aliases / `priority` / `extra_args`, restart: `sudo systemctl restart gguf-switchboard`.
- `deploy.sh` leaves the unit stopped when no GGUF or Safetensors models are registered. Pull either format, then re-run deploy with the backend skip flags if no engine update is needed.
- Every successful run ends with a summary of llama.cpp, gguf-switchboard, vLLM, indexed models, service state, and the available `ggs` operational commands.

```bash
# Background service status and logs
ggs status
ggs logs
ggs logs watch
ggs logs --tail 250
```

#### Shell alias (optional)

Add a short `ggs` alias so you can type `ggs` instead of `gguf-switchboard`. `deploy.sh` offers to add this automatically on Linux without conflicting with the common `gs='git status'` alias.

**Linux (bash):**

```bash
echo "alias ggs='gguf-switchboard'" >> ~/.bashrc
source ~/.bashrc
```

**Linux (zsh):**

```bash
echo "alias ggs='gguf-switchboard'" >> ~/.zshrc
source ~/.zshrc
```

After that:

```bash
ggs models search "Qwen3.5"
ggs models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M
ggs config.toml
ggs status    # prints whether the background systemd service is running or stopped
ggs stop      # sudo systemctl stop gguf-switchboard
ggs restart   # sudo systemctl restart gguf-switchboard
ggs logs      # latest 100 journal entries
ggs logs watch
ggs logs --tail 250
```

#### Troubleshooting first install

| Symptom | Likely fix |
|---------|------------|
| llama.cpp setup fails | Confirm `nvcc`, the NVIDIA driver, and CUDA toolkit are usable; rerun `./deploy.sh` after correcting the toolchain |
| vLLM setup fails | Review the `uv sync` error for Python, wheel, glibc, CUDA, or platform incompatibility; rerun `./deploy.sh` after correcting it |
| Service unhealthy / no models | Pull either a GGUF or Safetensors model, then rerun deploy |
| First install has no models | Not fatal. The installer leaves the service stopped and prints both model-pull paths |
| Empty `/v1/models` | Check `models_dir` in `/opt/gguf-switchboard/models.toml`; enable `auto_discover = true`; `ggs restart` |
| Deploy "lost" my edits | `git stash list` — deploy stashes dirty trees before pull |
| Port 9090 in use | Change `bind` in `/opt/gguf-switchboard/config.toml` and `ggs restart` |

### Further documentation

| Doc | Contents |
|-----|----------|
| **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** | `config.toml`, `models.toml`, `[fit]` section, discovery, context sizing, CLI |
| **[docs/USAGE.md](docs/USAGE.md)** | API examples (OpenAI + Anthropic), SDKs, IDE setup, monitoring, local run |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | Scheduler/backend overview, ModelFitPlanner, kind guard, project layout |
| **[docs/COMPARISON.md](docs/COMPARISON.md)** | Landscape vs Ollama / llama-swap / others |
| **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** | Throughput, swap latency, bench script |
| **[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)** | OpenAI + Anthropic endpoint coverage, feature matrix |
| **[docs/QUANT_SCORING.md](docs/QUANT_SCORING.md)** | FIT/SPEED/BALANCED/PRECISION scoring formulas and sources |

#### Configuration (short)

Two runtime files under **`/opt/gguf-switchboard/`**: **`config.toml`** (bind, idle timeout, `vram_gb`, `[fit]` section) and **`models.toml`** (aliases → GGUF paths). Models live in **`/var/lib/gguf-switchboard/models/`**. Full reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

```bash
# After install, tweak models then restart
sudo systemctl restart gguf-switchboard
```

#### Try the API

```bash
# OpenAI Chat Completions
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"YOUR_ALIAS","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

# Anthropic Messages API
curl http://localhost:9090/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: not-needed" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"YOUR_ALIAS","max_tokens":64,"messages":[{"role":"user","content":"Hello"}]}'
```

Swagger UI: **http://localhost:9090/swagger-ui/** — more examples in [docs/USAGE.md](docs/USAGE.md).
Conformance console: **http://localhost:9090/swagger-ui/conformance.html** — diagnose tool-calling / chat-template behavior of a local or external OpenAI-compatible model; runs are saved to `conformance.db`.

## License

MIT
