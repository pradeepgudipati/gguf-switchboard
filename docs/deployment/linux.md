# Install on Linux

> [← Back to README](../../README.md)

Linux with NVIDIA/CUDA is the primary deployment target. The default installer sets up CUDA llama.cpp, an isolated vLLM environment, gguf-switchboard, and its systemd service.

## Prerequisites

gguf-switchboard is a **model router**; llama.cpp and vLLM perform inference. The default systemd installer provisions both engines. Model weights remain opt-in, and an empty first install leaves the service installed but stopped until either a GGUF or Safetensors model is registered.

| Requirement | Notes |
|-------------|--------|
| **Linux** | Ubuntu/Debian for `deploy.sh` (`apt`). Other distros: install build deps yourself. |
| **NVIDIA GPU + CUDA toolkit** | `nvcc` and the NVIDIA driver must be usable. CPU-only llama.cpp works but is slow. |
| **llama.cpp** | Installed or updated by `./deploy.sh` into `/usr/local`; serves GGUF models. |
| **vLLM** | Installed or updated by `./deploy.sh` in `/opt/gguf-switchboard/vllm-runtime`; serves Safetensors models. Verify with `/usr/local/bin/uv run --project /opt/gguf-switchboard/vllm-runtime vllm --version`. See the [official GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) for non-default environments. |
| **Rust** | Installed automatically by `deploy.sh` if missing; otherwise [rustup](https://rustup.rs/). |
| **Model weights** | Pulled separately after installation. GGUF defaults to `/var/lib/gguf-switchboard/models`; vLLM weights use the adjacent managed Safetensors directory. |

## Primary installer (`deploy.sh`)

Clone the repo and run the default dual-backend deployment:

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
./deploy.sh
```

### What `deploy.sh` does

1. Pulls latest `main` (stashes dirty working tree first — see [Updating](#updating))
2. Creates system user `ggs` and directories under `/opt/gguf-switchboard` + `/var/lib/gguf-switchboard`
3. Installs build dependencies and Rust if needed
4. Checks the configured llama.cpp release channel and rebuilds only when required
5. Builds the release switchboard binary → `/usr/local/bin/gguf-switchboard`
6. Syncs the project into `/opt/gguf-switchboard`
7. Checks the latest compatible vLLM release and synchronizes only when required
8. Writes runtime configuration without replacing user-owned values
9. Installs the systemd unit, validates the required engines as `ggs`, and checks `/health`

### How llama.cpp updates work

`deploy.sh` calls `scripts/update-llama-cpp.sh` with service management disabled. The helper defaults to the latest manually cut upstream `vMAJOR.MINOR.PATCH` release and rebuilds only when that release differs from the last successfully installed marker, or when `llama-server` is missing or broken. Automated `bNNNNN` snapshots are available only through the explicit nightly channel. A required rebuild checks CUDA, builds a CUDA Release, verifies GPU discovery, installs under `/usr/local`, fixes runtime library paths, and refreshes the linker cache. If the release check is unavailable while the installed runtime remains healthy, deploy retains it and prints a warning.

### How vLLM updates work

The same deployment installs `uv` when needed, checks PyPI for the latest stable vLLM release allowed by `vllm-runtime/pyproject.toml`, and runs `uv sync` only when that version changed or the environment is missing or broken. Use `--skip-llama-cpp` or `--skip-vllm` to bypass even the corresponding update check.

### Environment variable overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_DIR` | `~/llama.cpp` | llama.cpp source checkout path |
| `PREFIX` | `/usr/local` | Install prefix for llama-server binary |
| `SERVICE` | `gguf-switchboard` | systemd service name |
| `LLAMA_RELEASE_CHANNEL` | `stable` | `stable` (semver tags) or `nightly` (build snapshots) |
| `SKIP_PULL` | `0` | Set `1` to skip `git pull` |
| `SKIP_SERVICE` | `0` | Set `1` to skip systemd operations |
| `FORCE_REBUILD` | `0` | Set `1` to rebuild llama.cpp even if the marker matches |

### Deploy flags

| Flag | Description |
|------|-------------|
| `--migrate-models` | Copy legacy `~/models` into the system models dir (never deletes the source) |
| `--refresh-models` | Discover from an alternate directory while still registering the canonical `models_dir` |
| `--skip-llama-cpp` | Skip the llama.cpp update check entirely |
| `--skip-vllm` | Skip the vLLM update check entirely |

```bash
# Optional: copy legacy ~/models into the system models dir
./deploy.sh --migrate-models

# Discover from an alternate directory
MODELS_DIR=/path/to/gguf-files ./deploy.sh --refresh-models
```

### After install

Then open **http://localhost:9090/swagger-ui/**.

If the binary is not on `PATH`, point the registry at it:

```toml
# models.toml
[defaults]
llama_server = "/path/to/llama-server"
```

## Manual llama.cpp (optional)

Prefer `./scripts/update-llama-cpp.sh` on CUDA hosts — it installs shared libs correctly and strips RUNPATH. Manual one-shot (binary copy only):

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j"$(nproc)"
sudo cp build/bin/llama-server /usr/local/bin/
llama-server --version
```

## Prebuilt binary (Linux)

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

## Build without systemd

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

## Install models

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

FIT is a continuous 0–100 memory-fit score (replaces the old binary "Supported: Yes/No"). SPEED/PRECISION show whichever quant maximizes each dimension for your machine; BALANCED is the one quant that gives up the least on both instead of maxing out either extreme — see [docs/QUANT_SCORING.md](QUANT_SCORING.md) for the exact formulas, sources, and how to override RAM bandwidth (`--ram-bandwidth-gbps`) with a measured value.

Public downloads automatically use `aria2c` when available, then verify the expected size, Hugging Face LFS checksum, and GGUF metadata before registration. If Hugging Face rejects parallel range requests, the native downloader resumes the partial file. Authenticated downloads using `HF_TOKEN`, or systems without `aria2c`, use the native downloader directly. A successful pull refreshes a running gguf-switchboard server automatically and (unless `--no-bench`) runs a short chat completion to print prompt and generation tok/s.

Or download manually — any `.gguf` file in `/var/lib/gguf-switchboard/models` works:

```bash
# Example layout after manual download:
#   /var/lib/gguf-switchboard/models/Qwen3.5-9B-Q4_K_M.gguf
#   /var/lib/gguf-switchboard/models/gemma-4-E4B-it-Q4_K_M.gguf
```

If you downloaded models manually, run `./deploy.sh --refresh-models` so discovery registers them.

For Safetensors models, use the vLLM search and pull lane. The pull validates that the repository contains `config.json` and Safetensors weights, downloads the tokenizer/configuration files, detects supported quantization metadata, and registers the isolated vLLM backend:

```bash
gguf-switchboard models search vllm "Qwen 7B Instruct"
gguf-switchboard models pull vllm Qwen/Qwen2.5-7B-Instruct \
  --registry /opt/gguf-switchboard/models.toml
./deploy.sh --skip-llama-cpp --skip-vllm
```

One alias may contain both a GGUF and a Safetensors source. Unless the registry pins `backend`, gguf-switchboard prefers vLLM when its weights fit available VRAM and falls back to llama.cpp when they do not.

## Verify

```bash
curl -s http://localhost:9090/health
curl -s http://localhost:9090/status | jq .   # includes gpus[] and host CPU/RAM usage
curl -s http://localhost:9090/v1/models | jq '.data[].id'

# NVIDIA processes with the loaded GGUF model name
./scripts/nvidia-smi-models.sh
./scripts/nvidia-smi-models.sh --watch 2
```

The model-aware NVIDIA view prints the standard `nvidia-smi` dashboard, then adds a process table that joins GPU usage with each process's `-m` or `--model` argument from `/proc`. Run it as the same user as `llama-server`, or with sufficient permission to read that process's command line. Processes whose command line is inaccessible show `-` for the model.

## Updating

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

## Shell alias (optional)

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

## Troubleshooting first install

| Symptom | Likely fix |
|---------|------------|
| llama.cpp setup fails | Confirm `nvcc`, the NVIDIA driver, and CUDA toolkit are usable; rerun `./deploy.sh` after correcting the toolchain |
| vLLM setup fails | Review the `uv sync` error for Python, wheel, glibc, CUDA, or platform incompatibility; rerun `./deploy.sh` after correcting it |
| Service unhealthy / no models | Pull either a GGUF or Safetensors model, then rerun deploy |
| First install has no models | Not fatal. The installer leaves the service stopped and prints both model-pull paths |
| Empty `/v1/models` | Check `models_dir` in `/opt/gguf-switchboard/models.toml`; enable `auto_discover = true`; `ggs restart` |
| Deploy "lost" my edits | `git stash list` — deploy stashes dirty trees before pull |
| Port 9090 in use | Change `bind` in `/opt/gguf-switchboard/config.toml` and `ggs restart` |
