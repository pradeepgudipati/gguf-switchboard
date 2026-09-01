# Configuration

> [← Back to README](../README.md)

Configuration is split across two files:

| File | Purpose |
|------|---------|
| **`config.toml`** | Server bind address, idle timeout, GPU VRAM, database path |
| **`models.toml`** | Model registry: aliases, GGUF and Safetensors sources, backend selection, priorities, per-model overrides |

Default runtime paths after `deploy.sh`:

| Path | Role |
|------|------|
| `/opt/gguf-switchboard/config.toml` | Server config |
| `/opt/gguf-switchboard/models.toml` | Model registry |
| `/var/lib/gguf-switchboard/models/` | GGUF files |
| `/opt/gguf-switchboard/vllm-runtime/` | Managed uv/vLLM environment |
| `/usr/local/share/gguf-switchboard/llama-cpp-release` | Last successfully installed numbered llama.cpp release |
| `/var/lib/gguf-switchboard/usage.db` | Token usage SQLite |
| `/usr/local/bin/gguf-switchboard` | Binary (root-owned) |

Tracked defaults live in `config.example.toml` and `models.example.toml`. The systemd unit runs as `User=ggs` / `Group=ggs`.

### Server configuration (`config.toml`)

```toml
bind = "0.0.0.0:9090"
startup_timeout = 60
idle_timeout = 600
default_backend = "llama.cpp"

# GPU VRAM in GB — sizes per-model context (-c) when not set in models.toml
# RTX 3060 = 12; lower if you share VRAM with a display or other apps
vram_gb = 12

# Opt-in: pick -ngl at load from free VRAM (nvidia-smi or vram_gb) + GGUF size
auto_ngl = false

database_path = "/var/lib/gguf-switchboard/usage.db"

# Model registry (absolute path — does not depend on WorkingDirectory)
models_file = "/opt/gguf-switchboard/models.toml"
```

If `models_file` is omitted but a sibling `models.toml` exists next to your config, it is loaded automatically.

See [Context size (`-c`)](#context-size-c) for how `vram_gb` affects per-model `-c` values.
See [Auto GPU layers (`auto_ngl`)](#auto-gpu-layers-auto_ngl) for load-time `-ngl` heuristics.

### Model configuration (`models.toml`)

```toml
version = 1

[defaults]
models_dir = "/models"                        # Root directory for GGUF files
llama_server = "/usr/local/bin/llama-server"  # Backend binary (auto-detected on discover)
host = "127.0.0.1"                            # llama-server bind host
base_port = 18081                              # First internal model port; others increment from here
context_size = 16384                          # Safe default for consumer GPUs; raise if you have VRAM headroom
ngl = 999                                     # Default -ngl (GPU layers)
backend = "llama.cpp"
vllm_command = "/usr/local/bin/uv"
vllm_project = "/opt/gguf-switchboard/vllm-runtime"

auto_discover = true    # Also register any .gguf under models_dir not listed in [[models]]

[[models]]
alias = "gemma-4-e4b"     # API model id — use this in Cursor, Cline, etc.
file = "gemma-4-E4B-it-Q4_K_M.gguf"
display_name = "Gemma 4 E4B"
kind = "chat"             # chat | coder | vision | embedding (inferred when omitted)
enabled = true            # false = hide from /v1/models and scheduling
priority = true           # Auto-load after idle_timeout (only one should be true)
# port = 18081            # Internal assigned port; refresh normalizes the sequence
# context_size = 32768    # Override VRAM-based default from config.toml vram_gb (optional)
# ngl = 40                # Pin GPU layers (disables auto_ngl for this model)
# description = "..."     # Optional blurb for /v1/models + Swagger (or run sync-hf-metadata)
# max_context_length = 131072  # Model max context from HF/GGUF metadata
# min_vram_gb = 6         # Approximate minimum VRAM (GB)
# capabilities = ["tools", "vision"]
# hf_repo = "lmstudio-community/Qwen3.5-9B-GGUF"
# extra_args = ["--jinja"]  # Extra llama-server flags (optional)
```

A Safetensors source uses the same alias schema and selects vLLM:

```toml
[[models]]
alias = "qwen2-5-7b-instruct"
file = ""
display_name = "Qwen 2.5 7B Instruct"
kind = "chat"
backend = "vllm"
vllm_file = "/var/lib/gguf-switchboard/vllm-models/Qwen--Qwen2.5-7B-Instruct"
vllm_hf_repo = "Qwen/Qwen2.5-7B-Instruct"
# quantization = "awq"
# attention_backend = "FLASH_ATTN"
# tensor_parallel_size = 2
# gpu_memory_utilization = 0.9
# served_model_name = "qwen2-5-7b-instruct"
```

The CLI writes these fields with `ggs models pull vllm <repo>`. One entry may also retain a GGUF `file`/`hf_repo`. Without an explicit `backend` pin, vLLM is preferred when its weights fit VRAM and llama.cpp is the fallback.

| Field | Description |
|-------|-------------|
| `version` | Registry schema version (currently `1`) |
| `defaults.models_dir` | Directory (or comma-separated directories) scanned for llama.cpp-loadable GGUF files |
| `defaults.llama_server` | Path to `llama-server` binary |
| `defaults.vllm_command` | System-visible `uv` executable used to run vLLM |
| `defaults.vllm_project` | Isolated uv project containing the managed vLLM installation |
| `defaults.base_port` | Starting port; model at index *N* uses `base_port + N` unless `port` is set |
| `defaults.context_size` | Fallback/ceiling context window when `vram_gb` heuristics do not apply (default `16384`; raise for 24 GB+ GPUs) |
| `auto_discover` | When `true`, any `.gguf` under `models_dir` not listed in `[[models]]` is registered at runtime |
| `[[models]].alias` | Short id used in API requests (`model` field) |
| `[[models]].file` | GGUF filename relative to `models_dir`, or absolute path |
| `[[models]].display_name` | Human-readable name; defaults to a title-cased alias |
| `[[models]].kind` | `chat`, `coder`, `vision`, `embedding`, or `reranker` — inferred from alias/file when omitted |
| `[[models]].enabled` | When `false`, model is omitted from `/v1/models` and scheduling |
| `[[models]].priority` | If `true`, this model loads automatically after `idle_timeout` |
| `[[models]].port` | Normalized internal backend port; discovery and refresh rewrite it from `defaults.base_port` |
| `[[models]].context_size` | Override per-model `-c` (otherwise sized from `vram_gb`) |
| `[[models]].ngl` | Override per-model `-ngl` (pins against `auto_ngl`) |
| `[[models]].description` | Optional description shown in `/v1/models` and Swagger |
| `[[models]].max_context_length` | Model max context from HF/GGUF metadata (not the serving `-c`) |
| `[[models]].min_vram_gb` | Approximate minimum VRAM in GB (weights floor) |
| `[[models]].capabilities` | Tags such as `tools`, `vision`, `reasoning` |
| `[[models]].hf_repo` | Matched Hugging Face repo id after `sync-hf-metadata` |
| `[[models]].extra_args` | Extra flags appended to `llama-server` launch args |
| `[[models]].backend` | Optional `llama.cpp` or `vllm` pin; automatic source selection applies when omitted |
| `[[models]].vllm_file` | Absolute or model-directory-relative Safetensors repository directory |
| `[[models]].vllm_hf_repo` | Hugging Face repository id used when no local vLLM directory is present |
| `[[models]].quantization` | vLLM quantization mode detected from `config.json` or set explicitly |
| `[[models]].attention_backend` | Optional vLLM attention backend override |
| `[[models]].draft_model` | Hugging Face repo or local path for speculative decoding |
| `[[models]].num_speculative_tokens` | vLLM speculative token count |
| `[[models]].tensor_parallel_size` | Number of GPUs used by vLLM tensor parallelism |
| `[[models]].gpu_memory_utilization` | vLLM GPU allocator fraction from `0.0` to `1.0` |
| `[[models]].served_model_name` | Model id exposed by the internal vLLM server; defaults to the alias |

Duplicate `[[models]]` entries (same alias or file) are merged automatically on load and during `discover-models --merge`. Only one model may be `priority = true`; extras are cleared with a warning.

Kind is enforced at request time: chat/completions/responses accept `chat`/`coder`/`vision`; `/v1/embeddings` accepts `embedding` only.

#### Portable `models.json`

`discover-models` writes a sibling **`models.json`** alongside `models.toml`. The running server also exposes it at:

```bash
curl http://localhost:9090/v1/models/registry.json -o models.json
```

Example shape:

```json
{
  "version": 1,
  "models_dir": "/home/pradeep/models",
  "models": [
    {
      "id": "gemma-4-e4b",
      "file": "gemma-4-E4B-it-Q4_K_M.gguf",
      "display_name": "Gemma 4 E4B",
      "kind": "chat",
      "enabled": true,
      "priority": true,
      "context_size": null,
      "description": null,
      "max_context_length": 131072,
      "min_vram_gb": 6,
      "capabilities": ["tools"],
      "hf_repo": "lmstudio-community/gemma-4-E4B-it-GGUF",
      "tags": ["chat", "priority"]
    }
  ]
}
```

`GET /v1/models` returns the same metadata on each OpenAI-style model object (`kind`, `description`, `max_context_length`, `min_vram_gb`, `capabilities`, `hf_repo`).

Export manually:

```bash
./gguf-switchboard export-registry models.toml -o models.json
```

`models_file` in `config.toml` may also point directly at a **`models.json`** registry (portable subset — no `llama_server` / port defaults; those fall back to built-in defaults).

#### `kind` inference

When `kind` is omitted on a `[[models]]` entry, it is inferred from the alias and filename:

| Pattern in alias/file | Inferred `kind` |
|-----------------------|-----------------|
| `embed`, `granite-embedding` | `embedding` |
| `-vl`, `vision`, `mmproj` | `vision` |
| `coder`, `-code` | `coder` |
| (default) | `chat` |

Set `enabled = false` to keep a model in the registry but hide it from `/v1/models` and scheduling (useful for vision models missing an `mmproj` sidecar, or models you are still downloading).

#### Port assignment

Ports are internal implementation details assigned sequentially from `defaults.base_port` after models are sorted deterministically:

| Index | Port (default base 18081) |
|-------|--------------------------|
| 0 | 18081 |
| 1 | 18082 |
| 2 | 18083 |

Discovery and refresh replace existing per-model `port` values with this normalized sequence. Set `defaults.base_port` to move the entire internal range.

#### Alias generation

When models are discovered from filenames, aliases are derived automatically:

1. Take the filename stem (without `.gguf`)
2. Lowercase
3. Strip common suffixes (`-instruct`, `-it`, `-gguf`, quant tags like `-Q4_K_M`, `-bf16`, etc.)
4. Replace `_` with `-`

Examples:

| GGUF filename | Generated alias |
|---------------|-----------------|
| `Qwen3.5-9B-Q4_K_M.gguf` | `qwen3.5-9b` |
| `gemma-3-4b-it-Q4_K_M.gguf` | `gemma-3-4b` |
| `llama-3.2-3b.gguf` | `llama-3.2-3b` |

Duplicate aliases get a numeric suffix (`model-2`, `model-3`, …).

#### Auto-discover at runtime

With `auto_discover = true`, the runtime scans every directory listed in `models_dir` on startup and registers any llama.cpp-loadable `.gguf` file not already listed in `[[models]]`. Explicit entries let you pin aliases, display names, or priorities for specific files; everything else is picked up automatically. Pins that fail the same validation checks are **skipped with a warning** (they are not registered).

`models_dir` must exist at startup — no fallback directories are searched. Use a comma-separated list to scan multiple folders:

```toml
[defaults]
models_dir = "/models,/home/you/extra-gguf"
auto_discover = true
```

Discovery is recursive. Before a file is registered (auto-discover **or** explicit `[[models]]` pin), it must pass a cheap **prefix-only** validation ladder — never a full multi-GB read:

1. **Filename** — reject sidecars/adapters (`mmproj*`, `mtp-*`, `*projector*`, `*adapter*`, `*tokenizer*`, `ggml-vocab*`, LoRA/`-vocab` names)
2. **Header** — `GGUF` magic, version `2` or `3`, `tensor_count > 0`
3. **Metadata** — require `general.architecture`; reject encoder/sidecar arches (`clip`, `siglip`, `vit`, …) and `general.type` of `lora`/`vocab`; if `{arch}.block_count` is present and `0`, reject

Embedding architectures remain discoverable (for `/v1/embeddings`). Passing this ladder means the file looks like a standalone model — **GPU load success is still proven later** when `llama-server` starts and passes health checks.

With a single `models_dir`, nested paths are stored relative to that root; with multiple directories, discovered files are stored as absolute paths.

You can omit `[[models]]` entirely and rely on auto-discover, or add entries only for models you want to customize.

#### Deploy-time auto-generation

`./deploy.sh` installs a system-wide layout:

- `/opt/gguf-switchboard/config.toml` + `models.toml`
- `/var/lib/gguf-switchboard/models/` for GGUF files
- systemd unit as `User=ggs`

It generates `/opt/gguf-switchboard/models.toml` when:

- **First install** — no existing `models.toml` (auto-discovers from GGUF files)
- **`--refresh-models`** — explicitly regenerate from disk, merging with the existing registry

Subsequent deploys without `--refresh-models` keep the existing registry unchanged.

When generation runs:

1. Builds the release binary and installs it to `/usr/local/bin/gguf-switchboard`
2. Requires `/usr/local/bin/llama-server`
3. Runs `discover-models` as `ggs` against `/var/lib/gguf-switchboard/models`
4. Merges with the existing registry when present — preserves `alias`, `display_name`, `priority`, `port`, `context_size`, `kind`, `enabled`, and `extra_args` per file; **duplicate alias/file entries are deduplicated**
5. Installs the result to `/opt/gguf-switchboard/models.toml` and writes sibling **`models.json`**
6. Prints a checklist + configured models after `/health` succeeds

**Models directory**:

- Canonical: `/var/lib/gguf-switchboard/models`
- Optional discover override: `MODELS_DIR=/path ./deploy.sh --refresh-models` (registry `models_dir` remains canonical)
- Legacy `~/models`: copy with `./deploy.sh --migrate-models` (never deletes the source)

An empty first installation completes successfully without starting the systemd service. Deploy prints concrete `ggs models search` and `ggs models pull` commands; after downloading a model, run `./deploy.sh --refresh-models` to generate the registry and start the service.

```bash
MODELS_DIR=/path/to/models ./deploy.sh --refresh-models
./deploy.sh --migrate-models
```

#### `models` CLI (search, files, pull)

Search, browse, and download GGUF models from Hugging Face:

```bash
# Search HF Hub for GGUF models
gguf-switchboard models search "Qwen3.5 9B"

# Limit results
gguf-switchboard models search "Qwen3.5 9B" --limit 5

# Override RAM bandwidth for speed estimates (GB/s)
gguf-switchboard models search "Qwen3.5 9B" --ram-bandwidth-gbps 50

# List .gguf files in a repo (with size and quantization)
gguf-switchboard models files lmstudio-community/Qwen3.5-9B-GGUF

# Download, validate, and register a model in one step
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M

# Specify a destination directory and registry file
gguf-switchboard models pull bartowski/Qwen3.5-9B-GGUF --quant Q4_K_M --dir /models --registry models.toml

# Tune parallel aria2 connections (default 8, maximum 16)
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --connections 8

# Skip the post-pull speed test
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --no-bench

# Dry-run: show what the fit planner would generate without downloading
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --fit-dry-run
```

**`models search` flags:**

| Flag | Description |
|------|-------------|
| `--limit N` | Maximum number of repositories to return (default `10`) |
| `--ram-bandwidth-gbps N` | Override auto-detected RAM bandwidth for speed estimates (useful after `mbw`/`likwid-bench` measurement) |

Search prints FIT/SPEED/BALANCED/PRECISION scores for each result — see [docs/QUANT_SCORING.md](QUANT_SCORING.md) for the exact formulas.

**`models pull` flags:**

| Flag | Description |
|------|-------------|
| `--quant QUANT` | Quantization to download: exact label (`Q4_K_M`), family (`Q4`), predictable alias (`K_M`), or `auto` (largest fitting quant) |
| `--dir PATH` | Destination directory for GGUF files (default: `models_dir` from registry) |
| `--registry PATH` | Registry file to merge into (default: `models.toml` next to config) |
| `--connections N` | Parallel aria2c connections (1–16, default `8`) |
| `--no-bench` | Skip the post-pull speed test |
| `--fit-dry-run` | Show what the fit planner would generate (context_size, ngl, extra_args) without downloading |

`models pull` performs the complete workflow: fetches the repo tree, resolves `--quant` case-insensitively, streams the download with progress, validates the GGUF header, generates an alias, runs the fit planner to generate context_size/ngl/extra_args, and merges into `models.toml`. Use an exact label such as `Q4_K_M`, a family such as `Q4` (preference order: `Q4_K_M`, `Q4_K_S`, `Q4_0`, `Q4_1`), `K_M` as a predictable alias for `Q4_K_M`, or `auto` to select the largest standalone quant that fits total system RAM plus NVIDIA VRAM with 20% runtime headroom. The running server picks up the new entry on the next `POST /v1/models/refresh` or restart.

Set `HF_TOKEN` in the environment for gated models:

```bash
HF_TOKEN=hf_... gguf-switchboard models pull meta-llama/Llama-3-70B-GGUF --quant Q4_K_M
```

**`models search` output format:**

```
Hardware: System RAM 32.0 GiB | NVIDIA VRAM 24.0 GiB | Total 56.0 GiB
Speed model inputs: GPU bandwidth 1008 GB/s (NVIDIA GeForce RTX 4090) | RAM bandwidth 40 GB/s (assumed) | GPU efficiency 0.55 | CPU efficiency 0.35

REPO                                               | FILES |     SIZE | FIT | CONTEXT    | ARCH  | SPEED            | BALANCED               | PRECISION    | QUANT
bartowski/Qwen3.5-9B-GGUF                          |    24 |  9421 MB | 100 | 32768 tok  | qwen3 | Q4_K_M ~127tok/s | Q5_K_M ~91tok/s/~98.9% | Q6_K ~99.6%  | Q2_K,Q3_K_M,Q4_K_M,Q5_K_M,Q6_K,Q8_0
unsloth/Muse-Glimmer-30B-GGUF                      |    20 | 27855 MB | 100 | 131072 tok | muse-  | IQ2_XXS ~52tok/s | Q5_K_XL ~25tok/s/~98.6%| Q8_0 ~99.9%  | IQ2_XXS,IQ2_XS,IQ2_M,Q2_K_XL,...
                                                     glimmer
FIT: 0-100 memory-fit score (100 = comfortable headroom; 0 = does not fit RAM+VRAM). SPEED/PRECISION: the quant that maximizes each — tok/s from a memory-bandwidth model (verify against `llama-bench` on your machine), quality % from published per-quant perplexity measurements ("~" = extrapolated, not directly measured for this architecture). BALANCED: the quant with the best average of speed and quality, both normalized to this model's own quant options — a middle ground when you don't want either extreme. See docs/QUANT_SCORING.md for methodology and sources; override RAM bandwidth with --ram-bandwidth-gbps if you've measured your own.
Try: ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q4_K_M   (fastest, ~127 tok/s est.)
     ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q5_K_M   (balanced, ~91 tok/s / ~98.9% quality est.)
     ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q6_K   (least precision loss, ~99.6% quality est.)
```

| Column | Description |
|--------|-------------|
| `FIT` | 0–100 memory-fit score (100 = comfortable headroom, 0 = doesn't fit) |
| `SPEED` | Fastest quant with estimated tok/s |
| `BALANCED` | Quant at the size midpoint of fitting options (speed/precision trade-off) |
| `PRECISION` | Least-lossy quant with quality score (% of fp16 quality retained) |
| `QUANT` | All fitting quants ordered from smallest to largest |

The footer legend explains each column. The `"~"` prefix on SPEED and PRECISION values means the estimate is extrapolated, not directly measured for that architecture. The `Try:` lines at the end suggest the fastest, balanced, and least precision loss quants with pull commands.

When a repo has FIT=0 (doesn't fit RAM+VRAM), SPEED/BALANCED/PRECISION show `-` and QUANT is empty — the repo is listed but no recommendations are made.

#### `discover-models`, `sync-hf-metadata`, and `export-registry` CLI

Generate or refresh `models.toml` without a full deploy:

```bash
# Fresh discover from a directory (also writes models.json)
./gguf-switchboard discover-models /models -o models.toml

# Merge with an existing registry (preserves customizations by file path)
./gguf-switchboard discover-models /models -o models.toml --merge models.toml

# Enrich empty description / max_context_length / min_vram_gb / capabilities / hf_repo from Hugging Face
./gguf-switchboard sync-hf-metadata models.toml

# Export portable JSON from an existing registry
./gguf-switchboard export-registry models.toml -o models.json
```

`sync-hf-metadata` also runs automatically on **server launch** and on **`POST /v1/models/refresh`** (and the periodic rescan watcher). Failures are logged and the server continues with the local registry. The standalone CLI remains available for offline/manual runs.

`sync-hf-metadata` matches each local GGUF against the Hub (`filter=gguf`), prefers exact sibling filenames and `lmstudio-community` repos, and **only fills empty fields** (explicit `kind`, `context_size`, `extra_args`, etc. are never overwritten). Swagger Try-it-out then shows a live model dropdown from `/api-docs/openapi.json`.

`discover-models`:

- Recursively scans for `.gguf` files
- Detects `llama-server` via `command -v llama-server` (falls back to `/usr/local/bin/llama-server`)
- Writes `version = 1`, `[defaults]`, and `[[models]]` entries with aliases, display names, and inferred `kind`
- Sets `auto_discover = true` on fresh output
- Writes a sibling **`models.json`** next to the output TOML path
- Marks the first suitable model as `priority` unless an existing merge already defines one (embedding models are never auto-priority)
- Deduplicates entries with the same alias or file on merge

#### Docker (`models.docker.toml`)

For Docker deployments, use `models.docker.toml` (mounted by `docker-compose` alongside `config.docker.toml`). The same schema applies; paths are container paths (`/models`, `/usr/local/bin/llama-server`). Example entries for thinking models are included in the repo template.

#### Customizing aliases and priorities

1. Edit `models.toml` — set `alias`, `display_name`, `priority`, `kind`, `enabled`, `context_size`, or `extra_args` on `[[models]]` entries
2. Re-run `./deploy.sh --refresh-models` (or `discover-models --merge`) to pick up new GGUF files while keeping your edits
3. Restart the service: `sudo systemctl restart gguf-switchboard`

Only one model should have `priority = true` (the idle-timeout default). If multiple are set, the runtime keeps the first and clears the rest with a warning. If none is set after discovery, the best-matching chat model is marked priority (embeddings are skipped).

**Tip — Llama 3.1 tool-call behavior:** Some Meta Llama 3.1 GGUFs ship a tool-use chat template that makes `/v1/chat/completions` return JSON like `{"name":"...","parameters":{...}}` instead of a normal answer (even with no `tools` in the request). gguf-switchboard auto-adds `--chat-template llama3` for Llama 3.1 models unless you already set `--chat-template`, `--chat-template-file`, or `--jinja` in `extra_args`.

To opt into GGUF tool-calling templates instead:

```toml
extra_args = ["--jinja"]
```

Raw `/v1/completions` (no chat template) is unaffected.

### Inline model config (advanced)

You can still define models directly in `config.toml` when you need full control over backend args:

```toml
bind = "0.0.0.0:9090"        # Address to listen on
startup_timeout = 60           # Max seconds to wait for model health
idle_timeout = 600             # Seconds before priority model auto-loads
default_backend = "llama.cpp"  # Default backend engine
vram_gb = 12                   # GPU VRAM for context sizing (see models.toml path)

[models.local-gemma-code]
backend = "llama.cpp"
display_name = "Gemma 3 Coding Model"
command = "/usr/local/bin/llama-server"
args = [
    "-m", "/models/gemma-3-4b.gguf",
    "--host", "127.0.0.1",
    "--port", "18081",
    "-c", "16384",
    "-ngl", "999",
]
backend_url = "http://127.0.0.1:18081/v1"
health_url = "http://127.0.0.1:18081/health"
priority = true                # Auto-load after idle timeout

[models.local-qwen-coder]
backend = "llama.cpp"
display_name = "Qwen 2.5 Coder"
command = "/usr/local/bin/llama-server"
args = [
    "-m", "/models/qwen2.5-coder-7b.gguf",
    "--host", "127.0.0.1",
    "--port", "18082",
    "-c", "16384",
    "-ngl", "999",
]
backend_url = "http://127.0.0.1:18082/v1"
health_url = "http://127.0.0.1:18082/health"
priority = false
```

### Fields

| Field | Description |
|-------|-------------|
| `bind` | Socket address for the HTTP server |
| `startup_timeout` | Seconds to wait for a backend to become healthy |
| `idle_timeout` | Seconds of inactivity before the priority model loads |
| `default_backend` | Fallback backend engine name |
| `vram_gb` | Assumed GPU capacity in GB — heuristic for per-model `-c` when not set in `models.toml` (default: `12` for RTX 3060); also VRAM fallback when `auto_ngl` cannot query `nvidia-smi` |
| `auto_ngl` | When `true`, pick `-ngl` at load from free VRAM + GGUF size (default: `false`) |
| `models_file` | Path to model registry (`models.toml` or `models.json`) |
| `models.<id>.backend` | Engine type (`llama.cpp` or `vllm`) |
| `models.<id>.display_name` | Human-readable name shown in `/v1/models` |
| `models.<id>.command` | Path to the backend binary |
| `models.<id>.args` | Command-line arguments (model path, port, context size, etc.) |
| `models.<id>.backend_url` | Base URL for the backend's OpenAI-compatible API |
| `models.<id>.health_url` | Health check endpoint URL |
| `models.<id>.priority` | If `true`, auto-loads after `idle_timeout` |
| `memory_warning_threshold` | RAM usage % that logs a warning |
| `memory_critical_threshold` | RAM usage % that auto-unloads the active model |
| `memory_check_interval_secs` | Seconds between RAM pressure checks |
| `context_fallback_min` | Lowest `-c` value used when auto-reducing context after an OOM-class load failure |
| `switch_drain_timeout_secs` | Seconds to wait for in-flight requests before switching models (default `120`) |
| `switch_strategy` | `unload_first` (default): stop the resident model before starting the next so it gets the whole GPU; previous model is re-loaded if the switch fails. `load_first`: start the next model while the previous is still resident — only sensible when VRAM can hold both, otherwise the new model OOMs into the fallback ladder and loads slowly / partly on CPU |
| `prewarm_recent_models` | After each load, re-read the GGUF files of the N most recently used other models into the OS page cache (background, cancelled when a real load starts). Speeds up switching back when RAM ≫ combined model sizes. Default `0` (off) |
| `priority_load_cooldown_secs` | Seconds to skip priority-model reload after a failed load (default `300`) |
| `models_rescan_interval_secs` | Seconds between automatic model-directory rescans (default `86400` = daily). `0` disables. |

### Context size (`-c`)

Per-model context is chosen in this order:

1. `context_size` on the `[[models]]` entry (explicit override)
2. **Capacity heuristic** from `vram_gb` in `config.toml` (default `12`) using model file size and kind
3. `defaults.context_size` in `models.toml` as the ceiling/fallback (bundled default **`16384`** — raise to `32768` or `65536` in `models.toml` if you have spare VRAM)

```toml
# config.toml — set to your GPU VRAM (RTX 3060 = 12)
vram_gb = 12
```

Typical results with `vram_gb = 12` (when `context_size` is not set per model):

| Model class | Suggested `-c` |
|-------------|----------------|
| Embedding | 8192 |
| 8B chat (Q4, ~5 GB file) | 32768 |
| 30B MoE / large GGUF (≥12 GB file) | 16384 |

Explicit `context_size` on a `[[models]]` entry always wins. Inline `[models.*]` blocks in `config.toml` use whatever `-c` you set in `args` directly.

**After changing `-c`**, restart the runtime (or trigger a model reload) so `llama-server` picks up the new value:

```bash
sudo systemctl restart gguf-switchboard
# or
./deploy.sh
```

**VRAM tradeoff:** larger context uses more GPU memory. On constrained GPUs (e.g. 12 GB), you may need to lower `-c` per model if loads fail or you hit OOM — especially for larger quantised models.

**Automatic fallback:** when a model load fails with an OOM-class error (detected from stderr), the runtime halves the context size and retries until it succeeds or reaches `context_fallback_min` (default `8192`). Missing files, port conflicts, and other non-OOM failures do not reduce context. The reduced value applies for the rest of the process lifetime (it is not written back to `config.toml`).

```toml
context_fallback_min = 8192
```

### Auto GPU layers (`auto_ngl`)

When `auto_ngl = true` in `config.toml`, each model load picks `-ngl` from:

1. Free VRAM via `nvidia-smi` (first GPU), or `vram_gb * 1024` if unavailable (macOS/Metal has no live probe yet)
2. ~80% of that as usable VRAM (KV/overhead reserve)
3. GGUF file size and `block_count` from the GGUF header

If the file fits in usable VRAM, all layers go on GPU; otherwise `-ngl` is scaled by `usable_vram / file_size`. This is a **heuristic**, not live layer telemetry — oversized KV or concurrent GPU use can still OOM.

**Overrides (win over auto):** `[[models]].ngl`, or `-ngl` / `--n-gpu-layers` in `extra_args`. Default remains `defaults.ngl = 999` when `auto_ngl` is false.

```toml
# config.toml
auto_ngl = true
vram_gb = 12   # fallback when nvidia-smi is missing
```

### ModelFitPlanner (`[fit]` section)

An opt-in hardware-aware preflight planner that inspects GPU topology, free VRAM, and model metadata before every load to produce a safe launch profile. On OOM, it advances through a bounded fallback ladder instead of blindly retrying.

```toml
# config.toml
[fit]
enabled = false          # opt-in (default: false)
vram_reserve_mb = 2048   # safety headroom subtracted from free VRAM (MB)
multi_gpu = "auto"       # "auto" detects GPUs and computes tensor-split from free VRAM
split_mode = "layer"     # "layer" (recommended), "row", or "none"
max_attempts = 5         # maximum load attempts before giving up
cache_profiles = true    # persist successful profiles to model-profiles.json
```

| Field | Description |
|-------|-------------|
| `fit.enabled` | When `true`, the planner runs before every model load to produce safe launch params. Default `false`. |
| `fit.vram_reserve_mb` | MB of free VRAM reserved for KV cache and overhead. Default `2048`. |
| `fit.multi_gpu` | Multi-GPU strategy: `"auto"` detects GPUs and computes tensor-split from free VRAM. Default `"auto"`. |
| `fit.split_mode` | GPU split mode for multi-GPU: `"layer"` (recommended), `"row"`, or `"none"`. Default `"layer"`. |
| `fit.max_attempts` | Maximum fallback ladder attempts before giving up. Default `5`. |
| `fit.cache_profiles` | When `true`, successful load profiles are persisted to `/var/lib/gguf-switchboard/model-profiles.json` so subsequent loads skip the fallback ladder entirely. Default `true`. |

When `fit.enabled = true`, the planner produces a `FitPlan` with context size, nGL, split mode, and KV cache type. On OOM, it advances through a bounded degradation sequence:

1. Requested context + default KV + auto-fit GPU
2. Requested context + Q8 KV + auto-fit GPU
3. 75% context + Q8 KV + auto-fit GPU
4. 50% context + Q8 KV + auto-fit GPU
5. 25% context + Q8 KV + reduced GPU offload

The minimum context is clamped to 4096. Cached profiles (`model-profiles.json`) let subsequent loads skip the ladder entirely when the same model + context combination has been loaded successfully before.

**Overrides:** Per-model `ngl` or `-ngl` in `extra_args` pins GPU layers and disables auto-fitting for that model. `context_size` on a `[[models]]` entry also overrides the planner's context recommendation.

### Per-model advanced fields

These fields on `[[models]]` entries control the fit planner and embedding behavior per model:

```toml
[[models]]
alias = "my-model"
file = "my-model-Q4_K_M.gguf"
# ... standard fields ...

# Fit planner overrides
gpu_fit = "auto"           # "auto" or "manual" — override fit planner for this model
split_mode = "layer"       # per-model split mode override ("layer", "row", "none")
kv_cache_type = "q8_0"     # per-model KV cache type override ("q8_0", "q4_0")

# Embedding batch overrides
batch_size = 2048          # logical batch size (-b); important for large embedding inputs
ubatch_size = 2048         # physical micro-batch size (-ub); must be <= batch_size
```

| Field | Description |
|-------|-------------|
| `[[models]].gpu_fit` | Override the fit planner's GPU strategy for this model: `"auto"` (use planner) or `"manual"` (skip planning). |
| `[[models]].split_mode` | Per-model GPU split mode override: `"layer"`, `"row"`, or `"none"`. Overrides `fit.split_mode`. |
| `[[models]].kv_cache_type` | Per-model KV cache type override: `"q8_0"`, `"q4_0"`, etc. Overrides the planner's KV cache recommendation. |
| `[[models]].batch_size` | Logical batch size (`-b`). Controls maximum tokens processed in a single batch. Important for embedding models with large inputs. Default: llama.cpp built-in. |
| `[[models]].ubatch_size` | Physical micro-batch size (`-ub`). Must be `<= batch_size`. Critical for embedding models. Default: llama.cpp built-in. |

`batch_size` and `ubatch_size` are primarily useful for embedding models where large input arrays need to be split into manageable batches. When set, they are passed as `-b` and `-ub` flags to `llama-server`.

### Balanced embedding VRAM profiles

Embedding models use a hardware-aware balanced profile by default, independently of the general `[fit]` switch. The planner reads live free VRAM when available, reserves the larger of 15% or 1536 MB, subtracts GGUF weight size, and selects bounded context, batch, and micro-batch values from the remaining headroom.

```toml
[embedding_fit]
enabled = true
profile = "balanced"
vram_reserve_percent = 15
vram_reserve_min_mb = 1536
queue_timeout_secs = 30
```

| Headroom after reserve and weights | Context ceiling | Batch | Micro-batch | Request concurrency |
|---:|---:|---:|---:|---:|
| under 1.5 GB | 2048 | 256 | 128 | 1 |
| 1.5–3 GB | 4096 | 512 | 256 | 1 |
| 3–5 GB | 8192 | 1024 | 512 | 1 |
| 5–8 GB | 8192 | 2048 | 1024 | 2 |
| over 8 GB | 16384 | 4096 | 2048 | 2 |

Requests beyond the active model's concurrency enter a bounded in-process queue. When `queue_timeout_secs` expires, the API returns `429` with `Retry-After` instead of forwarding more simultaneous work to `llama-server`. The active runtime profile reports `batch_size`, `ubatch_size`, and `embedding_concurrency`; Prometheus exports queue depth, queue wait, and rejected-request metrics.
