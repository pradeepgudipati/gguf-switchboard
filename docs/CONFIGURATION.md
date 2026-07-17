# Configuration

> [← Back to README](../README.md)

Configuration is split across two files:

| File | Purpose |
|------|---------|
| **`config.toml`** | Server bind address, idle timeout, GPU VRAM, database path |
| **`models.toml`** | Model registry — aliases, GGUF paths, priorities, per-model overrides |

Default paths after `deploy.sh`: `config.toml` and `models.toml` in the repo checkout.  
Machine-specific copies are also synced to **`models.local.toml`** and **`models.local.json`** (gitignored).

Override the config directory (optional):

```bash
GGUF_SWITCHBOARD_CONFIG_DIR=/etc/gguf-switchboard ./deploy.sh
```

### Server configuration (`config.toml`)

```toml
bind = "0.0.0.0:9090"
startup_timeout = 60
idle_timeout = 600
default_backend = "llama.cpp"

# GPU VRAM in GB — sizes per-model context (-c) when not set in models.toml
# RTX 3060 = 12; lower if you share VRAM with a display or other apps
vram_gb = 12

database_path = "/var/lib/gguf-switchboard/usage.db"

# Model registry (TOML or portable JSON)
models_file = "models.toml"
```

If `models_file` is omitted but a sibling `models.toml` exists next to your config, it is loaded automatically.

See [Context size (`-c`)](#context-size-c) for how `vram_gb` affects per-model `-c` values.

### Model configuration (`models.toml`)

```toml
version = 1

[defaults]
models_dir = "/models"                        # Root directory for GGUF files
llama_server = "/usr/local/bin/llama-server"  # Backend binary (auto-detected on discover)
host = "127.0.0.1"                            # llama-server bind host
base_port = 8081                              # First model port; others increment from here
context_size = 16384                          # Safe default for consumer GPUs; raise if you have VRAM headroom
ngl = 999                                     # Default -ngl (GPU layers)
backend = "llama.cpp"

auto_discover = true    # Also register any .gguf under models_dir not listed in [[models]]

[[models]]
alias = "gemma-4-e4b"     # API model id — use this in Cursor, Cline, etc.
file = "gemma-4-E4B-it-Q4_K_M.gguf"
display_name = "Gemma 4 E4B"
kind = "chat"             # chat | coder | vision | embedding (inferred when omitted)
enabled = true            # false = hide from /v1/models and scheduling
priority = true           # Auto-load after idle_timeout (only one should be true)
# port = 8085             # Override auto-assigned port (optional)
# context_size = 32768    # Override VRAM-based default from config.toml vram_gb (optional)
# description = "..."     # Optional blurb for /v1/models + Swagger (or run sync-hf-metadata)
# max_context_length = 131072  # Model max context from HF/GGUF metadata
# min_vram_gb = 6         # Approximate minimum VRAM (GB)
# capabilities = ["tools", "vision"]
# hf_repo = "lmstudio-community/Qwen3.5-9B-GGUF"
# extra_args = ["--jinja"]  # Extra llama-server flags (optional)
```

| Field | Description |
|-------|-------------|
| `version` | Registry schema version (currently `1`) |
| `defaults.models_dir` | Directory (or comma-separated directories) scanned for llama.cpp-loadable GGUF files |
| `defaults.llama_server` | Path to `llama-server` binary |
| `defaults.base_port` | Starting port; model at index *N* uses `base_port + N` unless `port` is set |
| `defaults.context_size` | Fallback/ceiling context window when `vram_gb` heuristics do not apply (default `16384`; raise for 24 GB+ GPUs) |
| `auto_discover` | When `true`, any `.gguf` under `models_dir` not listed in `[[models]]` is registered at runtime |
| `[[models]].alias` | Short id used in API requests (`model` field) |
| `[[models]].file` | GGUF filename relative to `models_dir`, or absolute path |
| `[[models]].display_name` | Human-readable name; defaults to a title-cased alias |
| `[[models]].kind` | `chat`, `coder`, `vision`, or `embedding` — inferred from alias/file when omitted |
| `[[models]].enabled` | When `false`, model is omitted from `/v1/models` and scheduling |
| `[[models]].priority` | If `true`, this model loads automatically after `idle_timeout` |
| `[[models]].port` | Override the auto-assigned backend port |
| `[[models]].context_size` | Override per-model `-c` (otherwise sized from `vram_gb`) |
| `[[models]].description` | Optional description shown in `/v1/models` and Swagger |
| `[[models]].max_context_length` | Model max context from HF/GGUF metadata (not the serving `-c`) |
| `[[models]].min_vram_gb` | Approximate minimum VRAM in GB (weights floor) |
| `[[models]].capabilities` | Tags such as `tools`, `vision`, `reasoning` |
| `[[models]].hf_repo` | Matched Hugging Face repo id after `sync-hf-metadata` |
| `[[models]].extra_args` | Extra flags appended to `llama-server` launch args |

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

After `deploy.sh`, copies land in the repo as **`models.local.toml`** and **`models.local.json`** (gitignored) so `git pull` updates do not conflict with your machine-specific registry.

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

Ports are assigned sequentially from `defaults.base_port`:

| Index | Port (default base 8081) |
|-------|--------------------------|
| 0 | 8081 |
| 1 | 8082 |
| 2 | 8083 |

Set `port` on a specific `[[models]]` entry to pin a backend to a fixed port.

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

`./deploy.sh` generates `$CONFIG_DIR/models.toml` (default: repo checkout) when:

- **First install** — no existing `models.toml` in the config dir (auto-discovers from GGUF files)
- **`--refresh-models`** — explicitly regenerate from disk, merging with the existing registry

Subsequent deploys without `--refresh-models` keep the existing registry unchanged.

When generation runs:

1. Builds the release binary (required before `discover-models`)
2. Detects the models directory (see below)
3. Runs `discover-models` to scan for `.gguf` files
4. Merges with the existing registry when present — preserves `alias`, `display_name`, `priority`, `port`, `context_size`, `kind`, `enabled`, and `extra_args` per file; **duplicate alias/file entries are deduplicated**
5. Installs the result to `$CONFIG_DIR/models.toml` and writes sibling **`models.json`**
6. Syncs copies to **`models.local.toml`** / **`models.local.json`** in the repo (gitignored)
7. Prints a table of configured models after the service is healthy

**Models directory detection**:

1. `$MODELS_DIR` environment variable (may be comma-separated)
2. `models_dir` from existing `models.toml` (deploy target or repo copy)

If no directory is configured or discovery fails, deploy warns and copies the template `models.toml` if needed — deploy does not fail.

Set `models_dir` in `models.toml` or pass `MODELS_DIR` when models live outside `/models`:

```bash
MODELS_DIR=/path/to/models ./deploy.sh --refresh-models
```

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
    "--port", "8081",
    "-c", "16384",
    "-ngl", "999",
]
backend_url = "http://127.0.0.1:8081/v1"
health_url = "http://127.0.0.1:8081/health"
priority = true                # Auto-load after idle timeout

[models.local-qwen-coder]
backend = "llama.cpp"
display_name = "Qwen 2.5 Coder"
command = "/usr/local/bin/llama-server"
args = [
    "-m", "/models/qwen2.5-coder-7b.gguf",
    "--host", "127.0.0.1",
    "--port", "8082",
    "-c", "16384",
    "-ngl", "999",
]
backend_url = "http://127.0.0.1:8082/v1"
health_url = "http://127.0.0.1:8082/health"
priority = false
```

### Fields

| Field | Description |
|-------|-------------|
| `bind` | Socket address for the HTTP server |
| `startup_timeout` | Seconds to wait for a backend to become healthy |
| `idle_timeout` | Seconds of inactivity before the priority model loads |
| `default_backend` | Fallback backend engine name |
| `vram_gb` | Assumed GPU capacity in GB — heuristic for per-model `-c` when not set in `models.toml` (default: `12` for RTX 3060); does not query live GPU memory |
| `models_file` | Path to model registry (`models.toml` or `models.json`) |
| `models.<id>.backend` | Engine type (`llama.cpp`) |
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
| `priority_load_cooldown_secs` | Seconds to skip priority-model reload after a failed load (default `300`) |

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
