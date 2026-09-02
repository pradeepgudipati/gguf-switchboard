# Usage

> [← Back to README](../README.md)

API examples, SDK snippets, IDE setup, and monitoring.

## Running locally

```bash
# Create user-owned runtime configuration once
cp config.example.toml config.toml
cp models.example.toml models.toml

# With cargo
cargo run --release -- config.toml

# With environment-based log level
RUST_LOG=debug cargo run --release -- config.toml

# With custom port
# (edit config.toml bind = "0.0.0.0:3000")
```

### Pre-commit checks

Install git hooks to run standard Rust checks before each commit (format, clippy with denied warnings, build, tests):

```bash
./scripts/install-hooks.sh
```

Run the same checks manually:

```bash
./precommit.sh
```

## API Documentation

See dedicated API documentation:

- [OpenAI API](api/openai-api.md) — Chat Completions, Completions, Embeddings, Responses, Audio examples
- [Anthropic API](api/anthropic-api.md) — Anthropic Messages API translation

## IDE Integration

See dedicated integration guides:

- [OpenCode](integrations/opencode.md)
- [Cursor](integrations/cursor.md)
- [Cline](integrations/cline.md)
- [Continue](integrations/continue.md)
- [OpenAI SDK](integrations/openai-sdk.md)

## Model Management

See dedicated model documentation:

- [Model Search](models/model-search.md) — Hardware-aware Hugging Face search
- [GGUF Models](models/gguf.md) — GGUF format, quantizations, pulling models
- [SafeTensors Models](models/safetensors.md) — SafeTensors format, vLLM support

## Runtime Documentation

See dedicated runtime documentation:

- [Runtime Overview](runtimes/overview.md) — llama.cpp and vLLM backend selection
- [Model Switching](runtimes/model-switching.md) — Single-slot swapping, drain, rollback
- [VRAM Management](runtimes/vram-management.md) — Hardware-aware fit planning, OOM fallback

## Systemd service

Native install is recommended: the runtime spawns `llama-server` or vLLM as a child and needs direct GPU and model-file access. Plain `./deploy.sh` installs both engines; use `--skip-llama-cpp` or `--skip-vllm` only to retain an existing engine during an update.

**Install or upgrade:** see [Install on Linux](deployment/linux.md) for deployment and update instructions. Day-to-day:

```bash
ggs status
ggs logs
ggs logs watch
ggs logs --tail 250
ggs restart
```

`ggs status` reports whether the background systemd service is `running` or
`stopped`. `ggs logs` prints the latest 100 journal entries, `watch` follows new
entries, and `--tail N` prints the latest positive number `N` without a pager.

## Hardware-aware model search

`ggs models search` scores every discovered quant against your detected hardware:

```bash
ggs models search "gemma"
ggs models search "Qwen3.5 9B" --limit 5
ggs models search "Qwen3.5 9B" --ram-bandwidth-gbps 50
```

Search prints total system RAM, NVIDIA VRAM, and their combined capacity before an aligned table with FIT/SPEED/BALANCED/PRECISION columns:

```
Hardware: System RAM 32.0 GiB | NVIDIA VRAM 24.0 GiB | Total 56.0 GiB
Speed model inputs: GPU bandwidth 1008 GB/s (NVIDIA GeForce RTX 4090) | RAM bandwidth 40 GB/s (assumed) | GPU efficiency 0.55 | CPU efficiency 0.35

REPO                                               | FILES |     SIZE | FIT | CONTEXT    | ARCH  | SPEED            | BALANCED               | PRECISION    | QUANT
bartowski/Qwen3.5-9B-GGUF                          |    24 |  9421 MB | 100 | 32768 tok  | qwen3 | Q4_K_M ~127tok/s | Q5_K_M ~91tok/s/~98.9% | Q6_K ~99.6%  | Q2_K,Q3_K_M,Q4_K_M,Q5_K_M,Q6_K,Q8_0
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

The footer legend explains each column. The `"~"` prefix on SPEED and PRECISION values means the estimate is extrapolated, not directly measured for that architecture. The `Try:` lines suggest the fastest, balanced, and least precision loss quants with pull commands.

When a repo has FIT=0 (doesn't fit RAM+VRAM), SPEED/BALANCED/PRECISION show `-` and QUANT is empty — the repo is listed but no recommendations are made.

See [docs/QUANT_SCORING.md](QUANT_SCORING.md) for the exact formulas, sources, and how to override RAM bandwidth (`--ram-bandwidth-gbps`) with a measured value.

## Model management CLI

```bash
# Search Hugging Face for GGUF models
ggs models search "Qwen3.5 9B"

# Browse available files in a repo
ggs models files lmstudio-community/Qwen3.5-9B-GGUF

# Download, validate, and register a model (runs a speed test if the server is up)
ggs models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --dir /var/lib/gguf-switchboard/models

# Tune parallel aria2 connections (default 8, maximum 16)
ggs models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --connections 8

# Skip the post-pull speed test
ggs models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --no-bench

# Dry-run: show what the fit planner would generate
ggs models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --fit-dry-run

# Search and pull Hugging Face Safetensors models for vLLM
ggs models search vllm "Qwen 7B Instruct"
ggs models pull vllm Qwen/Qwen2.5-7B-Instruct \
  --registry /opt/gguf-switchboard/models.toml
```

`models pull` performs the complete workflow: fetches the repo tree, resolves `--quant` case-insensitively, streams the download with progress, validates the GGUF header, generates an alias, runs the fit planner to generate context_size/ngl/extra_args, and merges into `models.toml`. A successful pull refreshes a running gguf-switchboard server automatically.

`models pull vllm` requires `config.json` and Safetensors weights. It downloads weights plus tokenizer/configuration files, detects AWQ/GPTQ/FP8-style quantization metadata when declared by the repository, writes the vLLM source and launch options into the registry, and refreshes a running server. It never enables `trust_remote_code` automatically.

When an alias has both source types and no explicit backend pin, startup prefers vLLM if the Safetensors weights fit detected VRAM; otherwise it uses the GGUF source through llama.cpp.

## Model inventory and removal

```bash
# Numbered inventory of every GGUF file / Safetensors dir under the model dirs,
# with the registered alias (if any). Add --json for machine output.
ggs models list

# Delete by alias/name or by the number from `models list` — prompts for
# confirmation (add --yes to skip), removes the file/dir and the models.toml entry
ggs models delete qwen3-embedding-4b
ggs models delete 2 --yes
```

## Monitoring

### Structured Logging

Logs are emitted as JSON to stdout:

```json
{
    "timestamp": "2025-01-15T10:30:00.000Z",
    "level": "INFO",
    "message": "Model loaded and healthy",
    "model": "local-gemma-code",
    "elapsed_ms": 3420,
    "request_id": "abc-123"
}
```

Set `RUST_LOG` to control verbosity:

```bash
RUST_LOG=info          # Default
RUST_LOG=debug         # Verbose
RUST_LOG=gguf_switchboard=debug,tower_http=info  # Per-crate
```

### Prometheus Metrics

See [OpenAI API — Prometheus Metrics](api/openai-api.md#prometheus-metrics) for the full metrics reference.
