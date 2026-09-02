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

### Hardware-aware model search

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

### Model management CLI

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

### Model inventory and removal

```bash
# Numbered inventory of every GGUF file / Safetensors dir under the model dirs,
# with the registered alias (if any). Add --json for machine output.
ggs models list

# Delete by alias/name or by the number from `models list` — prompts for
# confirmation (add --yes to skip), removes the file/dir and the models.toml entry
ggs models delete qwen3-embedding-4b
ggs models delete 2 --yes
```

## Systemd service

Native install is recommended: the runtime spawns `llama-server` or vLLM as a child and needs direct GPU and model-file access. Plain `./deploy.sh` installs both engines; use `--skip-llama-cpp` or `--skip-vllm` only to retain an existing engine during an update.

**Install or upgrade:** see [Install on Linux](INSTALL-LINUX.md) for deployment and update instructions. Day-to-day:

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

## API Examples

### Chat Completions

```bash
curl http://localhost:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local-gemma-code",
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a binary search in Rust."}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }'
```

### Thinking models

`gemma-4-e4b` and `qwen3.5-9b` are thinking models served by llama.cpp with **reasoning enabled**. They emit chain-of-thought in a `reasoning_content` field on assistant messages (and stream deltas). The final answer is in `content` when the model finishes; if `max_tokens` is too low, reasoning may consume the budget and `content` can be empty — the runtime promotes `reasoning_content` into `content` in that case but keeps both fields when present.

Use **`max_tokens` 2048 or higher** for substantive questions so the model has room to think and answer. Short prompts with `max_tokens: 50` often return only thinking traces.

```bash
curl http://localhost:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4-e4b",
        "messages": [
            {"role": "user", "content": "Is Rust faster than Python for backend services?"}
        ],
        "max_tokens": 2048,
        "stream": false
    }'
```

Optional: pass template kwargs through to llama-server (model-specific), e.g. `chat_template_kwargs` in the request body when your client supports it.

### Streaming Chat

```bash
curl http://localhost:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local-gemma-code",
        "messages": [
            {"role": "user", "content": "Explain ownership in Rust."}
        ],
        "stream": true
    }'
```

### Text Completions

```bash
curl http://localhost:9090/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local-gemma-code",
        "prompt": "fn fibonacci(n: u64) -> u64 {",
        "max_tokens": 256,
        "temperature": 0.2
    }'
```

### Embeddings

```bash
curl http://localhost:9090/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local-gemma-code",
        "input": "The quick brown fox jumps over the lazy dog."
    }'
```

### List Models

After deploy, `./deploy.sh` prints configured models in the terminal. You can also query the API:

```bash
# OpenAI-compatible model list (enabled models only)
curl http://localhost:9090/v1/models

# Portable registry JSON (all entries, with kind/tags)
curl http://localhost:9090/v1/models/registry.json
```

### Individual Model Info

```bash
# Get detailed info for a specific model (includes tools_verified)
curl http://localhost:9090/v1/models/gemma-4-e4b

# Get runtime profile (effective context_size, ngl, split_mode, kv_cache_type, profile_source)
curl http://localhost:9090/v1/models/gemma-4-e4b/runtime
```

The `/v1/models/{model_id}` endpoint returns the model's `tools_verified` field — `true` if a real tool call succeeded at load time, `false` if the probe failed, `null` if not yet probed.

The `/v1/models/{model_id}/runtime` endpoint returns the `RuntimeProfileInfo` with the effective launch parameters:

```json
{
    "context_size": 32768,
    "ngl": 40,
    "split_mode": "layer",
    "kv_cache_type": "q8_0",
    "batch_size": null,
    "ubatch_size": null,
    "profile_source": "fit_planner"
}
```

`profile_source` indicates where the launch parameters came from: `"fit_planner"`, `"config"`, `"registry"`, or `"fallback"`.

### Hot-reload Model Registry

After adding or removing GGUF files from the models directory, trigger a hot-reload without restarting the service:

```bash
curl -X POST http://localhost:9090/v1/models/refresh
```

This re-scans the configured `models_dir`, merges new discoveries with the existing registry, syncs HF metadata, and updates the running model list. Also available in Swagger UI as **Rescan Models**.

A periodic rescan watcher runs every `models_rescan_interval_secs` (default: daily) to pick up new models automatically.

### Responses API

```bash
curl http://localhost:9090/v1/responses \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local-gemma-code",
        "input": "What is the capital of France?",
        "instructions": "Answer concisely."
    }'
```

### Anthropic Messages API

gguf-switchboard translates the Anthropic Messages API onto the loaded `llama-server` OpenAI backend. Streaming and tool calling are supported.

```bash
# Non-streaming
curl http://localhost:9090/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: not-needed" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
        "model": "gemma-4-e4b",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Explain the difference between threads and processes."}
        ]
    }'

# Streaming
curl http://localhost:9090/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: not-needed" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
        "model": "gemma-4-e4b",
        "max_tokens": 1024,
        "stream": true,
        "messages": [
            {"role": "user", "content": "Write a haiku about Rust programming."}
        ]
    }'

# With tool calling
curl http://localhost:9090/v1/messages \
    -H "Content-Type: application/json" \
    -H "x-api-key: not-needed" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
        "model": "gemma-4-e4b",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "What is the weather in San Francisco?"}
        ]
    }'
```

The request is translated to OpenAI format, forwarded to `llama-server`, and the response is translated back to Anthropic format. Tool definitions, tool calls, and content blocks are mapped bidirectionally.

### Responses API — Function Tools

The Responses API supports function tools with `tool_choice` and streaming:

```bash
curl http://localhost:9090/v1/responses \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4-e4b",
        "input": "What is the weather in Tokyo?",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ],
        "tool_choice": "auto"
    }'
```

Function tools are translated to/from Chat Completions format. Function calls are returned as top-level output items. Streaming events include `response.output_item.added`, `response.content_part.delta`, and `response.completed`.

### API Explorer (Swagger UI)

After starting the runtime, open the interactive API docs in your browser:

- **Swagger UI:** http://localhost:9090/swagger-ui/
- **OpenAPI spec:** http://localhost:9090/api-docs/openapi.json
- **Model registry JSON:** http://localhost:9090/v1/models/registry.json
- **Root redirect:** http://localhost:9090/ → Swagger UI


All endpoints are listed and testable from the Swagger UI — health, models, chat completions, embeddings, usage, and more.

A **Model** dropdown and **models.json** download link appear in the top bar. The selected model is persisted in the browser and applied to the `model` field on send. Request body textareas are editable — your changes are preserved until you edit them again; only Swagger placeholder values are sanitized when a request is sent.

### Health & Status

```bash
# Liveness probe
curl http://localhost:9090/health

# Detailed status
curl http://localhost:9090/status

# Prometheus metrics
curl http://localhost:9090/metrics

# NVIDIA processes with the loaded GGUF model name
./scripts/nvidia-smi-models.sh
./scripts/nvidia-smi-models.sh --watch 2
```

The model-aware NVIDIA view prints the standard `nvidia-smi` dashboard, then adds a process table that joins GPU usage with each process's `-m` or `--model` argument from `/proc`. Run it as the same user as `llama-server`, or with sufficient permission to read that process's command line. Processes whose command line is inaccessible show `-` for the model.

## SDK Examples

### Python (openai)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9090/v1",
    api_key="not-needed",  # any string works
)

# Chat completion
response = client.chat.completions.create(
    model="local-gemma-code",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="local-gemma-code",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print()
```

### Node.js (openai)

```javascript
import OpenAI from "openai";

const client = new OpenAI({
    baseURL: "http://localhost:9090/v1",
    apiKey: "not-needed",
});

// Chat completion
const response = await client.chat.completions.create({
    model: "local-gemma-code",
    messages: [
        { role: "system", content: "You are a helpful assistant." },
        { role: "user", content: "Hello!" },
    ],
});
console.log(response.choices[0].message.content);

// Streaming
const stream = await client.chat.completions.create({
    model: "local-gemma-code",
    messages: [{ role: "user", content: "Tell me a story." }],
    stream: true,
});
for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
console.log();
```

## IDE Integration

### Cursor

In Cursor settings, add a custom OpenAI-compatible model:

1. Open **Settings** → **Models** → **Add Model**
2. Set **API Base URL** to `http://localhost:9090/v1`
3. Set **API Key** to any string (e.g., `sk-local`)
4. Set **Model Name** to your model id (e.g., `local-gemma-code`)

### Cline (VS Code)

In Cline settings:

1. Select **OpenAI Compatible** as the API provider
2. Set **Base URL** to `http://localhost:9090/v1` (must include `/v1`)
3. Set **API Key** to any non-empty string (e.g., `sk-local`) — the runtime does not validate keys, but Cline requires the field
4. Set **Model** to your model id (must match `config.toml`, e.g., `gemma-4-e4b`)

If **"Use different models for Plan and Act modes"** is enabled, configure both modes separately (API key and base URL in each).

**Context errors:** Cline agent prompts can be large (30k+ tokens). If you see `exceed_context_size_error`, either start a fresh Cline task to reduce prompt size, or increase `-c` in `config.toml` and restart the runtime (see [Context size](#context-size-c) above).

### Continue (VS Code / JetBrains)

In `~/.continue/config.json`:

```json
{
    "models": [
        {
            "title": "Local Gemma Code",
            "provider": "openai",
            "model": "local-gemma-code",
            "apiBase": "http://localhost:9090/v1",
            "apiKey": "not-needed"
        }
    ]
}
```

## Monitoring

### Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `gguf_switchboard_requests_total` | Counter | — | Total HTTP requests |
| `gguf_switchboard_inference_latency_seconds` | Histogram | — | Inference time once the model is resident (streams are observed when they finish). Model load/switch wait is **not** included — see below |
| `gguf_switchboard_request_model_wait_seconds` | Histogram | `model` | Time a request waited for its model to become resident. Only observed on a miss, so `_count` = number of requests that paid a load, `_sum` = total seconds users spent waiting on loads |
| `gguf_switchboard_request_model_hit_total` | Counter | `model`, `result=hit\|miss` | Whether the requested model was already resident |
| `gguf_switchboard_model_switch_seconds` | Histogram | `model` (target), `result` | Whole switch: drain + unload previous + plan + every load attempt (+ rollback) |
| `gguf_switchboard_model_switch_phase_seconds` | Histogram | `model`, `phase=drain\|unload_previous\|plan\|spawn_to_healthy\|rollback` | Where the switch time goes |
| `gguf_switchboard_model_load_seconds` | Histogram | `model`, `result=ok\|oom_retry\|timeout\|error` | One `llama-server` spawn → `/health` attempt. `oom_retry` samples mean the fallback ladder is running |
| `gguf_switchboard_model_load_attempts_total` | Counter | `model`, `result` | Load attempts; `rate(...{result!="ok"})` is a good alert |
| `gguf_switchboard_model_last_load_seconds` | Gauge | `model` | Most recent successful spawn→healthy time per model |
| `gguf_switchboard_model_load_latency_seconds` | Histogram | — | Successful spawn→healthy time across all models (kept for existing dashboards) |
| `gguf_switchboard_model_switches_total` | Counter | `from`, `to`, `trigger=request\|priority`, `result` | Residency changes — shows ping-pong between models and how often the idle priority reload evicts a model |
| `gguf_switchboard_model_unloads_total` | Counter | `model`, `reason=switch\|idle_priority\|memory_pressure\|unhealthy\|registry_refresh\|shutdown` | Why models were unloaded |
| `gguf_switchboard_loaded_model` | Gauge | — | Whether a model is loaded (0/1) |
| `gguf_switchboard_loaded_model_info` | Gauge | `model` | `1` on the series of the resident model |
| `gguf_switchboard_active_requests` | Gauge | — | Current in-flight requests |
| `gguf_switchboard_streaming_requests` | Gauge | — | Active streaming connections |
| `gguf_switchboard_backend_healthy` | Gauge | — | Backend health status (0/1) |
| `gguf_switchboard_memory_usage_percent` | Gauge | — | System RAM usage |

Histogram buckets for the load/switch family go from 250 ms to 10 min; the inference family from 50 ms to 10 min.

Useful queries:

```promql
# p50 cold-switch cost per target model over the last hour
histogram_quantile(0.5, sum by (le, model) (rate(gguf_switchboard_model_switch_seconds_bucket{result="ok"}[1h])))

# Which phase dominates a switch?
sum by (phase) (rate(gguf_switchboard_model_switch_phase_seconds_sum[1h]))
  / sum by (phase) (rate(gguf_switchboard_model_switch_phase_seconds_count[1h]))

# OOM fallback ladder firing (should be ~0 with switch_strategy = "unload_first")
increase(gguf_switchboard_model_load_attempts_total{result="oom_retry"}[1d])

# Seconds users spent waiting on model loads, per model
increase(gguf_switchboard_request_model_wait_seconds_sum[1d])

# Idle priority reloads evicting the model people actually use
increase(gguf_switchboard_model_unloads_total{reason="idle_priority"}[1d])
```

`GET /status` also returns `last_switch` — a millisecond breakdown (`drain_ms`, `unload_previous_ms`,
`load_ms`, `rollback_ms`, `total_ms`) of the most recent switch — and every switch logs a
`Model switch finished` line with the same fields.

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
