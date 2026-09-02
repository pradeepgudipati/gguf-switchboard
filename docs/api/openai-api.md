# OpenAI API

> [← Back to README](../../README.md)

GGUF Switchboard exposes an OpenAI-compatible HTTP API. All endpoints work identically regardless of whether the loaded model runs through llama.cpp (GGUF) or vLLM (SafeTensors).

## Chat Completions

### Basic request

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

### Streaming

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

## Text Completions

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

## Embeddings

```bash
curl http://localhost:9090/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local-gemma-code",
        "input": "The quick brown fox jumps over the lazy dog."
    }'
```

## Responses API

### Basic request

```bash
curl http://localhost:9090/v1/responses \
    -H "Content-Type: application/json" \
    -d '{
        "model": "local-gemma-code",
        "input": "What is the capital of France?",
        "instructions": "Answer concisely."
    }'
```

### Function tools

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

## Model Management

### List models

```bash
# OpenAI-compatible model list (enabled models only)
curl http://localhost:9090/v1/models

# Portable registry JSON (all entries, with kind/tags)
curl http://localhost:9090/v1/models/registry.json
```

### Individual model info

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

### Hot-reload model registry

After adding or removing GGUF files from the models directory, trigger a hot-reload without restarting the service:

```bash
curl -X POST http://localhost:9090/v1/models/refresh
```

This re-scans the configured `models_dir`, merges new discoveries with the existing registry, syncs HF metadata, and updates the running model list. Also available in Swagger UI as **Rescan Models**.

A periodic rescan watcher runs every `models_rescan_interval_secs` (default: daily) to pick up new models automatically.

## Health & Status

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

## API Explorer (Swagger UI)

After starting the runtime, open the interactive API docs in your browser:

- **Swagger UI:** http://localhost:9090/swagger-ui/
- **OpenAPI spec:** http://localhost:9090/api-docs/openapi.json
- **Model registry JSON:** http://localhost:9090/v1/models/registry.json
- **Root redirect:** http://localhost:9090/ → Swagger UI

All endpoints are listed and testable from the Swagger UI — health, models, chat completions, embeddings, usage, and more.

A **Model** dropdown and **models.json** download link appear in the top bar. The selected model is persisted in the browser and applied to the `model` field on send. Request body textareas are editable — your changes are preserved until you edit them again; only Swagger placeholder values are sanitized when a request is sent.

## Prometheus Metrics

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

`GET /status` also returns `last_switch` — a millisecond breakdown (`drain_ms`, `unload_previous_ms`, `load_ms`, `rollback_ms`, `total_ms`) of the most recent switch — and every switch logs a `Model switch finished` line with the same fields.

## Structured Logging

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
