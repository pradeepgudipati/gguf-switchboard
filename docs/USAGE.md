# Usage

> [← Back to README](../README.md)

API examples, SDK snippets, IDE setup, and monitoring.

## Running locally

```bash
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

## Systemd service

Native install is recommended: the runtime spawns `llama-server` as a child and needs direct GPU + model-file access.

**Install or upgrade:** see [Fresh machine](../README.md#fresh-machine-linux--systemd) and [Updating](../README.md#updating) in the README. Day-to-day:

```bash
sudo systemctl status gguf-switchboard
sudo journalctl -u gguf-switchboard -f
sudo systemctl restart gguf-switchboard
```

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

### API Explorer (Swagger UI)

After starting the runtime, open the interactive API docs in your browser:

- **Swagger UI:** http://localhost:9090/swagger-ui/
- **OpenAPI spec:** http://localhost:9090/api-docs/openapi.json
- **Model registry JSON:** http://localhost:9090/v1/models/registry.json
- **Root redirect:** http://localhost:9090/ → Swagger UI

![Swagger UI with model dropdown](swagger-ui.png)

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
```

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

| Metric | Type | Description |
|--------|------|-------------|
| `gguf_switchboard_requests_total` | Counter | Total HTTP requests |
| `gguf_switchboard_inference_latency_seconds` | Histogram | End-to-end inference latency |
| `gguf_switchboard_model_load_latency_seconds` | Histogram | Model cold-start time |
| `gguf_switchboard_active_requests` | Gauge | Current in-flight requests |
| `gguf_switchboard_loaded_model` | Gauge | Whether a model is loaded (0/1) |
| `gguf_switchboard_backend_healthy` | Gauge | Backend health status (0/1) |
| `gguf_switchboard_streaming_requests` | Gauge | Active streaming connections |

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
