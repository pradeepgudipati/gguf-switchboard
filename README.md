# OpenAI Runtime

A **100% OpenAI API-compatible** local inference runtime that dynamically loads and unloads GGUF models on demand. Point any OpenAI SDK or tool at it — Python, Node, Cursor, Cline, Continue — and it Just Works.

## Why

When running local LLMs, you typically manage models manually: start `llama-server` for one model, stop it, start another. If you use multiple models — a code model in Cursor, a general model in Continue, an embedding model for search — you're juggling processes, ports, and GPU memory yourself.

OpenAI Runtime eliminates that. It sits between your tools and `llama-server`, presenting a single OpenAI-compatible endpoint. When a request comes in for a model that isn't loaded, it automatically unloads the current model, loads the requested one, and proxies the request. Your tools never know the difference — they just see an OpenAI API that happens to serve any configured model on demand.

The result: one endpoint, many models, zero manual process management.

```
┌─────────────────────────────────────────────────────────────┐
│                       OpenAI Runtime                         │
│                                                              │
│  /v1/chat/completions  ─┐                                    │
│  /v1/completions       ─┤                                    │
│  /v1/embeddings        ─┼──▶ Scheduler ──▶ Backend          │
│  /v1/responses         ─┤      (LRU)       (llama.cpp)      │
│  /v1/audio/*           ─┘                                    │
│                                                              │
│  Dynamic model loading: A→B→A without restart                │
│  Priority model auto-loads after configurable idle timeout   │
│  Prometheus metrics at /metrics                              │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Drop-in OpenAI API** — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/models`, `/v1/audio/*`
- **Dynamic model loading** — models are loaded/unloaded on demand; no restart needed
- **LRU eviction** — automatically unloads the least-recently-used model when capacity is reached
- **Priority model** — auto-loads your preferred model after a configurable idle timeout
- **SSE streaming** — full `text/event-stream` support with proper `[DONE]` termination
- **Prometheus metrics** — request counts, latency histograms, active request gauges
- **Graceful shutdown** — SIGTERM/SIGINT handling with backend cleanup
- **Production-ready** — structured JSON logging, request IDs, error responses matching OpenAI format

## Quick Start

### Prerequisites

- [Rust 1.85+](https://rustup.rs/) (edition 2024)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built with server support
- A GGUF model file

### Install & Run

```bash
# Clone
git clone https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard

# Edit config.toml to point to your llama-server binary and model
cp config.toml config.local.toml
$EDITOR config.local.toml

# Build and run
cargo run --release -- config.local.toml
```

The server starts on `0.0.0.0:9090` by default.

## Architecture

```
Client Request
     │
     ▼
┌──────────┐    ┌───────────────┐    ┌──────────────┐
│  Axum    │───▶│   Scheduler   │───▶│   Backend    │
│  Router  │    │               │    │  (llama.cpp) │
│          │    │ • LRU queue   │    │              │
│ /v1/...  │    │ • Load lock   │    │ • child proc │
│ /health  │    │ • Priority    │    │ • health ck  │
│ /metrics │    │   watcher     │    │ • HTTP proxy │
└──────────┘    └───────────────┘    └──────────────┘
```

**Scheduler** is the core component:
1. Request arrives for model `X`
2. If `X` is loaded → forward immediately
3. If model `Y` is loaded → unload `Y` → load `X` → wait for health → forward
4. After `idle_timeout` seconds with no requests, the priority model auto-loads

**Backend** (llama.cpp implementation):
- Spawns `llama-server` as a child process using the configured `command` + `args`
- Polls the health endpoint until healthy or timeout
- Proxies all OpenAI-compatible HTTP requests to the backend URL
- Parses SSE streams and re-emits them with proper framing

## Configuration

Configuration is a TOML file (default: `config.toml`):

```toml
bind = "0.0.0.0:9090"        # Address to listen on
startup_timeout = 60           # Max seconds to wait for model health
idle_timeout = 600             # Seconds before priority model auto-loads
default_backend = "llama.cpp"  # Default backend engine

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
    "-c", "32768",
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
| `models.<id>.backend` | Engine type (`llama.cpp`) |
| `models.<id>.display_name` | Human-readable name shown in `/v1/models` |
| `models.<id>.command` | Path to the backend binary |
| `models.<id>.args` | Command-line arguments (model path, port, context size, etc.) |
| `models.<id>.backend_url` | Base URL for the backend's OpenAI-compatible API |
| `models.<id>.health_url` | Health check endpoint URL |
| `models.<id>.priority` | If `true`, auto-loads after `idle_timeout` |

## Running Locally

```bash
# With cargo
cargo run --release -- config.toml

# With environment-based log level
RUST_LOG=debug cargo run --release -- config.toml

# With custom port
# (edit config.toml bind = "0.0.0.0:3000")
```

## Systemd Setup

```bash
# Build
cargo build --release
sudo cp target/release/openai-runtime /usr/local/bin/

# Create user
sudo useradd --system --create-home --shell /bin/bash openai-runtime

# Copy config
sudo mkdir -p /etc/openai-runtime
sudo cp config.toml /etc/openai-runtime/config.toml

# Install service
sudo cp openai-runtime.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now openai-runtime

# Check status
sudo systemctl status openai-runtime
sudo journalctl -u openai-runtime -f
```

## Systemd Setup (Recommended)

The native install is recommended because the runtime spawns `llama-server` as a child process and needs direct access to your GPU and model files.

```bash
# Build
cargo build --release

# Install binary
sudo cp target/release/openai-runtime /usr/local/bin/

# Create directories
sudo mkdir -p /etc/openai-runtime /var/lib/openai-runtime

# Copy and edit config
sudo cp config.toml /etc/openai-runtime/config.toml
$EDITOR /etc/openai-runtime/config.toml

# Create systemd service
sudo tee /etc/systemd/system/openai-runtime.service > /dev/null << 'EOF'
[Unit]
Description=OpenAI Runtime - Local LLM Inference Server
After=network.target

[Service]
Type=simple
User=pradeep
ExecStart=/usr/local/bin/openai-runtime /etc/openai-runtime/config.toml
Restart=on-failure
RestartSec=5
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now openai-runtime

# Check status
sudo systemctl status openai-runtime
sudo journalctl -u openai-runtime -f
```

### How It Works

```
Client Request
     │
     ▼
┌──────────────────────────────────────────────────┐
│              OpenAI Runtime (:9090)               │
│                                                   │
│  1. Request arrives for model "X"                 │
│  2. If model "Y" loaded → SIGTERM "Y"             │
│  3. Spawn llama-server for "X" → wait for health  │
│  4. Proxy request → return response               │
│  5. After idle_timeout → load priority model      │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │   llama-server   │
              │   (GPU loaded)   │
              │   One at a time  │
              └──────────────────┘
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

```bash
curl http://localhost:9090/v1/models
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
2. Set **Base URL** to `http://localhost:9090/v1`
3. Set **API Key** to any string
4. Set **Model** to your model id

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
| `openai_runtime_requests_total` | Counter | Total HTTP requests |
| `openai_runtime_inference_latency_seconds` | Histogram | End-to-end inference latency |
| `openai_runtime_model_load_latency_seconds` | Histogram | Model cold-start time |
| `openai_runtime_active_requests` | Gauge | Current in-flight requests |
| `openai_runtime_loaded_model` | Gauge | Whether a model is loaded (0/1) |
| `openai_runtime_backend_healthy` | Gauge | Backend health status (0/1) |
| `openai_runtime_streaming_requests` | Gauge | Active streaming connections |

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
RUST_LOG=openai_runtime=debug,tower_http=info  # Per-crate
```

## Benchmarks

### Throughput Test

```bash
# Install hey: go install github.com/rakyll/hey@latest

# Non-streaming throughput
hey -n 100 -c 4 \
    -m POST \
    -H "Content-Type: application/json" \
    -d '{"model":"local-gemma-code","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}' \
    http://localhost:9090/v1/chat/completions

# Streaming throughput
hey -n 50 -c 2 \
    -m POST \
    -H "Content-Type: application/json" \
    -d '{"model":"local-gemma-code","messages":[{"role":"user","content":"Count to 10"}],"stream":true,"max_tokens":100}' \
    http://localhost:9090/v1/chat/completions
```

### Model Switching Latency

```bash
# Time a cold model switch
time curl -s http://localhost:9090/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"local-qwen-coder","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}' \
    > /dev/null
```

## Project Structure

```
.
├── Cargo.toml              # Dependencies and build config
├── config.toml             # Example configuration
├── openai-runtime.service  # Systemd unit file
├── .github/workflows/
│   ├── ci.yml              # CI: check, clippy, build, test
│   └── release.yml         # Multi-platform release builds
└── src/
    ├── main.rs             # Entry point, signal handling
    ├── config/mod.rs       # TOML configuration loading
    ├── errors/mod.rs       # OpenAI-compatible error responses
    ├── types/              # Request/response type definitions
    │   ├── mod.rs          # Shared types (ModelInfo, Usage, etc.)
    │   ├── chat.rs         # Chat completion types
    │   ├── completions.rs  # Text completion types
    │   ├── embeddings.rs   # Embedding types
    │   ├── models.rs       # Model permission types
    │   └── responses.rs    # Responses API types
    ├── backend/
    │   ├── mod.rs          # Backend trait definition
    │   └── llama_cpp.rs    # llama.cpp backend implementation
    ├── scheduler/mod.rs    # Core scheduler with LRU + priority
    ├── state/mod.rs        # Shared application state
    ├── memory/mod.rs       # System memory pressure monitoring
    ├── db/mod.rs           # Token usage tracking (SQLite)
    ├── proxy/mod.rs        # SSE proxy helpers
    ├── metrics/mod.rs      # Prometheus metric collectors
    └── api/
        ├── mod.rs          # Router setup
        ├── chat.rs         # POST /v1/chat/completions
        ├── completions.rs  # POST /v1/completions
        ├── embeddings.rs   # POST /v1/embeddings
        ├── models.rs       # GET /v1/models
        ├── responses.rs    # POST /v1/responses
        ├── audio.rs        # POST /v1/audio/*
        ├── health.rs       # GET /health, /status
        ├── metrics.rs      # GET /metrics
        └── usage.rs        # GET /v1/usage
```

## License

MIT
