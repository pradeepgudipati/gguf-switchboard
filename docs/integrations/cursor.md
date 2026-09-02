# Cursor Integration

> [← Back to README](../../README.md)

Use GGUF Switchboard as a custom OpenAI-compatible model provider in Cursor.

## What this enables

Cursor connects to GGUF Switchboard's OpenAI-compatible API, giving you access to any locally loaded GGUF or SafeTensors model for code completion, chat, and editing.

## Prerequisites

- GGUF Switchboard running on `http://localhost:9090`
- At least one model registered and loaded
- Cursor installed

## Verify available models

```bash
curl -s http://localhost:9090/v1/models | jq -r '.data[].id'
```

## Configure Cursor

1. Open **Settings** → **Models** → **Add Model**
2. Set **API Base URL** to `http://localhost:9090/v1`
3. Set **API Key** to any string (e.g., `sk-local`)
4. Set **Model Name** to your model id (e.g., `qwen3.5-9b`)

## Select a model

Use the model alias configured in your `models.toml`. The model name in Cursor should match the alias.

## Test the connection

Start a chat or code completion in Cursor. The request will be routed through GGUF Switchboard to the appropriate backend.

## Switching models

Change the model in Cursor's model selector. GGUF Switchboard handles the backend switching automatically.

## Backend considerations

- **GGUF models** run through llama.cpp. Best for quantized models, constrained VRAM.
- **SafeTensors models** run through vLLM. Best for higher-throughput GPU inference.

The client experience is identical regardless of backend.

## Context size considerations

Cursor agent prompts can be large. If you encounter context size errors:

1. Check your model's context size: `curl http://localhost:9090/v1/models/YOUR_MODEL/runtime`
2. Increase context size in `models.toml` if needed
3. Restart GGUF Switchboard

## Recommended configuration

For coding tasks:

- **Qwen 3.5 9B** (GGUF Q4_K_M) — good balance of speed and quality
- **Gemma 4 E4B** — fast, capable coding model

## Troubleshooting

**Model not found:** Ensure the model alias in Cursor matches an alias in `models.toml`.

**Connection refused:** Verify GGUF Switchboard is running: `curl http://localhost:9090/health`

**Context errors:** Increase context size in `models.toml` or use a model with larger context support.
