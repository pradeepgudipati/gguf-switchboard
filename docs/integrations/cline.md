# Cline Integration

> [← Back to README](../../README.md)

Use GGUF Switchboard as an OpenAI Compatible provider in Cline (VS Code extension).

## What this enables

Cline connects to GGUF Switchboard's OpenAI-compatible API, giving you access to any locally loaded GGUF or SafeTensors model for AI-assisted coding in VS Code.

## Prerequisites

- GGUF Switchboard running on `http://localhost:9090`
- At least one model registered and loaded
- Cline extension installed in VS Code

## Verify available models

```bash
curl -s http://localhost:9090/v1/models | jq -r '.data[].id'
```

## Configure Cline

1. Open Cline settings in VS Code
2. Select **OpenAI Compatible** as the API provider
3. Set **Base URL** to `http://localhost:9090/v1` (must include `/v1`)
4. Set **API Key** to any non-empty string (e.g., `sk-local`) — the runtime does not validate keys, but Cline requires the field
5. Set **Model** to your model id (must match `config.toml`, e.g., `gemma-4-e4b`)

If **"Use different models for Plan and Act modes"** is enabled, configure both modes separately (API key and base URL in each).

## Select a model

Use the model alias configured in your `models.toml`. The model name in Cline should match the alias.

## Test the connection

Start a conversation in Cline. The request will be routed through GGUF Switchboard to the appropriate backend.

## Switching models

Change the model in Cline's settings. GGUF Switchboard handles the backend switching automatically.

## Backend considerations

- **GGUF models** run through llama.cpp. Best for quantized models, constrained VRAM.
- **SafeTensors models** run through vLLM. Best for higher-throughput GPU inference.

## Context size considerations

Cline agent prompts can be large (30k+ tokens). If you see `exceed_context_size_error`:

1. Start a fresh Cline task to reduce prompt size, or
2. Increase `-c` in `config.toml` and restart the runtime
3. Check your model's context size: `curl http://localhost:9090/v1/models/YOUR_MODEL/runtime`

## Recommended configuration

For coding tasks:

- **Qwen 3.5 9B** (GGUF Q4_K_M) — good balance of speed and quality
- **Gemma 4 E4B** — fast, capable coding model

## Troubleshooting

**Model not found:** Ensure the model alias in Cline matches an alias in `models.toml`.

**Connection refused:** Verify GGUF Switchboard is running: `curl http://localhost:9090/health`

**Context errors:** Increase context size in `models.toml` or start a fresh Cline task.
