# OpenCode Integration

> [← Back to README](../../README.md)

Use GGUF Switchboard as the model provider for OpenCode, an AI coding assistant.

## What this enables

OpenCode connects to GGUF Switchboard's OpenAI-compatible API, giving you access to any locally loaded GGUF or SafeTensors model through the same endpoint. Switch between models without changing your OpenCode configuration.

## Prerequisites

- GGUF Switchboard running on `http://localhost:9090`
- At least one model registered and loaded
- OpenCode installed

## Verify available models

```bash
curl -s http://localhost:9090/v1/models | jq -r '.data[].id'
```

## Configure OpenCode

Add GGUF Switchboard as an OpenAI-compatible provider in your OpenCode configuration:

```json
{
  "providers": {
    "local": {
      "type": "openai",
      "apiKey": "not-needed",
      "baseUrl": "http://localhost:9090/v1",
      "models": {
        "qwen3.5-9b": {
          "name": "Qwen 3.5 9B"
        },
        "gemma-4-e4b": {
          "name": "Gemma 4 E4B"
        }
      }
    }
  }
}
```

## Select a model

Use the model alias configured in your `models.toml`. The model name in OpenCode should match the alias:

```bash
# Check available aliases
curl -s http://localhost:9090/v1/models | jq -r '.data[].id'
```

## Test the connection

Start a conversation in OpenCode. The request will be routed through GGUF Switchboard to the appropriate backend (llama.cpp for GGUF, vLLM for SafeTensors).

## Switching models

Change the model in OpenCode's model selector. GGUF Switchboard handles the backend switching automatically — unloading the previous model and loading the requested one.

## Backend considerations

- **GGUF models** run through llama.cpp. Best for quantized models, constrained VRAM, CPU/GPU offloading.
- **SafeTensors models** run through vLLM. Best for higher-throughput GPU inference, modern architectures.

The client experience is identical regardless of backend.

## Recommended configuration

For coding tasks, consider:

- **Qwen 3.5 9B** (GGUF Q4_K_M) — good balance of speed and quality
- **Gemma 4 E4B** — fast, capable coding model
- **DeepSeek Coder V2 Lite** — strong coding performance

## Troubleshooting

**Model not found:** Ensure the model alias in OpenCode matches an alias in `models.toml`.

**Connection refused:** Verify GGUF Switchboard is running: `curl http://localhost:9090/health`

**Slow responses:** The first request after a model switch includes model loading time. Subsequent requests are fast.
