# Continue Integration

> [← Back to README](../../README.md)

Use GGUF Switchboard as a model provider in Continue (VS Code / JetBrains extension).

## What this enables

Continue connects to GGUF Switchboard's OpenAI-compatible API, giving you access to any locally loaded GGUF or SafeTensors model for code completion, chat, and editing.

## Prerequisites

- GGUF Switchboard running on `http://localhost:9090`
- At least one model registered and loaded
- Continue extension installed in VS Code or JetBrains

## Verify available models

```bash
curl -s http://localhost:9090/v1/models | jq -r '.data[].id'
```

## Configure Continue

Add GGUF Switchboard as an OpenAI-compatible provider in `~/.continue/config.json`:

```json
{
    "models": [
        {
            "title": "Local Qwen 3.5 9B",
            "provider": "openai",
            "model": "qwen3.5-9b",
            "apiBase": "http://localhost:9090/v1",
            "apiKey": "not-needed"
        }
    ]
}
```

## Multiple model configuration

You can configure multiple models for different purposes:

```json
{
    "models": [
        {
            "title": "Chat Model",
            "provider": "openai",
            "model": "qwen3.5-9b",
            "apiBase": "http://localhost:9090/v1",
            "apiKey": "not-needed"
        },
        {
            "title": "Fast Model",
            "provider": "openai",
            "model": "gemma-4-e4b",
            "apiBase": "http://localhost:9090/v1",
            "apiKey": "not-needed"
        }
    ],
    "tabAutocompleteModel": {
        "title": "Autocomplete",
        "provider": "openai",
        "model": "gemma-4-e4b",
        "apiBase": "http://localhost:9090/v1",
        "apiKey": "not-needed"
    }
}
```

## Select a model

Use the model alias configured in your `models.toml`. The model name in Continue should match the alias.

## Test the connection

Start a chat or code completion in Continue. The request will be routed through GGUF Switchboard to the appropriate backend.

## Switching models

Change the model in Continue's model selector. GGUF Switchboard handles the backend switching automatically.

## Backend considerations

- **GGUF models** run through llama.cpp. Best for quantized models, constrained VRAM.
- **SafeTensors models** run through vLLM. Best for higher-throughput GPU inference.

## Recommended configuration

For coding tasks:

- **Qwen 3.5 9B** (GGUF Q4_K_M) — good balance of speed and quality for chat
- **Gemma 4 E4B** — fast model for autocomplete

## Troubleshooting

**Model not found:** Ensure the model alias in Continue matches an alias in `models.toml`.

**Connection refused:** Verify GGUF Switchboard is running: `curl http://localhost:9090/health`

**Autocomplete slow:** Use a smaller, faster model for `tabAutocompleteModel`.
