# Quick Start

> [← Back to README](../../README.md)

Get GGUF Switchboard running and send your first request in 5 minutes.

## Prerequisites

- Linux with NVIDIA GPU and CUDA toolkit
- `nvcc` and NVIDIA driver installed
- Git

## Install

```bash
git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git
cd gguf-switchboard
./deploy.sh
```

This installs:
- CUDA llama.cpp (for GGUF models)
- Isolated vLLM environment (for SafeTensors models)
- gguf-switchboard binary
- systemd service

## Pull a model

```bash
# GGUF served by llama.cpp
gguf-switchboard models pull lmstudio-community/Qwen3.5-9B-GGUF \
  --quant Q4_K_M \
  --dir /var/lib/gguf-switchboard/models \
  --registry /opt/gguf-switchboard/models.toml

# SafeTensors served by vLLM
gguf-switchboard models pull vllm Qwen/Qwen2.5-7B-Instruct \
  --dir /var/lib/gguf-switchboard/vllm-models \
  --registry /opt/gguf-switchboard/models.toml
```

## Verify

```bash
# Check service status
ggs status

# List available models
curl -s http://localhost:9090/v1/models | jq -r '.data[].id'
```

## Send a request

```bash
curl http://localhost:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Name the model you want. GGUF Switchboard selects its registered backend, unloads the resident model when necessary, starts the requested model, and forwards the request through the same API.

## What's next?

- [Model Search](../models/model-search.md) — find models that fit your hardware
- [Configuration](configuration.md) — customize behavior
- [Integrations](../integrations/opencode.md) — connect your AI coding tools
