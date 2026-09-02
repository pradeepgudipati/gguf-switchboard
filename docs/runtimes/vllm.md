# vLLM Runtime

> [← Back to README](../../README.md)

SafeTensors models are served through vLLM.

## Overview

vLLM provides:

- Higher-throughput GPU inference
- Modern transformer architectures
- AWQ/GPTQ/FP8 quantization support
- Multi-GPU execution (tensor parallelism)

## Configuration

vLLM models are configured in `models.toml`:

```toml
[[models]]
alias = "qwen2-5-7b-instruct"
file = ""
display_name = "Qwen 2.5 7B Instruct"
kind = "chat"
backend = "vllm"
vllm_file = "/var/lib/gguf-switchboard/vllm-models/Qwen--Qwen2.5-7B-Instruct"
vllm_hf_repo = "Qwen/Qwen2.5-7B-Instruct"
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `vllm_file` | Path to SafeTensors model directory |
| `vllm_hf_repo` | Hugging Face repository ID |
| `quantization` | Quantization format (awq, gptq, fp8) |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism |
| `gpu_memory_utilization` | GPU memory utilization (0.0-1.0) |
| `max_model_len` | Maximum model context length |
| `attention_backend` | Attention backend (FLASH_ATTN, etc.) |

## uv-managed environment

vLLM runs through a uv-managed Python environment:

```
GGUF Switchboard
      │
      ▼
detect SafeTensors model
      │
      ▼
ensure uv/vLLM runtime
      │
      ▼
launch isolated vLLM process
      │
      ▼
health check
      │
      ▼
route OpenAI-compatible traffic
```

### Why uv is used

- Isolated Python environment for vLLM
- No conflicts with system Python
- Pinned vLLM version
- Reproducible builds

### Environment location

The uv environment is created at:

```
/opt/gguf-switchboard/vllm-runtime/
```

### Managing the environment

```bash
# Check vLLM version
/usr/local/bin/uv run --project /opt/gguf-switchboard/vllm-runtime vllm --version

# Recreate environment
rm -rf /opt/gguf-switchboard/vllm-runtime/.venv
/usr/local/bin/uv sync --project /opt/gguf-switchboard/vllm-runtime
```

## GPU selection

vLLM uses all available GPUs by default. For multi-GPU setups:

```toml
[[models]]
alias = "my-model"
tensor_parallel_size = 2
```

## Health checks

vLLM exposes a health endpoint at `/health`. GGUF Switchboard polls this endpoint until the model is healthy or timeout.

## Troubleshooting

See [vLLM Troubleshooting](../troubleshooting/vllm.md) for common issues.
