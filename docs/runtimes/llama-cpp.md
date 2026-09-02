# llama.cpp Runtime

> [← Back to README](../../README.md)

GGUF models are served through llama.cpp's `llama-server`.

## Overview

llama.cpp provides:

- Quantized model support (Q2_K through Q8_0)
- CPU/GPU offloading
- Flexible context sizing
- Broad hardware compatibility
- Embedding and reranking support

## Configuration

llama.cpp models are configured in `models.toml`:

```toml
[[models]]
alias = "qwen3.5-9b"
file = "Qwen3.5-9B-Q4_K_M.gguf"
display_name = "Qwen 3.5 9B"
kind = "chat"
enabled = true
priority = true
context_size = 32768
ngl = 999
backend = "llama.cpp"
```

### Key parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `context_size` | Maximum context length | 16384 |
| `ngl` | GPU layers to offload (999 = all) | 999 |
| `extra_args` | Additional llama-server flags | [] |

### Context size

Context size determines how many tokens the model can process at once. Larger context sizes require more VRAM.

Default context size is set in `config.toml` based on `vram_gb`:

```toml
vram_gb = 12
```

Per-model override in `models.toml`:

```toml
[[models]]
alias = "my-model"
context_size = 32768
```

### GPU layers (ngl)

The `ngl` parameter controls how many model layers are offloaded to GPU:

- `ngl = 999` — offload all layers to GPU (default)
- `ngl = 0` — run on CPU only
- `ngl = 20` — offload first 20 layers to GPU

### Extra arguments

Additional llama-server flags can be passed via `extra_args`:

```toml
[[models]]
alias = "my-model"
extra_args = ["--jinja", "--chat-template-file", "/path/to/template.jinja"]
```

## ModelFitPlanner

The ModelFitPlanner (opt-in, `[fit]` section in `config.toml`) plans llama.cpp launches:

- Before every model load, inspects GPU topology / free VRAM, model metadata, and requested context
- Produces a safe launch profile (context size, nGL, split mode, KV cache type)
- On OOM, advances through a bounded degradation sequence

See [VRAM Management](vram-management.md) for details.

## Health checks

llama.cpp exposes a health endpoint at `/health`. GGUF Switchboard polls this endpoint until the model is healthy or timeout.

## Troubleshooting

See [Model Loading Troubleshooting](../troubleshooting/model-loading.md) for common issues.
