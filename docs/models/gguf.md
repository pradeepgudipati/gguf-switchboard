# GGUF Models

> [← Back to README](../../README.md)

GGUF (GPT-Generated Unified Format) models are served through llama.cpp.

## What are GGUF models?

GGUF is a file format for storing large language models. It includes:

- Model weights (quantized)
- Tokenizer configuration
- Metadata (architecture, context size, etc.)

GGUF models are commonly used for:

- Local inference on consumer hardware
- CPU/GPU offloading
- Constrained VRAM systems
- Broad hardware compatibility

## Supported quantizations

Common quantization levels:

| Quant | Description | Quality | Size |
|-------|-------------|---------|------|
| Q2_K | 2-bit quantization | Lowest | Smallest |
| Q3_K_M | 3-bit quantization | Low | Small |
| Q4_K_M | 4-bit quantization | Good | Medium |
| Q5_K_M | 5-bit quantization | Better | Larger |
| Q6_K | 6-bit quantization | High | Large |
| Q8_0 | 8-bit quantization | Very high | Largest |

Q4_K_M is the most common choice for balancing quality and size.

## Search for GGUF models

```bash
ggs models search "Qwen 7B"
ggs models search "gemma"
```

## Pull a GGUF model

```bash
ggs models pull lmstudio-community/Qwen3.5-9B-GGUF \
  --quant Q4_K_M \
  --dir /var/lib/gguf-switchboard/models \
  --registry /opt/gguf-switchboard/models.toml
```

This command:

1. Fetches the repo tree from Hugging Face
2. Resolves `--quant` case-insensitively
3. Streams the download with progress
4. Validates the GGUF header
5. Generates an alias
6. Runs the fit planner to generate context_size/ngl/extra_args
7. Merges into `models.toml`

A successful pull refreshes a running gguf-switchboard server automatically.

## Model configuration

After pulling, the model is automatically configured in `models.toml`:

```toml
[[models]]
alias = "qwen3.5-9b"
file = "Qwen3.5-9B-Q4_K_M.gguf"
display_name = "Qwen 3.5 9B"
kind = "chat"
priority = true
context_size = 32768
ngl = 999
backend = "llama.cpp"
```

## Manual configuration

You can also manually add models to `models.toml`:

```toml
[[models]]
alias = "my-model"
file = "my-model.gguf"
display_name = "My Model"
kind = "chat"
enabled = true
priority = false
context_size = 16384
ngl = 999
backend = "llama.cpp"
```

## Context size

Context size determines how many tokens the model can process at once. Larger context sizes require more VRAM.

Default context size is set in `config.toml`:

```toml
vram_gb = 12
```

Per-model override in `models.toml`:

```toml
[[models]]
alias = "my-model"
context_size = 32768
```

## GPU layers (ngl)

The `ngl` parameter controls how many model layers are offloaded to GPU:

- `ngl = 999` — offload all layers to GPU (default)
- `ngl = 0` — run on CPU only
- `ngl = 20` — offload first 20 layers to GPU

## See also

- [Model Search](model-search.md) — find models that fit your hardware
- [llama.cpp Runtime](../runtimes/llama-cpp.md) — llama.cpp backend details
- [Configuration](../getting-started/configuration.md) — full configuration reference
