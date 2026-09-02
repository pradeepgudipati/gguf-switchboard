# SafeTensors Models

> [← Back to README](../../README.md)

SafeTensors models are served through vLLM.

## What are SafeTensors models?

SafeTensors is a file format for storing large language model weights. It was designed for:

- Safety (no arbitrary code execution)
- Speed (fast loading)
- Simplicity (standard format)

SafeTensors models are commonly used for:

- Hugging Face model distribution
- Higher-throughput GPU inference
- Modern transformer architectures
- Multi-GPU execution

## Supported architectures

vLLM supports a wide range of model architectures. Check the [vLLM documentation](https://docs.vllm.ai/en/latest/models/supported_models.html) for the full list.

## Supported quantizations

vLLM supports several quantization formats:

- **AWQ** — Activation-aware Weight Quantization
- **GPTQ** — GPT Quantization
- **FP8** — 8-bit floating point
- **compressed-tensors** — Compressed tensor format

## Search for SafeTensors models

```bash
ggs models search vllm "Qwen 7B Instruct"
ggs models search vllm "Muse"
```

## Pull a SafeTensors model

```bash
ggs models pull vllm Qwen/Qwen2.5-7B-Instruct \
  --dir /var/lib/gguf-switchboard/vllm-models \
  --registry /opt/gguf-switchboard/models.toml
```

This command:

1. Downloads weights plus tokenizer/configuration files
2. Detects AWQ/GPTQ/FP8-style quantization metadata
3. Writes the vLLM source and launch options into the registry
4. Refreshes a running server

It never enables `trust_remote_code` automatically.

## Model configuration

After pulling, the model is automatically configured in `models.toml`:

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

## Dual-source models

When an alias has both GGUF and SafeTensors sources and no explicit backend pin, startup prefers vLLM if the SafeTensors weights fit detected VRAM; otherwise it uses the GGUF source through llama.cpp.

## vLLM requirements

SafeTensors models require:

- Python 3.10-3.14
- uv (automatically installed by `deploy.sh`)
- NVIDIA GPU with CUDA support
- Sufficient VRAM for model weights

## See also

- [Model Search](model-search.md) — find models that fit your hardware
- [vLLM Runtime](../runtimes/vllm.md) — vLLM backend details
- [Configuration](../getting-started/configuration.md) — full configuration reference
