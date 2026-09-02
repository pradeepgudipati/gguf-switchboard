# vLLM Troubleshooting

> [← Back to README](../../README.md)

vLLM-specific errors and uv environment issues.

## vLLM process exited before healthy

**Error:** `vLLM process exited before healthy`

**Causes:**

- vLLM not installed
- CUDA version mismatch
- Model incompatible with vLLM
- Insufficient VRAM

**Solutions:**

1. Check vLLM installation: `/usr/local/bin/uv run --project /opt/gguf-switchboard/vllm-runtime vllm --version`
2. Check CUDA: `nvcc --version`
3. Check model compatibility
4. Check logs: `ggs logs`

## uv environment creation failed

**Error:** `uv environment creation failed`

**Causes:**

- Python not installed
- uv not installed
- Network issues

**Solutions:**

1. Install Python 3.10-3.14
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Check network connectivity

## SafeTensors architecture unsupported

**Error:** `SafeTensors architecture unsupported by vLLM`

**Cause:** Model architecture not supported by vLLM.

**Solutions:**

1. Check vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models.html
2. Use a supported model architecture
3. Use GGUF format with llama.cpp instead

## CUDA version mismatch

**Error:** CUDA version mismatch between vLLM and driver.

**Solutions:**

1. Check driver CUDA version: `nvidia-smi`
2. Reinstall vLLM with matching CUDA version
3. Update NVIDIA driver

## Recreating the uv environment

If the environment is corrupted:

```bash
rm -rf /opt/gguf-switchboard/vllm-runtime/.venv
/usr/local/bin/uv sync --project /opt/gguf-switchboard/vllm-runtime
```

## See also

- [Model Loading](model-loading.md) — general loading troubleshooting
- [vLLM Runtime](../runtimes/vllm.md) — vLLM backend details
