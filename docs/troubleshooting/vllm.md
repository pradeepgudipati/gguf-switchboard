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

1. Check vLLM installation (import probe — safe without a GPU): `/usr/local/bin/uv run --project /opt/gguf-switchboard/vllm-runtime python -c "import importlib.metadata; print(importlib.metadata.version('vllm'))"`
2. Check CUDA: `nvcc --version`
3. Check model compatibility
4. Check logs: `ggs logs`

## `Can't initialize NVML` / `Failed to infer device type`

**Cause:** torch cannot see an NVIDIA driver/GPU, so vLLM's device
inference fails during CLI startup — even for version checks. The
import probe above still succeeds; only `vllm serve` (and the old
`vllm --version` check) crash.

**Solutions:**

1. `nvidia-smi -L` must list a GPU; `ls -l /dev/nvidia*` must exist.
2. Unset an empty `CUDA_VISIBLE_DEVICES` (it hides all GPUs).
3. Confirm torch sees CUDA: `/usr/local/bin/uv run --project /opt/gguf-switchboard/vllm-runtime python -c "import torch; print(torch.cuda.is_available())"`.
4. vLLM requires a CUDA GPU — there is no CPU fallback for serving.

## Unexpected dependency upgrades (tokenspeed-triton, xgrammar, ...)

**Cause:** without a lockfile, `uv sync` re-resolves floating
transitive pins on every deploy.

**Solutions:**

1. Deploys sync `--frozen` against the committed `vllm-runtime/uv.lock`.
   Refresh it deliberately with `uv lock --project vllm-runtime --python 3.12`
   (Python is capped at `<3.13`: torch/vLLM wheels lag new CPython releases).
2. Verify drift with `./deploy.sh` (the locked fast path re-syncs to the lock).

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
