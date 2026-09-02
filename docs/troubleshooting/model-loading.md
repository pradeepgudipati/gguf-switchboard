# Model Loading Troubleshooting

> [← Back to README](../../README.md)

Common model loading errors and solutions.

## Model backend exited before healthy

**Error:** `model backend exited before healthy`

**Causes:**

- llama-server binary not found
- Model file not found
- Insufficient VRAM
- Port conflict

**Solutions:**

1. Verify llama-server is installed: `which llama-server`
2. Verify model file exists: `ls /var/lib/gguf-switchboard/models/`
3. Check VRAM: `nvidia-smi`
4. Check logs: `ggs logs`

## CUDA out of memory

**Error:** `CUDA out of memory`

**Causes:**

- Model too large for VRAM
- Context size too large
- Other processes using VRAM

**Solutions:**

1. Use a smaller model or quantization
2. Reduce context size in `models.toml`
3. Close other GPU processes
4. Enable ModelFitPlanner: `[fit] enabled = true`

## Model failed to load

**Error:** `model failed to load`

**Causes:**

- Corrupted model file
- Incompatible model format
- Missing dependencies

**Solutions:**

1. Re-download the model
2. Check model format compatibility
3. Check logs: `ggs logs`

## Port already in use

**Error:** `address already in use`

**Causes:**

- Another process using the port
- Previous llama-server not cleaned up

**Solutions:**

1. Check for running processes: `lsof -i :18081`
2. Kill stale processes
3. Restart GGUF Switchboard: `ggs restart`

## Context size exceeds available memory

**Error:** `context size exceeds available memory`

**Causes:**

- Requested context too large for VRAM
- Model + context don't fit

**Solutions:**

1. Reduce context size in `models.toml`
2. Use a smaller model
3. Enable ModelFitPlanner for automatic fallback

## See also

- [Out of Memory](out-of-memory.md) — OOM-specific troubleshooting
- [vLLM Issues](vllm.md) — vLLM-specific troubleshooting
