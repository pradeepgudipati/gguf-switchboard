# Out of Memory Troubleshooting

> [← Back to README](../../README.md)

OOM errors, context reduction, and fallback strategies.

## OOM during weight loading

**Error:** `out of memory` / `cannot allocate` during weight loading

**Cause:** Model weights don't fit in VRAM.

**Solutions:**

1. Use a smaller quantization (e.g., Q4_K_M instead of Q6_K)
2. Use a smaller model
3. Enable auto_ngl to offload some layers to CPU
4. Enable ModelFitPlanner for automatic fallback

## OOM during KV cache allocation

**Error:** `kv_cache` / `alloc` failure after weights loaded

**Cause:** Context size too large for remaining VRAM after weights.

**Solutions:**

1. Reduce context size in `models.toml`
2. Enable ModelFitPlanner for automatic context reduction
3. Use a smaller model

## ModelFitPlanner fallback

When enabled, ModelFitPlanner automatically tries smaller context sizes:

1. Requested context + default KV + auto-fit GPU
2. Requested context + Q8 KV + auto-fit GPU
3. 75% context + Q8 KV + auto-fit GPU
4. 50% context + Q8 KV + auto-fit GPU
5. 25% context + Q8 KV + reduced GPU offload

Enable in `config.toml`:

```toml
[fit]
enabled = true
```

## Memory-pressure eviction

A system memory-pressure monitor unloads the resident model past a critical threshold:

```toml
# config.toml
[eviction]
memory_usage_threshold = 90
```

## See also

- [Model Loading](model-loading.md) — general loading troubleshooting
- [VRAM Management](../runtimes/vram-management.md) — fit planning details
