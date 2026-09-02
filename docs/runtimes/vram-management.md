# VRAM Management

> [← Back to README](../../README.md)

GGUF Switchboard includes hardware-aware model fit planning and OOM fallback.

## VRAM detection

GGUF Switchboard detects available VRAM via nvidia-smi:

```toml
# config.toml
vram_gb = 12
```

Set this to your GPU's VRAM in GB. Lower if you share VRAM with a display or other apps.

## Auto GPU layers (auto_ngl)

When `auto_ngl = true`, GGUF Switchboard automatically picks the number of GPU layers based on free VRAM and model size:

```toml
# config.toml
auto_ngl = true
```

## ModelFitPlanner

The ModelFitPlanner (opt-in, `[fit]` section in `config.toml`) plans llama.cpp launches:

### What it does

- Before every model load, inspects GPU topology / free VRAM, model metadata, and requested context
- Produces a safe launch profile (context size, nGL, split mode, KV cache type)
- Caches known-good profiles to `model-profiles.json`

### Configuration

```toml
# config.toml
[fit]
enabled = true
vram_reserve_mb = 512
max_attempts = 5
cache_profiles = true
```

### OOM fallback ladder

On OOM, the ModelFitPlanner advances through a bounded degradation sequence:

1. Requested context + default KV + auto-fit GPU
2. Requested context + Q8 KV + auto-fit GPU
3. 75% context + Q8 KV + auto-fit GPU
4. 50% context + Q8 KV + auto-fit GPU
5. 25% context + Q8 KV + reduced GPU offload

Each attempt is tried until success or all attempts exhausted.

### Profile caching

Known-good profiles are cached to `model-profiles.json`. Subsequent loads skip the fallback ladder entirely.

## Memory-pressure eviction

A system memory-pressure monitor unloads the resident model past a critical threshold:

```toml
# config.toml
[eviction]
memory_usage_threshold = 90
```

## Configuration reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vram_gb` | GPU VRAM in GB | 12 |
| `auto_ngl` | Auto-pick GPU layers | false |
| `fit.enabled` | Enable ModelFitPlanner | false |
| `fit.vram_reserve_mb` | VRAM reserved for system | 512 |
| `fit.max_attempts` | Max OOM fallback attempts | 5 |
| `fit.cache_profiles` | Cache known-good profiles | true |

## See also

- [Model Switching](model-switching.md) — switch behavior and strategies
- [Configuration](../getting-started/configuration.md) — full configuration reference
