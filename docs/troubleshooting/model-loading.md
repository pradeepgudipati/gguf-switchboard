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

## Model times out waiting for health (`did not become healthy`)

**Error:** `Model 'qwen3.5-9b' did not become healthy within 60s` → 504 → rollback

**Cause:** usually VRAM pressure in disguise, not a dead server. Weights + KV
cache don't fit free VRAM, `llama-server` spills to CPU, and the spill makes
the load crawl past `startup_timeout`. Concrete shape on an RTX 3060 12 GB
(~6.7 GB free): a 5.4 GB Qwen3.5-9B with a 32K context leaves ~1.3 GB for a KV
cache that needs far more, so the load takes 60.4s against a 60s timeout.

**What the proxy does:** a health timeout now triggers the same fit-reduction
retry as an OOM (context first, then `-ngl`), instead of rolling straight back
to the previous model. The reduced profile is persisted to `models.toml` /
`model-profiles.json` so the next load starts small.

**Immediate workaround** (before the retry lands you a small profile, or if
you want to pin it yourself):

```toml
# models.toml — qwen3.5-9b on a 12 GB card
[[models]]
alias = "qwen3.5-9b"
context_size = 8192
```

and/or raise the ceiling in `config.toml`:

```toml
startup_timeout = 180
```

**Second cause — stale `llama-server`:** Qwen3.5 uses a Gated DeltaNet (GDN)
recurrent architecture whose kernels only exist in recent llama.cpp. If the
load fails fast with unknown-architecture / missing-tensor errors rather than
a slow timeout, update the backend first:

```bash
# stable channel (default); Qwen3.5 GDN needs a build newer than ~Sep 2026 —
# use the nightly channel until the next stable cut includes it
LLAMA_RELEASE_CHANNEL=nightly ./scripts/update-llama-cpp.sh
llama-server --version
```

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
