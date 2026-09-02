# Model Switching

> [← Back to README](../../README.md)

GGUF Switchboard runs one model at a time and switches between them on demand.

## How switching works

```
Request for model B while A is loaded
      │
      ▼
Drain in-flight requests on A
      │
      ▼
Unload A (frees VRAM)
      │
      ▼
Load B
      │
      ▼
Wait for health check
      │
      ▼
Forward request
```

## Drain behavior

When a switch is requested:

1. New requests for the current model are queued
2. In-flight requests are allowed to complete
3. After drain timeout, remaining requests are interrupted
4. The model is unloaded

## Rollback on failure

If the new model fails to load:

1. The previous model is re-loaded
2. The switch is aborted
3. The request returns an error

## Priority model warm-up

After `idle_timeout` seconds with no requests, the priority model auto-loads:

```toml
# config.toml
idle_timeout = 600
```

```toml
# models.toml
[[models]]
alias = "qwen3.5-9b"
priority = true
```

Only one model should have `priority = true`. If multiple are set, the runtime keeps the first and clears the rest with a warning.

## Switch strategies

Two switch strategies are available:

### unload_first (default)

```
Unload A → Load B
```

- Requires VRAM for only one model at a time
- Slower switch time
- Lower VRAM requirement

### load_first

```
Load B next to A → Unload A
```

- Requires VRAM for both models simultaneously
- Faster switch time
- Higher VRAM requirement

Configure in `config.toml`:

```toml
switch_strategy = "unload_first"
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `idle_timeout` | Seconds before priority model warms up | 600 |
| `switch_drain_timeout_secs` | Max seconds to drain in-flight requests | 30 |
| `switch_strategy` | "unload_first" or "load_first" | "unload_first" |

## Monitoring switches

Track switches via Prometheus metrics:

```promql
# Total switches
gguf_switchboard_model_switches_total

# Switch duration
gguf_switchboard_model_switch_seconds

# Switch phases
gguf_switchboard_model_switch_phase_seconds
```

`GET /status` also returns `last_switch` — a millisecond breakdown (`drain_ms`, `unload_previous_ms`, `load_ms`, `rollback_ms`, `total_ms`) of the most recent switch.

## See also

- [VRAM Management](vram-management.md) — hardware-aware fit planning
- [Architecture Overview](../architecture/overview.md) — scheduler details
