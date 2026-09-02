# Backend Selection

> [← Back to README](../../README.md)

GGUF Switchboard automatically selects the appropriate backend based on model format.

## Selection logic

```
Requested model
      │
      ├── GGUF?
      │      │
      │      └──► llama.cpp
      │
      └── SafeTensors?
             │
             └──► vLLM
```

## Explicit backend pin

You can explicitly set the backend in `models.toml`:

```toml
[[models]]
alias = "my-model"
backend = "llama.cpp"  # or "vllm"
```

## Dual-source fallback

When an alias has both GGUF and SafeTensors sources and no explicit backend pin:

1. Startup prefers vLLM if the SafeTensors weights fit detected VRAM
2. Otherwise, it uses the GGUF source through llama.cpp

## Unsupported combinations

- GGUF through vLLM is not currently supported
- SafeTensors through llama.cpp is not supported

## See also

- [Runtime Overview](../runtimes/overview.md) — backend details
- [Architecture Overview](overview.md) — scheduler details
