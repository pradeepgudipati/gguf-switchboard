# Runtime Overview

> [← Back to README](../../README.md)

GGUF Switchboard supports two model execution backends: llama.cpp for GGUF models and vLLM for SafeTensors models.

## Backend selection

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

The backend is automatically selected based on the model format:

- **GGUF models** → llama.cpp
- **SafeTensors models** → vLLM

When an alias has both sources and no explicit backend pin, startup prefers vLLM if the SafeTensors weights fit detected VRAM; otherwise it uses the GGUF source through llama.cpp.

## Unified client experience

Externally, users see:

```
OpenCode ─────┐
Claude Code ──┤
Cursor ───────┤
Cline ────────┼──► :9090 ──► GGUF Switchboard
Continue ─────┤                    │
Agents ───────┤           ┌────────┴────────┐
OpenAI SDK ───┘           │                 │
                       llama.cpp           vLLM
                          │                  │
                        GGUF            SafeTensors
```

The client experience is identical regardless of backend.

## llama.cpp

GGUF models run through llama.cpp, which provides:

- Quantized model support (Q2_K through Q8_0)
- CPU/GPU offloading
- Flexible context sizing
- Broad hardware compatibility

See [llama.cpp Runtime](llama-cpp.md) for details.

## vLLM

SafeTensors models run through vLLM, which provides:

- Higher-throughput GPU inference
- Modern transformer architectures
- AWQ/GPTQ/FP8 quantization support
- Multi-GPU execution

See [vLLM Runtime](vllm.md) for details.

## Model switching

GGUF Switchboard runs one model at a time. When a request arrives for a different model:

1. Drain in-flight requests on the current model
2. Unload the current model (frees VRAM)
3. Load the requested model
4. Wait for health check
5. Forward the request

See [Model Switching](model-switching.md) for details.

## VRAM management

GGUF Switchboard includes hardware-aware model fit planning:

- VRAM detection via nvidia-smi
- Context size planning
- GPU layer count optimization
- OOM fallback with bounded degradation

See [VRAM Management](vram-management.md) for details.
