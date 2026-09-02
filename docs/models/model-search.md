# Model Search

> [← Back to README](../../README.md)

Find models that actually fit your hardware.

## Hardware-aware search

`ggs models search` scores every discovered quant against your detected hardware:

```bash
ggs models search "gemma"
ggs models search "Qwen3.5 9B" --limit 5
ggs models search "Qwen3.5 9B" --ram-bandwidth-gbps 50
```

Search prints total system RAM, NVIDIA VRAM, and their combined capacity before an aligned table with FIT/SPEED/BALANCED/PRECISION columns:

```
Hardware: System RAM 32.0 GiB | NVIDIA VRAM 24.0 GiB | Total 56.0 GiB
Speed model inputs: GPU bandwidth 1008 GB/s (NVIDIA GeForce RTX 4090) | RAM bandwidth 40 GB/s (assumed) | GPU efficiency 0.55 | CPU efficiency 0.35

REPO                                               | FILES |     SIZE | FIT | CONTEXT    | ARCH  | SPEED            | BALANCED               | PRECISION    | QUANT
bartowski/Qwen3.5-9B-GGUF                          |    24 |  9421 MB | 100 | 32768 tok  | qwen3 | Q4_K_M ~127tok/s | Q5_K_M ~91tok/s/~98.9% | Q6_K ~99.6%  | Q2_K,Q3_K_M,Q4_K_M,Q5_K_M,Q6_K,Q8_0
FIT: 0-100 memory-fit score (100 = comfortable headroom; 0 = does not fit RAM+VRAM). SPEED/PRECISION: the quant that maximizes each — tok/s from a memory-bandwidth model (verify against `llama-bench` on your machine), quality % from published per-quant perplexity measurements ("~" = extrapolated, not directly measured for this architecture). BALANCED: the quant with the best average of speed and quality, both normalized to this model's own quant options — a middle ground when you don't want either extreme. See docs/QUANT_SCORING.md for methodology and sources; override RAM bandwidth with --ram-bandwidth-gbps if you've measured your own.
Try: ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q4_K_M   (fastest, ~127 tok/s est.)
     ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q5_K_M   (balanced, ~91 tok/s / ~98.9% quality est.)
     ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q6_K   (least precision loss, ~99.6% quality est.)
```

## Column descriptions

| Column | Description |
|--------|-------------|
| `FIT` | 0–100 memory-fit score (100 = comfortable headroom, 0 = doesn't fit) |
| `SPEED` | Fastest quant with estimated tok/s |
| `BALANCED` | Quant at the size midpoint of fitting options (speed/precision trade-off) |
| `PRECISION` | Least-lossy quant with quality score (% of fp16 quality retained) |
| `QUANT` | All fitting quants ordered from smallest to largest |

The footer legend explains each column. The `"~"` prefix on SPEED and PRECISION values means the estimate is extrapolated, not directly measured for that architecture. The `Try:` lines suggest the fastest, balanced, and least precision loss quants with pull commands.

When a repo has FIT=0 (doesn't fit RAM+VRAM), SPEED/BALANCED/PRECISION show `-` and QUANT is empty — the repo is listed but no recommendations are made.

## Search SafeTensors/vLLM models

```bash
ggs models search vllm "Qwen 7B Instruct"
ggs models search vllm "Muse"
```

## Override RAM bandwidth

If you've measured your system's RAM bandwidth:

```bash
ggs models search "Qwen3.5 9B" --ram-bandwidth-gbps 50
```

## Hardware evaluation factors

The search evaluates models based on:

- GPU VRAM
- System RAM
- Model size
- Quantization
- Context size
- CPU/GPU offloading
- GPU memory bandwidth
- RAM bandwidth
- Backend compatibility

## See also

- [Quant Scoring](../QUANT_SCORING.md) — exact formulas and sources
- [GGUF Models](gguf.md) — pulling and configuring GGUF models
- [SafeTensors Models](safetensors.md) — pulling and configuring SafeTensors models
