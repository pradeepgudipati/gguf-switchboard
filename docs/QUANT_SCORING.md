# Quant scoring: FIT, SPEED, PRECISION

`gguf-switchboard models search` and `models files` score every discovered GGUF
quant against your detected hardware instead of reporting a flat
`Supported: Yes/No`. This doc explains what each number means, exactly how
it's computed, where the constants come from, and how to check them against
your own machine. Implementation: [`src/quant_profile.rs`](../src/quant_profile.rs).

## FIT (0–100)

A continuous replacement for the old binary support check. Same underlying
rule (model fits within total RAM + VRAM with 20% reserved headroom), but
degrades smoothly either side of that line instead of a hard cutoff:

| Utilization (model size ÷ RAM+VRAM) | Score |
|---|---|
| ≤ 80% | 100 (comfortable headroom for KV-cache/context) |
| 80% → 83.3% | 100 → 60 (83.3% is the old hard cutoff) |
| 83.3% → 100% | 60 → 20 (fits, but tight — little room for context) |
| > 100% | 0 |

The score shown for a repository is the **best** achievable across every
quant it offers — i.e. "how well does your best option fit," not an average.

## SPEED (tokens/sec estimate)

llama.cpp token generation is memory-bandwidth bound: each token requires
streaming the (offloaded) weights through VRAM or system RAM once. So:

```
tokens/sec ≈ (effective_bandwidth_GB/s ÷ model_size_GB) × efficiency_factor
```

This is the same approach used by [llmfit](https://github.com/AlexsJones/llmfit)
(`bandwidth_GB_s / model_size_GB × efficiency_factor`, default efficiency 0.55).

Three modes, chosen by whether the quant fits in free VRAM:

- **GPU** (fits in ≤90% of free VRAM, 10% reserved for KV-cache/context):
  `bandwidth = <your GPU's memory bandwidth>`, `efficiency = 0.55`.
- **GPU+CPU** (exceeds free VRAM): bandwidth and efficiency are linearly
  blended between the GPU and RAM figures by the fraction of the model that
  overflows VRAM.
- **CPU** (no GPU detected): `bandwidth = RAM bandwidth`, `efficiency = 0.35`
  (lower — CPU generation is compute-bound as well as bandwidth-bound).

GPU memory bandwidth comes from a lookup table of ~35 common NVIDIA cards
(vendor datasheet figures — substring-matched against the `nvidia-smi` device
name, most-specific pattern first). An unrecognized NVIDIA card falls back to
a 400 GB/s mid-range guess. Detection is NVIDIA-only, matching this
codebase's existing `nvidia-smi`-based GPU probing (`src/gpu.rs`) — AMD/Apple
GPUs aren't detected anywhere else in gguf-switchboard either.

RAM bandwidth defaults to **40 GB/s** — a conservative estimate for a modern
dual-channel DDR4-3200..DDR5-5600 desktop after controller/refresh overhead
(well below the DDR5-5600 theoretical peak of 89.6 GB/s). Override it with
`--ram-bandwidth-gbps <value>` on `models search` if you've measured your own
(e.g. with `mbw` or `likwid-bench`).

**Every estimate ships its inputs.** Run with `RUST_LOG=debug` to see the
exact GPU name, bandwidth, mode, and efficiency factor used for each quant —
so a number can be checked against `llama-bench` on the actual machine rather
than trusted blindly. `models pull` already does this for real, post-download
(`--no-bench` to skip): that measured number is the ground truth; the search
estimate is only meant to rank options *before* you download anything.

## PRECISION (0–100 quality score)

Derived from published perplexity-increase measurements per quant, relative
to fp16/bf16 (`quality_score = 100 − ppl_increase_pct`, clamped to [0, 100]):

- **k-quants PR #1684** (LLaMA-7B, 2023):
  <https://github.com/ggml-org/llama.cpp/pull/1684>
- **"Which Quantization Should I Use? A Unified Evaluation of llama.cpp
  Quantization on Llama-3.1-8B-Instruct"** (2026): <https://arxiv.org/abs/2601.14277>

The two sources disagree in absolute terms — Llama-3.1-8B (2026, more heavily
trained, less redundant weight to compress) loses measurably *more* per quant
step than 2023-era LLaMA-7B did. The default table uses the Llama-3.1-8B
numbers as more representative of current dense models:

| Quant | PPL increase | Quality score | Confidence |
|---|---|---|---|
| F32/F16/BF16 | 0.0% | 100.0 | Measured |
| Q8_0 | 0.1% | 99.9 | Measured |
| Q6_K | 0.4% | 99.6 | Measured |
| Q5_K_M | 1.1% | 98.9 | Measured |
| Q5_K_S / Q5_0 / Q5_1 | 1.5% | 98.5 | Measured |
| Q4_K_M | 3.3% | 96.7 | Measured |
| Q4_K_S | 4.1% | 95.9 | Measured |
| Q4_1 | 5.5% | 94.5 | Measured |
| Q4_0 | 5.7% | 94.3 | Measured |
| Q3_K_L | 6.7% | 93.3 | Measured |
| Q3_K_M | 8.7% | 91.3 | Measured |
| Q3_K_S | 22.4% | 77.6 | Measured |
| Q2_K | ~35% | ~65 | **Extrapolated** |
| IQ4_NL/IQ4_XS, IQ3_*, IQ2_*, IQ1_* | various | various | **Extrapolated** |

Entries not covered by either paper (Q2_K, most IQ-quants) are extrapolated
from the k-quants PR's measured Q2_K/Q3_K_S ratio (≈1.58×) applied to the
Llama-3.1-8B Q3_K_S figure, and are marked `Extrapolated` — treat those as
"roughly this order of magnitude," not measured fact for whatever model
you're actually looking at. An unrecognized quant label falls back to the
same-bit-width family average, or a conservative mid-table guess if the
family itself is unrecognized.

**This is a generic table, not per-model data.** If you have actual measured
per-model perplexity or KL-divergence numbers (an imatrix run, a quantizer's
model card with published figures), that is strictly better than this table
*for that specific model* — quantization loss varies by architecture and
training. There is currently no automated way to pull that from the Hugging
Face search API for an arbitrary repository, so it isn't wired in. The lookup
is a single function (`quant_profile::quant_quality`), so plugging in a
measured-data source is a small, contained change if/when one becomes
reliably available per-model.

## What this deliberately does not do

- No AMD/Apple GPU bandwidth table — matches this codebase's existing
  NVIDIA-only detection scope (`src/gpu.rs`). Adding AMD (`rocm-smi`) or
  Apple (`system_profiler`) detection is a separate, larger change.
- No multi-GPU tensor-parallel throughput model — SPEED uses the single GPU
  with the most free VRAM as the reference device; multi-GPU split
  throughput is non-linear and isn't modeled.
- No blended single "composite score" across FIT/SPEED/PRECISION — the ask
  this was built for was "best quant for speed" and "best quant for least
  precision loss" as two separate, inspectable recommendations, not one
  opaque number.
