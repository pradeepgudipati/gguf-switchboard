# Quant scoring: FIT, SPEED, BALANCED, PRECISION

`gguf-switchboard models search` and `models files` score every discovered GGUF
quant against your detected hardware instead of reporting a flat
`Supported: Yes/No`. This doc explains what each number means, exactly how
it's computed, where the constants come from, and how to check them against
your own machine. Implementation: [`src/quant_profile.rs`](../src/quant_profile.rs)
(FIT/SPEED/PRECISION) and [`src/config/models_cmd.rs`](../src/config/models_cmd.rs)
(`balanced_quant`, BALANCED).

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

## BALANCED (a single middle-ground quant)

FIT/SPEED/PRECISION each answer "what's the *best* option for this one
dimension" — which, taken alone, always points at one of the two extremes
(the fastest/smallest quant, or the slowest/largest one). BALANCED answers a
different question: "if I don't want to go to either extreme, what's the one
quant that gives up the least on both?"

Within a repository's set of quants that fit your hardware, file size is
almost perfectly anti-correlated between speed and precision: a smaller
quant is both faster (less to stream per token, see SPEED above) *and*
lossier (more aggressively quantized, see PRECISION above), and a larger one
is both slower and more precise. That means the size range from the smallest
to the largest fitting quant already traces the speed/precision trade-off
curve — so BALANCED is simply **the quant whose file size is closest to the
midpoint of that range**. Ties (equidistant from two quants) prefer the
smaller, cheaper-to-run one, then quant name, for determinism.

An earlier version of this instead averaged a 0–100-normalized speed score
with the quality score. That was rejected: normalizing speed against the
fastest candidate in the set gives it a much wider spread (often 3-4x from
worst to best) than quality typically has among realistic quants (usually
well under 2x from worst to best fitting quant), so an equal-weight average
is dominated by speed and keeps picking one of the fastest/lossiest
options — the opposite of "balanced." The size-midpoint approach sidesteps
having to reconcile two differently-scaled scores at all.

`models search` prints BALANCED as its own table column and, when it differs
from both the fastest and least-lossy picks, as a third `ggs models pull`
suggestion line.

## Output format

`models search` prints a header with hardware detection and speed model
inputs, then an aligned table with these columns:

| Column | Content |
|--------|---------|
| `REPO` | Hugging Face repository id |
| `FILES` | Number of `.gguf` files in the repo |
| `SIZE` | Total size of all standalone GGUF files |
| `FIT` | 0–100 memory-fit score (100 = comfortable headroom; 0 = doesn't fit) |
| `CONTEXT` | Maximum context window from GGUF metadata |
| `ARCH` | Model architecture from GGUF metadata |
| `SPEED` | Fastest quant with estimated tok/s (e.g. `Q4_K_M ~127tok/s`) |
| `BALANCED` | Middle-ground quant with tok/s and quality % (e.g. `Q5_K_M ~91tok/s/~98.9%`) |
| `PRECISION` | Least-lossy quant with quality % (e.g. `Q6_K ~99.6%`) |
| `QUANT` | All fitting quants ordered from smallest to largest |

The footer legend explains each column in plain language:

```
FIT: 0-100 memory-fit score (100 = comfortable headroom; 0 = does not fit RAM+VRAM). SPEED/PRECISION: the quant that maximizes each — tok/s from a memory-bandwidth model (verify against `llama-bench` on your machine), quality % from published per-quant perplexity measurements ("~" = extrapolated, not directly measured for this architecture). BALANCED: the quant with the best average of speed and quality, both normalized to this model's own quant options — a middle ground when you don't want either extreme. See docs/QUANT_SCORING.md for methodology and sources; override RAM bandwidth with --ram-bandwidth-gbps if you've measured your own.
```

The `"~"` prefix on SPEED and PRECISION values means the estimate is
extrapolated, not directly measured for that architecture. The `Try:` lines
at the end suggest the fastest, balanced, and least precision loss quants
with pull commands:

```
Try: ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q4_K_M   (fastest, ~127 tok/s est.)
     ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q5_K_M   (balanced, ~91 tok/s / ~98.9% quality est.)
     ggs models pull bartowski/Qwen3.5-9B-GGUF --quant Q6_K   (least precision loss, ~99.6% quality est.)
```

When a repo has FIT=0 (doesn't fit RAM+VRAM), SPEED/BALANCED/PRECISION show
`-` and QUANT is empty — the repo is listed but no recommendations are made.
Auxiliary speculative-decoding drafters require a separate target model, so
they show FIT=0 and all recommendation columns as `-` even when they fit
memory, because they cannot be loaded standalone.

## What this deliberately does not do

- No AMD/Apple GPU bandwidth table — matches this codebase's existing
  NVIDIA-only detection scope (`src/gpu.rs`). Adding AMD (`rocm-smi`) or
  Apple (`system_profiler`) detection is a separate, larger change.
- No multi-GPU tensor-parallel throughput model — SPEED uses the single GPU
  with the most free VRAM as the reference device; multi-GPU split
  throughput is non-linear and isn't modeled.
- No blended numeric score across FIT/SPEED/PRECISION — BALANCED picks a
  quant, not a score; it never averages SPEED's tokens/sec against
  PRECISION's quality points into one opaque number (see above for why that
  was tried and rejected). FIT, SPEED, and PRECISION remain three separate,
  inspectable recommendations; BALANCED is a fourth, not a merger of them.
