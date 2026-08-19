//! Per-quant speed and precision-loss scoring for `models search` / `models files`.
//!
//! This replaces the old binary "Supported: Yes/No" signal with two things a
//! GGUF quant actually differs on:
//!
//! 1. **Speed** — estimated tokens/sec, from a memory-bandwidth model. Token
//!    generation with llama.cpp is memory-bandwidth bound: every token requires
//!    streaming the full (offloaded) weight set through the GPU or system RAM
//!    once. So:
//!
//!    `tokens/sec ≈ (effective_bandwidth_GB_s / model_size_GB) × efficiency_factor`
//!
//!    This is the same approach used by [llmfit](https://github.com/AlexsJones/llmfit)
//!    (`bandwidth_GB_s / model_size_GB × efficiency_factor`, default efficiency
//!    0.55). Every [`SpeedEstimate`] carries the exact inputs it used
//!    (`gpu_name`, `bandwidth_gbps`, `mode`, `efficiency_factor`) so a number can
//!    be checked against `llama-bench` on the actual machine rather than trusted
//!    blindly — run with `RUST_LOG=debug` to see them logged per model.
//!
//! 2. **Precision loss** — a 0–100 quality score derived from published
//!    perplexity-increase measurements per quant, relative to fp16/bf16:
//!
//!    - k-quants PR #1684 (LLaMA-7B, 2023):
//!      <https://github.com/ggml-org/llama.cpp/pull/1684>
//!    - "Which Quantization Should I Use? A Unified Evaluation of llama.cpp
//!      Quantization on Llama-3.1-8B-Instruct" (2026): <https://arxiv.org/abs/2601.14277>
//!
//!    The two sources disagree in absolute terms — the newer, more heavily
//!    trained Llama-3.1-8B loses measurably *more* per quant step than 2023-era
//!    LLaMA-7B did, because there's less redundant weight left to compress.
//!    [`QUANT_PPL_INCREASE_PCT`] below uses the Llama-3.1-8B numbers as the
//!    default table (more representative of current dense models). Entries not
//!    covered by either source (Q2_K, most IQ-quants) are extrapolated from the
//!    k-quants ratio and marked [`Confidence::Extrapolated`] — treat those as
//!    "roughly this order of magnitude," not measured fact.
//!
//!    If you have *actual* measured per-model perplexity/KL-divergence data
//!    (an imatrix run, a quantizer's model card with numbers), that is strictly
//!    better than this generic table for that specific model — there is
//!    currently no automated way to pull that from the HF search API for an
//!    arbitrary repo, so it isn't wired in here. The lookup is a single function
//!    ([`quant_quality`]) so a measured-data override is a small, contained
//!    change if/when a reliable source shows up.

use tracing::debug;

use crate::gpu::GpuDeviceInfo;

// ── Confidence ───────────────────────────────────────────────────────────────

/// How much to trust a given number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Backed by a published measurement (see module docs for sources).
    Measured,
    /// Derived by extrapolating from a measured ratio, or a fallback constant
    /// for hardware/quant types with no direct data. Order-of-magnitude only.
    Extrapolated,
}

// ── Precision-loss scoring ──────────────────────────────────────────────────

/// Perplexity increase (%) vs fp16/bf16, per quant label.
///
/// Primary source: Llama-3.1-8B-Instruct, WikiText-2 (arXiv:2601.14277), fp16
/// baseline PPL 7.32. Entries not in that paper are extrapolated from the
/// k-quants PR #1684 LLaMA-7B ratios (documented inline) and marked
/// [`Confidence::Extrapolated`].
const QUANT_PPL_INCREASE_PCT: &[(&str, f32, Confidence)] = &[
    ("F32", 0.0, Confidence::Measured),
    ("F16", 0.0, Confidence::Measured),
    ("BF16", 0.0, Confidence::Measured),
    ("Q8_0", 0.1, Confidence::Measured),
    ("Q6_K", 0.4, Confidence::Measured),
    ("Q5_K_M", 1.1, Confidence::Measured),
    ("Q5_K_S", 1.5, Confidence::Measured),
    ("Q5_1", 1.5, Confidence::Measured),
    ("Q5_0", 1.5, Confidence::Measured),
    // IQ4_* generally match or beat legacy Q4 at the same/lower bpw thanks to the
    // importance matrix; placed just above their legacy counterparts.
    ("IQ4_NL", 3.0, Confidence::Extrapolated),
    ("Q4_K_M", 3.3, Confidence::Measured),
    ("IQ4_XS", 3.8, Confidence::Extrapolated),
    ("Q4_K_S", 4.1, Confidence::Measured),
    ("Q4_1", 5.5, Confidence::Measured),
    ("Q4_0", 5.7, Confidence::Measured),
    ("Q3_K_L", 6.7, Confidence::Measured),
    ("IQ3_M", 8.0, Confidence::Extrapolated),
    ("Q3_K_M", 8.7, Confidence::Measured),
    ("IQ3_S", 10.5, Confidence::Extrapolated),
    ("IQ3_XXS", 13.0, Confidence::Extrapolated),
    ("Q3_K_S", 22.4, Confidence::Measured),
    // Q2_K/IQ2_*/IQ1_* extrapolated: PR #1684 measured Q2_K/Q3_K_S ppl-increase
    // ratio ≈ 1.58x on LLaMA-7B (0.8698 / 0.5505); applied to the Q3_K_S figure
    // above (22.4 * 1.58 ≈ 35.4, rounded).
    ("IQ2_M", 27.0, Confidence::Extrapolated),
    ("Q2_K", 35.0, Confidence::Extrapolated),
    ("IQ2_XS", 33.0, Confidence::Extrapolated),
    ("IQ2_XXS", 40.0, Confidence::Extrapolated),
    ("IQ1_M", 55.0, Confidence::Extrapolated),
    ("IQ1_S", 70.0, Confidence::Extrapolated),
];

/// Bit-width family prefixes, longest/most-specific first, used when a quant
/// label isn't an exact match (e.g. an unseen suffix variant like `Q4_K_XL`).
const FAMILY_PREFIXES: &[&str] = &[
    "IQ1", "IQ2", "IQ3", "IQ4", "Q2", "Q3", "Q4", "Q5", "Q6", "Q8",
];

/// Precision-loss score for a single quant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantQuality {
    /// Estimated perplexity increase (%) vs fp16, lower is better.
    pub ppl_increase_pct: f32,
    /// 0–100 quality score, 100 = indistinguishable from fp16.
    pub quality_score: f32,
    pub confidence: Confidence,
}

/// Look up (or estimate) the precision-loss score for a quant label.
///
/// Exact match first; otherwise falls back to the same bit-width family
/// (e.g. an unrecognized `Q4_K_XL` falls back to the `Q4_*` entries' average)
/// with [`Confidence::Extrapolated`]; otherwise a conservative mid-table guess.
pub fn quant_quality(quant: &str) -> QuantQuality {
    let upper = quant.trim().to_ascii_uppercase();

    if let Some(&(_, pct, confidence)) = QUANT_PPL_INCREASE_PCT
        .iter()
        .find(|(label, _, _)| *label == upper)
    {
        return build_quality(pct, confidence);
    }

    if let Some(prefix) = FAMILY_PREFIXES.iter().find(|p| upper.starts_with(**p)) {
        let matches: Vec<f32> = QUANT_PPL_INCREASE_PCT
            .iter()
            .filter(|(label, _, _)| label.starts_with(prefix))
            .map(|(_, pct, _)| *pct)
            .collect();
        if !matches.is_empty() {
            let avg = matches.iter().sum::<f32>() / matches.len() as f32;
            return build_quality(avg, Confidence::Extrapolated);
        }
    }

    // Totally unrecognized quant label: assume a middling loss rather than
    // claiming near-lossless or unusable.
    build_quality(10.0, Confidence::Extrapolated)
}

fn build_quality(ppl_increase_pct: f32, confidence: Confidence) -> QuantQuality {
    QuantQuality {
        ppl_increase_pct,
        quality_score: (100.0 - ppl_increase_pct).clamp(0.0, 100.0),
        confidence,
    }
}

// ── Speed scoring ────────────────────────────────────────────────────────────

/// Memory bandwidth (GB/s) by GPU name substring, most-specific pattern first
/// so e.g. "RTX 4080 SUPER" is checked before the bare "RTX 4080" would match.
/// NVIDIA only, matching this codebase's existing `nvidia-smi`-based detection
/// in [`crate::gpu`]. Figures are from vendor datasheets (GDDR/HBM effective
/// bandwidth), not independently benchmarked — verify with `llama-bench` for
/// your exact card.
const GPU_BANDWIDTH_GBPS: &[(&str, f64)] = &[
    ("H200", 4800.0),
    ("H100 SXM", 3350.0),
    ("H100", 2000.0),
    ("A100 80GB", 2039.0),
    ("A100", 1555.0),
    ("RTX 5090", 1792.0),
    ("RTX 5080", 960.0),
    ("RTX 5070 TI", 896.0),
    ("RTX 5070", 672.0),
    ("RTX 4090", 1008.0),
    ("RTX 4080 SUPER", 736.0),
    ("RTX 4080", 716.8),
    ("RTX 4070 TI SUPER", 672.0),
    ("RTX 4070 TI", 504.2),
    ("RTX 4070 SUPER", 504.2),
    ("RTX 4070", 504.2),
    ("RTX 4060 TI", 288.0),
    ("RTX 4060", 272.0),
    ("RTX 3090 TI", 1008.0),
    ("RTX 3090", 936.2),
    ("RTX 3080 TI", 912.4),
    ("RTX 3080", 760.3),
    ("RTX 3070 TI", 608.3),
    ("RTX 3070", 448.0),
    ("RTX 3060 TI", 448.0),
    ("RTX 3060", 360.0),
    ("TITAN RTX", 672.0),
    ("RTX 2080 TI", 616.0),
    ("RTX 2080", 448.0),
    ("L40S", 864.0),
    ("L40", 864.0),
    ("L4", 300.0),
    ("A6000", 768.0),
    ("A5000", 768.0),
    ("A4000", 448.0),
    ("V100", 900.0),
    ("T4", 320.0),
];

/// Bandwidth fallback (GB/s) for an unrecognized NVIDIA GPU — a conservative
/// mid-range consumer-card guess. Always [`Confidence::Extrapolated`].
const GPU_BANDWIDTH_FALLBACK_GBPS: f64 = 400.0;

/// Default effective system RAM bandwidth (GB/s) used for CPU-only inference
/// and the CPU-offloaded portion of a partial-GPU load: roughly what a modern
/// dual-channel DDR4-3200..DDR5-5600 desktop sustains in practice, after
/// controller/refresh overhead (well below the DDR5-5600 theoretical 89.6
/// GB/s). Override with `--ram-bandwidth-gbps` if you've measured your own
/// (e.g. via `mbw` or `likwid-bench`).
pub const RAM_BANDWIDTH_GBPS_DEFAULT: f64 = 40.0;

/// Fraction of peak bandwidth actually realized for GPU-resident generation:
/// accounts for kernel launch overhead, KV-cache reads, and memory-controller
/// inefficiency. Matches llmfit's default (0.55).
const GPU_EFFICIENCY_FACTOR: f64 = 0.55;

/// Lower efficiency factor for CPU-bound (or CPU-offloaded) generation: CPU
/// token generation is compute-bound as well as bandwidth-bound, so realized
/// throughput is further below the naive bandwidth/size ratio than on GPU.
const CPU_EFFICIENCY_FACTOR: f64 = 0.35;

/// Fraction of free VRAM budgeted before falling back to CPU offload, leaving
/// headroom for KV-cache and context.
const VRAM_USABLE_FRACTION: f64 = 0.9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeedMode {
    /// Entire model fits in free VRAM.
    FullGpu,
    /// Model exceeds free VRAM; part of it is offloaded to system RAM.
    PartialOffload,
    /// No GPU detected.
    CpuOnly,
}

impl SpeedMode {
    pub fn label(self) -> &'static str {
        match self {
            SpeedMode::FullGpu => "GPU",
            SpeedMode::PartialOffload => "GPU+CPU",
            SpeedMode::CpuOnly => "CPU",
        }
    }
}

/// Estimated generation speed for one quant on the detected/assumed hardware.
/// Carries every input the formula used, per the module's transparency goal.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeedEstimate {
    pub tokens_per_sec: f64,
    pub mode: SpeedMode,
    pub gpu_name: Option<String>,
    pub bandwidth_gbps: f64,
    pub efficiency_factor: f64,
    pub confidence: Confidence,
}

/// Detected (or overridden) hardware context used for speed estimation.
#[derive(Debug, Clone, Default)]
pub struct HardwareCtx {
    pub gpus: Vec<GpuDeviceInfo>,
    pub ram_bandwidth_gbps: f64,
}

impl HardwareCtx {
    /// Probe live hardware (NVIDIA GPUs via `nvidia-smi`; RAM bandwidth uses
    /// the documented default unless overridden).
    pub fn detect(ram_bandwidth_override: Option<f64>) -> Self {
        Self {
            gpus: crate::gpu::probe_all_gpus(),
            ram_bandwidth_gbps: ram_bandwidth_override.unwrap_or(RAM_BANDWIDTH_GBPS_DEFAULT),
        }
    }

    fn free_vram_mb(&self) -> u64 {
        self.gpus.iter().map(|g| g.free_mb).sum()
    }

    fn best_gpu(&self) -> Option<&GpuDeviceInfo> {
        self.gpus.iter().max_by_key(|g| g.free_mb)
    }
}

/// GPU memory bandwidth (GB/s) for a device name, by substring match against
/// [`GPU_BANDWIDTH_GBPS`]. Falls back to [`GPU_BANDWIDTH_FALLBACK_GBPS`].
pub fn gpu_bandwidth_gbps(name: &str) -> (f64, Confidence) {
    let upper = name.to_ascii_uppercase();
    for (pattern, bandwidth) in GPU_BANDWIDTH_GBPS {
        if upper.contains(pattern) {
            return (*bandwidth, Confidence::Measured);
        }
    }
    (GPU_BANDWIDTH_FALLBACK_GBPS, Confidence::Extrapolated)
}

/// Estimate tokens/sec for a quant of `model_bytes` on `hw`.
pub fn estimate_speed(model_bytes: u64, hw: &HardwareCtx) -> SpeedEstimate {
    let model_gb = model_bytes as f64 / 1_000_000_000.0;
    if model_gb <= 0.0 {
        return SpeedEstimate {
            tokens_per_sec: 0.0,
            mode: SpeedMode::CpuOnly,
            gpu_name: None,
            bandwidth_gbps: 0.0,
            efficiency_factor: 0.0,
            confidence: Confidence::Extrapolated,
        };
    }

    let Some(best_gpu) = hw.best_gpu() else {
        let tokens_per_sec = (hw.ram_bandwidth_gbps / model_gb) * CPU_EFFICIENCY_FACTOR;
        let estimate = SpeedEstimate {
            tokens_per_sec,
            mode: SpeedMode::CpuOnly,
            gpu_name: None,
            bandwidth_gbps: hw.ram_bandwidth_gbps,
            efficiency_factor: CPU_EFFICIENCY_FACTOR,
            confidence: Confidence::Extrapolated,
        };
        log_estimate(model_gb, &estimate);
        return estimate;
    };

    let (bandwidth_gbps, bw_confidence) = gpu_bandwidth_gbps(&best_gpu.name);
    // MiB (as reported by nvidia-smi) treated as ~GB; within a few percent at
    // this scale, well inside the error bars of the rest of this estimate.
    let usable_vram_gb = (hw.free_vram_mb() as f64 / 1024.0) * VRAM_USABLE_FRACTION;

    let estimate = if model_gb <= usable_vram_gb {
        SpeedEstimate {
            tokens_per_sec: (bandwidth_gbps / model_gb) * GPU_EFFICIENCY_FACTOR,
            mode: SpeedMode::FullGpu,
            gpu_name: Some(best_gpu.name.clone()),
            bandwidth_gbps,
            efficiency_factor: GPU_EFFICIENCY_FACTOR,
            confidence: bw_confidence,
        }
    } else {
        let offload_fraction = ((model_gb - usable_vram_gb) / model_gb).clamp(0.0, 1.0);
        let effective_bandwidth =
            (1.0 - offload_fraction) * bandwidth_gbps + offload_fraction * hw.ram_bandwidth_gbps;
        let effective_efficiency = (1.0 - offload_fraction) * GPU_EFFICIENCY_FACTOR
            + offload_fraction * CPU_EFFICIENCY_FACTOR;
        SpeedEstimate {
            tokens_per_sec: (effective_bandwidth / model_gb) * effective_efficiency,
            mode: SpeedMode::PartialOffload,
            gpu_name: Some(best_gpu.name.clone()),
            bandwidth_gbps: effective_bandwidth,
            efficiency_factor: effective_efficiency,
            confidence: Confidence::Extrapolated,
        }
    };
    log_estimate(model_gb, &estimate);
    estimate
}

fn log_estimate(model_gb: f64, estimate: &SpeedEstimate) {
    debug!(
        model_gb,
        tokens_per_sec = estimate.tokens_per_sec,
        mode = estimate.mode.label(),
        gpu_name = estimate.gpu_name.as_deref().unwrap_or("none"),
        bandwidth_gbps = estimate.bandwidth_gbps,
        efficiency_factor = estimate.efficiency_factor,
        confidence = ?estimate.confidence,
        "quant_profile: speed estimate inputs"
    );
}

// ── Fit scoring ──────────────────────────────────────────────────────────────

/// Continuous 0–100 replacement for the old binary "Supported: Yes/No".
///
/// Mirrors the old threshold (model fits within `capacity_bytes` with 20%
/// reserved headroom → previously "Yes") at the score's midpoint, but degrades
/// smoothly either side of it instead of dropping off a cliff:
///
/// - ≤80% utilization: 100 (comfortable headroom for KV-cache/context)
/// - 80%..83.3% (the old 20%-headroom cutoff): 100 → 60
/// - 83.3%..100%: 60 → 20 (fits, but tight — little room for context)
/// - \>100%: 0 (does not fit RAM+VRAM at all)
pub fn fit_score(model_bytes: u64, capacity_bytes: u64) -> f32 {
    if capacity_bytes == 0 {
        return 0.0;
    }
    let utilization = model_bytes as f64 / capacity_bytes as f64;
    let old_cutoff = 1.0 / 1.2; // ≈ 0.8333, matches `is_supported`'s 20% headroom

    let score = if utilization > 1.0 {
        0.0
    } else if utilization > old_cutoff {
        lerp(60.0, 20.0, (utilization - old_cutoff) / (1.0 - old_cutoff))
    } else if utilization > 0.8 {
        lerp(100.0, 60.0, (utilization - 0.8) / (old_cutoff - 0.8))
    } else {
        100.0
    };
    score as f32
}

fn lerp(from: f64, to: f64, t: f64) -> f64 {
    from + (to - from) * t.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, free_mb: u64) -> GpuDeviceInfo {
        GpuDeviceInfo {
            index: 0,
            name: name.to_string(),
            total_mb: free_mb,
            free_mb,
            used_mb: 0,
        }
    }

    #[test]
    fn quality_exact_match() {
        let q = quant_quality("Q4_K_M");
        assert_eq!(q.confidence, Confidence::Measured);
        assert!((q.quality_score - 96.7).abs() < 0.01);
    }

    #[test]
    fn quality_case_insensitive() {
        let q = quant_quality("q8_0");
        assert_eq!(q.confidence, Confidence::Measured);
        assert!(q.quality_score > 99.0);
    }

    #[test]
    fn quality_family_fallback_for_unknown_suffix() {
        let q = quant_quality("Q4_K_XL");
        assert_eq!(q.confidence, Confidence::Extrapolated);
        // Should land in the Q4 family's range, not defer to the generic guess.
        assert!(q.ppl_increase_pct > 2.0 && q.ppl_increase_pct < 6.0);
    }

    #[test]
    fn quality_totally_unknown_quant_is_conservative_guess() {
        let q = quant_quality("XYZ_MADEUP");
        assert_eq!(q.confidence, Confidence::Extrapolated);
        assert!((q.quality_score - 90.0).abs() < 0.01);
    }

    #[test]
    fn quality_orders_quants_by_bits() {
        // Higher bit-width should never score worse than a lower one.
        assert!(quant_quality("Q8_0").quality_score > quant_quality("Q6_K").quality_score);
        assert!(quant_quality("Q6_K").quality_score > quant_quality("Q5_K_M").quality_score);
        assert!(quant_quality("Q5_K_M").quality_score > quant_quality("Q4_K_M").quality_score);
        assert!(quant_quality("Q4_K_M").quality_score > quant_quality("Q3_K_M").quality_score);
        assert!(quant_quality("Q3_K_M").quality_score > quant_quality("Q2_K").quality_score);
    }

    #[test]
    fn bandwidth_lookup_matches_specific_before_generic() {
        let (bw, conf) = gpu_bandwidth_gbps("NVIDIA GeForce RTX 4080 SUPER");
        assert_eq!(conf, Confidence::Measured);
        assert!((bw - 736.0).abs() < 0.01);

        let (bw, _) = gpu_bandwidth_gbps("NVIDIA GeForce RTX 4080");
        assert!((bw - 716.8).abs() < 0.01);
    }

    #[test]
    fn bandwidth_lookup_unknown_gpu_falls_back() {
        let (bw, conf) = gpu_bandwidth_gbps("Some Future GPU 9000");
        assert_eq!(conf, Confidence::Extrapolated);
        assert!((bw - GPU_BANDWIDTH_FALLBACK_GBPS).abs() < 0.01);
    }

    #[test]
    fn speed_full_gpu_when_model_fits_in_vram() {
        let hw = HardwareCtx {
            gpus: vec![gpu("NVIDIA GeForce RTX 4090", 20_000)],
            ram_bandwidth_gbps: RAM_BANDWIDTH_GBPS_DEFAULT,
        };
        // 5 GB model, plenty of free VRAM.
        let estimate = estimate_speed(5_000_000_000, &hw);
        assert_eq!(estimate.mode, SpeedMode::FullGpu);
        assert_eq!(
            estimate.gpu_name.as_deref(),
            Some("NVIDIA GeForce RTX 4090")
        );
        // (1008 / 5) * 0.55 ≈ 110.9 tok/s
        assert!((estimate.tokens_per_sec - 110.88).abs() < 1.0);
    }

    #[test]
    fn speed_partial_offload_when_model_exceeds_vram() {
        let hw = HardwareCtx {
            gpus: vec![gpu("NVIDIA GeForce RTX 3060", 8_000)],
            ram_bandwidth_gbps: RAM_BANDWIDTH_GBPS_DEFAULT,
        };
        // 20 GB model vs ~7.2 GB usable VRAM: mostly CPU-offloaded.
        let estimate = estimate_speed(20_000_000_000, &hw);
        assert_eq!(estimate.mode, SpeedMode::PartialOffload);
        assert!(estimate.tokens_per_sec > 0.0);
        // Should be slower than a fully-GPU-resident estimate at the same size.
        let full_gpu_hw = HardwareCtx {
            gpus: vec![gpu("NVIDIA GeForce RTX 3060", 40_000)],
            ram_bandwidth_gbps: RAM_BANDWIDTH_GBPS_DEFAULT,
        };
        let full_estimate = estimate_speed(20_000_000_000, &full_gpu_hw);
        assert!(estimate.tokens_per_sec < full_estimate.tokens_per_sec);
    }

    #[test]
    fn speed_cpu_only_when_no_gpu() {
        let hw = HardwareCtx {
            gpus: vec![],
            ram_bandwidth_gbps: RAM_BANDWIDTH_GBPS_DEFAULT,
        };
        let estimate = estimate_speed(8_000_000_000, &hw);
        assert_eq!(estimate.mode, SpeedMode::CpuOnly);
        assert!(estimate.gpu_name.is_none());
        // (40 / 8) * 0.35 = 1.75 tok/s
        assert!((estimate.tokens_per_sec - 1.75).abs() < 0.01);
    }

    #[test]
    fn fit_score_comfortable_utilization_is_perfect() {
        assert_eq!(fit_score(4_000, 10_000), 100.0);
    }

    #[test]
    fn fit_score_matches_old_cutoff_at_midpoint() {
        // Old is_supported cutoff: bytes * 1.2 <= capacity, i.e. utilization 1/1.2.
        let capacity = 10_000_u64;
        let bytes_at_cutoff = (capacity as f64 / 1.2) as u64;
        let score = fit_score(bytes_at_cutoff, capacity);
        assert!((score - 60.0).abs() < 2.0);
    }

    #[test]
    fn fit_score_zero_when_over_capacity() {
        assert_eq!(fit_score(12_000, 10_000), 0.0);
    }

    #[test]
    fn fit_score_zero_capacity_is_zero() {
        assert_eq!(fit_score(1, 0), 0.0);
    }
}
