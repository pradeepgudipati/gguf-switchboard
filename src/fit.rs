//! ModelFitPlanner — hardware-aware preflight planning with bounded fallback ladder.
//!
//! Before every model load, the planner inspects GPU topology / free VRAM, model
//! metadata, and requested context, then produces a [`FitPlan`] that is safe to
//! attempt.  On OOM the planner advances through a bounded degradation sequence:
//!
//! 1. Requested context + default KV + auto-fit GPU
//! 2. Requested context + Q8 KV  + auto-fit GPU
//! 3. 75 % context  + Q8 KV  + auto-fit GPU
//! 4. 50 % context  + Q8 KV  + auto-fit GPU
//! 5. 25 % context  + Q8 KV  + reduced GPU offload
//!
//! The minimum context is clamped to the configured floor (default 4096).

use std::fmt;

use tracing::{debug, info, warn};

use crate::gpu::GpuDeviceInfo;

// ── Configuration types (simplified; real values come from Config) ────────────

/// Fit-related configuration, deserialized from the `[fit]` TOML section.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FitConfig {
    #[serde(default = "default_fit_enabled")]
    pub enabled: bool,
    #[serde(default = "default_vram_reserve_mb")]
    pub vram_reserve_mb: u32,
    #[serde(default = "default_multi_gpu")]
    pub multi_gpu: String,
    #[serde(default = "default_split_mode")]
    pub split_mode: String,
    #[serde(default)]
    pub max_attempts: u32,
    #[serde(default = "default_true")]
    pub cache_profiles: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            enabled: default_fit_enabled(),
            vram_reserve_mb: default_vram_reserve_mb(),
            multi_gpu: default_multi_gpu(),
            split_mode: default_split_mode(),
            max_attempts: 5,
            cache_profiles: true,
        }
    }
}

impl FitConfig {
    /// KV cache fallback types (hard-coded for now; could be made configurable).
    pub fn kv_fallback_types(&self) -> Vec<String> {
        vec!["q8_0".to_string()]
    }

    /// Context reduction step ratios.
    pub fn context_steps(&self) -> Vec<f64> {
        vec![1.0, 0.75, 0.5, 0.25]
    }

    /// Minimum context size floor.
    pub fn context_minimum(&self) -> u32 {
        4096
    }
}

fn default_fit_enabled() -> bool {
    false
}
fn default_vram_reserve_mb() -> u32 {
    2048
}
fn default_multi_gpu() -> String {
    "auto".to_string()
}
fn default_split_mode() -> String {
    "layer".to_string()
}
fn default_true() -> bool {
    true
}

// ── Hardware ─────────────────────────────────────────────────────────────────

/// Snapshot of the machine's GPU and RAM state at a point in time.
#[derive(Debug, Clone)]
pub struct HardwareSummary {
    pub gpus: Vec<GpuDeviceInfo>,
    pub total_vram_mb: u64,
    pub free_vram_mb: u64,
    pub system_ram_mb: u64,
    pub hardware_fingerprint: String,
}

impl HardwareSummary {
    /// Probe the current hardware state.
    ///
    /// `vram_gb` is the fallback VRAM capacity when nvidia-smi is unavailable.
    pub fn probe(vram_gb: u32) -> Self {
        let gpus = crate::gpu::probe_all_gpus();
        let total_vram_mb = if gpus.is_empty() {
            u64::from(vram_gb.max(1)) * 1024
        } else {
            gpus.iter().map(|g| g.total_mb).sum()
        };
        let free_vram_mb = if gpus.is_empty() {
            u64::from(vram_gb.max(1)) * 1024
        } else {
            gpus.iter().map(|g| g.free_mb).sum()
        };
        let system_ram_mb = crate::memory::check_memory()
            .map(|s| s.total_mb)
            .unwrap_or(0);
        let hardware_fingerprint = compute_hardware_fingerprint(&gpus, total_vram_mb);

        Self {
            gpus,
            total_vram_mb,
            free_vram_mb,
            system_ram_mb,
            hardware_fingerprint,
        }
    }

    /// Usable VRAM budget after subtracting the safety reserve.
    pub fn usable_vram_mb(&self, reserve_mb: u32) -> u64 {
        self.free_vram_mb.saturating_sub(u64::from(reserve_mb))
    }

    /// True when more than one GPU is detected.
    pub fn is_multi_gpu(&self) -> bool {
        self.gpus.len() > 1
    }
}

fn compute_hardware_fingerprint(gpus: &[GpuDeviceInfo], total_vram_mb: u64) -> String {
    if gpus.is_empty() {
        return format!("cpu-{total_vram_mb}mb");
    }
    // Group by name → count, then format as "2xRTX4090-24564"
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for g in gpus {
        *counts.entry(&g.name).or_insert(0) += 1;
    }
    let mut parts = Vec::new();
    for (name, count) in &counts {
        // Shorten name: take last token (e.g. "RTX 4090" from "NVIDIA GeForce RTX 4090")
        let short = name.split_whitespace().last().unwrap_or(name);
        parts.push(format!("{count}x{short}"));
    }
    format!("{}-{}", parts.join("+"), total_vram_mb)
}

// ── Model metadata ───────────────────────────────────────────────────────────

/// Metadata extracted from the GGUF file and registry for fit planning.
#[derive(Debug, Clone)]
pub struct ModelSummary {
    pub file_path: String,
    pub file_size_mb: u64,
    pub block_count: Option<u32>,
    pub architecture: Option<String>,
    pub max_context_from_gguf: Option<u32>,
    pub requested_context: u32,
    pub model_fingerprint: String,
    pub ngl_pinned: bool,
    pub pinned_ngl: Option<u32>,
    /// Model kind: "chat", "embedding", "coder", "vision", etc.
    /// Used to apply kind-specific optimizations (e.g., batch sizes for embedding models).
    pub kind: Option<String>,
    /// Minimum context size this model must always keep, even after VRAM-
    /// pressure fallback reduces context for other models (`RegistryEntry.ctx`).
    /// `None` means no per-model floor beyond the existing global minimum.
    pub context_floor: Option<u32>,
}

impl ModelSummary {
    /// Build a ModelSummary from a file path and metadata already available in
    /// ModelConfig / GgufMetadata.
    pub fn from_file(
        file_path: &str,
        block_count: Option<u32>,
        architecture: Option<String>,
        max_context_from_gguf: Option<u32>,
        requested_context: u32,
        ngl_pinned: bool,
        pinned_ngl: Option<u32>,
    ) -> Self {
        let meta = std::fs::metadata(file_path).ok();
        let file_size_mb = meta.map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
        let file_name = std::path::Path::new(file_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| file_path.to_string());
        let model_fingerprint = format!("{file_name}:{file_size_mb}");

        Self {
            file_path: file_path.to_string(),
            file_size_mb,
            block_count,
            architecture,
            max_context_from_gguf,
            requested_context,
            model_fingerprint,
            ngl_pinned,
            pinned_ngl,
            kind: None,
            context_floor: None,
        }
    }

    /// Set the model kind for kind-specific optimizations.
    pub fn with_kind(mut self, kind: &str) -> Self {
        self.kind = Some(kind.to_string());
        self
    }

    /// Set the per-model minimum context floor (`RegistryEntry.ctx`).
    pub fn with_context_floor(mut self, floor: Option<u32>) -> Self {
        self.context_floor = floor;
        self
    }
}

// ── FitPlan ──────────────────────────────────────────────────────────────────

/// A single load profile to attempt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FitPlan {
    pub context_size: u32,
    pub ngl: u32,
    pub split_mode: Option<String>,
    pub tensor_split: Option<Vec<f64>>,
    pub cache_type_k: Option<String>,
    pub cache_type_v: Option<String>,
    pub reason: String,
    pub attempt: u32,
    /// Batch size for embedding models (`-b`). None for non-embedding models.
    pub batch_size: Option<u32>,
    /// Micro-batch size for embedding models (`-ub`). None for non-embedding models.
    pub ubatch_size: Option<u32>,
}

impl fmt::Display for FitPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ctx={} ngl={} kv={}/{} split={:?} ts={:?} batch={}/{} [attempt {} — {}]",
            self.context_size,
            self.ngl,
            self.cache_type_k.as_deref().unwrap_or("f16"),
            self.cache_type_v.as_deref().unwrap_or("f16"),
            self.split_mode,
            self.tensor_split,
            self.batch_size.unwrap_or(0),
            self.ubatch_size.unwrap_or(0),
            self.attempt,
            self.reason,
        )
    }
}

// ── FitPlanner ───────────────────────────────────────────────────────────────

/// State machine that produces a sequence of increasingly conservative [`FitPlan`]s.
pub struct FitPlanner {
    hardware: HardwareSummary,
    model: ModelSummary,
    config: FitConfig,
    plans: Vec<FitPlan>,
    current: usize,
}

impl FitPlanner {
    /// Create a new planner and pre-compute the full fallback ladder.
    pub fn new(hardware: HardwareSummary, model: ModelSummary, config: FitConfig) -> Self {
        let plans = build_fallback_ladder(&hardware, &model, &config);
        info!(
            model = %model.file_path,
            attempts = plans.len(),
            hardware = %hardware.hardware_fingerprint,
            "FitPlanner: built fallback ladder"
        );
        Self {
            hardware,
            model,
            config,
            plans,
            current: 0,
        }
    }

    /// The current plan to attempt.
    pub fn current_plan(&self) -> &FitPlan {
        &self.plans[self.current.min(self.plans.len() - 1)]
    }

    /// Advance to the next fallback plan.  Returns `None` when exhausted.
    pub fn advance(&mut self, failure_reason: &str) -> Option<&FitPlan> {
        debug!(
            attempt = self.current + 1,
            reason = failure_reason,
            "FitPlanner: advancing to next fallback"
        );
        self.current += 1;
        if self.current >= self.plans.len() {
            warn!(
                model = %self.model.file_path,
                attempts = self.plans.len(),
                "FitPlanner: all attempts exhausted"
            );
            return None;
        }
        Some(self.current_plan())
    }

    /// True when there are no more fallback plans.
    pub fn is_exhausted(&self) -> bool {
        self.current >= self.plans.len()
    }

    /// The total number of pre-computed plans.
    pub fn total_plans(&self) -> usize {
        self.plans.len()
    }

    /// Access all pre-computed plans (for dry-run display).
    pub fn all_plans(&self) -> &[FitPlan] {
        &self.plans
    }

    /// Access the hardware summary.
    pub fn hardware(&self) -> &HardwareSummary {
        &self.hardware
    }

    /// Access the model summary.
    pub fn model(&self) -> &ModelSummary {
        &self.model
    }

    /// Access the fit config.
    pub fn fit_config(&self) -> &FitConfig {
        &self.config
    }

    /// Apply the current plan to a base set of llama-server args.
    ///
    /// This rewrites `-c`, `-ngl`, `--split-mode`, `--tensor-split`,
    /// `--cache-type-k`, `--cache-type-v`, `-b`, and `-ub` as needed.
    pub fn apply_plan_to_args(base_args: &[String], plan: &FitPlan) -> Vec<String> {
        let mut args = base_args.to_vec();

        // Context size
        args = crate::context::with_context_size(&args, plan.context_size);

        // NGL (unless the model pins it)
        // The caller is responsible for checking ngl_pinned before calling this.
        args = crate::ngl::with_ngl(&args, plan.ngl);

        // Split mode
        if let Some(ref mode) = plan.split_mode {
            args = with_flag(&args, "--split-mode", mode);
        }

        // Tensor split
        if let Some(ref ts) = plan.tensor_split {
            let ts_str = ts
                .iter()
                .map(|v| format!("{v:.1}"))
                .collect::<Vec<_>>()
                .join(",");
            args = with_flag(&args, "--tensor-split", &ts_str);
        }

        // KV cache type (k)
        if let Some(ref ct) = plan.cache_type_k {
            args = with_flag(&args, "--cache-type-k", ct);
        }

        // KV cache type (v)
        if let Some(ref ct) = plan.cache_type_v {
            args = with_flag(&args, "--cache-type-v", ct);
        }

        // Batch size (for embedding models)
        if let Some(batch) = plan.batch_size {
            args = crate::batch::with_batch_size(&args, batch);
        }

        // Micro-batch size (for embedding models)
        if let Some(ubatch) = plan.ubatch_size {
            args = crate::batch::with_ubatch_size(&args, ubatch);
        }

        args
    }
}

/// Set or append a `--flag value` pair in args.
fn with_flag(args: &[String], flag: &str, value: &str) -> Vec<String> {
    let mut updated = args.to_vec();
    for i in 0..updated.len() {
        if updated[i] == flag && i + 1 < updated.len() {
            updated[i + 1] = value.to_string();
            return updated;
        }
    }
    updated.push(flag.to_string());
    updated.push(value.to_string());
    updated
}

/// Read the value following `flag` in an args list, if present.
fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

// ── vLLM fit planning ───────────────────────────────────────────────────────
//
// vLLM has a different set of serving knobs than llama.cpp — no `-ngl`/KV
// quant type, instead `--max-model-len` (context), `--gpu-memory-utilization`
// (fraction of VRAM vLLM is allowed to claim), and `--tensor-parallel-size`
// (GPU sharding). The fallback ladder here only varies context on OOM, since
// that is the one knob analogous to llama.cpp's context-halving that's safe
// to change after a failed load without re-deriving GPU topology.

/// Read `--max-model-len` from a vLLM args list.
pub fn vllm_max_model_len_from_args(args: &[String]) -> Option<u32> {
    flag_value(args, "--max-model-len").and_then(|v| v.parse().ok())
}

/// Read `--gpu-memory-utilization` from a vLLM args list.
pub fn vllm_gpu_memory_utilization_from_args(args: &[String]) -> Option<f32> {
    flag_value(args, "--gpu-memory-utilization").and_then(|v| v.parse().ok())
}

/// Read `--tensor-parallel-size` from a vLLM args list.
pub fn vllm_tensor_parallel_size_from_args(args: &[String]) -> Option<u32> {
    flag_value(args, "--tensor-parallel-size").and_then(|v| v.parse().ok())
}

/// Sum `*.safetensors` file sizes in a local directory, in MB. Returns 0 when
/// the directory doesn't exist or can't be read (treated as "unknown size,
/// don't block on it" by callers).
pub fn sum_safetensors_mb(dir: &std::path::Path) -> u64 {
    if !dir.is_dir() {
        return 0;
    }
    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return 0;
    };
    let total_bytes: u64 = read_dir
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        })
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();
    total_bytes / (1024 * 1024)
}

/// Estimate a vLLM model's on-disk weight size in MB: sums `*.safetensors`
/// file sizes in the model's local directory (the positional arg right after
/// `serve`). Returns 0 (treated as "unknown, assume it fits") when the model
/// is served straight from an HF repo id with no local directory.
pub fn vllm_model_size_mb_from_args(args: &[String]) -> u64 {
    let Some(model_ref) = args
        .iter()
        .position(|a| a == "serve")
        .and_then(|i| args.get(i + 1))
    else {
        return 0;
    };
    sum_safetensors_mb(std::path::Path::new(model_ref))
}

/// Coarse "can this possibly be served at all" check — not a full fit plan,
/// just a hard reject for models that are guaranteed to fail regardless of
/// context/quant tuning. Used to keep gguf-switchboard from registering (and
/// later trying, and OOM-looping on) a model the hardware fundamentally
/// cannot hold.
///
/// `allow_cpu_ram`: true for llama.cpp (which can offload to system RAM),
/// false for vLLM (GPU-resident weights only — no CPU fallback).
pub fn hardware_can_possibly_serve(
    weights_mb: u64,
    hardware: &HardwareSummary,
    allow_cpu_ram: bool,
) -> bool {
    if weights_mb == 0 {
        // Unknown size (e.g. no local file yet) — don't block registration on it.
        return true;
    }
    // 15% headroom for KV cache/activations on top of raw weight size.
    let required_mb = weights_mb.saturating_add(weights_mb / 7);
    let capacity_mb = if allow_cpu_ram {
        hardware.total_vram_mb.saturating_add(hardware.system_ram_mb)
    } else {
        hardware.total_vram_mb
    };
    required_mb <= capacity_mb
}

/// A single vLLM load profile to attempt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VllmFitPlan {
    pub max_model_len: u32,
    pub gpu_memory_utilization: f32,
    pub tensor_parallel_size: u32,
    pub reason: String,
    pub attempt: u32,
}

impl fmt::Display for VllmFitPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_model_len={} gpu_memory_utilization={:.2} tensor_parallel_size={} [attempt {} — {}]",
            self.max_model_len,
            self.gpu_memory_utilization,
            self.tensor_parallel_size,
            self.attempt,
            self.reason,
        )
    }
}

/// State machine producing a bounded, increasingly conservative sequence of
/// [`VllmFitPlan`]s — the vLLM analog of [`FitPlanner`].
pub struct VllmFitPlanner {
    plans: Vec<VllmFitPlan>,
    current: usize,
}

impl VllmFitPlanner {
    /// Compute an initial plan (and context-reduction fallback ladder) from
    /// hardware and model size. `pinned_tp`/`pinned_gmu` come from an explicit
    /// `--tensor-parallel-size`/`--gpu-memory-utilization` already present in
    /// the model's args (registry override) and are never auto-adjusted.
    pub fn new(
        hardware: &HardwareSummary,
        model_size_mb: u64,
        requested_context: u32,
        pinned_tensor_parallel_size: Option<u32>,
        pinned_gpu_memory_utilization: Option<f32>,
        config: &FitConfig,
    ) -> Self {
        let plans = build_vllm_fallback_ladder(
            hardware,
            model_size_mb,
            requested_context,
            pinned_tensor_parallel_size,
            pinned_gpu_memory_utilization,
            config,
        );
        info!(
            attempts = plans.len(),
            hardware = %hardware.hardware_fingerprint,
            "VllmFitPlanner: built fallback ladder"
        );
        Self { plans, current: 0 }
    }

    pub fn current_plan(&self) -> &VllmFitPlan {
        &self.plans[self.current.min(self.plans.len() - 1)]
    }

    pub fn advance(&mut self, failure_reason: &str) -> Option<&VllmFitPlan> {
        debug!(
            attempt = self.current + 1,
            reason = failure_reason,
            "VllmFitPlanner: advancing to next fallback"
        );
        self.current += 1;
        if self.current >= self.plans.len() {
            return None;
        }
        Some(self.current_plan())
    }

    pub fn is_exhausted(&self) -> bool {
        self.current >= self.plans.len()
    }

    pub fn total_plans(&self) -> usize {
        self.plans.len()
    }

    pub fn all_plans(&self) -> &[VllmFitPlan] {
        &self.plans
    }

    /// Apply a plan to a base vLLM args list, rewriting `--max-model-len`,
    /// `--gpu-memory-utilization`, and `--tensor-parallel-size`.
    pub fn apply_plan_to_args(base_args: &[String], plan: &VllmFitPlan) -> Vec<String> {
        let mut args = with_flag(base_args, "--max-model-len", &plan.max_model_len.to_string());
        args = with_flag(
            &args,
            "--gpu-memory-utilization",
            &format!("{:.2}", plan.gpu_memory_utilization),
        );
        args = with_flag(
            &args,
            "--tensor-parallel-size",
            &plan.tensor_parallel_size.to_string(),
        );
        args
    }
}

fn build_vllm_fallback_ladder(
    hardware: &HardwareSummary,
    model_size_mb: u64,
    requested_context: u32,
    pinned_tensor_parallel_size: Option<u32>,
    pinned_gpu_memory_utilization: Option<f32>,
    config: &FitConfig,
) -> Vec<VllmFitPlan> {
    let max_attempts = config.max_attempts.max(1) as usize;
    let usable = hardware.usable_vram_mb(config.vram_reserve_mb);

    // Tensor parallel size: shard across all GPUs when the model doesn't fit
    // on a single one; otherwise 1. Left as a heuristic — vLLM additionally
    // requires attention-head counts to divide evenly by this, which we
    // can't verify without the model's config.json at plan time.
    let tensor_parallel_size = pinned_tensor_parallel_size.unwrap_or_else(|| {
        if hardware.gpus.len() <= 1 || model_size_mb == 0 {
            1
        } else {
            let per_gpu_usable = usable / hardware.gpus.len().max(1) as u64;
            if model_size_mb <= per_gpu_usable.max(1) {
                1
            } else {
                hardware.gpus.len() as u32
            }
        }
    });

    // Initial GPU memory utilization: the fraction of *total* VRAM vLLM may
    // claim. Derived from how much is actually free right now, minus the
    // configured reserve, expressed as a fraction of total — capped to a
    // sane [0.5, 0.95] band so we neither starve vLLM nor overcommit.
    let gpu_memory_utilization = pinned_gpu_memory_utilization.unwrap_or_else(|| {
        if hardware.total_vram_mb == 0 {
            0.9
        } else {
            (usable as f32 / hardware.total_vram_mb as f32).clamp(0.5, 0.95)
        }
    });

    let min_ctx = config.context_minimum().max(512);
    let mut ctx_steps: Vec<u32> = config
        .context_steps()
        .iter()
        .map(|&ratio| ((requested_context as f64 * ratio).floor() as u32).max(min_ctx))
        .collect();
    ctx_steps.sort_unstable();
    ctx_steps.dedup();
    ctx_steps.reverse();
    ctx_steps.truncate(max_attempts.max(1));

    let mut plans = Vec::new();
    for &ctx in &ctx_steps {
        let attempt = plans.len() as u32 + 1;
        let reason = if ctx < requested_context {
            format!(
                "attempt {attempt}: max_model_len {requested_context}→{ctx} ({:.0}%)",
                (ctx as f64 / requested_context as f64) * 100.0
            )
        } else {
            format!("attempt {attempt}: requested parameters")
        };
        plans.push(VllmFitPlan {
            max_model_len: ctx,
            gpu_memory_utilization,
            tensor_parallel_size,
            reason,
            attempt,
        });
    }

    if plans.is_empty() {
        plans.push(VllmFitPlan {
            max_model_len: requested_context.max(min_ctx),
            gpu_memory_utilization,
            tensor_parallel_size,
            reason: "default (no fallbacks configured)".to_string(),
            attempt: 1,
        });
    }

    plans
}

// ── Fallback ladder construction ─────────────────────────────────────────────

fn build_fallback_ladder(
    hardware: &HardwareSummary,
    model: &ModelSummary,
    config: &FitConfig,
) -> Vec<FitPlan> {
    let max_attempts = config.max_attempts.max(1) as usize;
    let embedding_headroom = is_embedding(model).then(|| {
        hardware
            .usable_vram_mb(config.vram_reserve_mb)
            .saturating_sub(model.file_size_mb)
    });
    let requested_ctx = embedding_headroom
        .map(balanced_embedding_context)
        .map(|cap| model.requested_context.min(cap))
        .unwrap_or(model.requested_context);
    let min_ctx = if is_embedding(model) {
        2048
    } else {
        config.context_minimum().max(512)
    };
    let min_ctx = min_ctx.max(model.context_floor.unwrap_or(0));

    // Compute default NGL: full offload unless model is larger than usable VRAM.
    let usable = hardware.usable_vram_mb(config.vram_reserve_mb);
    let default_ngl = if model.ngl_pinned {
        model.pinned_ngl.unwrap_or(999)
    } else if model.file_size_mb <= usable || model.block_count.is_none() {
        999
    } else {
        let layers = model.block_count.unwrap_or(999).max(1);
        let n = (u64::from(layers).saturating_mul(usable)) / model.file_size_mb.max(1);
        u32::try_from(n).unwrap_or(layers).min(layers)
    };

    // Multi-GPU tensor split
    let (default_split_mode, default_tensor_split) =
        compute_gpu_split(hardware, config, default_ngl);

    // Compute batch sizes for embedding models based on hardware.
    let (default_batch, default_ubatch, batch_steps) =
        compute_embedding_batch_sizes(hardware, model, config.vram_reserve_mb);

    // Build context steps (unique, descending, clamped to minimum)
    let mut ctx_steps: Vec<u32> = config
        .context_steps()
        .iter()
        .map(|&ratio| {
            let ctx = (requested_ctx as f64 * ratio).floor() as u32;
            ctx.max(min_ctx)
        })
        .collect();
    ctx_steps.sort_unstable();
    ctx_steps.dedup();
    ctx_steps.reverse();

    // KV fallback sequence: [None (default), ...fallback_types]
    let mut kv_sequence: Vec<Option<String>> = vec![None];
    for kv in config.kv_fallback_types() {
        kv_sequence.push(Some(kv));
    }

    // Generate plans: outer loop = context steps, inner loop = KV types.
    // Capped at max_attempts.
    let mut plans = Vec::new();

    for &ctx in &ctx_steps {
        for kv in &kv_sequence {
            if plans.len() >= max_attempts {
                break;
            }

            let attempt = plans.len() as u32 + 1;
            let (cache_type_k, cache_type_v) = match kv {
                None => (None, None),
                Some(t) => (Some(t.clone()), Some(t.clone())),
            };

            // Last attempt: try reduced GPU offload if we have layers to spare.
            let (ngl, split_mode, tensor_split) =
                if plans.len() == max_attempts - 1 && default_ngl > 1 && !model.ngl_pinned {
                    let reduced = (default_ngl * 3) / 4; // 75% of layers
                    let reason_extra = "reduced GPU offload";
                    debug!(
                        ngl = reduced,
                        reason = reason_extra,
                        "Reducing ngl for final attempt"
                    );
                    (
                        reduced,
                        default_split_mode.clone(),
                        default_tensor_split.clone(),
                    )
                } else {
                    (
                        default_ngl,
                        default_split_mode.clone(),
                        default_tensor_split.clone(),
                    )
                };

            // Compute batch size for this attempt (embedding models only).
            let (batch_size, ubatch_size) = if default_batch.is_some() {
                let step_idx = plans.len().min(batch_steps.len().saturating_sub(1));
                let batch = batch_steps.get(step_idx).copied().unwrap_or(256);
                let ubatch = default_ubatch
                    .map(|default| default.min((batch / 2).max(128)))
                    .unwrap_or(128);
                (Some(batch), Some(ubatch))
            } else {
                (None, None)
            };

            let batch_info = batch_size
                .zip(default_batch)
                .map(|(current, default)| BatchInfo { current, default });
            let reason = format_reason(
                ctx,
                requested_ctx,
                &cache_type_k,
                ngl,
                default_ngl,
                attempt,
                batch_info,
            );

            plans.push(FitPlan {
                context_size: ctx,
                ngl,
                split_mode,
                tensor_split,
                cache_type_k,
                cache_type_v,
                reason,
                attempt,
                batch_size,
                ubatch_size,
            });
        }
    }

    // Ensure at least one plan exists.
    if plans.is_empty() {
        plans.push(FitPlan {
            context_size: requested_ctx.max(min_ctx),
            ngl: default_ngl,
            split_mode: default_split_mode,
            tensor_split: default_tensor_split,
            cache_type_k: None,
            cache_type_v: None,
            reason: "default (no fallbacks configured)".to_string(),
            attempt: 1,
            batch_size: default_batch,
            ubatch_size: default_ubatch,
        });
    }

    plans
}

/// Compute batch and micro-batch sizes from VRAM left after reserve and model weights.
///
/// Returns `(default_batch, default_ubatch, fallback_steps)` where:
/// - `default_batch` / `default_ubatch` are `None` for non-embedding models
/// - `fallback_steps` is a descending list of batch sizes to try on OOM
///
/// Balanced headroom tiers keep activation memory bounded on consumer GPUs.
fn compute_embedding_batch_sizes(
    hardware: &HardwareSummary,
    model: &ModelSummary,
    reserve_mb: u32,
) -> (Option<u32>, Option<u32>, Vec<u32>) {
    if !is_embedding(model) {
        return (None, None, Vec::new());
    }

    let headroom_mb = hardware
        .usable_vram_mb(reserve_mb)
        .saturating_sub(model.file_size_mb);

    // Determine base batch size from VRAM tier.
    let (base_batch, base_ubatch) = if headroom_mb > 8 * 1024 {
        (4096, 2048)
    } else if headroom_mb >= 5 * 1024 {
        (2048, 1024)
    } else if headroom_mb >= 3 * 1024 {
        (1024, 512)
    } else if headroom_mb >= 1536 {
        (512, 256)
    } else {
        (256, 128)
    };

    // Build fallback steps: base, 75%, 50%, 25% (clamped to 256 minimum).
    let steps = vec![
        base_batch,
        (base_batch * 3) / 4,
        base_batch / 2,
        base_batch / 4,
    ];
    let steps: Vec<u32> = steps
        .into_iter()
        .map(|s| s.max(256))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut steps = steps;
    steps.sort_unstable();
    steps.dedup();
    steps.reverse();

    debug!(
        headroom_mb,
        reserve_mb,
        base_batch,
        base_ubatch,
        fallback_steps = ?steps,
        "Computed embedding model batch sizes"
    );

    (Some(base_batch), Some(base_ubatch), steps)
}

fn is_embedding(model: &ModelSummary) -> bool {
    model.kind.as_deref() == Some("embedding")
}

fn balanced_embedding_context(headroom_mb: u64) -> u32 {
    if headroom_mb > 8 * 1024 {
        16384
    } else if headroom_mb >= 3 * 1024 {
        8192
    } else if headroom_mb >= 1536 {
        4096
    } else {
        2048
    }
}

fn compute_gpu_split(
    hardware: &HardwareSummary,
    config: &FitConfig,
    _ngl: u32,
) -> (Option<String>, Option<Vec<f64>>) {
    if !hardware.is_multi_gpu() || config.multi_gpu != "auto" {
        return (None, None);
    }

    let split_mode = Some(config.split_mode.clone());

    // Compute tensor-split proportional to free VRAM.
    let min_free = hardware
        .gpus
        .iter()
        .map(|g| g.free_mb)
        .min()
        .filter(|&m| m > 0);
    let tensor_split = min_free.map(|min_free| {
        hardware
            .gpus
            .iter()
            .map(|g| {
                if min_free == 0 {
                    1.0
                } else {
                    let ratio = g.free_mb as f64 / min_free as f64;
                    (ratio * 10.0).round() / 10.0 // one decimal place
                }
            })
            .collect()
    });

    (split_mode, tensor_split)
}

/// Batch size info for format_reason (current and default, to show reductions).
struct BatchInfo {
    current: u32,
    default: u32,
}

fn format_reason(
    ctx: u32,
    requested_ctx: u32,
    kv: &Option<String>,
    ngl: u32,
    default_ngl: u32,
    attempt: u32,
    batch: Option<BatchInfo>,
) -> String {
    let mut parts = Vec::new();
    if ctx < requested_ctx {
        parts.push(format!(
            "context {}→{} ({:.0}%)",
            requested_ctx,
            ctx,
            (ctx as f64 / requested_ctx as f64) * 100.0
        ));
    }
    if let Some(kv_type) = kv {
        parts.push(format!("KV={kv_type}"));
    }
    if ngl < default_ngl {
        parts.push(format!("ngl {default_ngl}→{ngl}"));
    }
    // Show batch size reduction for embedding models.
    if let Some(BatchInfo { current, default }) = batch
        && current < default
    {
        parts.push(format!("batch {default}→{current}"));
    }
    if parts.is_empty() {
        format!("attempt {attempt}: requested parameters")
    } else {
        format!("attempt {}: {}", attempt, parts.join(", "))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_gpu(name: &str, total: u64, free: u64) -> GpuDeviceInfo {
        GpuDeviceInfo {
            index: 0,
            name: name.to_string(),
            total_mb: total,
            free_mb: free,
            used_mb: total - free,
        }
    }

    fn single_gpu_hardware() -> HardwareSummary {
        HardwareSummary {
            gpus: vec![fake_gpu("NVIDIA GeForce RTX 4090", 24564, 22000)],
            total_vram_mb: 24564,
            free_vram_mb: 22000,
            system_ram_mb: 128_000,
            hardware_fingerprint: "1xRTX4090-24564".to_string(),
        }
    }

    fn dual_gpu_hardware() -> HardwareSummary {
        HardwareSummary {
            gpus: vec![
                fake_gpu("NVIDIA GeForce RTX 4090", 24564, 22000),
                GpuDeviceInfo {
                    index: 1,
                    name: "NVIDIA GeForce RTX 4090".to_string(),
                    total_mb: 24564,
                    free_mb: 22000,
                    used_mb: 2564,
                },
            ],
            total_vram_mb: 49128,
            free_vram_mb: 44000,
            system_ram_mb: 128_000,
            hardware_fingerprint: "2xRTX4090-49128".to_string(),
        }
    }

    fn small_model() -> ModelSummary {
        ModelSummary {
            file_path: "/models/small.gguf".to_string(),
            file_size_mb: 4000,
            block_count: Some(32),
            architecture: Some("llama".to_string()),
            max_context_from_gguf: Some(32768),
            requested_context: 32768,
            model_fingerprint: "small.gguf:4000".to_string(),
            ngl_pinned: false,
            pinned_ngl: None,
            kind: Some("chat".to_string()),
            context_floor: None,
        }
    }

    fn large_model() -> ModelSummary {
        ModelSummary {
            file_path: "/models/large.gguf".to_string(),
            file_size_mb: 42000,
            block_count: Some(80),
            architecture: Some("llama".to_string()),
            max_context_from_gguf: Some(131072),
            requested_context: 32768,
            model_fingerprint: "large.gguf:42000".to_string(),
            ngl_pinned: false,
            pinned_ngl: None,
            kind: Some("chat".to_string()),
            context_floor: None,
        }
    }

    fn embedding_model() -> ModelSummary {
        ModelSummary {
            file_path: "/models/nomic-embed.gguf".to_string(),
            file_size_mb: 500,
            block_count: Some(12),
            architecture: Some("nomic-bert".to_string()),
            max_context_from_gguf: Some(8192),
            requested_context: 8192,
            model_fingerprint: "nomic-embed.gguf:500".to_string(),
            ngl_pinned: false,
            pinned_ngl: None,
            kind: Some("embedding".to_string()),
            context_floor: None,
        }
    }

    #[test]
    fn small_model_single_gpu_fits_immediately() {
        let hw = single_gpu_hardware();
        let model = small_model();
        let config = FitConfig::default();
        let planner = FitPlanner::new(hw, model, config);

        let plan = planner.current_plan();
        assert_eq!(plan.context_size, 32768);
        assert_eq!(plan.ngl, 999);
        assert!(plan.cache_type_k.is_none());
    }

    #[test]
    fn large_model_generates_fallback_ladder() {
        let hw = single_gpu_hardware();
        let model = large_model();
        let config = FitConfig::default();
        let planner = FitPlanner::new(hw, model, config);

        // Should have multiple plans
        assert!(planner.total_plans() > 1);

        // First plan: full context, default KV
        let first = planner.current_plan();
        assert_eq!(first.context_size, 32768);
        assert!(first.cache_type_k.is_none());
    }

    #[test]
    fn dual_gpu_computes_tensor_split() {
        let hw = dual_gpu_hardware();
        let model = small_model();
        let config = FitConfig {
            multi_gpu: "auto".to_string(),
            split_mode: "layer".to_string(),
            ..FitConfig::default()
        };
        let planner = FitPlanner::new(hw, model, config);

        let plan = planner.current_plan();
        assert!(plan.split_mode.is_some());
        assert!(plan.tensor_split.is_some());
        let ts = plan.tensor_split.as_ref().unwrap();
        assert_eq!(ts.len(), 2);
        // Both GPUs have equal free VRAM, so ratio should be 1.0, 1.0
        assert!((ts[0] - 1.0).abs() < 0.01);
        assert!((ts[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn pinned_ngl_is_respected() {
        let hw = single_gpu_hardware();
        let mut model = small_model();
        model.ngl_pinned = true;
        model.pinned_ngl = Some(24);
        let config = FitConfig::default();
        let planner = FitPlanner::new(hw, model, config);

        let plan = planner.current_plan();
        assert_eq!(plan.ngl, 24);
    }

    #[test]
    fn advance_exhausts_plans() {
        let hw = single_gpu_hardware();
        let model = large_model();
        let config = FitConfig {
            max_attempts: 3,
            ..FitConfig::default()
        };
        let mut planner = FitPlanner::new(hw, model, config);
        assert_eq!(planner.total_plans(), 3);

        assert!(planner.advance("OOM").is_some());
        assert!(planner.advance("OOM").is_some());
        assert!(planner.advance("OOM").is_none());
        assert!(planner.is_exhausted());
    }

    #[test]
    fn context_clamped_to_minimum() {
        let hw = single_gpu_hardware();
        let model = ModelSummary {
            requested_context: 8192,
            ..large_model()
        };
        // Default context_steps are [1.0, 0.75, 0.5, 0.25]; 0.25 * 8192 = 2048 < 4096 minimum
        let config = FitConfig::default();
        let planner = FitPlanner::new(hw, model, config);

        // All plans should have context >= 4096 (the minimum)
        for plan in &planner.plans {
            assert!(
                plan.context_size >= 4096,
                "ctx={} < 4096",
                plan.context_size
            );
        }
    }

    #[test]
    fn apply_plan_to_args_rewrites_flags() {
        let base = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "-c".to_string(),
            "32768".to_string(),
            "-ngl".to_string(),
            "999".to_string(),
        ];
        let plan = FitPlan {
            context_size: 16384,
            ngl: 50,
            split_mode: Some("layer".to_string()),
            tensor_split: Some(vec![1.0, 1.0]),
            cache_type_k: Some("q8_0".to_string()),
            cache_type_v: Some("q8_0".to_string()),
            reason: "test".to_string(),
            attempt: 2,
            batch_size: None,
            ubatch_size: None,
        };
        let args = FitPlanner::apply_plan_to_args(&base, &plan);
        let args_str = args.join(" ");
        assert!(args_str.contains("-c 16384"), "args: {args_str}");
        assert!(args_str.contains("-ngl 50"), "args: {args_str}");
        assert!(args_str.contains("--split-mode layer"), "args: {args_str}");
        assert!(
            args_str.contains("--tensor-split 1.0,1.0"),
            "args: {args_str}"
        );
        assert!(args_str.contains("--cache-type-k q8_0"), "args: {args_str}");
        assert!(args_str.contains("--cache-type-v q8_0"), "args: {args_str}");
    }

    #[test]
    fn hardware_fingerprint_single_gpu() {
        let gpus = vec![fake_gpu("NVIDIA GeForce RTX 4090", 24564, 22000)];
        let fp = compute_hardware_fingerprint(&gpus, 24564);
        // Last token of "NVIDIA GeForce RTX 4090" is "4090"
        assert_eq!(fp, "1x4090-24564");
    }

    #[test]
    fn hardware_fingerprint_dual_gpu() {
        let gpus = vec![
            fake_gpu("NVIDIA GeForce RTX 4090", 24564, 22000),
            GpuDeviceInfo {
                index: 1,
                name: "NVIDIA GeForce RTX 4090".to_string(),
                total_mb: 24564,
                free_mb: 22000,
                used_mb: 2564,
            },
        ];
        let fp = compute_hardware_fingerprint(&gpus, 49128);
        assert_eq!(fp, "2x4090-49128");
    }

    #[test]
    fn embedding_model_gets_batch_sizes() {
        let hw = single_gpu_hardware(); // 24564 MB VRAM
        let model = embedding_model();
        let config = FitConfig::default();
        let planner = FitPlanner::new(hw, model, config);

        let plan = planner.current_plan();
        // A small embedding model leaves more than 8 GB headroom.
        assert_eq!(plan.batch_size, Some(4096));
        assert_eq!(plan.ubatch_size, Some(2048));
    }

    #[test]
    fn chat_model_has_no_batch_sizes() {
        let hw = single_gpu_hardware();
        let model = small_model();
        let config = FitConfig::default();
        let planner = FitPlanner::new(hw, model, config);

        let plan = planner.current_plan();
        assert_eq!(plan.batch_size, None);
        assert_eq!(plan.ubatch_size, None);
    }

    #[test]
    fn embedding_batch_scales_with_headroom_after_model_residency() {
        // 4 GB free - 2 GB reserve - 0.5 GB weights leaves about 1.5 GB.
        let hw_small = HardwareSummary {
            gpus: vec![fake_gpu("GTX 1650", 4096, 4096)],
            total_vram_mb: 4096,
            free_vram_mb: 4096,
            system_ram_mb: 16000,
            hardware_fingerprint: "1x1650-4096".to_string(),
        };
        let model = embedding_model();
        let config = FitConfig::default();
        let planner = FitPlanner::new(hw_small, model.clone(), config.clone());
        let plan = planner.current_plan();
        assert_eq!(plan.batch_size, Some(512));
        assert_eq!(plan.ubatch_size, Some(256));

        // 8 GB free - 2 GB reserve - 0.5 GB weights leaves 5.5 GB.
        let hw_medium = HardwareSummary {
            gpus: vec![fake_gpu("RTX 3060", 8192, 8192)],
            total_vram_mb: 8192,
            free_vram_mb: 8192,
            system_ram_mb: 32000,
            hardware_fingerprint: "1x3060-8192".to_string(),
        };
        let planner = FitPlanner::new(hw_medium, model.clone(), config.clone());
        let plan = planner.current_plan();
        assert_eq!(plan.batch_size, Some(2048));
        assert_eq!(plan.ubatch_size, Some(1024));

        // Large VRAM (24 GB) -> 4096 batch
        let hw_large = single_gpu_hardware();
        let planner = FitPlanner::new(hw_large, model, config);
        let plan = planner.current_plan();
        assert_eq!(plan.batch_size, Some(4096));
        assert_eq!(plan.ubatch_size, Some(2048));
    }

    #[test]
    fn twelve_gb_gpu_with_six_gb_embedding_uses_balanced_profile() {
        let hw = HardwareSummary {
            gpus: vec![fake_gpu("RTX 3060", 12288, 12288)],
            total_vram_mb: 12288,
            free_vram_mb: 12288,
            system_ram_mb: 32000,
            hardware_fingerprint: "1x3060-12288".to_string(),
        };
        let model = ModelSummary {
            file_size_mb: 6144,
            requested_context: 8192,
            ..embedding_model()
        };
        let planner = FitPlanner::new(hw, model, FitConfig::default());
        let plan = planner.current_plan();

        assert_eq!(plan.context_size, 8192);
        assert_eq!(plan.batch_size, Some(1024));
        assert_eq!(plan.ubatch_size, Some(512));
    }

    #[test]
    fn constrained_embedding_profile_reduces_context_to_2048() {
        let hw = HardwareSummary {
            gpus: vec![fake_gpu("RTX", 8192, 8192)],
            total_vram_mb: 8192,
            free_vram_mb: 8192,
            system_ram_mb: 32000,
            hardware_fingerprint: "1xRTX-8192".to_string(),
        };
        let model = ModelSummary {
            file_size_mb: 5000,
            requested_context: 8192,
            ..embedding_model()
        };
        let planner = FitPlanner::new(hw, model, FitConfig::default());

        assert_eq!(planner.current_plan().context_size, 2048);
    }

    #[test]
    fn dual_4090_embedding_profile_uses_high_headroom_tier() {
        let hw = dual_gpu_hardware();
        let model = ModelSummary {
            file_size_mb: 6144,
            requested_context: 16384,
            ..embedding_model()
        };
        let planner = FitPlanner::new(hw, model, FitConfig::default());
        let plan = planner.current_plan();

        assert_eq!(plan.context_size, 16384);
        assert_eq!(plan.batch_size, Some(4096));
        assert_eq!(plan.ubatch_size, Some(2048));
        assert!(plan.tensor_split.is_some());
    }

    #[test]
    fn embedding_batch_fallback_ladder_descends() {
        let hw = single_gpu_hardware(); // 24 GB -> base 4096
        let model = embedding_model();
        let config = FitConfig {
            max_attempts: 4,
            ..FitConfig::default()
        };
        let planner = FitPlanner::new(hw, model, config);

        let plans = planner.all_plans();
        // First plan should have highest batch size
        let first_batch = plans[0].batch_size.unwrap();
        assert_eq!(first_batch, 4096);

        // Later plans should have smaller batch sizes
        for plan in &plans[1..] {
            let batch = plan.batch_size.unwrap();
            assert!(batch <= first_batch, "batch {batch} > first {first_batch}");
        }
    }

    #[test]
    fn embedding_batch_minimum_is_256() {
        // Very small VRAM (1 GB) -> 512 base, 25% = 128 clamped to 256
        let hw_tiny = HardwareSummary {
            gpus: vec![fake_gpu("GT 730", 1024, 800)],
            total_vram_mb: 1024,
            free_vram_mb: 800,
            system_ram_mb: 8000,
            hardware_fingerprint: "1x730-1024".to_string(),
        };
        let model = embedding_model();
        let config = FitConfig::default();
        let planner = FitPlanner::new(hw_tiny, model, config);

        for plan in planner.all_plans() {
            let batch = plan.batch_size.unwrap();
            assert!(batch >= 256, "batch {batch} < 256 minimum");
        }
    }

    #[test]
    fn apply_plan_writes_batch_flags() {
        let base = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "-c".to_string(),
            "8192".to_string(),
            "-ngl".to_string(),
            "999".to_string(),
        ];
        let plan = FitPlan {
            context_size: 8192,
            ngl: 999,
            split_mode: None,
            tensor_split: None,
            cache_type_k: None,
            cache_type_v: None,
            reason: "test".to_string(),
            attempt: 1,
            batch_size: Some(2048),
            ubatch_size: Some(2048),
        };
        let args = FitPlanner::apply_plan_to_args(&base, &plan);
        let args_str = args.join(" ");
        assert!(args_str.contains("-b 2048"), "args: {args_str}");
        assert!(args_str.contains("-ub 2048"), "args: {args_str}");
    }

    #[test]
    fn apply_plan_omits_batch_for_chat() {
        let base = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "-c".to_string(),
            "32768".to_string(),
        ];
        let plan = FitPlan {
            context_size: 32768,
            ngl: 999,
            split_mode: None,
            tensor_split: None,
            cache_type_k: None,
            cache_type_v: None,
            reason: "test".to_string(),
            attempt: 1,
            batch_size: None,
            ubatch_size: None,
        };
        let args = FitPlanner::apply_plan_to_args(&base, &plan);
        let args_str = args.join(" ");
        assert!(!args_str.contains("-b "), "args: {args_str}");
        assert!(!args_str.contains("-ub "), "args: {args_str}");
    }
}
