//! Helpers for reading/adjusting llama-server GPU layer count (`-ngl`) and the auto_ngl heuristic.

use tracing::{info, warn};

use crate::memory;

const NGL_FLAGS: &[&str] = &["-ngl", "--n-gpu-layers", "--gpu-layers"];

/// Fraction of free VRAM reserved for KV cache / overhead.
/// ponytail: ~20% ceiling; upgrade to per-layer tensor byte estimates when needed.
const VRAM_USABLE_FRACTION: f64 = 0.8;

/// Extract `-m` / `--model` path from backend args.
pub fn model_path_from_args(args: &[String]) -> Option<&str> {
    for i in 0..args.len() {
        if (args[i] == "-m" || args[i] == "--model") && i + 1 < args.len() {
            return Some(args[i + 1].as_str());
        }
    }
    None
}

/// Read the configured GPU layer count from backend args, if present.
pub fn get_ngl(args: &[String]) -> Option<u32> {
    let (_, value_idx) = ngl_value_index(args)?;
    args.get(value_idx)?.parse().ok()
}

/// Return a copy of `args` with the `-ngl` value set to `ngl`.
pub fn with_ngl(args: &[String], ngl: u32) -> Vec<String> {
    let mut updated = args.to_vec();
    if let Some((_, value_idx)) = ngl_value_index(&updated) {
        updated[value_idx] = ngl.to_string();
        return updated;
    }

    updated.push("-ngl".to_string());
    updated.push(ngl.to_string());
    updated
}

fn ngl_value_index(args: &[String]) -> Option<(usize, usize)> {
    for (idx, arg) in args.iter().enumerate() {
        if NGL_FLAGS.contains(&arg.as_str()) && idx + 1 < args.len() {
            return Some((idx, idx + 1));
        }
    }
    None
}

// ── Split mode helpers ───────────────────────────────────────────────────────

const SPLIT_MODE_FLAG: &str = "--split-mode";

pub fn get_split_mode(args: &[String]) -> Option<&str> {
    flag_value(args, SPLIT_MODE_FLAG)
}

pub fn with_split_mode(args: &[String], mode: &str) -> Vec<String> {
    set_flag_value(args, SPLIT_MODE_FLAG, mode)
}

// ── Tensor split helpers ─────────────────────────────────────────────────────

const TENSOR_SPLIT_FLAG: &str = "--tensor-split";

pub fn get_tensor_split(args: &[String]) -> Option<&str> {
    flag_value(args, TENSOR_SPLIT_FLAG)
}

pub fn with_tensor_split(args: &[String], split: &str) -> Vec<String> {
    set_flag_value(args, TENSOR_SPLIT_FLAG, split)
}

// ── KV cache type helpers ────────────────────────────────────────────────────

const CACHE_TYPE_K_FLAG: &str = "--cache-type-k";
const CACHE_TYPE_V_FLAG: &str = "--cache-type-v";

pub fn get_cache_type_k(args: &[String]) -> Option<&str> {
    flag_value(args, CACHE_TYPE_K_FLAG)
}

pub fn with_cache_type_k(args: &[String], t: &str) -> Vec<String> {
    set_flag_value(args, CACHE_TYPE_K_FLAG, t)
}

pub fn get_cache_type_v(args: &[String]) -> Option<&str> {
    flag_value(args, CACHE_TYPE_V_FLAG)
}

pub fn with_cache_type_v(args: &[String], t: &str) -> Vec<String> {
    set_flag_value(args, CACHE_TYPE_V_FLAG, t)
}

// ── Generic flag helpers ─────────────────────────────────────────────────────

/// Read the value of `--flag` from args.
fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(&args[i + 1]);
        }
    }
    None
}

/// Return a copy of `args` with `--flag value` set (replaced or appended).
fn set_flag_value(args: &[String], flag: &str, value: &str) -> Vec<String> {
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

/// Inputs for the auto_ngl heuristic.
#[derive(Debug, Clone, Copy)]
pub struct AutoNglInput {
    pub free_vram_mb: u64,
    pub file_size_mb: u64,
    pub block_count: Option<u32>,
    pub available_ram_mb: Option<u64>,
}

/// Compute a suggested `-ngl` from free VRAM and GGUF size.
pub fn suggest_ngl(input: AutoNglInput) -> u32 {
    let usable_vram_mb = ((input.free_vram_mb as f64) * VRAM_USABLE_FRACTION).floor() as u64;
    let usable_vram_mb = usable_vram_mb.max(1);

    let layers = input.block_count.unwrap_or(999).max(1);

    let ngl = if input.file_size_mb <= usable_vram_mb {
        layers
    } else if input.block_count.is_none() {
        // No layer count: all-or-nothing.
        0
    } else {
        let n = (u64::from(layers).saturating_mul(usable_vram_mb)) / input.file_size_mb.max(1);
        u32::try_from(n).unwrap_or(layers).min(layers)
    };

    if let Some(available_ram_mb) = input.available_ram_mb {
        let overflow_mb = input.file_size_mb.saturating_sub(usable_vram_mb);
        let ram_budget = available_ram_mb.saturating_mul(80) / 100;
        if overflow_mb > ram_budget {
            warn!(
                overflow_mb,
                available_ram_mb,
                ngl,
                "auto_ngl: CPU offload may exceed available RAM; keeping VRAM-based ngl"
            );
        }
    }

    ngl
}

/// Resolve free VRAM: live nvidia-smi, else `vram_gb * 1024`.
pub fn resolve_free_vram_mb(vram_gb: u32) -> u64 {
    crate::gpu::free_vram_mb().unwrap_or_else(|| u64::from(vram_gb.max(1)).saturating_mul(1024))
}

/// Build auto_ngl input for a model file and apply `suggest_ngl`, logging the choice.
pub fn compute_auto_ngl(
    model_id: &str,
    model_path: &str,
    vram_gb: u32,
    block_count: Option<u32>,
) -> Option<u32> {
    let meta = std::fs::metadata(model_path).ok()?;
    let file_size_mb = meta.len() / (1024 * 1024);
    let free_vram_mb = resolve_free_vram_mb(vram_gb);
    let available_ram_mb = memory::check_memory().map(|s| s.available_mb);

    let input = AutoNglInput {
        free_vram_mb,
        file_size_mb,
        block_count,
        available_ram_mb,
    };
    let ngl = suggest_ngl(input);
    info!(
        model = %model_id,
        ngl,
        free_vram_mb,
        file_size_mb,
        block_count,
        ?available_ram_mb,
        "auto_ngl: selected GPU layers"
    );
    Some(ngl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_and_with_ngl() {
        let args = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "-ngl".to_string(),
            "999".to_string(),
        ];
        assert_eq!(get_ngl(&args), Some(999));
        let updated = with_ngl(&args, 24);
        assert_eq!(get_ngl(&updated), Some(24));
    }

    #[test]
    fn with_ngl_appends_when_missing() {
        let args = vec!["-m".to_string(), "model.gguf".to_string()];
        let updated = with_ngl(&args, 12);
        assert_eq!(get_ngl(&updated), Some(12));
    }

    #[test]
    fn suggest_ngl_fits_all_layers() {
        let ngl = suggest_ngl(AutoNglInput {
            free_vram_mb: 12_000,
            file_size_mb: 5_000,
            block_count: Some(32),
            available_ram_mb: Some(32_000),
        });
        // usable = 9600 > 5000 → all layers
        assert_eq!(ngl, 32);
    }

    #[test]
    fn suggest_ngl_half_vram() {
        let ngl = suggest_ngl(AutoNglInput {
            free_vram_mb: 10_000,
            file_size_mb: 16_000,
            block_count: Some(40),
            available_ram_mb: Some(64_000),
        });
        // usable = 8000; 40 * 8000 / 16000 = 20
        assert_eq!(ngl, 20);
    }

    #[test]
    fn suggest_ngl_no_block_count_all_or_nothing() {
        assert_eq!(
            suggest_ngl(AutoNglInput {
                free_vram_mb: 12_000,
                file_size_mb: 5_000,
                block_count: None,
                available_ram_mb: None,
            }),
            999
        );
        assert_eq!(
            suggest_ngl(AutoNglInput {
                free_vram_mb: 4_000,
                file_size_mb: 10_000,
                block_count: None,
                available_ram_mb: None,
            }),
            0
        );
    }

    #[test]
    fn split_mode_get_and_set() {
        let args = vec![
            "-m".to_string(),
            "model.gguf".to_string(),
            "--split-mode".to_string(),
            "row".to_string(),
        ];
        assert_eq!(get_split_mode(&args), Some("row"));
        let updated = with_split_mode(&args, "layer");
        assert_eq!(get_split_mode(&updated), Some("layer"));
    }

    #[test]
    fn split_mode_append_when_missing() {
        let args = vec!["-m".to_string(), "model.gguf".to_string()];
        let updated = with_split_mode(&args, "layer");
        assert_eq!(get_split_mode(&updated), Some("layer"));
    }

    #[test]
    fn tensor_split_get_and_set() {
        let args = vec!["--tensor-split".to_string(), "1,1".to_string()];
        assert_eq!(get_tensor_split(&args), Some("1,1"));
        let updated = with_tensor_split(&args, "2,1");
        assert_eq!(get_tensor_split(&updated), Some("2,1"));
    }

    #[test]
    fn cache_type_k_get_and_set() {
        let args = vec!["--cache-type-k".to_string(), "f16".to_string()];
        assert_eq!(get_cache_type_k(&args), Some("f16"));
        let updated = with_cache_type_k(&args, "q8_0");
        assert_eq!(get_cache_type_k(&updated), Some("q8_0"));
    }

    #[test]
    fn cache_type_v_get_and_set() {
        let args = vec!["--cache-type-v".to_string(), "f16".to_string()];
        assert_eq!(get_cache_type_v(&args), Some("f16"));
        let updated = with_cache_type_v(&args, "q4_0");
        assert_eq!(get_cache_type_v(&updated), Some("q4_0"));
    }
}
