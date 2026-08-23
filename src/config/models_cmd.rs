//! CLI handlers for `gguf-switchboard models {search,files,pull}`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};

use futures::{StreamExt, stream};
use serde_json::Value;

use super::hf_download::{self, HfTreeEntry, format_bytes};
use super::models_registry::{
    ModelsRegistry, RegistryEntry, alias_from_filename, validate_gguf_model,
};
use crate::errors::RuntimeError;
use crate::quant_profile::{self, HardwareCtx};

const HF_MODELS_API: &str = "https://huggingface.co/api/models";

// ── models search ────────────────────────────────────────────────────────────

/// `gguf-switchboard models search <query> [--limit N] [--ram-bandwidth-gbps N]`
/// `gguf-switchboard models search vllm <query> [--limit N]`
pub async fn cmd_search(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.get(1).is_some_and(|a| a == "vllm") {
        return cmd_search_vllm(&args[1..]).await;
    }

    let mut query: Option<String> = None;
    let mut limit: u32 = 10;
    let mut ram_bandwidth_override: Option<f64> = None;

    let mut i = 1; // args[0] is "search"
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                if let Some(val) = args.get(i + 1) {
                    limit = val
                        .parse()
                        .map_err(|_| "models search: invalid value for --limit")?;
                    i += 2;
                } else {
                    return Err("models search: missing value for --limit".into());
                }
            }
            "--ram-bandwidth-gbps" => {
                if let Some(val) = args.get(i + 1) {
                    ram_bandwidth_override = Some(
                        val.parse()
                            .map_err(|_| "models search: invalid value for --ram-bandwidth-gbps")?,
                    );
                    i += 2;
                } else {
                    return Err("models search: missing value for --ram-bandwidth-gbps".into());
                }
            }
            arg if arg.starts_with('-') => {
                return Err(format!("models search: unknown flag '{arg}'").into());
            }
            val => {
                query = Some(val.to_string());
                i += 1;
            }
        }
    }

    let query = query.ok_or("models search: missing search query")?;

    let client = hf_download::build_hf_client()?;
    let hits = search_hf_models_with_siblings(&client, &query, limit).await?;

    if hits.is_empty() {
        println!("No GGUF models found for \"{query}\"");
        return Ok(());
    }

    let total_ram_mb = crate::memory::check_memory()
        .map(|stats| stats.total_mb)
        .unwrap_or(0);
    let total_vram_mb = crate::gpu::total_vram_mb().unwrap_or(0);
    let capacity_bytes = total_ram_mb
        .saturating_add(total_vram_mb)
        .saturating_mul(1024 * 1024);
    let hw = HardwareCtx::detect(ram_bandwidth_override);

    let estimates = stream::iter(hits.iter().enumerate().map(|(index, hit)| {
        let client = client.clone();
        let hw = hw.clone();
        let repo = hit.get("id").and_then(Value::as_str).map(str::to_string);
        async move {
            let assessment = match repo {
                Some(repo) => match hf_download::fetch_repo_tree(&client, &repo).await {
                    Ok(entries) => assess_repository(hit, &entries, capacity_bytes, &hw),
                    Err(_) => SearchAssessment::default(),
                },
                None => SearchAssessment::default(),
            };
            (index, assessment)
        }
    }))
    .buffer_unordered(4)
    .collect::<Vec<_>>()
    .await;

    let mut assessments = vec![SearchAssessment::default(); hits.len()];
    for (index, assessment) in estimates {
        assessments[index] = assessment;
    }

    println!("{}", format_hardware_summary(total_ram_mb, total_vram_mb));
    if let Some(gpu) = hw.gpus.first() {
        let (bandwidth, confidence) = quant_profile::gpu_bandwidth_gbps(&gpu.name);
        println!(
            "Speed model inputs: GPU bandwidth {:.0} GB/s ({}{}) | RAM bandwidth {:.0} GB/s (assumed) | GPU efficiency 0.55 | CPU efficiency 0.35",
            bandwidth,
            gpu.name,
            if confidence == quant_profile::Confidence::Extrapolated {
                ", unrecognized model — fallback estimate"
            } else {
                ""
            },
            hw.ram_bandwidth_gbps,
        );
    } else {
        println!(
            "Speed model inputs: no NVIDIA GPU detected — CPU-only estimate at {:.0} GB/s RAM bandwidth (assumed) | CPU efficiency 0.35",
            hw.ram_bandwidth_gbps
        );
    }
    println!();
    print!("{}", render_search_table(&hits, &assessments));
    println!(
        "FIT: 0-100 memory-fit score (100 = comfortable headroom; 0 = does not fit RAM+VRAM). \
         SPEED/PRECISION: the quant that maximizes each — tok/s from a memory-bandwidth model \
         (verify against `llama-bench` on your machine), quality % from published per-quant \
         perplexity measurements (\"~\" = extrapolated, not directly measured for this \
         architecture). BALANCED: the quant with the best average of speed and quality, both \
         normalized to this model's own quant options — a middle ground when you don't want \
         either extreme. See docs/QUANT_SCORING.md for methodology and sources; override RAM \
         bandwidth with --ram-bandwidth-gbps if you've measured your own."
    );
    if let Some(command) = sample_pull_command(&hits, &assessments) {
        println!("{command}");
    }

    // vLLM preference: when a safetensors build of the same query exists,
    // recommend it over GGUF/llama.cpp — vLLM generally has better throughput
    // and native speculative-decoding/quantization support on GPU hardware.
    if let Some(vllm_repo) = quick_check_vllm_alternative(&client, &query).await {
        println!();
        println!(
            "★ A vLLM-servable (safetensors) build is also available: {vllm_repo}\n  \
             vLLM is preferred when both exist — see: ggs models search vllm \"{query}\""
        );
    }

    print!("{}", search_commands_help());
    Ok(())
}

/// Lightweight check (one HF API call, no per-repo tree fetch) for whether a
/// safetensors build of `query` exists, to nudge users toward the preferred
/// vLLM backend from the plain (GGUF) search path.
async fn quick_check_vllm_alternative(client: &reqwest::Client, query: &str) -> Option<String> {
    let url = reqwest::Url::parse_with_params(
        HF_MODELS_API,
        &[("search", query), ("limit", "5"), ("expand", "siblings")],
    )
    .ok()?;
    let resp = client.get(url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let hits: Vec<Value> = resp.json().await.ok()?;
    hits.into_iter().find_map(|hit| {
        let siblings = hit.get("siblings")?.as_array()?;
        let has_safetensors = siblings.iter().any(|s| {
            s.get("rfilename")
                .and_then(Value::as_str)
                .is_some_and(|f| f.ends_with(".safetensors"))
        });
        has_safetensors
            .then(|| hit.get("id").and_then(Value::as_str).map(str::to_string))
            .flatten()
    })
}

/// Shared "what commands exist" footer for both search modes.
fn search_commands_help() -> String {
    "\nCommands:\n  \
     ggs models search <query>                 GGUF/llama.cpp models\n  \
     ggs models search vllm <query>            safetensors/vLLM models (preferred when available)\n  \
     ggs models pull <repo> [--quant Q]         download + register a GGUF model\n  \
     ggs models pull vllm <repo> [--draft ...]  download + register a vLLM model\n  \
     ggs models files <repo>                    list a repo's files\n"
        .to_string()
}

/// `gguf-switchboard models search vllm <query> [--limit N]` — searches for
/// safetensors repos vLLM can serve directly, showing quantization and
/// speculative-decoding (dspark draft model) pairing instead of GGUF quants.
async fn cmd_search_vllm(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut query: Option<String> = None;
    // Higher than GGUF search's default: HF's relevance ranking for a bare
    // query tends to surface official full-precision repos first, so a small
    // limit can miss the quantized (AWQ/GPTQ/FP8) variants most likely to fit
    // a consumer GPU. Cast a wider net before the fit filter narrows it down.
    let mut limit: u32 = 30;

    let mut i = 1; // args[0] is "vllm"
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                if let Some(val) = args.get(i + 1) {
                    limit = val
                        .parse()
                        .map_err(|_| "models search vllm: invalid value for --limit")?;
                    i += 2;
                } else {
                    return Err("models search vllm: missing value for --limit".into());
                }
            }
            arg if arg.starts_with('-') => {
                return Err(format!("models search vllm: unknown flag '{arg}'").into());
            }
            val => {
                query = Some(val.to_string());
                i += 1;
            }
        }
    }
    let query = query.ok_or("models search vllm: missing search query")?;

    let client = hf_download::build_hf_client()?;
    let url = reqwest::Url::parse_with_params(
        HF_MODELS_API,
        &[
            ("search", query.as_str()),
            ("limit", &limit.to_string()),
            ("expand", "siblings"),
            ("expand", "tags"),
            ("expand", "pipeline_tag"),
        ],
    )
    .map_err(|e| RuntimeError::InternalError(e.to_string()))?;
    let resp = client.get(url).send().await.map_err(RuntimeError::from)?;
    if !resp.status().is_success() {
        return Err(format!("HF API search failed: HTTP {}", resp.status()).into());
    }
    let hits: Vec<Value> = resp.json().await.map_err(RuntimeError::from)?;

    // Keep only repos that actually look like servable safetensors models
    // (has weights + config.json), fetching each repo's tree to check.
    let candidates = stream::iter(hits.into_iter().map(|hit| {
        let client = client.clone();
        async move {
            let repo = hit.get("id").and_then(Value::as_str)?.to_string();
            let entries = hf_download::fetch_repo_tree_all(&client, &repo)
                .await
                .ok()?;
            if !super::vllm_meta::is_safetensors_repo(&entries) {
                return None;
            }
            let meta = super::vllm_meta::detect_vllm_metadata(&client, &repo)
                .await
                .unwrap_or_default();
            let size_bytes: u64 = entries
                .iter()
                .filter(|e| e.path.ends_with(".safetensors"))
                .map(|e| e.size)
                .sum();
            Some((repo, meta, size_bytes))
        }
    }))
    .buffer_unordered(4)
    .filter_map(|item| async move { item })
    .collect::<Vec<_>>()
    .await;

    if candidates.is_empty() {
        println!("No vLLM-servable (safetensors) models found for \"{query}\"");
        return Ok(());
    }

    // Fit gate: flag (don't hide) models this hardware cannot possibly hold
    // (vLLM needs weights fully GPU-resident — no CPU offload like llama.cpp).
    // Showing every result with a FITS column beats silently dropping rows —
    // a query that finds nothing but full-precision repos should say so, not
    // look identical to a query that found nothing at all.
    let hardware = crate::fit::HardwareSummary::probe(12);
    let fits = |size_bytes: u64| {
        crate::fit::hardware_can_possibly_serve(size_bytes / (1024 * 1024), &hardware, false)
    };
    let any_fits = candidates
        .iter()
        .any(|(_, _, size_bytes)| fits(*size_bytes));

    if !any_fits {
        println!(
            "Found {} safetensors repo(s) for \"{query}\", but none fit this hardware's \
             {:.1} GiB total VRAM (vLLM requires GPU-resident weights) — listed below anyway \
             so you can see what was found. Try a more specific query for a quantized variant \
             (e.g. `ggs models search vllm \"{query} awq\"` or `\"{query} gptq\"`), a higher \
             `--limit` (currently {limit}), or `ggs models search {query}` for GGUF instead.",
            candidates.len(),
            hardware.total_vram_mb as f64 / 1024.0
        );
    }

    println!(
        "{:<48} | {:<10} | {:<5} | {:<10} | {:<10} | {:<24} | DRAFT MODEL (speculative decoding)",
        "REPO", "SIZE", "FITS", "QUANT", "MAX CTX", "ARCHITECTURE"
    );
    for (repo, meta, size_bytes) in &candidates {
        let size = if *size_bytes > 0 {
            format_bytes(*size_bytes)
        } else {
            "-".to_string()
        };
        let fits_str = if *size_bytes == 0 {
            "?"
        } else if fits(*size_bytes) {
            "yes"
        } else {
            "no"
        };
        let quant = meta.quantization.as_deref().unwrap_or("none (fp16/bf16)");
        let ctx = meta
            .max_position_embeddings
            .map(|c| c.to_string())
            .unwrap_or_else(|| "-".to_string());
        let arch = meta.architecture.as_deref().unwrap_or("-");
        let draft = meta.draft_model.as_deref().unwrap_or("-");
        println!(
            "{repo:<48} | {size:<10} | {fits_str:<5} | {quant:<10} | {ctx:<10} | {arch:<24} | {draft}"
        );
    }

    let Some((first_fitting, ..)) = candidates
        .iter()
        .find(|(_, _, size_bytes)| fits(*size_bytes))
    else {
        print!("{}", search_commands_help());
        return Ok(());
    };
    println!(
        "\nTry: ggs models pull vllm {first_fitting}   (pulls safetensors + config, writes models.toml with backend = \"vllm\")"
    );
    print!("{}", search_commands_help());

    Ok(())
}

/// Search HF models API with siblings and gguf expansion.
async fn search_hf_models_with_siblings(
    client: &reqwest::Client,
    query: &str,
    limit: u32,
) -> Result<Vec<Value>, RuntimeError> {
    let url = reqwest::Url::parse_with_params(
        HF_MODELS_API,
        &[
            ("search", query.to_string()),
            ("filter", "gguf".to_string()),
            ("limit", limit.to_string()),
            ("expand", "siblings".to_string()),
            ("expand", "gguf".to_string()),
        ],
    )
    .map_err(|e| RuntimeError::InternalError(e.to_string()))?;

    let resp = client.get(url).send().await.map_err(RuntimeError::from)?;
    if !resp.status().is_success() {
        return Err(RuntimeError::ProxyError(format!(
            "HF API search failed: HTTP {}",
            resp.status()
        )));
    }
    let hits: Vec<Value> = resp.json().await.map_err(RuntimeError::from)?;
    Ok(hits)
}

fn format_hardware_summary(total_ram_mb: u64, total_vram_mb: u64) -> String {
    let total_mb = total_ram_mb.saturating_add(total_vram_mb);
    format!(
        "Hardware: System RAM {:.1} GiB | NVIDIA VRAM {:.1} GiB | Total {:.1} GiB",
        total_ram_mb as f64 / 1024.0,
        total_vram_mb as f64 / 1024.0,
        total_mb as f64 / 1024.0
    )
}

fn sample_pull_command(hits: &[Value], assessments: &[SearchAssessment]) -> Option<String> {
    hits.iter().zip(assessments).find_map(|(hit, assessment)| {
        let repository = hit.get("id").and_then(Value::as_str)?;
        let fastest = assessment.fastest.as_ref()?;
        let mut shown_quants = vec![fastest.quant.as_str()];
        let mut lines = vec![format!(
            "Try: ggs models pull {repository} --quant {}   (fastest, ~{:.0} tok/s est.)",
            fastest.quant, fastest.tokens_per_sec
        )];
        if let Some(balanced) = assessment.balanced.as_ref()
            && !shown_quants.contains(&balanced.quant.as_str())
        {
            shown_quants.push(&balanced.quant);
            lines.push(format!(
                "     ggs models pull {repository} --quant {}   (balanced, ~{:.0} tok/s / ~{:.1}% quality est.)",
                balanced.quant, balanced.tokens_per_sec, balanced.quality_score
            ));
        }
        if let Some(precision) = assessment.best_precision.as_ref()
            && !shown_quants.contains(&precision.quant.as_str())
        {
            lines.push(format!(
                "     ggs models pull {repository} --quant {}   (least precision loss, ~{:.1}% quality est.)",
                precision.quant, precision.quality_score
            ));
        }
        Some(lines.join("\n"))
    })
}

fn render_search_table(hits: &[Value], assessments: &[SearchAssessment]) -> String {
    let rows = hits
        .iter()
        .enumerate()
        .map(|(index, hit)| {
            let id = hit
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string();
            let siblings = hit
                .get("siblings")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter(|s| {
                            s.get("rfilename")
                                .and_then(|v| v.as_str())
                                .is_some_and(|f| f.ends_with(".gguf"))
                        })
                        .count()
                })
                .unwrap_or(0)
                .to_string();
            let size_mb = hit
                .get("gguf")
                .and_then(|g| g.get("total"))
                .and_then(|v| v.as_u64())
                .map(format_mb)
                .unwrap_or_default();
            let context = hit
                .get("gguf")
                .and_then(|g| g.get("context_length"))
                .and_then(|v| v.as_u64())
                .map(|v| format!("{v} tok"))
                .unwrap_or_default();
            let arch = hit
                .get("gguf")
                .and_then(|g| g.get("architecture"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let assessment = assessments.get(index);
            let fit = assessment
                .map(|value| format!("{:.0}", value.fit_score))
                .unwrap_or_else(|| "-".to_string());
            let speed = assessment
                .and_then(|value| value.fastest.as_ref())
                .map(|f| format!("{} ~{:.0}tok/s", f.quant, f.tokens_per_sec))
                .unwrap_or_else(|| "-".to_string());
            let balanced = assessment
                .and_then(|value| value.balanced.as_ref())
                .map(|b| {
                    format!(
                        "{} ~{:.0}tok/s/~{:.1}%",
                        b.quant, b.tokens_per_sec, b.quality_score
                    )
                })
                .unwrap_or_else(|| "-".to_string());
            let precision = assessment
                .and_then(|value| value.best_precision.as_ref())
                .map(|p| format!("{} ~{:.1}%", p.quant, p.quality_score))
                .unwrap_or_else(|| "-".to_string());
            let quants = assessment
                .filter(|value| !value.quants.is_empty())
                .map(|value| {
                    value
                        .quants
                        .iter()
                        .map(|option| option.quant.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_else(|| "-".to_string());
            (
                id, siblings, size_mb, fit, context, arch, speed, balanced, precision, quants,
            )
        })
        .collect::<Vec<_>>();

    let width = |heading: &str, column: usize| {
        rows.iter().fold(heading.len(), |width, row| {
            let value = match column {
                0 => &row.0,
                1 => &row.1,
                2 => &row.2,
                3 => &row.3,
                4 => &row.4,
                5 => &row.5,
                6 => &row.6,
                7 => &row.7,
                _ => &row.8,
            };
            width.max(value.len())
        })
    };
    let repo_width = width("REPO", 0);
    let files_width = width("FILES", 1);
    let size_width = width("SIZE", 2);
    let fit_width = width("FIT", 3);
    let context_width = width("CONTEXT", 4);
    let arch_width = width("ARCH", 5);
    let speed_width = width("SPEED", 6);
    let balanced_width = width("BALANCED", 7);
    let precision_width = width("PRECISION", 8);

    let mut output = String::new();
    writeln!(
        output,
        "{:<repo_width$} | {:>files_width$} | {:>size_width$} | {:>fit_width$} | {:<context_width$} | {:<arch_width$} | {:<speed_width$} | {:<balanced_width$} | {:<precision_width$} | QUANT",
        "REPO", "FILES", "SIZE", "FIT", "CONTEXT", "ARCH", "SPEED", "BALANCED", "PRECISION"
    )
    .expect("writing to a String cannot fail");
    for (repo, files, size, fit, context, arch, speed, balanced, precision, quants) in rows {
        writeln!(
            output,
            "{repo:<repo_width$} | {files:>files_width$} | {size:>size_width$} | {fit:>fit_width$} | {context:<context_width$} | {arch:<arch_width$} | {speed:<speed_width$} | {balanced:<balanced_width$} | {precision:<precision_width$} | {quants}"
        )
        .expect("writing to a String cannot fail");
    }

    output
}

// ── models files ─────────────────────────────────────────────────────────────

/// `gguf-switchboard models files <repo-id>`
pub async fn cmd_files(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let repo = args
        .get(1)
        .ok_or("models files: missing repository id (e.g. bartowski/Qwen3.5-9B-GGUF)")?;

    let client = hf_download::build_hf_client()?;
    let entries = hf_download::fetch_repo_tree(&client, repo).await?;

    if entries.is_empty() {
        println!("No .gguf files found in {repo}");
        return Ok(());
    }

    // Split main models from mmproj / projector files.
    let (models, mmproj): (Vec<_>, Vec<_>) = entries.iter().partition(|e| is_model_gguf(&e.path));

    if !models.is_empty() {
        let hw = HardwareCtx::detect(None);
        println!(
            "  {:<44} {:<12} {:<10} {:<10} QUANT",
            "FILENAME", "SIZE", "SPEED", "PRECISION"
        );
        for entry in &models {
            let size = format_bytes(entry.size);
            let quant = extract_quant(&entry.path);
            let speed = quant_profile::estimate_speed(entry.size, &hw);
            let quality = quant_profile::quant_quality(&quant);
            println!(
                "  {:<44} {:<12} {:<10} {:<10} {}",
                entry.path,
                size,
                format!("~{:.0}tok/s", speed.tokens_per_sec),
                format!("~{:.1}%", quality.quality_score),
                quant
            );
        }
    }

    if !mmproj.is_empty() {
        if !models.is_empty() {
            println!();
        }
        println!("  Multimodal projectors:");
        for entry in &mmproj {
            let size = format_bytes(entry.size);
            println!("  {:<44} {}", entry.path, size);
        }
    }

    Ok(())
}

// ── models pull ──────────────────────────────────────────────────────────────

/// `gguf-switchboard models pull <repo-id> [--quant Q4_K_M] [--dir /path] [--no-bench]`
/// `gguf-switchboard models pull vllm <repo-id> [--dir /path] [--draft <repo>] ...`
pub async fn cmd_pull(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.get(1).is_some_and(|a| a == "vllm") {
        return cmd_pull_vllm(&args[1..]).await;
    }

    let mut repo: Option<String> = None;
    let mut quant: Option<String> = None;
    let mut dest_override: Option<String> = None;
    let mut models_file: Option<String> = None;
    let mut connections: u16 = 8;
    let mut no_bench = false;
    let mut fit_dry_run = false;

    let mut i = 1; // args[0] is "pull"
    while i < args.len() {
        match args[i].as_str() {
            "--quant" => {
                if let Some(val) = args.get(i + 1) {
                    quant = Some(val.to_uppercase());
                    i += 2;
                } else {
                    return Err("models pull: missing value for --quant".into());
                }
            }
            "--dir" => {
                if let Some(val) = args.get(i + 1) {
                    dest_override = Some(val.clone());
                    i += 2;
                } else {
                    return Err("models pull: missing value for --dir".into());
                }
            }
            "--registry" => {
                if let Some(val) = args.get(i + 1) {
                    models_file = Some(val.clone());
                    i += 2;
                } else {
                    return Err("models pull: missing value for --registry".into());
                }
            }
            "--connections" => {
                if let Some(val) = args.get(i + 1) {
                    connections = parse_connections(val)?;
                    i += 2;
                } else {
                    return Err("models pull: missing value for --connections".into());
                }
            }
            "--no-bench" => {
                no_bench = true;
                i += 1;
            }
            "--fit-dry-run" => {
                fit_dry_run = true;
                i += 1;
            }
            arg if arg.starts_with('-') => {
                return Err(format!("models pull: unknown flag '{arg}'").into());
            }
            val => {
                repo = Some(val.to_string());
                i += 1;
            }
        }
    }

    let repo = repo.ok_or("models pull: missing repository id (e.g. bartowski/Qwen3.5-9B-GGUF)")?;

    // 1. Fetch repo tree.
    let client = hf_download::build_hf_client()?;

    // vLLM preference: this exact repo may ship both GGUF and safetensors
    // (some repos bundle both formats). If so, nudge toward vLLM before
    // committing to a GGUF download — vLLM is preferred when both exist.
    if let Ok(all_entries) = hf_download::fetch_repo_tree_all(&client, &repo).await
        && super::vllm_meta::is_safetensors_repo(&all_entries)
    {
        println!(
            "★ {repo} also ships safetensors weights — vLLM is preferred when available.\n  \
             Run instead: ggs models pull vllm {repo}\n  \
             Continuing with the GGUF/llama.cpp pull you requested..."
        );
    }

    let entries = hf_download::fetch_repo_tree(&client, &repo).await?;
    let model_entries: Vec<&HfTreeEntry> =
        entries.iter().filter(|e| is_model_gguf(&e.path)).collect();

    if model_entries.is_empty() {
        return Err(format!("No .gguf model files found in {repo}").into());
    }

    // 2. Select file by quant.
    let selected = match &quant {
        Some(q) => {
            let capacity_bytes = if q.eq_ignore_ascii_case("auto") {
                let total_ram_mb = crate::memory::check_memory()
                    .ok_or("models pull: could not detect system RAM for --quant auto")?
                    .total_mb;
                total_ram_mb
                    .saturating_add(crate::gpu::total_vram_mb().unwrap_or(0))
                    .saturating_mul(1024 * 1024)
            } else {
                0
            };
            match select_quant_entry(&model_entries, q, capacity_bytes, &repo) {
                Ok(entry) => entry,
                Err(reason) => {
                    println!("Could not resolve quant \"{q}\": {reason}. Available:");
                    for entry in &model_entries {
                        println!("  {} ({})", entry.path, extract_quant(&entry.path));
                    }
                    return Err("models pull: no matching quantization".into());
                }
            }
        }
        None => {
            if model_entries.len() == 1 {
                model_entries[0]
            } else {
                println!("Multiple .gguf files found; specify --quant:");
                for entry in &model_entries {
                    println!("  {} ({})", entry.path, extract_quant(&entry.path));
                }
                return Err("models pull: --quant is required when multiple files exist".into());
            }
        }
    };

    // 3. Resolve destination directory.
    let dest_dir = match &dest_override {
        Some(dir) => PathBuf::from(dir),
        None => resolve_default_models_dir()?,
    };

    // 4. Download.
    println!("Repository: {repo}");
    println!("Selected: {}", selected.path);
    println!("Size: {}", format_bytes(selected.size));
    let destination_name = Path::new(&selected.path)
        .file_name()
        .unwrap_or_else(|| selected.path.as_ref());
    println!("Destination: {}", dest_dir.join(destination_name).display());
    let downloaded =
        hf_download::download_file_auto(&client, &repo, selected, &dest_dir, connections).await?;

    // 5. Validate GGUF.
    let gguf_meta = match validate_gguf_model(&downloaded) {
        Ok(meta) => {
            println!("✓ GGUF metadata validated");
            meta
        }
        Err(reason) => {
            return Err(format!("Downloaded file is not a valid GGUF model: {reason}").into());
        }
    };

    // 6. Hardware fit preflight — compute best-fit params for this machine.
    let default_context = 16384u32;
    let model_path_str = downloaded.to_string_lossy().into_owned();
    let block_count = gguf_meta.block_count.and_then(|n| u32::try_from(n).ok());
    let max_context = gguf_meta.context_length.and_then(|n| u32::try_from(n).ok());

    // These will be written into the registry entry so the first load uses them.
    let fit_context_size: Option<u32>;
    let fit_ngl: Option<u32>;
    let fit_extra_args: Vec<String>;

    {
        use crate::fit::{FitConfig, FitPlanner, HardwareSummary, ModelSummary};

        let hardware = HardwareSummary::probe(12); // default vram_gb fallback
        let model = ModelSummary::from_file(
            &model_path_str,
            block_count,
            gguf_meta.architecture.clone(),
            max_context,
            default_context,
            false,
            None,
        );

        println!("── Hardware Fit ──");
        if hardware.gpus.is_empty() {
            println!(
                "RAM: {:.1} GiB | GPU: not detected (CPU-only)",
                hardware.system_ram_mb as f64 / 1024.0
            );
        } else {
            for g in &hardware.gpus {
                println!(
                    "GPU{}: {} — {:.1}/{:.1} GiB free",
                    g.index,
                    g.name,
                    g.free_mb as f64 / 1024.0,
                    g.total_mb as f64 / 1024.0,
                );
            }
        }
        println!(
            "Model: {:.1} GiB | Blocks: {} | Max context: {}",
            model.file_size_mb as f64 / 1024.0,
            block_count
                .map(|b| b.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            max_context
                .map(|c| c.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
        );

        let fit_config = FitConfig::default();
        let planner = FitPlanner::new(hardware, model, fit_config);
        let initial = planner.current_plan();
        println!(
            "Estimated profile: ctx={} ngl={} kv={}/{} split={:?} ts={:?}",
            initial.context_size,
            initial.ngl,
            initial.cache_type_k.as_deref().unwrap_or("f16"),
            initial.cache_type_v.as_deref().unwrap_or("f16"),
            initial.split_mode,
            initial.tensor_split,
        );
        if initial.context_size < default_context {
            println!(
                "⚠ Context reduced from {} to {} to fit available VRAM",
                default_context, initial.context_size
            );
        }
        if planner.total_plans() > 1 {
            println!(
                "Fallback ladder: {} attempts configured (run with fit.enabled=true to use)",
                planner.total_plans()
            );
        }

        if fit_dry_run {
            println!();
            println!("── Fit Dry Run ──");
            println!("Full fallback ladder:");
            for plan in planner.all_plans() {
                println!("  Attempt {}: {}", plan.attempt, plan);
            }
        }

        // Capture the best-fit params so they're written into the registry entry.
        fit_context_size = Some(initial.context_size);
        fit_ngl = if initial.ngl < 999 {
            Some(initial.ngl)
        } else {
            None
        };
        let mut extra = Vec::new();
        if let Some(ref mode) = initial.split_mode {
            extra.push("--split-mode".to_string());
            extra.push(mode.clone());
        }
        if let Some(ref ts) = initial.tensor_split {
            extra.push("--tensor-split".to_string());
            extra.push(
                ts.iter()
                    .map(|v| format!("{v:.1}"))
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        if let Some(ref ct) = initial.cache_type_k {
            extra.push("--cache-type-k".to_string());
            extra.push(ct.clone());
        }
        if let Some(ref ct) = initial.cache_type_v {
            extra.push("--cache-type-v".to_string());
            extra.push(ct.clone());
        }
        fit_extra_args = extra;
    }

    println!();

    // 7. Generate alias and register.
    let alias = alias_from_filename(&downloaded);
    let file_ref = downloaded
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| selected.path.clone());

    let registry_path = models_file
        .clone()
        .or_else(|| {
            let candidate = dest_dir.join("models.toml");
            if candidate.is_file() {
                Some(candidate.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "models.toml".to_string());

    let mut registry = if Path::new(&registry_path).is_file() {
        ModelsRegistry::load(&registry_path)?
    } else {
        ModelsRegistry {
            defaults: super::models_registry::RegistryDefaults {
                models_dir: dest_dir.to_string_lossy().into_owned(),
                ..Default::default()
            },
            auto_discover: true,
            models: Vec::new(),
            ..Default::default()
        }
    };

    let kind = infer_kind_from_filename(&selected.path);

    // Check if file is already registered.
    if let Some(existing) = registry.models.iter().find(|e| e.file == file_ref) {
        let registered_alias = existing.alias.clone();
        println!("✓ Already registered as: {registered_alias}");
        let refreshed = refresh_after_pull().await;
        maybe_bench_after_pull(&registered_alias, &kind, no_bench, refreshed).await;
        return Ok(());
    }

    // Merge into an existing vLLM-only entry with the same derived alias
    // (e.g. this model was already pulled via `models pull vllm`), rather
    // than creating a second, disconnected registry entry. `backend` stays
    // unset either way, so expand() keeps auto-preferring vLLM when it fits
    // and only uses this GGUF source as the fallback.
    if let Some(existing) = registry
        .models
        .iter_mut()
        .find(|e| e.alias == alias && e.file.is_empty() && e.has_vllm_source())
    {
        existing.file = file_ref;
        existing.hf_repo = existing.hf_repo.clone().or_else(|| Some(repo.clone()));
        existing.min_vram_gb = existing
            .min_vram_gb
            .or_else(|| Some(((selected.size as f64 / 1_000_000_000.0).ceil() as u32).max(1)));
        existing.context_size = existing.context_size.or(fit_context_size);
        existing.ngl = existing.ngl.or(fit_ngl);
        if existing.extra_args.is_empty() {
            existing.extra_args = fit_extra_args;
        }
        let alias = existing.alias.clone();
        registry.write(&registry_path)?;
        println!(
            "✓ Merged GGUF source into existing alias: {alias} \
             (this model now has both GGUF and vLLM sources — vLLM is preferred when it fits)"
        );
        let refreshed = refresh_after_pull().await;
        maybe_bench_after_pull(&alias, &kind, no_bench, refreshed).await;
        return Ok(());
    }

    let used_aliases: HashSet<String> = registry.models.iter().map(|e| e.alias.clone()).collect();
    let alias = dedupe_alias(&alias, &used_aliases);

    let entry = RegistryEntry {
        alias: alias.clone(),
        file: file_ref,
        display_name: Some(display_name_from_alias(&alias)),
        kind: Some(kind.clone()),
        enabled: true,
        hf_repo: Some(repo.clone()),
        min_vram_gb: Some(((selected.size as f64 / 1_000_000_000.0).ceil() as u32).max(1)),
        context_size: fit_context_size,
        ngl: fit_ngl,
        extra_args: fit_extra_args,
        ..Default::default()
    };
    registry.models.push(entry);

    registry.write(&registry_path)?;
    println!("✓ Registered as: {alias}");
    let refreshed = refresh_after_pull().await;
    maybe_bench_after_pull(&alias, &kind, no_bench, refreshed).await;

    Ok(())
}

/// `gguf-switchboard models pull vllm <repo-id> [--dir /path] [--registry /path]
///     [--draft <repo>] [--num-speculative-tokens N] [--attention-backend NAME]
///     [--tensor-parallel-size N] [--gpu-memory-utilization F] [--served-model-name NAME]
///     [--connections N] [--force]`
///
/// Downloads a safetensors repo (weights + config/tokenizer files) into a
/// `vllm-models` directory sibling to the GGUF models dir, detects
/// quantization/speculative-decoding metadata from `config.json`, and writes
/// a `models.toml` entry with `backend = "vllm"` and the exact parameters
/// vLLM needs to serve it. Short-circuits before touching the network if
/// this exact repo is already downloaded and registered — pass `--force` to
/// re-pull anyway (e.g. after a partial/corrupted download).
async fn cmd_pull_vllm(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut repo: Option<String> = None;
    let mut dest_override: Option<String> = None;
    let mut models_file: Option<String> = None;
    let mut connections: u16 = 8;
    let mut draft_repo: Option<String> = None;
    let mut num_speculative_tokens: Option<u32> = None;
    let mut attention_backend: Option<String> = None;
    let mut tensor_parallel_size: Option<u32> = None;
    let mut gpu_memory_utilization: Option<f32> = None;
    let mut served_model_name: Option<String> = None;
    let mut force = false;

    let mut i = 1; // args[0] is "vllm"
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                dest_override = Some(require_val(args, &mut i, "--dir")?);
            }
            "--registry" => {
                models_file = Some(require_val(args, &mut i, "--registry")?);
            }
            "--connections" => {
                connections = parse_connections(&require_val(args, &mut i, "--connections")?)?;
            }
            "--draft" => {
                draft_repo = Some(require_val(args, &mut i, "--draft")?);
            }
            "--num-speculative-tokens" => {
                let val = require_val(args, &mut i, "--num-speculative-tokens")?;
                num_speculative_tokens =
                    Some(val.parse().map_err(
                        |_| "models pull vllm: invalid value for --num-speculative-tokens",
                    )?);
            }
            "--attention-backend" => {
                attention_backend = Some(require_val(args, &mut i, "--attention-backend")?);
            }
            "--tensor-parallel-size" => {
                let val = require_val(args, &mut i, "--tensor-parallel-size")?;
                tensor_parallel_size =
                    Some(val.parse().map_err(
                        |_| "models pull vllm: invalid value for --tensor-parallel-size",
                    )?);
            }
            "--gpu-memory-utilization" => {
                let val = require_val(args, &mut i, "--gpu-memory-utilization")?;
                gpu_memory_utilization =
                    Some(val.parse().map_err(
                        |_| "models pull vllm: invalid value for --gpu-memory-utilization",
                    )?);
            }
            "--served-model-name" => {
                served_model_name = Some(require_val(args, &mut i, "--served-model-name")?);
            }
            "--force" => {
                force = true;
                i += 1;
            }
            arg if arg.starts_with('-') => {
                return Err(format!("models pull vllm: unknown flag '{arg}'").into());
            }
            val => {
                repo = Some(val.to_string());
                i += 1;
            }
        }
    }
    let repo =
        repo.ok_or("models pull vllm: missing repository id (e.g. Qwen/Qwen2.5-7B-Instruct)")?;

    // 1. Deploy check up front — fail before downloading anything multi-GB.
    let registry_path_hint = models_file
        .clone()
        .unwrap_or_else(|| "models.toml".to_string());
    let (vllm_command, vllm_project) = if Path::new(&registry_path_hint).is_file() {
        let existing = ModelsRegistry::load(&registry_path_hint)?;
        (
            existing.defaults.vllm_command.clone(),
            existing.defaults.vllm_project.clone(),
        )
    } else {
        ("uv".to_string(), None)
    };
    if let Err(reason) =
        super::models_registry::check_vllm_available(&vllm_command, vllm_project.as_deref())
    {
        return Err(reason.into());
    }
    println!("✓ vLLM environment OK ({vllm_command} run vllm)");

    // 2. Resolve destination + registry, then check whether this exact repo
    // is already downloaded and registered before touching the network at
    // all. aria2c/the native downloader already skip re-fetching individual
    // files that are complete, but that still means a repo-tree fetch, N
    // per-file redirect round-trips, a second metadata fetch, and a
    // registry rewrite on every re-run of an already-pulled model — wasted
    // work `--force` is what should trigger it again, not running the same
    // command twice.
    let base_dest = match &dest_override {
        Some(dir) => PathBuf::from(dir),
        None => resolve_default_vllm_models_dir()?,
    };
    let dest_dir = base_dest.join(sanitize_repo_dirname(&repo));
    let alias = alias_from_repo(&repo);
    let registry_path = models_file
        .clone()
        .or_else(|| {
            let candidate = base_dest.join("models.toml");
            if candidate.is_file() {
                Some(candidate.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "models.toml".to_string());
    let mut registry = if Path::new(&registry_path).is_file() {
        ModelsRegistry::load(&registry_path)?
    } else {
        ModelsRegistry {
            auto_discover: true,
            models: Vec::new(),
            ..Default::default()
        }
    };

    if !force
        && let Some(existing) = registry
            .models
            .iter()
            .find(|e| e.vllm_hf_repo.as_deref() == Some(repo.as_str()))
        && let Some(existing_dir) = existing.vllm_file.as_deref()
        && crate::fit::sum_safetensors_mb(Path::new(existing_dir)) > 0
    {
        println!(
            "✓ Already registered as: {} (vLLM source) — {repo} is already downloaded at {existing_dir}.\n  \
             Use --force to re-pull (e.g. after a partial/corrupted download).",
            existing.alias
        );
        return Ok(());
    }

    // 3. Fetch repo tree and validate it's a servable safetensors repo.
    let client = hf_download::build_hf_client()?;
    let entries = hf_download::fetch_repo_tree_all(&client, &repo).await?;
    if !super::vllm_meta::is_safetensors_repo(&entries) {
        return Err(format!(
            "{repo} does not look like a vLLM-servable safetensors repo \
             (no *.safetensors + config.json found). Use `ggs models search vllm <query>` \
             to find one, or `ggs models pull {repo}` for GGUF/llama.cpp instead."
        )
        .into());
    }
    let serving_files: Vec<&HfTreeEntry> = entries
        .iter()
        .filter(|e| super::vllm_meta::is_vllm_serving_file(&e.path))
        .collect();

    println!("Repository: {repo}");
    println!(
        "Files: {} ({} total)",
        serving_files.len(),
        format_bytes(serving_files.iter().map(|e| e.size).sum())
    );
    println!("Destination: {}", dest_dir.display());

    // 4. Download every serving file.
    for entry in serving_files.iter().copied() {
        println!("  {} ({})", entry.path, format_bytes(entry.size));
        hf_download::download_file_auto(&client, &repo, entry, &dest_dir, connections).await?;
    }
    println!("✓ Download complete");

    let weights_mb: u64 = serving_files
        .iter()
        .filter(|e| e.path.ends_with(".safetensors"))
        .map(|e| e.size)
        .sum::<u64>()
        / (1024 * 1024);

    // 4b. Hardware fit preview — same planner the scheduler runs at load time,
    // so what's printed here is what will actually be requested of vLLM.
    {
        use crate::fit::{HardwareSummary, VllmFitPlanner};
        let hardware = HardwareSummary::probe(12);
        println!("── Hardware Fit ──");
        if hardware.gpus.is_empty() {
            println!(
                "RAM: {:.1} GiB | GPU: not detected (CPU-only)",
                hardware.system_ram_mb as f64 / 1024.0
            );
        } else {
            for g in &hardware.gpus {
                println!(
                    "GPU{}: {} — {:.1}/{:.1} GiB free",
                    g.index,
                    g.name,
                    g.free_mb as f64 / 1024.0,
                    g.total_mb as f64 / 1024.0,
                );
            }
        }
        println!("Model weights: {:.1} GiB", weights_mb as f64 / 1024.0);
        let default_context = 16384u32;
        let planner = VllmFitPlanner::new(
            &hardware,
            weights_mb,
            default_context,
            tensor_parallel_size,
            gpu_memory_utilization,
            &crate::fit::FitConfig::default(),
        );
        let plan = planner.current_plan();
        println!(
            "Estimated profile: max_model_len={} gpu_memory_utilization={:.2} tensor_parallel_size={}",
            plan.max_model_len, plan.gpu_memory_utilization, plan.tensor_parallel_size
        );
        println!(
            "(The scheduler re-runs this at load time against live free VRAM and retries \
             with a reduced max_model_len on OOM — these are the settings the running \
             server actually starts with.)"
        );
    }
    println!();

    // 5. Detect quantization / speculative-decoding metadata from config.json.
    let meta = super::vllm_meta::detect_vllm_metadata(&client, &repo)
        .await
        .unwrap_or_default();
    if let Some(q) = &meta.quantization {
        println!("Detected quantization: {q}");
    }
    if let Some(d) = &meta.draft_model {
        println!("Detected Speculators draft/target pairing: {d}");
    }

    // 6. Optionally pull a draft (dspark) model for speculative decoding.
    let resolved_draft = if let Some(draft_repo) = &draft_repo {
        let draft_entries = hf_download::fetch_repo_tree_all(&client, draft_repo).await?;
        let draft_dest = base_dest.join(sanitize_repo_dirname(draft_repo));
        println!("Draft model: {draft_repo}");
        for entry in draft_entries
            .iter()
            .filter(|e| super::vllm_meta::is_vllm_serving_file(&e.path))
        {
            println!("  {} ({})", entry.path, format_bytes(entry.size));
            hf_download::download_file_auto(&client, draft_repo, entry, &draft_dest, connections)
                .await?;
        }
        Some(draft_dest.to_string_lossy().into_owned())
    } else {
        None
    };

    // 7. Register in models.toml (alias/registry_path/registry resolved in step 2).
    let kind = infer_kind_from_filename(&format!(
        "{repo} {}",
        meta.architecture.as_deref().unwrap_or("")
    ));
    let min_vram_gb = Some(((weights_mb as f64 / 1024.0).ceil() as u32).max(1));
    let vllm_dest_str = dest_dir.to_string_lossy().into_owned();

    // Merge into an existing entry with the same derived alias when one
    // exists (e.g. this model was already pulled as GGUF) rather than
    // creating a second, disconnected registry entry: `backend` is left
    // unset so expand() auto-prefers vLLM, falling back to the existing
    // GGUF source only if vLLM's weights don't fit this hardware.
    if let Some(existing) = registry.models.iter_mut().find(|e| e.alias == alias) {
        existing.vllm_file = Some(vllm_dest_str);
        existing.vllm_hf_repo = Some(repo.clone());
        existing.min_vram_gb = existing.min_vram_gb.or(min_vram_gb);
        existing.quantization = meta.quantization.clone().or(existing.quantization.clone());
        existing.attention_backend = attention_backend.or(existing.attention_backend.clone());
        existing.draft_model = resolved_draft
            .or(meta.draft_model.clone())
            .or(existing.draft_model.clone());
        existing.num_speculative_tokens = num_speculative_tokens
            .or(meta.num_speculative_tokens)
            .or(existing.num_speculative_tokens);
        existing.tensor_parallel_size = tensor_parallel_size.or(existing.tensor_parallel_size);
        existing.gpu_memory_utilization =
            gpu_memory_utilization.or(existing.gpu_memory_utilization);
        existing.served_model_name = served_model_name.or(existing.served_model_name.clone());
        existing.max_context_length = existing.max_context_length.or(meta.max_position_embeddings);
        registry.write(&registry_path)?;
        println!(
            "✓ Merged vLLM source into existing alias: {alias} \
             (this model now has both GGUF and vLLM sources — vLLM is preferred when it fits)"
        );
        // refresh_after_pull() already prints a "start or restart" hint on failure.
        refresh_after_pull().await;
        return Ok(());
    }

    let used_aliases: HashSet<String> = registry.models.iter().map(|e| e.alias.clone()).collect();
    let alias = dedupe_alias(&alias, &used_aliases);

    let entry = RegistryEntry {
        alias: alias.clone(),
        display_name: Some(display_name_from_alias(&alias)),
        kind: Some(kind.clone()),
        enabled: true,
        vllm_file: Some(vllm_dest_str),
        vllm_hf_repo: Some(repo.clone()),
        min_vram_gb,
        quantization: meta.quantization.clone(),
        attention_backend,
        draft_model: resolved_draft.or(meta.draft_model.clone()),
        num_speculative_tokens: num_speculative_tokens.or(meta.num_speculative_tokens),
        tensor_parallel_size,
        gpu_memory_utilization,
        served_model_name,
        max_context_length: meta.max_position_embeddings,
        ..Default::default()
    };
    registry.models.push(entry);
    registry.write(&registry_path)?;
    println!("✓ Registered as: {alias} (vLLM source)");

    // refresh_after_pull() already prints a "start or restart" hint on failure.
    refresh_after_pull().await;

    Ok(())
}

/// Shift-and-fetch the value following a flag, advancing `i` by 2. Errors
/// with a `models pull vllm: missing value for <flag>` message when absent.
fn require_val(args: &[String], i: &mut usize, flag: &str) -> Result<String, String> {
    match args.get(*i + 1) {
        Some(val) => {
            let val = val.clone();
            *i += 2;
            Ok(val)
        }
        None => Err(format!("models pull vllm: missing value for {flag}")),
    }
}

fn parse_connections(value: &str) -> Result<u16, &'static str> {
    let connections = value
        .parse::<u16>()
        .map_err(|_| "models pull: --connections must be an integer from 1 to 16")?;
    if !(1..=16).contains(&connections) {
        return Err("models pull: --connections must be an integer from 1 to 16");
    }
    Ok(connections)
}

fn client_address_from_config(config_path: &Path) -> Result<String, RuntimeError> {
    let raw = std::fs::read_to_string(config_path).map_err(|e| {
        RuntimeError::ConfigError(format!("Cannot read '{}': {e}", config_path.display()))
    })?;
    let config: toml::Value = toml::from_str(&raw)
        .map_err(|e| RuntimeError::ConfigError(format!("Invalid config.toml: {e}")))?;
    let bind = config
        .get("bind")
        .and_then(toml::Value::as_str)
        .ok_or_else(|| RuntimeError::ConfigError("config.toml has no bind address".to_string()))?;
    Ok(match bind.parse::<SocketAddr>() {
        Ok(address) if address.ip().is_unspecified() => {
            SocketAddr::new(IpAddr::from([127, 0, 0, 1]), address.port()).to_string()
        }
        _ => bind.to_string(),
    })
}

fn refresh_url_from_config(config_path: &Path) -> Result<String, RuntimeError> {
    Ok(format!(
        "http://{}/v1/models/refresh",
        client_address_from_config(config_path)?
    ))
}

fn chat_url_from_config(config_path: &Path) -> Result<String, RuntimeError> {
    Ok(format!(
        "http://{}/v1/chat/completions",
        client_address_from_config(config_path)?
    ))
}

async fn refresh_running_server(
    client: &reqwest::Client,
    config_path: &Path,
) -> Result<(), RuntimeError> {
    let url = refresh_url_from_config(config_path)?;
    let response = client.post(&url).send().await.map_err(RuntimeError::from)?;
    if !response.status().is_success() {
        return Err(RuntimeError::ProxyError(format!(
            "model refresh failed: HTTP {}",
            response.status()
        )));
    }
    Ok(())
}

/// Returns true when a running gguf-switchboard accepted the registry refresh.
async fn refresh_after_pull() -> bool {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            eprintln!(
                "Warning: model downloaded and registered, but refresh setup failed: {error}"
            );
            eprintln!("Start or restart gguf-switchboard to load the updated registry.");
            return false;
        }
    };
    match refresh_running_server(&client, &resolve_config_toml_path()).await {
        Ok(()) => {
            println!("✓ Running server refreshed");
            true
        }
        Err(error) => {
            eprintln!("Warning: model downloaded and registered, but live refresh failed: {error}");
            eprintln!("Start or restart gguf-switchboard to load the updated registry.");
            false
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SpeedStats {
    prompt_tps: f64,
    prompt_tokens: u64,
    gen_tps: f64,
    gen_tokens: u64,
    from_timings: bool,
}

/// Extract prompt/generation tok/s from a chat-completions JSON body.
/// Prefers llama.cpp `timings`; falls back to `usage` / wall-clock seconds.
fn extract_speed_stats(body: &Value, wall_secs: f64) -> Option<SpeedStats> {
    if let Some(timings) = body.get("timings") {
        let prompt_tokens = timings
            .get("prompt_n")
            .and_then(Value::as_u64)
            .or_else(|| {
                body.get("usage")
                    .and_then(|u| u.get("prompt_tokens"))
                    .and_then(Value::as_u64)
            })
            .unwrap_or(0);
        let gen_tokens = timings
            .get("predicted_n")
            .and_then(Value::as_u64)
            .or_else(|| {
                body.get("usage")
                    .and_then(|u| u.get("completion_tokens"))
                    .and_then(Value::as_u64)
            })
            .unwrap_or(0);

        let prompt_tps = timings
            .get("prompt_per_second")
            .and_then(Value::as_f64)
            .or_else(|| {
                let ms = timings.get("prompt_ms").and_then(Value::as_f64)?;
                if ms > 0.0 && prompt_tokens > 0 {
                    Some(prompt_tokens as f64 / (ms / 1000.0))
                } else {
                    None
                }
            });
        let gen_tps = timings
            .get("predicted_per_second")
            .and_then(Value::as_f64)
            .or_else(|| {
                let ms = timings.get("predicted_ms").and_then(Value::as_f64)?;
                if ms > 0.0 && gen_tokens > 0 {
                    Some(gen_tokens as f64 / (ms / 1000.0))
                } else {
                    None
                }
            });

        if let (Some(prompt_tps), Some(gen_tps)) = (prompt_tps, gen_tps) {
            return Some(SpeedStats {
                prompt_tps,
                prompt_tokens,
                gen_tps,
                gen_tokens,
                from_timings: true,
            });
        }
    }

    if wall_secs <= 0.0 {
        return None;
    }
    let usage = body.get("usage")?;
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let gen_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if prompt_tokens == 0 && gen_tokens == 0 {
        return None;
    }
    Some(SpeedStats {
        prompt_tps: prompt_tokens as f64 / wall_secs,
        prompt_tokens,
        gen_tps: gen_tokens as f64 / wall_secs,
        gen_tokens,
        from_timings: false,
    })
}

fn print_speed_stats(alias: &str, stats: &SpeedStats) {
    println!("✓ Speed test ({alias})");
    println!(
        "  Prompt:     {:>7.1} tok/s  ({} tokens)",
        stats.prompt_tps, stats.prompt_tokens
    );
    println!(
        "  Generation: {:>7.1} tok/s  ({} tokens)",
        stats.gen_tps, stats.gen_tokens
    );
    if !stats.from_timings {
        println!("  (wall-clock estimate — llama.cpp timings not present in response)");
    }
}

async fn maybe_bench_after_pull(alias: &str, kind: &str, no_bench: bool, refreshed: bool) {
    if no_bench {
        return;
    }
    if kind.eq_ignore_ascii_case("embedding") || kind.eq_ignore_ascii_case("reranker") {
        eprintln!("Speed test skipped: {kind} models have no chat generation");
        return;
    }
    if !refreshed {
        eprintln!(
            "Speed test skipped: gguf-switchboard is not running (start it, then re-pull or chat)"
        );
        return;
    }
    if let Err(reason) = bench_pulled_model(alias).await {
        eprintln!("Speed test skipped: {reason}");
    }
}

async fn bench_pulled_model(alias: &str) -> Result<(), String> {
    let config_path = resolve_config_toml_path();
    let url = chat_url_from_config(&config_path).map_err(|e| e.to_string())?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| format!("HTTP client: {e}"))?;

    let body = serde_json::json!({
        "model": alias,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 64,
        "stream": false
    });

    println!("Running speed test (first load may take a while)...");
    let started = std::time::Instant::now();
    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("chat request failed: {e}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        let detail = detail.trim();
        if detail.is_empty() {
            return Err(format!("chat request failed: HTTP {status}"));
        }
        return Err(format!("chat request failed: HTTP {status}: {detail}"));
    }
    let json: Value = response
        .json()
        .await
        .map_err(|e| format!("invalid chat response: {e}"))?;
    let wall_secs = started.elapsed().as_secs_f64();
    let stats = extract_speed_stats(&json, wall_secs)
        .ok_or_else(|| "response missing usable timings/usage".to_string())?;
    print_speed_stats(alias, &stats);
    Ok(())
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Find the default models directory from an existing registry, or fall back to common paths.
fn resolve_default_models_dir() -> Result<PathBuf, RuntimeError> {
    // Try loading models.toml from the current directory.
    if Path::new("models.toml").is_file()
        && let Ok(registry) = ModelsRegistry::load("models.toml")
        && let Ok(dirs) =
            super::models_registry::resolve_models_dirs_with_fallback(&registry.defaults.models_dir)
        && let Some(dir) = dirs.first()
    {
        return Ok(dir.clone());
    }

    // Try MODELS_DIR env var.
    if let Ok(dirs) = std::env::var("MODELS_DIR") {
        let dirs = dirs.trim();
        if !dirs.is_empty() {
            let first = dirs.split(',').next().unwrap_or(dirs).trim();
            if !first.is_empty() {
                let path = PathBuf::from(first);
                if path.is_dir() {
                    return Ok(path);
                }
            }
        }
    }

    // Fall back to ~/models.
    if let Some(home) = std::env::var_os("HOME") {
        let home_models = PathBuf::from(home).join("models");
        if home_models.is_dir() {
            return Ok(home_models);
        }
    }

    Err(RuntimeError::ConfigError(
        "No models directory found; use --dir or create ~/models".to_string(),
    ))
}

/// Locate the running server's `config.toml` for the live-refresh/benchmark
/// calls `ggs models pull [vllm]` makes after registering a model. `ggs` is
/// installed system-wide (`/usr/local/bin/gguf-switchboard`) and must work
/// from any `$PWD`, so this can't just assume `./config.toml` — that only
/// happened to work when run from the install directory.
///
/// Checked in order: `$GGUF_SWITCHBOARD_CONFIG_DIR` (the directory containing
/// `config.toml` — already documented as the override for a runtime config
/// kept outside the checkout, see docs/superpowers/specs/2026-08-08-deploy-
/// runtime-config-design.md, but never actually read anywhere until now),
/// `./config.toml` (still honored for custom/dev setups run from their own
/// directory), then `/opt/gguf-switchboard/config.toml` — the actual install
/// path per `deploy.sh` and `gguf-switchboard.service`. Falls back to the
/// literal `"config.toml"` (today's behavior) so a legitimately custom,
/// undetectable layout still gets the same error message as before.
fn resolve_config_toml_path() -> PathBuf {
    if let Ok(dir) = std::env::var("GGUF_SWITCHBOARD_CONFIG_DIR") {
        let path = PathBuf::from(dir).join("config.toml");
        if path.is_file() {
            return path;
        }
    }
    for candidate in [
        PathBuf::from("config.toml"),
        PathBuf::from("/opt/gguf-switchboard/config.toml"), // matches deploy.sh / gguf-switchboard.service
        PathBuf::from("/etc/gguf-switchboard/config.toml"), // legacy location deploy.sh migrates from
    ] {
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("config.toml")
}

/// Default destination for vLLM safetensors pulls: a `vllm-models` directory
/// *sibling* to wherever GGUF models live (not nested inside it), so the two
/// backends' downloads stay clearly separated on disk.
fn resolve_default_vllm_models_dir() -> Result<PathBuf, RuntimeError> {
    let gguf_dir = resolve_default_models_dir()?;
    let sibling = gguf_dir
        .parent()
        .map(|parent| parent.join("vllm-models"))
        .unwrap_or_else(|| PathBuf::from("vllm-models"));
    Ok(sibling)
}

/// Sanitize a HF repo id (`org/name`) into a single path segment.
fn sanitize_repo_dirname(repo: &str) -> String {
    repo.replace('/', "__")
}

/// Derive a short alias from a HF repo id (e.g. `Qwen/Qwen2.5-7B-Instruct` ->
/// `qwen2.5-7b-instruct`). Unlike `alias_from_filename`, this doesn't treat
/// `.` as an extension separator — repo names routinely contain literal dots
/// (version numbers) that `Path::file_stem()` would otherwise mis-split on.
fn alias_from_repo(repo: &str) -> String {
    let name = repo.rsplit('/').next().unwrap_or(repo);
    let lower = name.to_ascii_lowercase().replace('_', "-");
    let alias: String = lower
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    alias.trim_matches('-').to_string()
}

/// Return true if the filename looks like a main model GGUF (not mmproj/lora/projector).
fn is_model_gguf(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    !(lower.contains("mmproj")
        || lower.contains("-lora")
        || lower.contains("-projector")
        || lower.contains("projector")
        || lower.contains("-adapter")
        || lower.contains("tokenizer")
        || lower.contains("vocab"))
}

fn is_auxiliary_model_file(path: &str) -> bool {
    let filename = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_ascii_lowercase();
    let tokens = filename
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let has_dspark = tokens.contains(&"dspark");
    let quant_tokens = filename
        .trim_end_matches(".gguf")
        .split(['-', '.'])
        .filter(|token| is_quant_token(&token.to_ascii_uppercase()))
        .count();
    (tokens.contains(&"mtp") && quant_tokens > 1)
        || (has_dspark && (tokens.first() == Some(&"dspark") || tokens.contains(&"support")))
}

fn split_model_part(path: &str) -> Option<(&str, usize, usize)> {
    let stem = path.get(..path.len().checked_sub(5)?)?;
    if !path.get(path.len() - 5..)?.eq_ignore_ascii_case(".gguf") {
        return None;
    }
    let (before_total, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = before_total.rsplit_once('-')?;
    let index = index.parse::<usize>().ok()?;
    let total = total.parse::<usize>().ok()?;
    (index > 0 && total > 0 && index <= total).then_some((prefix, index, total))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelOption {
    quant: String,
    bytes: u64,
}

/// A quant recommended for a specific reason (fastest, or least precision loss).
#[derive(Debug, Clone, PartialEq)]
struct QuantRecommendation {
    quant: String,
    tokens_per_sec: f64,
    quality_score: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct SearchAssessment {
    supported: bool,
    quants: Vec<ModelOption>,
    recommended_quant: Option<String>,
    /// 0–100 memory-fit score for the best-fitting discovered quant. See
    /// [`quant_profile::fit_score`] — continuous replacement for the old
    /// binary "Supported: Yes/No".
    fit_score: f32,
    /// The supported quant with the highest estimated tokens/sec.
    fastest: Option<QuantRecommendation>,
    /// The supported quant with the highest precision-retention score.
    best_precision: Option<QuantRecommendation>,
    /// The supported quant that best balances speed and precision — see
    /// [`balanced_quant`]. `None` only when `quants` is empty.
    balanced: Option<QuantRecommendation>,
}

/// Pick the quant that best balances speed against precision loss.
///
/// Within a repo's candidate quants, file size is almost perfectly
/// anti-correlated between the two dimensions — a bigger quant is (almost)
/// always both slower *and* more precise, because size is the direct input
/// to both the speed model (smaller = less to stream per token) and the
/// precision table (smaller = more aggressively quantized). So the size
/// range from the smallest to the largest fitting quant already traces the
/// speed/precision trade-off curve, and "balanced" is the quant whose size
/// sits closest to the midpoint of that range — not too close to either the
/// smallest (fastest, lossiest) or largest (slowest, most precise) end.
///
/// An earlier version of this tried to average a 0–100-normalized speed
/// score with the quality score, but normalizing speed against the fastest
/// candidate gives it a much wider spread (often 3–4x across the range) than
/// quality typically has (usually well under 2x from worst to best fitting
/// quant), so the average was dominated by speed and kept picking one of the
/// fastest/lossiest options — the opposite of "balanced." Size-midpoint
/// avoids needing to reconcile those two different scales at all.
///
/// Ties (a size exactly equidistant from two quants) prefer the smaller
/// (cheaper-to-run) one, then quant name for determinism.
fn balanced_quant(
    scored: &[(
        &ModelOption,
        quant_profile::SpeedEstimate,
        quant_profile::QuantQuality,
    )],
) -> Option<QuantRecommendation> {
    let (min_bytes, max_bytes) = scored
        .iter()
        .fold((u64::MAX, 0_u64), |(min, max), (o, ..)| {
            (min.min(o.bytes), max.max(o.bytes))
        });
    if min_bytes > max_bytes {
        return None;
    }
    let midpoint = min_bytes / 2 + max_bytes / 2; // avoid overflow on the sum

    scored
        .iter()
        .min_by(|a, b| {
            let distance = |bytes: u64| bytes.abs_diff(midpoint);
            distance(a.0.bytes)
                .cmp(&distance(b.0.bytes))
                .then_with(|| a.0.bytes.cmp(&b.0.bytes))
                .then_with(|| a.0.quant.cmp(&b.0.quant))
        })
        .map(|(option, speed, quality)| QuantRecommendation {
            quant: option.quant.clone(),
            tokens_per_sec: speed.tokens_per_sec,
            quality_score: quality.quality_score,
        })
}

fn complete_model_options(entries: &[HfTreeEntry]) -> Vec<ModelOption> {
    let mut options: HashMap<String, u64> = HashMap::new();
    let mut split_groups: HashMap<&str, (usize, BTreeMap<usize, u64>, bool)> = HashMap::new();

    for entry in entries
        .iter()
        .filter(|entry| is_model_gguf(&entry.path) && !is_auxiliary_model_file(&entry.path))
    {
        if let Some((prefix, index, total)) = split_model_part(&entry.path) {
            let group = split_groups
                .entry(prefix)
                .or_insert_with(|| (total, BTreeMap::new(), true));
            if group.0 != total || group.1.insert(index, entry.size).is_some() {
                group.2 = false;
            }
        } else {
            let quant = extract_quant(&entry.path);
            if !quant.is_empty() {
                options
                    .entry(quant)
                    .and_modify(|bytes| *bytes = (*bytes).min(entry.size))
                    .or_insert(entry.size);
            }
        }
    }

    for (prefix, (expected, shards, valid)) in split_groups {
        if valid && shards.len() == expected && shards.keys().copied().eq(1..=expected) {
            let size = shards
                .values()
                .try_fold(0_u64, |total, size| total.checked_add(*size));
            if let Some(size) = size {
                let quant = extract_quant(prefix);
                if !quant.is_empty() {
                    options
                        .entry(quant)
                        .and_modify(|bytes| *bytes = (*bytes).min(size))
                        .or_insert(size);
                }
            }
        }
    }

    let mut options = options
        .into_iter()
        .map(|(quant, bytes)| ModelOption { quant, bytes })
        .collect::<Vec<_>>();
    options.sort_by(|left, right| {
        left.bytes
            .cmp(&right.bytes)
            .then_with(|| left.quant.cmp(&right.quant))
    });
    options
}

fn is_supported(model_bytes: Option<u64>, capacity_bytes: u64) -> bool {
    model_bytes.is_some_and(|bytes| u128::from(bytes) * 120 <= u128::from(capacity_bytes) * 100)
}

fn select_quant_entry<'a>(
    entries: &'a [&'a HfTreeEntry],
    selector: &str,
    capacity_bytes: u64,
    repository: &str,
) -> Result<&'a HfTreeEntry, String> {
    let selector = selector.trim().to_ascii_uppercase();

    if selector == "AUTO" {
        if contains_auxiliary_token(repository) {
            return Err("auto selection is disabled for an auxiliary repository".to_string());
        }
        return entries
            .iter()
            .copied()
            .filter(|entry| !is_auxiliary_model_file(&entry.path))
            .filter(|entry| !contains_auxiliary_token(&entry.path))
            .filter(|entry| split_model_part(&entry.path).is_none())
            .filter(|entry| !extract_quant(&entry.path).is_empty())
            .filter(|entry| is_supported(Some(entry.size), capacity_bytes))
            .max_by_key(|entry| entry.size)
            .ok_or_else(|| "no quantization fits the detected hardware capacity".to_string());
    }

    let target = if selector == "K_M" {
        "Q4_K_M".to_string()
    } else if selector.len() >= 2
        && selector.starts_with('Q')
        && selector[1..]
            .chars()
            .all(|character| character.is_ascii_digit())
    {
        let preferences = [
            format!("{selector}_K_M"),
            format!("{selector}_K_S"),
            format!("{selector}_0"),
            format!("{selector}_1"),
        ];
        preferences
            .into_iter()
            .find(|preference| {
                entries
                    .iter()
                    .any(|entry| extract_quant(&entry.path) == *preference)
            })
            .ok_or_else(|| format!("no preferred {selector} quantization is available"))?
    } else {
        selector
    };

    let matches = entries
        .iter()
        .copied()
        .filter(|entry| extract_quant(&entry.path) == target)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [entry] => Ok(*entry),
        [] => Err(format!("quantization {target} is not available")),
        _ => Err(format!(
            "multiple files provide {target}; use a repository with one file per quantization"
        )),
    }
}

fn contains_auxiliary_token(value: &str) -> bool {
    value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "drafter" | "speculator"
            )
        })
}

fn is_standalone_model(hit: &Value, entries: &[HfTreeEntry]) -> bool {
    let has_auxiliary_tag = hit
        .get("tags")
        .and_then(Value::as_array)
        .is_some_and(|tags| {
            tags.iter().filter_map(Value::as_str).any(|tag| {
                matches!(
                    tag.to_ascii_lowercase().as_str(),
                    "draft-model" | "auxiliary-model"
                )
            })
        });
    if has_auxiliary_tag {
        return false;
    }

    let architecture = hit
        .get("gguf")
        .and_then(|gguf| gguf.get("architecture"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    if matches!(architecture.as_str(), "dflash" | "deepseek4-dspark")
        || architecture.contains("draft")
        || architecture.contains("speculator")
    {
        return false;
    }

    let repository_is_auxiliary = hit
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(contains_auxiliary_token);
    let filename_is_auxiliary = entries
        .iter()
        .any(|entry| contains_auxiliary_token(&entry.path));
    !(repository_is_auxiliary || filename_is_auxiliary)
}

fn assess_repository(
    hit: &Value,
    entries: &[HfTreeEntry],
    capacity_bytes: u64,
    hw: &HardwareCtx,
) -> SearchAssessment {
    if !is_standalone_model(hit, entries) {
        return SearchAssessment::default();
    }

    let all_options = complete_model_options(entries);
    // Best-case fit score across every discovered quant (not just the ones
    // that clear the hard capacity filter below), so a repo that's *close* to
    // fitting still reports a meaningful score instead of a flat 0.
    let fit_score = all_options
        .iter()
        .map(|option| quant_profile::fit_score(option.bytes, capacity_bytes))
        .fold(0.0_f32, f32::max);

    let quants = all_options
        .into_iter()
        .filter(|option| is_supported(Some(option.bytes), capacity_bytes))
        .collect::<Vec<_>>();
    let recommended_quant = quants
        .iter()
        .find(|option| option.quant == "Q4_K_M")
        .or_else(|| quants.last())
        .map(|option| option.quant.clone());

    let mut fastest: Option<QuantRecommendation> = None;
    let mut best_precision: Option<QuantRecommendation> = None;
    let mut scored = Vec::with_capacity(quants.len());
    for option in &quants {
        let speed = quant_profile::estimate_speed(option.bytes, hw);
        let quality = quant_profile::quant_quality(&option.quant);
        if fastest
            .as_ref()
            .is_none_or(|current| speed.tokens_per_sec > current.tokens_per_sec)
        {
            fastest = Some(QuantRecommendation {
                quant: option.quant.clone(),
                tokens_per_sec: speed.tokens_per_sec,
                quality_score: quality.quality_score,
            });
        }
        if best_precision
            .as_ref()
            .is_none_or(|current| quality.quality_score > current.quality_score)
        {
            best_precision = Some(QuantRecommendation {
                quant: option.quant.clone(),
                tokens_per_sec: speed.tokens_per_sec,
                quality_score: quality.quality_score,
            });
        }
        scored.push((option, speed, quality));
    }
    let balanced = balanced_quant(&scored);

    SearchAssessment {
        supported: !quants.is_empty(),
        quants,
        recommended_quant,
        fit_score,
        fastest,
        best_precision,
        balanced,
    }
}

/// Extract a quantization label from a GGUF filename.
fn extract_quant(filename: &str) -> String {
    let stem = Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename);

    // Look for common quant patterns: Q4_K_M, Q5_K_S, IQ4_NL, BF16, etc.
    let parts: Vec<&str> = stem.split(['-', '.']).collect();
    for part in parts.iter().rev() {
        let upper = part.to_ascii_uppercase();
        if is_quant_token(&upper) {
            return upper;
        }
    }
    String::new()
}

fn is_quant_token(value: &str) -> bool {
    if matches!(
        value,
        "BF16" | "FP16" | "FP32" | "F16" | "F32" | "MXFP4" | "NVFP4"
    ) {
        return true;
    }
    let bytes = value.as_bytes();
    (bytes.len() >= 4
        && bytes[0] == b'I'
        && bytes[1] == b'Q'
        && bytes[2].is_ascii_digit()
        && bytes[3] == b'_')
        || (bytes.len() >= 3 && bytes[0] == b'Q' && bytes[1].is_ascii_digit() && bytes[2] == b'_')
}

/// Duplicate of `models_registry::display_name_from_alias` (not pub).
fn display_name_from_alias(alias: &str) -> String {
    alias
        .split(['-', '_'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let rest: String = chars.collect();
                    format!("{}{}", first.to_ascii_uppercase(), rest)
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Infer model kind from a GGUF filename.
fn infer_kind_from_filename(filename: &str) -> String {
    let lower = filename.to_ascii_lowercase();
    if lower.contains("rerank") {
        "reranker".to_string()
    } else if lower.contains("embed") {
        "embedding".to_string()
    } else if lower.contains("-vl") || lower.contains("vision") {
        "vision".to_string()
    } else if lower.contains("coder") || lower.contains("-code") {
        "coder".to_string()
    } else {
        "chat".to_string()
    }
}

/// Format bytes as MB (e.g. "5243 MB").
fn format_mb(bytes: u64) -> String {
    let mb = (bytes as f64 / 1_000_000.0).ceil() as u64;
    format!("{mb} MB")
}

/// Deduplicate an alias against a set of used names, appending -2, -3, etc.
fn dedupe_alias(alias: &str, used: &HashSet<String>) -> String {
    if !used.contains(alias) {
        return alias.to_string();
    }
    let mut counter = 2;
    loop {
        let candidate = format!("{alias}-{counter}");
        if !used.contains(&candidate) {
            return candidate;
        }
        counter += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ModelOption, QuantRecommendation, SearchAssessment, assess_repository, balanced_quant,
        chat_url_from_config, complete_model_options, extract_quant, extract_speed_stats,
        format_hardware_summary, is_standalone_model, is_supported, parse_connections,
        refresh_url_from_config, render_search_table, sample_pull_command, select_quant_entry,
    };
    use crate::config::hf_download::HfTreeEntry;
    use crate::quant_profile;
    use crate::quant_profile::HardwareCtx;

    fn tree_entry(path: &str, size: u64) -> HfTreeEntry {
        HfTreeEntry {
            r#type: "file".to_string(),
            path: path.to_string(),
            size,
            lfs: None,
        }
    }

    #[test]
    fn standalone_eligibility_rejects_auxiliary_tags() {
        let entries = [tree_entry("model-q4.gguf", 4_000)];
        for tag in ["draft-model", "auxiliary-model"] {
            let hit = serde_json::json!({"id": "org/model", "tags": [tag], "gguf": {"architecture": "llama"}});
            assert!(!is_standalone_model(&hit, &entries), "tag: {tag}");
        }
    }

    #[test]
    fn standalone_eligibility_rejects_auxiliary_architectures() {
        let entries = [tree_entry("model-q4.gguf", 4_000)];
        for architecture in [
            "dflash",
            "deepseek4-dspark",
            "deepseek_v4_flash_dspark_draft",
            "qwen_speculator_v2",
        ] {
            let hit = serde_json::json!({"id": "org/model", "tags": [], "gguf": {"architecture": architecture}});
            assert!(
                !is_standalone_model(&hit, &entries),
                "architecture: {architecture}"
            );
        }
    }

    #[test]
    fn standalone_eligibility_rejects_strong_repository_and_filename_tokens() {
        let normal_entries = [tree_entry("model-q4.gguf", 4_000)];
        let drafter_hit = serde_json::json!({"id": "org/model-drafter-GGUF", "tags": [], "gguf": {"architecture": "llama"}});
        assert!(!is_standalone_model(&drafter_hit, &normal_entries));

        let normal_hit = serde_json::json!({"id": "org/model-GGUF", "tags": [], "gguf": {"architecture": "llama"}});
        let speculator_entries = [tree_entry("model-speculator-q4.gguf", 4_000)];
        assert!(!is_standalone_model(&normal_hit, &speculator_entries));
    }

    #[test]
    fn standalone_eligibility_preserves_full_dspark_target_models() {
        let entries = [tree_entry(
            "DeepSeek-V4-Flash-DSpark-support-q4.gguf",
            4_000,
        )];
        let hit = serde_json::json!({
            "id": "org/DeepSeek-V4-Flash-DSpark-draft-GGUF",
            "tags": ["speculative-decoding", "dspark"],
            "gguf": {"architecture": "deepseek4"}
        });

        assert!(is_standalone_model(&hit, &entries));
    }

    #[test]
    fn complete_model_options_extracts_named_normal_quants() {
        let entries = [
            tree_entry("model-Q4_K_M.gguf", 4_000),
            tree_entry("model-Q8_0.gguf", 8_000),
            tree_entry("model.gguf", 2_000),
        ];

        let options = complete_model_options(&entries);

        assert_eq!(
            options
                .iter()
                .map(|option| (option.quant.as_str(), option.bytes))
                .collect::<Vec<_>>(),
            vec![("Q4_K_M", 4_000), ("Q8_0", 8_000)]
        );
    }

    #[test]
    fn extract_quant_handles_dot_before_quant_label() {
        assert_eq!(extract_quant("nomic-embed-text-v1.5.Q4_K_M.gguf"), "Q4_K_M");
    }

    #[test]
    fn extract_quant_rejects_descriptive_q_tokens() {
        assert_eq!(extract_quant("model-Q4KExperts-Q8Out-chat.gguf"), "");
        assert_eq!(extract_quant("model-Q2KDown-chat.gguf"), "");
    }

    #[test]
    fn quant_selector_prefers_an_exact_quant() {
        let entries = [
            tree_entry("model-Q4_K_S.gguf", 3_000),
            tree_entry("model-Q4_K_M.gguf", 4_000),
        ];
        let candidates = entries.iter().collect::<Vec<_>>();

        let selected = select_quant_entry(&candidates, "q4_k_m", 0, "org/model").unwrap();

        assert_eq!(selected.path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn quant_selector_uses_q4_family_preference_order() {
        let entries = [
            tree_entry("model-Q4_0.gguf", 3_000),
            tree_entry("model-Q4_K_S.gguf", 3_500),
            tree_entry("model-Q4_K_M.gguf", 4_000),
        ];
        let candidates = entries.iter().collect::<Vec<_>>();

        let selected = select_quant_entry(&candidates, "Q4", 0, "org/model").unwrap();

        assert_eq!(selected.path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn quant_selector_maps_k_m_to_q4_k_m() {
        let entries = [
            tree_entry("model-Q3_K_M.gguf", 3_000),
            tree_entry("model-Q4_K_M.gguf", 4_000),
            tree_entry("model-Q5_K_M.gguf", 5_000),
        ];
        let candidates = entries.iter().collect::<Vec<_>>();

        let selected = select_quant_entry(&candidates, "K_M", 0, "org/model").unwrap();

        assert_eq!(selected.path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn quant_selector_auto_chooses_largest_quant_with_headroom() {
        let entries = [
            tree_entry("model-Q3_K_M.gguf", 3_000),
            tree_entry("model-Q4_K_M.gguf", 4_000),
            tree_entry("model-Q5_K_M.gguf", 5_000),
        ];
        let candidates = entries.iter().collect::<Vec<_>>();

        let selected = select_quant_entry(&candidates, "auto", 4_800, "org/model").unwrap();

        assert_eq!(selected.path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn quant_selector_auto_rejects_hardware_with_no_fitting_quant() {
        let entries = [tree_entry("model-Q2_K.gguf", 3_000)];
        let candidates = entries.iter().collect::<Vec<_>>();

        let error = select_quant_entry(&candidates, "auto", 3_599, "org/model").unwrap_err();

        assert!(error.contains("no quantization fits"));
    }

    #[test]
    fn quant_selector_auto_ignores_auxiliary_and_split_files() {
        let entries = [
            tree_entry("model-Q8_0-00001-of-00002.GGUF", 8_000),
            tree_entry("model-speculator-Q6_K.gguf", 6_000),
            tree_entry("model-Q4_K_M.gguf", 4_000),
        ];
        let candidates = entries.iter().collect::<Vec<_>>();

        let selected = select_quant_entry(&candidates, "auto", 9_600, "org/model").unwrap();

        assert_eq!(selected.path, "model-Q4_K_M.gguf");
    }

    #[test]
    fn quant_selector_auto_rejects_auxiliary_repository() {
        let entries = [tree_entry("model-Q4_K_M.gguf", 4_000)];
        let candidates = entries.iter().collect::<Vec<_>>();

        let error =
            select_quant_entry(&candidates, "auto", 4_800, "org/model-drafter-GGUF").unwrap_err();

        assert!(error.contains("auxiliary repository"));
    }

    #[test]
    fn complete_model_options_excludes_auxiliary_dspark_and_mtp_files() {
        let entries = [
            tree_entry("dspark-DeepSeek-V4-Flash-Q8_0.gguf", 10_000),
            tree_entry("DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf", 3_500),
            tree_entry("DeepSeek-V4-Flash-Q4_K_M.gguf", 80_000),
            tree_entry("DeepSeek-V4-Pro-Qwen3.5-9B-MTP-Q4_K_M.gguf", 5_400),
        ];

        let options = complete_model_options(&entries);

        assert_eq!(options.len(), 1);
        assert_eq!(options[0].quant, "Q4_K_M");
        assert_eq!(options[0].bytes, 5_400);
    }

    #[test]
    fn complete_model_options_groups_splits_and_rejects_incomplete_sets() {
        let entries = [
            tree_entry("model-Q5_K_M-00001-of-00002.gguf", 2_000),
            tree_entry("model-Q5_K_M-00002-of-00002.gguf", 3_000),
            tree_entry("model-Q6_K-00001-of-00002.gguf", 3_000),
        ];

        let options = complete_model_options(&entries);

        assert_eq!(options.len(), 1);
        assert_eq!(options[0].quant, "Q5_K_M");
        assert_eq!(options[0].bytes, 5_000);
    }

    #[test]
    fn complete_model_options_deduplicates_quant_to_smallest_size() {
        let entries = [
            tree_entry("a-Q4_K_M.gguf", 4_500),
            tree_entry("b-Q4_K_M.gguf", 4_000),
        ];

        let options = complete_model_options(&entries);

        assert_eq!(options.len(), 1);
        assert_eq!(options[0].quant, "Q4_K_M");
        assert_eq!(options[0].bytes, 4_000);
    }

    #[test]
    fn search_assessment_filters_capacity_and_prefers_q4_k_m() {
        let hit =
            serde_json::json!({"id": "org/model", "tags": [], "gguf": {"architecture": "llama"}});
        let entries = [
            tree_entry("model-Q3_K_M.gguf", 3_000),
            tree_entry("model-Q4_K_M.gguf", 4_000),
            tree_entry("model-Q8_0.gguf", 8_000),
        ];

        let assessment = assess_repository(&hit, &entries, 4_800, &HardwareCtx::default());

        assert!(assessment.supported);
        assert_eq!(
            assessment
                .quants
                .iter()
                .map(|option| option.quant.as_str())
                .collect::<Vec<_>>(),
            vec!["Q3_K_M", "Q4_K_M"]
        );
        assert_eq!(assessment.recommended_quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn search_assessment_falls_back_to_largest_fitting_quant() {
        let hit =
            serde_json::json!({"id": "org/model", "tags": [], "gguf": {"architecture": "llama"}});
        let entries = [
            tree_entry("model-Q3_K_M.gguf", 3_000),
            tree_entry("model-Q6_K.gguf", 6_000),
        ];

        let assessment = assess_repository(&hit, &entries, 7_200, &HardwareCtx::default());

        assert_eq!(assessment.recommended_quant.as_deref(), Some("Q6_K"));
    }

    #[test]
    fn search_assessment_excludes_auxiliary_repositories() {
        let hit = serde_json::json!({"id": "org/drafter", "tags": ["draft-model"], "gguf": {"architecture": "dflash"}});
        let entries = [tree_entry("model-Q4_K_M.gguf", 4_000)];

        let assessment = assess_repository(&hit, &entries, 4_800, &HardwareCtx::default());

        assert!(!assessment.supported);
        assert!(assessment.quants.is_empty());
        assert_eq!(assessment.recommended_quant, None);
    }

    #[test]
    fn support_estimate_reserves_twenty_percent_headroom() {
        assert!(is_supported(Some(100), 120));
        assert!(!is_supported(Some(100), 119));
    }

    #[test]
    fn support_estimate_rejects_unknown_size_and_accepts_cpu_capacity() {
        assert!(!is_supported(None, 1_000));
        assert!(is_supported(Some(800), 960));
    }

    #[test]
    fn search_table_renders_fit_score_speed_and_precision() {
        let hits = vec![
            serde_json::json!({
                "id": "org/gemma-small",
                "siblings": [{"rfilename": "gemma-q4.gguf"}],
                "gguf": {"total": 4_000_000_000_u64, "context_length": 8192, "architecture": "gemma"}
            }),
            serde_json::json!({"id": "org/gemma-large", "siblings": [], "gguf": {}}),
        ];

        let assessments = [
            SearchAssessment {
                supported: true,
                quants: vec![ModelOption {
                    quant: "Q4_K_M".to_string(),
                    bytes: 4_000,
                }],
                recommended_quant: Some("Q4_K_M".to_string()),
                fit_score: 95.0,
                fastest: Some(QuantRecommendation {
                    quant: "Q4_K_M".to_string(),
                    tokens_per_sec: 42.0,
                    quality_score: 96.7,
                }),
                best_precision: Some(QuantRecommendation {
                    quant: "Q4_K_M".to_string(),
                    tokens_per_sec: 42.0,
                    quality_score: 96.7,
                }),
                balanced: Some(QuantRecommendation {
                    quant: "Q4_K_M".to_string(),
                    tokens_per_sec: 42.0,
                    quality_score: 96.7,
                }),
            },
            SearchAssessment::default(),
        ];
        let table = render_search_table(&hits, &assessments);

        let header = table.lines().next().unwrap();
        assert!(header.contains("FIT"));
        assert!(header.contains("SPEED"));
        assert!(header.contains("BALANCED"));
        assert!(header.contains("PRECISION"));
        assert!(!header.contains("SUPPORTED"));
        assert!(table.lines().any(|line| line.contains("org/gemma-small")
            && line.contains("95")
            && line.contains("42tok/s")
            && line.contains("96.7%")));
        assert!(
            table
                .lines()
                .any(|line| { line.contains("org/gemma-large") && line.trim_end().ends_with('-') })
        );
    }

    #[test]
    fn search_output_renders_hardware_capacity_in_binary_gib() {
        assert_eq!(
            format_hardware_summary(65_536, 24_576),
            "Hardware: System RAM 64.0 GiB | NVIDIA VRAM 24.0 GiB | Total 88.0 GiB"
        );
    }

    #[test]
    fn search_output_aligns_long_repository_rows_and_places_quant_last() {
        let hits = vec![
            serde_json::json!({"id": "org/short", "siblings": [], "gguf": {"total": 4_000, "architecture": "llama"}}),
            serde_json::json!({"id": "org/a-very-long-repository-name-that-used-to-overflow", "siblings": [], "gguf": {"total": 8_000, "architecture": "llama"}}),
        ];
        let assessments = [
            SearchAssessment {
                supported: true,
                quants: vec![ModelOption {
                    quant: "Q4_K_M".to_string(),
                    bytes: 4_000,
                }],
                recommended_quant: Some("Q4_K_M".to_string()),
                fit_score: 95.0,
                fastest: Some(QuantRecommendation {
                    quant: "Q4_K_M".to_string(),
                    tokens_per_sec: 42.0,
                    quality_score: 96.7,
                }),
                best_precision: None,
                balanced: None,
            },
            SearchAssessment::default(),
        ];

        let table = render_search_table(&hits, &assessments);
        let lines = table.lines().collect::<Vec<_>>();
        let delimiters = |line: &str| {
            line.match_indices(" | ")
                .map(|(index, _)| index)
                .collect::<Vec<_>>()
        };

        assert_eq!(delimiters(lines[0]), delimiters(lines[1]));
        assert_eq!(delimiters(lines[0]), delimiters(lines[2]));
        assert!(lines[0].ends_with("QUANT"));
        assert!(lines[1].ends_with("Q4_K_M"));
        assert!(lines[2].ends_with('-'));
    }

    #[test]
    fn search_output_builds_speed_balanced_and_precision_pull_commands() {
        let hits = vec![serde_json::json!({"id": "org/model"})];
        let three_distinct_winners = [SearchAssessment {
            supported: true,
            quants: vec![],
            recommended_quant: Some("Q4_K_M".to_string()),
            fit_score: 90.0,
            fastest: Some(QuantRecommendation {
                quant: "Q3_K_M".to_string(),
                tokens_per_sec: 55.0,
                quality_score: 91.3,
            }),
            balanced: Some(QuantRecommendation {
                quant: "Q4_K_M".to_string(),
                tokens_per_sec: 42.0,
                quality_score: 96.7,
            }),
            best_precision: Some(QuantRecommendation {
                quant: "Q6_K".to_string(),
                tokens_per_sec: 28.0,
                quality_score: 99.6,
            }),
        }];

        let command = sample_pull_command(&hits, &three_distinct_winners).expect("command");
        assert!(command.contains("--quant Q3_K_M") && command.contains("fastest"));
        assert!(command.contains("--quant Q4_K_M") && command.contains("balanced"));
        assert!(command.contains("--quant Q6_K") && command.contains("least precision loss"));
        assert_eq!(command.lines().count(), 3);

        // Same quant wins all three categories: only one line, no duplication.
        let single_winner = [SearchAssessment {
            supported: true,
            quants: vec![],
            recommended_quant: Some("Q4_K_M".to_string()),
            fit_score: 90.0,
            fastest: Some(QuantRecommendation {
                quant: "Q4_K_M".to_string(),
                tokens_per_sec: 42.0,
                quality_score: 96.7,
            }),
            balanced: Some(QuantRecommendation {
                quant: "Q4_K_M".to_string(),
                tokens_per_sec: 42.0,
                quality_score: 96.7,
            }),
            best_precision: Some(QuantRecommendation {
                quant: "Q4_K_M".to_string(),
                tokens_per_sec: 42.0,
                quality_score: 96.7,
            }),
        }];
        let command = sample_pull_command(&hits, &single_winner).expect("command");
        assert_eq!(command.lines().count(), 1);

        assert_eq!(
            sample_pull_command(&hits, &[SearchAssessment::default()]),
            None
        );
    }

    #[test]
    fn balanced_quant_favors_normalized_middle_ground_over_extremes() {
        let fast_low_quality = ModelOption {
            quant: "Q2_K".to_string(),
            bytes: 2_000,
        };
        let mid = ModelOption {
            quant: "Q4_K_M".to_string(),
            bytes: 4_000,
        };
        let slow_high_quality = ModelOption {
            quant: "Q8_0".to_string(),
            bytes: 8_000,
        };
        let scored = vec![
            (
                &fast_low_quality,
                quant_profile::SpeedEstimate {
                    tokens_per_sec: 100.0,
                    mode: quant_profile::SpeedMode::FullGpu,
                    gpu_name: None,
                    bandwidth_gbps: 0.0,
                    efficiency_factor: 0.0,
                    confidence: quant_profile::Confidence::Measured,
                },
                quant_profile::quant_quality("Q2_K"),
            ),
            (
                &mid,
                quant_profile::SpeedEstimate {
                    tokens_per_sec: 50.0,
                    mode: quant_profile::SpeedMode::FullGpu,
                    gpu_name: None,
                    bandwidth_gbps: 0.0,
                    efficiency_factor: 0.0,
                    confidence: quant_profile::Confidence::Measured,
                },
                quant_profile::quant_quality("Q4_K_M"),
            ),
            (
                &slow_high_quality,
                quant_profile::SpeedEstimate {
                    tokens_per_sec: 25.0,
                    mode: quant_profile::SpeedMode::FullGpu,
                    gpu_name: None,
                    bandwidth_gbps: 0.0,
                    efficiency_factor: 0.0,
                    confidence: quant_profile::Confidence::Measured,
                },
                quant_profile::quant_quality("Q8_0"),
            ),
        ];

        let balanced = balanced_quant(&scored).expect("a balanced pick");
        assert_eq!(balanced.quant, "Q4_K_M");
    }

    #[test]
    fn balanced_quant_empty_input_is_none() {
        assert!(balanced_quant(&[]).is_none());
    }

    #[test]
    fn connections_accept_supported_range() {
        assert_eq!(parse_connections("1"), Ok(1));
        assert_eq!(parse_connections("8"), Ok(8));
        assert_eq!(parse_connections("16"), Ok(16));
    }

    #[test]
    fn connections_reject_invalid_values() {
        assert!(parse_connections("0").is_err());
        assert!(parse_connections("17").is_err());
        assert!(parse_connections("fast").is_err());
    }

    #[test]
    fn refresh_url_normalizes_unspecified_bind_address() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.toml");
        std::fs::write(&config, "bind = \"0.0.0.0:9090\"\n").unwrap();

        assert_eq!(
            refresh_url_from_config(&config).unwrap(),
            "http://127.0.0.1:9090/v1/models/refresh"
        );
        assert_eq!(
            chat_url_from_config(&config).unwrap(),
            "http://127.0.0.1:9090/v1/chat/completions"
        );
    }

    #[test]
    fn extract_speed_prefers_llama_timings() {
        let body = serde_json::json!({
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "timings": {
                "prompt_n": 12,
                "prompt_ms": 30.0,
                "prompt_per_second": 400.0,
                "predicted_n": 64,
                "predicted_ms": 1600.0,
                "predicted_per_second": 40.0
            }
        });
        let stats = extract_speed_stats(&body, 10.0).expect("stats");
        assert!(stats.from_timings);
        assert_eq!(stats.prompt_tokens, 12);
        assert_eq!(stats.gen_tokens, 64);
        assert!((stats.prompt_tps - 400.0).abs() < f64::EPSILON);
        assert!((stats.gen_tps - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_speed_falls_back_to_usage_over_wall_clock() {
        let body = serde_json::json!({
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        });
        let stats = extract_speed_stats(&body, 2.0).expect("stats");
        assert!(!stats.from_timings);
        assert_eq!(stats.prompt_tokens, 100);
        assert_eq!(stats.gen_tokens, 50);
        assert!((stats.prompt_tps - 50.0).abs() < f64::EPSILON);
        assert!((stats.gen_tps - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn extract_speed_computes_from_timing_ms_when_per_second_missing() {
        let body = serde_json::json!({
            "timings": {
                "prompt_n": 100,
                "prompt_ms": 250.0,
                "predicted_n": 50,
                "predicted_ms": 1000.0
            }
        });
        let stats = extract_speed_stats(&body, 9.0).expect("stats");
        assert!(stats.from_timings);
        assert!((stats.prompt_tps - 400.0).abs() < 0.01);
        assert!((stats.gen_tps - 50.0).abs() < 0.01);
    }
}
