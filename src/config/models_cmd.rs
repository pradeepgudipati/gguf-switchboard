//! CLI handlers for `gguf-switchboard models {search,files,pull,check-updates,update}`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::io::IsTerminal as _;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};

use futures::{StreamExt, stream};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use super::hf_download::{self, HfTreeEntry, format_bytes};
use super::models_local::{resolve_registry, resolve_scan_dirs, resolve_target, scan_local_models};
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

    let client = hf_download::build_hf_client().await?;
    let hits = search_hf_models_with_siblings(&client, &query, limit).await?;

    if hits.is_empty() {
        println!("No GGUF models found for \"{query}\"");
        return Ok(());
    }
    // `org/model` queries mean the org's own releases — drop third-party
    // repacks (e.g. `Qwen/Qwen3` hides `bartowski/*`). Applied before the
    // per-repo tree fetches below so hidden repos cost no network.
    let (hits, org_note) = filter_hits_by_org(&query, hits);

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
    if let Some(note) = org_note {
        println!("{note}");
        println!();
    }
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

/// When the query names an org (`org/model`), keep only that org's repos so
/// third-party repacks don't drown out first-party releases.
///
/// Returns the filtered hits plus a display note (None when nothing was
/// filtered). When no repo of that org matches, the original hits are
/// returned with a note instead of an empty table — the query may still be
/// useful as a fuzzy search.
fn filter_hits_by_org(query: &str, hits: Vec<Value>) -> (Vec<Value>, Option<String>) {
    let Some(slash) = query.find('/') else {
        return (hits, None);
    };
    let org = query[..slash].trim();
    if org.is_empty() || org.contains(char::is_whitespace) {
        return (hits, None);
    }
    let prefix = format!("{}/", org.to_ascii_lowercase());
    let total = hits.len();
    let (kept, _dropped): (Vec<Value>, Vec<Value>) = hits.into_iter().partition(|hit| {
        hit.get("id")
            .and_then(Value::as_str)
            .is_some_and(|id| id.to_ascii_lowercase().starts_with(&prefix))
    });
    match (kept.len(), total) {
        (0, _) => (
            _dropped,
            Some(format!(
                "No '{org}/*' repos matched \"{query}\"; showing closest matches across all orgs."
            )),
        ),
        (k, t) if k < t => (
            kept,
            Some(format!(
                "Filtered to '{org}/*' ({k} of {t} repos shown); search without the org prefix to see all publishers."
            )),
        ),
        _ => (kept, None),
    }
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

    let client = hf_download::build_hf_client().await?;
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

    // `org/model` queries mean the org's own releases — same rule as the
    // GGUF path, applied before the per-repo tree fetches below.
    let (hits, org_note) = filter_hits_by_org(&query, hits);
    if let Some(note) = &org_note {
        println!("{note}");
        println!();
    }

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
            // Binary verdict from the original hardware-support spec: Yes iff
            // the repo is standalone-eligible AND at least one named quant fits
            // capacity (SearchAssessment.supported). FIT alongside is the
            // continuous score for the same data.
            let supported = assessment
                .map(|value| {
                    if value.supported {
                        "Yes".to_string()
                    } else {
                        "No".to_string()
                    }
                })
                .unwrap_or_else(|| "-".to_string());
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
                id, siblings, size_mb, supported, fit, context, arch, speed, balanced, precision,
                quants,
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
                8 => &row.8,
                _ => &row.9,
            };
            width.max(value.len())
        })
    };
    let repo_width = width("REPO", 0);
    let files_width = width("FILES", 1);
    let size_width = width("SIZE", 2);
    let supported_width = width("SUPPORTED", 3);
    let fit_width = width("FIT", 4);
    let context_width = width("CONTEXT", 5);
    let arch_width = width("ARCH", 6);
    let speed_width = width("SPEED", 7);
    let balanced_width = width("BALANCED", 8);
    let precision_width = width("PRECISION", 9);

    let mut output = String::new();
    writeln!(
        output,
        "{:<repo_width$} | {:>files_width$} | {:>size_width$} | {:<supported_width$} | {:>fit_width$} | {:<context_width$} | {:<arch_width$} | {:<speed_width$} | {:<balanced_width$} | {:<precision_width$} | QUANT",
        "REPO", "FILES", "SIZE", "SUPPORTED", "FIT", "CONTEXT", "ARCH", "SPEED", "BALANCED", "PRECISION"
    )
    .expect("writing to a String cannot fail");
    for (repo, files, size, supported, fit, context, arch, speed, balanced, precision, quants) in
        rows
    {
        writeln!(
            output,
            "{repo:<repo_width$} | {files:>files_width$} | {size:>size_width$} | {supported:<supported_width$} | {fit:>fit_width$} | {context:<context_width$} | {arch:<arch_width$} | {speed:<speed_width$} | {balanced:<balanced_width$} | {precision:<precision_width$} | {quants}"
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

    let client = hf_download::build_hf_client().await?;
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
    let client = hf_download::build_hf_client().await?;

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

    // Resolve the same way `models pull vllm` does: an explicit --registry wins,
    // then the canonical deployed registry whenever it exists, then a
    // models.toml beside the download dir, then config.toml's `models_file`.
    // `ggs` is commonly run from a source checkout while the running server
    // reads /opt/gguf-switchboard.
    let config_path = resolve_config_toml_path();
    let dest_dir_str = dest_dir.to_string_lossy().into_owned();
    let registry_path = resolve_vllm_registry_path(
        models_file.as_deref(),
        Some(dest_dir_str.as_str()),
        config_path.as_path(),
        Path::new("/opt/gguf-switchboard/models.toml"),
    );
    warn_on_stray_registries(&registry_path);

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
            "✓ Merged GGUF source into existing alias: {alias} (registry: {registry_path}) \
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
    println!("✓ Registered as: {alias} (registry: {registry_path})");
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

    // 1. Resolve the deployed registry before checking vLLM. `ggs` is commonly
    // invoked from the source checkout, while deploy keeps the live registry in
    // /opt/gguf-switchboard and points to it from config.toml.
    let config_path = resolve_config_toml_path();
    let registry_path = resolve_vllm_registry_path(
        models_file.as_deref(),
        dest_override.as_deref(),
        config_path.as_path(),
        Path::new("/opt/gguf-switchboard/models.toml"),
    );
    let existing_registry = if Path::new(&registry_path).is_file() {
        Some(ModelsRegistry::load(&registry_path)?)
    } else {
        None
    };
    let (vllm_command, vllm_project) = if let Some(existing) = existing_registry.as_ref() {
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
    let base_dest = match (&dest_override, existing_registry.as_ref()) {
        (Some(dir), _) => PathBuf::from(dir),
        (None, Some(existing)) => {
            let gguf_dir = existing
                .defaults
                .models_dir
                .split(',')
                .next()
                .unwrap_or(existing.defaults.models_dir.as_str())
                .trim();
            Path::new(gguf_dir)
                .parent()
                .map(|parent| parent.join("vllm-models"))
                .unwrap_or_else(|| PathBuf::from("vllm-models"))
        }
        (None, None) => resolve_default_vllm_models_dir()?,
    };
    let dest_dir = base_dest.join(sanitize_repo_dirname(&repo));
    let alias = alias_from_repo(&repo);
    let mut registry = if let Some(existing) = existing_registry {
        existing
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
    let client = hf_download::build_hf_client().await?;
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
            "✓ Merged vLLM source into existing alias: {alias} (registry: {registry_path}) \
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
    println!("✓ Registered as: {alias} (vLLM source, registry: {registry_path})");

    // refresh_after_pull() already prints a "start or restart" hint on failure.
    refresh_after_pull().await;

    Ok(())
}

// ── models check-updates / update ────────────────────────────────────────────

/// Upstream comparison result for one on-disk model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateStatus {
    UpToDate,
    UpdateAvailable,
    Renamed,
    NewFiles,
    MissingRemote,
    SkippedNoRepo,
    Error,
}

impl UpdateStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::UpToDate => "up-to-date",
            Self::UpdateAvailable => "update-available",
            Self::Renamed => "renamed",
            Self::NewFiles => "new-files",
            Self::MissingRemote => "missing-remote",
            Self::SkippedNoRepo => "skipped-no-repo",
            Self::Error => "error",
        }
    }
}

struct UpdateRow {
    index: usize,
    alias: String,
    path: PathBuf,
    local_bytes: u64,
    remote_bytes: Option<u64>,
    status: UpdateStatus,
    detail: String,
    repo: Option<String>,
    remote_file: Option<String>,
    is_safetensors_dir: bool,
}

struct CheckUpdatesOptions {
    dir_overrides: Vec<String>,
    registry_path: Option<String>,
    json: bool,
    verify: bool,
    target: Option<String>,
}

fn parse_check_updates_args(args: &[String]) -> Result<CheckUpdatesOptions, String> {
    let mut out = CheckUpdatesOptions {
        dir_overrides: Vec::new(),
        registry_path: None,
        json: false,
        verify: false,
        target: None,
    };
    let mut i = 1; // args[0] is "check-updates"
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "models check-updates: missing value for --dir".to_string())?;
                out.dir_overrides.push(v.clone());
                i += 2;
            }
            "--registry" => {
                let v = args.get(i + 1).ok_or_else(|| {
                    "models check-updates: missing value for --registry".to_string()
                })?;
                out.registry_path = Some(v.clone());
                i += 2;
            }
            "--json" => {
                out.json = true;
                i += 1;
            }
            "--verify" => {
                out.verify = true;
                i += 1;
            }
            flag if flag.starts_with('-') => {
                return Err(format!("models check-updates: unknown flag '{flag}'"));
            }
            other => {
                if out.target.is_some() {
                    return Err("models check-updates: expected at most one target".to_string());
                }
                out.target = Some(other.to_string());
                i += 1;
            }
        }
    }
    Ok(out)
}

/// Normalize an HF LFS oid (`sha256:<hex>` or bare hex) for comparison.
fn normalize_oid(oid: &str) -> String {
    oid.trim()
        .strip_prefix("sha256:")
        .unwrap_or_else(|| oid.trim())
        .trim()
        .to_ascii_lowercase()
}

/// Stream a local file through SHA-256 without loading it into memory.
fn sha256_hex_file(path: &Path) -> Result<String, String> {
    use std::io::Read as _;
    let file =
        std::fs::File::open(path).map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
    let mut reader = std::io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buf)
            .map_err(|e| format!("cannot hash '{}': {e}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>())
}

fn remote_oid(entry: &HfTreeEntry) -> Option<String> {
    entry
        .lfs
        .as_ref()
        .and_then(|lfs| lfs.oid.as_deref())
        .map(normalize_oid)
        .filter(|oid| !oid.is_empty())
}

/// Classify one GGUF file against its remote tree entry.
///
/// Size mismatch is always an update. When sizes match, only `--verify`
/// (local SHA-256 vs the remote LFS oid) can catch a same-size content
/// change; without it the file reports up-to-date.
fn classify_gguf_update(
    local_bytes: u64,
    local_path: &Path,
    remote: Option<&HfTreeEntry>,
    verify: bool,
) -> (UpdateStatus, Option<u64>, String) {
    let Some(remote) = remote else {
        return (
            UpdateStatus::MissingRemote,
            None,
            "local file no longer exists upstream".to_string(),
        );
    };
    if remote.size != local_bytes {
        let diff = remote.size as i64 - local_bytes as i64;
        return (
            UpdateStatus::UpdateAvailable,
            Some(remote.size),
            format!(
                "size changed: {} ({} B) -> {} ({} B), diff {:+} B",
                format_bytes(local_bytes),
                local_bytes,
                format_bytes(remote.size),
                remote.size,
                diff
            ),
        );
    }
    if verify {
        match (remote_oid(remote), sha256_hex_file(local_path)) {
            (Some(expected), Ok(actual)) if expected != actual => {
                return (
                    UpdateStatus::UpdateAvailable,
                    Some(remote.size),
                    "content hash changed (same size)".to_string(),
                );
            }
            (None, _) => {
                return (
                    UpdateStatus::UpToDate,
                    Some(remote.size),
                    "same size (no upstream checksum to verify)".to_string(),
                );
            }
            (Some(_), Err(reason)) => {
                return (UpdateStatus::Error, Some(remote.size), reason);
            }
            _ => {}
        }
    }
    (
        UpdateStatus::UpToDate,
        Some(remote.size),
        "matches upstream".to_string(),
    )
}

fn gguf_basename(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Upstream files that could be the renamed successor of a local GGUF: same
/// quantization label, main model files only (no mmproj/lora/projector or
/// auxiliary dspark/mtp shards).
fn find_renamed_matches<'a>(local_name: &str, tree: &'a [HfTreeEntry]) -> Vec<&'a HfTreeEntry> {
    let quant = extract_quant(local_name);
    if quant.is_empty() {
        return Vec::new();
    }
    tree.iter()
        .filter(|e| is_model_gguf(&e.path) && !is_auxiliary_model_file(&e.path))
        .filter(|e| extract_quant(&gguf_basename(Path::new(&e.path))) == quant)
        .collect()
}

/// `ggs models check-updates [--dir PATH] [--registry FILE] [--json] [--verify] [target]`
pub async fn cmd_check_updates(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let opts =
        parse_check_updates_args(args).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let (rows, registry_path) = collect_update_rows(
        &opts.dir_overrides,
        opts.registry_path.as_deref(),
        opts.target.as_deref(),
        opts.verify,
    )
    .await?;

    if opts.json {
        let arr: Vec<_> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "index": r.index,
                    "alias": r.alias,
                    "path": r.path.to_string_lossy(),
                    "local_bytes": r.local_bytes,
                    "remote_bytes": r.remote_bytes,
                    "status": r.status.as_str(),
                    "detail": r.detail,
                    "repo": r.repo,
                    "remote_file": r.remote_file,
                    "safetensors_dir": r.is_safetensors_dir,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }

    if rows.is_empty() {
        println!("No GGUF files or safetensors model directories found.");
        return Ok(());
    }
    match &registry_path {
        Some(path) => println!("Registry: {path}"),
        None => println!("Registry: (none found — every entry shows as unregistered)"),
    }

    println!();
    println!(
        "  {:>3}  {:<28}  {:<16}  {:>10}  {:>10}  DETAIL",
        "#", "ALIAS / NAME", "STATUS", "LOCAL", "REMOTE"
    );
    for r in &rows {
        let remote = r
            .remote_bytes
            .map(format_bytes)
            .unwrap_or_else(|| "-".to_string());
        println!(
            "  {:>3}  {:<28}  {:<16}  {:>10}  {:>10}  {}",
            r.index,
            truncate_update_name(&r.alias, 28),
            r.status.as_str(),
            format_bytes(r.local_bytes),
            remote,
            r.detail
        );
    }
    println!();
    let updates = rows
        .iter()
        .filter(|r| r.status == UpdateStatus::UpdateAvailable || r.status == UpdateStatus::Renamed)
        .count();
    let up_to_date = rows
        .iter()
        .filter(|r| r.status == UpdateStatus::UpToDate || r.status == UpdateStatus::NewFiles)
        .count();
    let skipped = rows
        .iter()
        .filter(|r| {
            r.status == UpdateStatus::SkippedNoRepo
                || r.status == UpdateStatus::MissingRemote
                || r.status == UpdateStatus::Error
        })
        .count();
    println!("{updates} update(s) available, {up_to_date} up-to-date, {skipped} skipped/error.");
    if updates > 0 {
        println!("Update with:  ggs models update <#|name|all> [--yes]");
    }
    Ok(())
}

fn truncate_update_name(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

/// Shared scan + upstream comparison used by `check-updates` and `update ... all`.
/// Returns the rows plus the registry path they were resolved against (for display).
async fn collect_update_rows(
    dir_overrides: &[String],
    registry_path: Option<&str>,
    target: Option<&str>,
    verify: bool,
) -> Result<(Vec<UpdateRow>, Option<String>), Box<dyn std::error::Error>> {
    let registry = resolve_registry(registry_path);
    let resolved_registry_path = registry.as_ref().map(|(p, _)| p.clone());
    let dirs = resolve_scan_dirs(dir_overrides, registry.as_ref().map(|(_, r)| r))
        .map_err(|e| e.replace("models list:", "models check-updates:"))?;
    let entries = scan_local_models(&dirs, registry.as_ref().map(|(_, r)| r));

    let wanted: Vec<super::models_local::LocalModelEntry> = match target {
        Some(t) => vec![
            resolve_target(t, &entries)
                .map_err(|e| e.replace("models delete", "models check-updates"))?
                .clone(),
        ],
        None => entries,
    };

    // Alias -> (hf_repo, vllm_hf_repo).
    let mut repos: HashMap<String, (Option<String>, Option<String>)> = HashMap::new();
    if let Some((_, ref reg)) = registry {
        for e in &reg.models {
            repos.insert(e.alias.clone(), (e.hf_repo.clone(), e.vllm_hf_repo.clone()));
        }
    }

    // Distinct upstream repos to fetch once.
    let mut fetch_jobs: Vec<(String, bool)> = Vec::new(); // (repo, full_tree)
    for entry in &wanted {
        let (hf_repo, vllm_repo) = entry
            .alias
            .as_deref()
            .and_then(|a| repos.get(a))
            .map(|(h, v)| (h.clone(), v.clone()))
            .unwrap_or((None, None));
        if entry.is_safetensors_dir {
            if let Some(repo) = vllm_repo
                && !fetch_jobs.iter().any(|(r, full)| r == &repo && *full)
            {
                fetch_jobs.push((repo, true));
            }
        } else if let Some(repo) = hf_repo
            && !fetch_jobs.iter().any(|(r, _)| r == &repo)
        {
            fetch_jobs.push((repo, false));
        }
    }

    let client = hf_download::build_hf_client().await?;
    let trees: HashMap<String, Result<Vec<HfTreeEntry>, String>> =
        stream::iter(fetch_jobs.into_iter().map(|(repo, full)| {
            let client = client.clone();
            async move {
                let result = if full {
                    hf_download::fetch_repo_tree_all(&client, &repo).await
                } else {
                    hf_download::fetch_repo_tree(&client, &repo).await
                }
                .map_err(|e| e.to_string());
                (repo, result)
            }
        }))
        .buffer_unordered(8)
        .collect()
        .await;

    // Local basenames per repo, for new-file detection.
    let mut local_names: HashSet<String> = HashSet::new();
    for e in &wanted {
        if !e.is_safetensors_dir {
            local_names.insert(gguf_basename(&e.path).to_ascii_lowercase());
        }
    }

    let mut rows = Vec::with_capacity(wanted.len());
    for entry in wanted {
        let alias = entry
            .alias
            .clone()
            .unwrap_or_else(|| "(unregistered)".to_string());
        let (hf_repo, vllm_repo) = entry
            .alias
            .as_deref()
            .and_then(|a| repos.get(a))
            .map(|(h, v)| (h.clone(), v.clone()))
            .unwrap_or((None, None));

        if entry.is_safetensors_dir {
            let Some(repo) = vllm_repo else {
                rows.push(UpdateRow {
                    index: entry.index,
                    alias,
                    path: entry.path.clone(),
                    local_bytes: entry.size_bytes,
                    remote_bytes: None,
                    status: UpdateStatus::SkippedNoRepo,
                    detail: "no vllm_hf_repo recorded; cannot check".to_string(),
                    repo: None,
                    remote_file: None,
                    is_safetensors_dir: true,
                });
                continue;
            };
            match trees.get(&repo) {
                Some(Ok(tree)) => {
                    let serving: Vec<&HfTreeEntry> = tree
                        .iter()
                        .filter(|e| super::vllm_meta::is_vllm_serving_file(&e.path))
                        .collect();
                    let mut changed = Vec::new();
                    let mut total_remote: u64 = 0;
                    for remote in &serving {
                        total_remote += remote.size;
                        let name = gguf_basename(Path::new(&remote.path));
                        let local = entry.path.join(&name);
                        let local_size = std::fs::metadata(&local).map(|m| m.len());
                        match local_size {
                            Ok(size) if size == remote.size => {
                                if verify
                                    && let Some(expected) = remote_oid(remote)
                                    && let Ok(actual) = sha256_hex_file(&local)
                                    && expected != actual
                                {
                                    changed.push(format!("{name} (hash changed)"));
                                }
                            }
                            Ok(size) => changed.push(format!(
                                "{name} ({} ({} B) -> {} ({} B))",
                                format_bytes(size),
                                size,
                                format_bytes(remote.size),
                                remote.size
                            )),
                            Err(_) => changed.push(format!("{name} (missing locally)")),
                        }
                    }
                    if changed.is_empty() {
                        rows.push(UpdateRow {
                            index: entry.index,
                            alias,
                            path: entry.path.clone(),
                            local_bytes: entry.size_bytes,
                            remote_bytes: Some(total_remote),
                            status: UpdateStatus::UpToDate,
                            detail: format!("{} file(s) match upstream", serving.len()),
                            repo: Some(repo),
                            remote_file: None,
                            is_safetensors_dir: true,
                        });
                    } else {
                        rows.push(UpdateRow {
                            index: entry.index,
                            alias,
                            path: entry.path.clone(),
                            local_bytes: entry.size_bytes,
                            remote_bytes: Some(total_remote),
                            status: UpdateStatus::UpdateAvailable,
                            detail: format!("{} changed: {}", changed.len(), changed.join(", ")),
                            repo: Some(repo),
                            remote_file: None,
                            is_safetensors_dir: true,
                        });
                    }
                }
                Some(Err(reason)) => rows.push(UpdateRow {
                    index: entry.index,
                    alias,
                    path: entry.path.clone(),
                    local_bytes: entry.size_bytes,
                    remote_bytes: None,
                    status: UpdateStatus::Error,
                    detail: reason.clone(),
                    repo: Some(repo),
                    remote_file: None,
                    is_safetensors_dir: true,
                }),
                None => rows.push(UpdateRow {
                    index: entry.index,
                    alias,
                    path: entry.path.clone(),
                    local_bytes: entry.size_bytes,
                    remote_bytes: None,
                    status: UpdateStatus::Error,
                    detail: "upstream fetch missing".to_string(),
                    repo: Some(repo),
                    remote_file: None,
                    is_safetensors_dir: true,
                }),
            }
            continue;
        }

        let Some(repo) = hf_repo else {
            rows.push(UpdateRow {
                index: entry.index,
                alias,
                path: entry.path.clone(),
                local_bytes: entry.size_bytes,
                remote_bytes: None,
                status: UpdateStatus::SkippedNoRepo,
                detail: "no hf_repo recorded; cannot check".to_string(),
                repo: None,
                remote_file: None,
                is_safetensors_dir: false,
            });
            continue;
        };
        let name = gguf_basename(&entry.path);
        let tree = trees.get(&repo).and_then(|r| r.as_ref().ok());
        let remote = tree.and_then(|tree| {
            tree.iter()
                .find(|e| gguf_basename(Path::new(&e.path)).eq_ignore_ascii_case(&name))
        });
        // Renamed upstream? Fall back to the unique same-quant file.
        let (remote, renamed_from) = match (remote, tree) {
            (Some(r), _) => (Some(r), None),
            (None, Some(tree)) => match find_renamed_matches(&name, tree).as_slice() {
                [one] => (Some(*one), Some(name.clone())),
                _ => (None, None),
            },
            (None, None) => (None, None),
        };
        match trees.get(&repo) {
            Some(Err(reason)) => rows.push(UpdateRow {
                index: entry.index,
                alias,
                path: entry.path.clone(),
                local_bytes: entry.size_bytes,
                remote_bytes: None,
                status: UpdateStatus::Error,
                detail: reason.clone(),
                repo: Some(repo),
                remote_file: Some(name),
                is_safetensors_dir: false,
            }),
            _ => {
                let (status, remote_bytes, mut detail) =
                    classify_gguf_update(entry.size_bytes, &entry.path, remote, verify);
                let mut remote_name = name.clone();
                if let (Some(old), Some(matched)) = (renamed_from, remote) {
                    let new = gguf_basename(Path::new(&matched.path));
                    remote_name = new.clone();
                    let size_note = match remote_bytes {
                        Some(rb) if rb != entry.size_bytes => format!(
                            "; size {} ({} B) -> {} ({} B)",
                            format_bytes(entry.size_bytes),
                            entry.size_bytes,
                            format_bytes(rb),
                            rb
                        ),
                        _ => "; same size".to_string(),
                    };
                    detail = format!(
                        "renamed upstream: '{old}' -> '{new}' (same {quant}{size_note})",
                        quant = extract_quant(&old),
                    );
                    rows.push(UpdateRow {
                        index: entry.index,
                        alias,
                        path: entry.path.clone(),
                        local_bytes: entry.size_bytes,
                        remote_bytes,
                        status: UpdateStatus::Renamed,
                        detail,
                        repo: Some(repo),
                        remote_file: Some(remote_name),
                        is_safetensors_dir: false,
                    });
                    continue;
                }
                if status == UpdateStatus::MissingRemote {
                    detail = match tree {
                        Some(tree) => {
                            let mut cands: Vec<String> = tree
                                .iter()
                                .filter(|e| super::models_local::is_model_gguf_public(&e.path))
                                .take(5)
                                .map(|e| gguf_basename(Path::new(&e.path)))
                                .collect();
                            if cands.is_empty() {
                                format!(
                                    "local file '{name}' not in upstream repo '{repo}' and no same-quant match; repo has no model files"
                                )
                            } else {
                                cands.sort();
                                format!(
                                    "local file '{name}' not in upstream repo '{repo}' and no unique same-quant match; upstream has: {} (see `ggs models files {repo}`)",
                                    cands.join(", ")
                                )
                            }
                        }
                        None => format!(
                            "local file '{name}' not in upstream repo '{repo}' (renamed or removed); see `ggs models files {repo}`"
                        ),
                    };
                }
                // New-file hint: same file matches but the repo ships other
                // quants not present locally.
                if status == UpdateStatus::UpToDate
                    && let Some(Ok(tree)) = trees.get(&repo)
                {
                    let others = tree
                        .iter()
                        .filter(|e| {
                            super::models_local::is_model_gguf_public(&e.path)
                                && !local_names.contains(
                                    &gguf_basename(Path::new(&e.path)).to_ascii_lowercase(),
                                )
                        })
                        .count();
                    if others > 0 {
                        detail =
                            format!("matches upstream; {others} other file(s) in repo not on disk");
                        rows.push(UpdateRow {
                            index: entry.index,
                            alias,
                            path: entry.path.clone(),
                            local_bytes: entry.size_bytes,
                            remote_bytes,
                            status: UpdateStatus::NewFiles,
                            detail,
                            repo: Some(repo),
                            remote_file: Some(name),
                            is_safetensors_dir: false,
                        });
                        continue;
                    }
                }
                rows.push(UpdateRow {
                    index: entry.index,
                    alias,
                    path: entry.path.clone(),
                    local_bytes: entry.size_bytes,
                    remote_bytes,
                    status,
                    detail,
                    repo: Some(repo),
                    remote_file: Some(remote_name),
                    is_safetensors_dir: false,
                });
            }
        }
    }
    rows.sort_by_key(|r| r.index);
    Ok((rows, resolved_registry_path))
}

struct UpdateOptions {
    dir_overrides: Vec<String>,
    registry_path: Option<String>,
    connections: u16,
    no_bench: bool,
    yes: bool,
    target: String,
}

fn parse_update_args(args: &[String]) -> Result<UpdateOptions, String> {
    let mut out = UpdateOptions {
        dir_overrides: Vec::new(),
        registry_path: None,
        connections: 8,
        no_bench: false,
        yes: false,
        target: String::new(),
    };
    let mut i = 1; // args[0] is "update"
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "models update: missing value for --dir".to_string())?;
                out.dir_overrides.push(v.clone());
                i += 2;
            }
            "--registry" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "models update: missing value for --registry".to_string())?;
                out.registry_path = Some(v.clone());
                i += 2;
            }
            "--connections" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| "models update: missing value for --connections".to_string())?;
                out.connections = parse_connections(v).map_err(str::to_string)?;
                i += 2;
            }
            "--no-bench" => {
                out.no_bench = true;
                i += 1;
            }
            "--yes" | "-y" => {
                out.yes = true;
                i += 1;
            }
            flag if flag.starts_with('-') => {
                return Err(format!("models update: unknown flag '{flag}'"));
            }
            other => {
                if !out.target.is_empty() {
                    return Err("models update: expected exactly one target".to_string());
                }
                out.target = other.to_string();
                i += 1;
            }
        }
    }
    if out.target.is_empty() {
        return Err(
            "models update: missing target — pass a name, the number from `ggs models list`, or `all`"
                .to_string(),
        );
    }
    Ok(out)
}

fn confirm_update(count: usize, yes: bool) -> Result<bool, String> {
    if yes {
        return Ok(true);
    }
    if !std::io::stdin().is_terminal() {
        return Err(
            "models update: refusing to replace files without --yes in a non-interactive shell"
                .to_string(),
        );
    }
    let noun = if count == 1 {
        "this file"
    } else {
        "these files"
    };
    print!("Replace {noun} with the upstream version? [y/N] ");
    std::io::Write::flush(&mut std::io::stdout()).map_err(|e| e.to_string())?;
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .map_err(|e| e.to_string())?;
    Ok(matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

/// `ggs models update <name|#|all> [--yes] [--connections N] [--no-bench]
/// [--dir PATH] [--registry FILE]`
pub async fn cmd_update(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let opts = parse_update_args(args).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    let update_all = opts.target.eq_ignore_ascii_case("all");

    // Resolve the target set. `all` reuses the check-updates comparison so a
    // single invocation never mixes stale decisions across repos.
    let targets: Vec<UpdateRow> = if update_all {
        collect_update_rows(
            &opts.dir_overrides,
            opts.registry_path.as_deref(),
            None,
            false,
        )
        .await?
        .0
        .into_iter()
        .filter(|r| r.status == UpdateStatus::UpdateAvailable || r.status == UpdateStatus::Renamed)
        .collect()
    } else {
        collect_update_rows(
            &opts.dir_overrides,
            opts.registry_path.as_deref(),
            Some(opts.target.as_str()),
            false,
        )
        .await?
        .0
    };

    if targets.is_empty() {
        if update_all {
            println!("All models are up to date.");
            return Ok(());
        }
        return Err("models update: no matching model found".into());
    }
    if update_all {
        println!("Update available for {} model(s):", targets.len());
        for t in &targets {
            println!("  #{} {} ({})", t.index, t.alias, t.detail);
        }
    } else {
        let t = &targets[0];
        if t.status == UpdateStatus::SkippedNoRepo {
            return Err(format!(
                "models update: '{}' has no upstream repo recorded (hf_repo/vllm_hf_repo); cannot update",
                t.alias
            )
            .into());
        }
        if t.status == UpdateStatus::MissingRemote {
            return Err(format!("models update: '{}': {}", t.alias, t.detail).into());
        }
        if t.status == UpdateStatus::Error {
            return Err(format!("models update: upstream check failed: {}", t.detail).into());
        }
        if t.status == UpdateStatus::UpToDate || t.status == UpdateStatus::NewFiles {
            println!("'{}' is already up to date ({}).", t.alias, t.detail);
            return Ok(());
        }
    }

    if !confirm_update(targets.len(), opts.yes)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?
    {
        println!("Aborted; nothing was updated.");
        return Ok(());
    }

    let client = hf_download::build_hf_client().await?;
    let mut updated = 0usize;
    for t in &targets {
        let repo = t
            .repo
            .clone()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("models update: '{}' has no upstream repo recorded", t.alias).into()
            })?;
        if t.is_safetensors_dir {
            update_vllm_dir(
                &client,
                t,
                &repo,
                opts.connections,
                opts.registry_path.as_deref(),
            )
            .await?;
        } else {
            update_gguf_file(
                &client,
                t,
                &repo,
                opts.connections,
                opts.registry_path.as_deref(),
            )
            .await?;
        }
        updated += 1;
        // Refresh + bench for single GGUF targets, like `models pull`.
        if !update_all && !t.is_safetensors_dir {
            let kind = infer_kind_from_filename(t.remote_file.as_deref().unwrap_or_default());
            let refreshed = refresh_after_pull().await;
            maybe_bench_after_pull(&t.alias, &kind, opts.no_bench, refreshed).await;
        }
    }
    if update_all {
        refresh_after_pull().await;
    }
    println!("Updated {updated} model(s).");
    Ok(())
}

/// Replace one GGUF file with its upstream entry, atomically.
///
/// Downloads into a staging directory on the same filesystem, validates the
/// GGUF header, then renames into place — the old file stays intact until
/// the new one is known good. When upstream renamed the file (check-updates
/// `renamed`), the download lands under the new name, the registry `file`
/// field follows it, and the old file is removed.
async fn update_gguf_file(
    client: &reqwest::Client,
    target: &UpdateRow,
    repo: &str,
    connections: u16,
    registry_path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = hf_download::fetch_repo_tree(client, repo).await?;
    let name = target
        .remote_file
        .clone()
        .unwrap_or_else(|| gguf_basename(&target.path));
    let remote = entries
        .iter()
        .find(|e| gguf_basename(Path::new(&e.path)).eq_ignore_ascii_case(&name))
        .ok_or_else(|| format!("'{repo}' no longer contains '{name}'; not updating"))?;
    let parent = target
        .path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let staging = parent.join(format!(".ggs-update-{}", std::process::id()));
    std::fs::create_dir_all(&staging)?;
    let staged = staging.join(&name);
    // A stale staged file from an aborted run must not be resumed into.
    let _ = std::fs::remove_file(&staged);
    let downloaded =
        hf_download::download_file_auto(client, repo, remote, &staging, connections).await?;
    if let Err(reason) = validate_gguf_model(&downloaded) {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(format!("Downloaded file is not a valid GGUF model: {reason}").into());
    }
    let local_name = gguf_basename(&target.path);
    let renamed = !name.eq_ignore_ascii_case(&local_name);
    // Persist the file move first so a registry-write failure never loses data.
    if renamed {
        std::fs::rename(&downloaded, parent.join(&name))?;
    } else {
        std::fs::rename(&downloaded, &target.path)?;
    }
    let _ = std::fs::remove_dir_all(&staging);

    let final_path = parent.join(&name);
    let new_bytes = std::fs::metadata(&final_path)
        .map(|m| m.len())
        .unwrap_or(remote.size);
    if renamed {
        // Old file is superseded only now that the new one is in place.
        let _ = std::fs::remove_file(&target.path);
        println!(
            "✓ Updated {}: renamed '{}' -> '{}' ({} -> {})",
            target.alias,
            local_name,
            name,
            format_bytes(target.local_bytes),
            format_bytes(new_bytes)
        );
    } else {
        println!(
            "✓ Updated {}: {} -> {}",
            target.alias,
            format_bytes(target.local_bytes),
            format_bytes(new_bytes)
        );
    }
    update_registry_after_gguf(&target.alias, &local_name, &name, new_bytes, registry_path);
    Ok(())
}

/// Re-download the changed shards of one safetensors directory.
async fn update_vllm_dir(
    client: &reqwest::Client,
    target: &UpdateRow,
    repo: &str,
    connections: u16,
    registry_path: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = hf_download::fetch_repo_tree_all(client, repo).await?;
    let serving: Vec<&HfTreeEntry> = entries
        .iter()
        .filter(|e| super::vllm_meta::is_vllm_serving_file(&e.path))
        .collect();
    if serving.is_empty() {
        return Err(format!("'{repo}' has no servable safetensors files; not updating").into());
    }
    let mut refreshed = 0usize;
    for remote in serving {
        let name = gguf_basename(Path::new(&remote.path));
        let local = target.path.join(&name);
        let stale = match std::fs::metadata(&local).map(|m| m.len()) {
            Ok(size) if size != remote.size => true,
            Ok(_) => match (remote_oid(remote), sha256_hex_file(&local)) {
                (Some(expected), Ok(actual)) => expected != actual,
                _ => false,
            },
            Err(_) => true,
        };
        if !stale {
            continue;
        }
        // Remove first: the native downloader resumes partial files, which
        // would corrupt a same-size content change or a shrunk shard.
        let _ = std::fs::remove_file(&local);
        hf_download::download_file_auto(client, repo, remote, &target.path, connections).await?;
        refreshed += 1;
    }
    println!("✓ Updated {}: {refreshed} file(s) refreshed", target.alias);
    let weights_mb: u64 = std::fs::read_dir(&target.path)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.ends_with(".safetensors") {
                e.metadata().ok().map(|m| m.len())
            } else {
                None
            }
        })
        .sum::<u64>()
        / (1024 * 1024);
    let new_gb = ((weights_mb as f64 / 1024.0).ceil() as u32).max(1);
    update_registry_vram(&target.alias, new_gb, registry_path);
    Ok(())
}

/// Best-effort registry update after a GGUF replace or rename. Follows the
/// `file` field to the new basename on renames so future checks keep
/// matching. Warns instead of failing the update when the registry cannot
/// be read or written.
fn update_registry_after_gguf(
    alias: &str,
    old_basename: &str,
    new_basename: &str,
    new_bytes: u64,
    registry_path: Option<&str>,
) {
    let Some((path, mut registry)) = resolve_registry(registry_path) else {
        return;
    };
    let old_key = old_basename.to_ascii_lowercase();
    let new_gb = ((new_bytes as f64 / 1_000_000_000.0).ceil() as u32).max(1);
    let mut changed = false;
    for entry in &mut registry.models {
        if !entry.alias.eq_ignore_ascii_case(alias) {
            continue;
        }
        let registered = Path::new(&entry.file)
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if registered != old_key && !entry.file.is_empty() {
            continue;
        }
        if entry.file != *new_basename && !new_basename.eq_ignore_ascii_case(&registered) {
            // Adopt the upstream filename (bare or relative form preserved).
            entry.file = if entry.file.contains('/') || entry.file.contains('\\') {
                let p = Path::new(&entry.file);
                match p.parent() {
                    Some(dir) if !dir.as_os_str().is_empty() => {
                        dir.join(new_basename).to_string_lossy().into_owned()
                    }
                    _ => new_basename.to_string(),
                }
            } else {
                new_basename.to_string()
            };
            changed = true;
        }
        if entry.min_vram_gb != Some(new_gb) {
            entry.min_vram_gb = Some(new_gb);
            changed = true;
        }
    }
    if changed && let Err(e) = registry.write(&path) {
        eprintln!("Warning: updated file but could not write registry '{path}': {e}");
    }
}

/// Best-effort vLLM VRAM update after a shard refresh.
fn update_registry_vram(alias: &str, new_gb: u32, registry_path: Option<&str>) {
    let Some((path, mut registry)) = resolve_registry(registry_path) else {
        return;
    };
    let mut changed = false;
    for entry in &mut registry.models {
        if entry.alias.eq_ignore_ascii_case(alias) && entry.min_vram_gb != Some(new_gb) {
            entry.min_vram_gb = Some(new_gb);
            changed = true;
        }
    }
    if changed && let Err(e) = registry.write(&path) {
        eprintln!("Warning: updated files but could not write registry '{path}': {e}");
    }
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
pub(crate) async fn refresh_after_pull() -> bool {
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

/// Find the default models directory for `models pull` downloads.
///
/// Order: the canonical deployed registry first (whenever it exists, so
/// pulls from a source checkout land where the running server reads them —
/// mirroring registry resolution), then `./models.toml`, then
/// `$MODELS_DIR`, then `~/models`.
fn resolve_default_models_dir() -> Result<PathBuf, RuntimeError> {
    resolve_default_models_dir_inner(Path::new("/opt/gguf-switchboard/models.toml"))
}

fn resolve_default_models_dir_inner(deployed_registry: &Path) -> Result<PathBuf, RuntimeError> {
    if deployed_registry.is_file()
        && let Ok(registry) = ModelsRegistry::load(&deployed_registry.to_string_lossy())
        && let Ok(dirs) = super::models_registry::resolve_models_dirs(&registry.defaults.models_dir)
        && let Some(dir) = dirs.first()
    {
        return Ok(dir.clone());
    }

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
/// The registry path the *running server* actually reads: `models_file` from
/// config.toml (resolved relative to the config when not absolute), falling back
/// to the canonical deployed registry.
fn server_registry_path(config_path: &Path) -> PathBuf {
    if let Ok(content) = std::fs::read_to_string(config_path)
        && let Ok(config) = toml::from_str::<toml::Value>(&content)
        && let Some(path) = config.get("models_file").and_then(toml::Value::as_str)
    {
        let path = PathBuf::from(path);
        return if path.is_absolute() {
            path
        } else {
            config_path.parent().unwrap_or(Path::new(".")).join(path)
        };
    }
    PathBuf::from("/opt/gguf-switchboard/models.toml")
}

/// The config the running service actually reads: the deployed one when this
/// machine has a system install, else the same lookup `models pull` uses.
/// (Resolving `./config.toml` from a source checkout is what produced the
/// misleading "server reads 'models.toml'" warning.)
fn resolve_server_config_path() -> PathBuf {
    let deployed = PathBuf::from("/opt/gguf-switchboard/config.toml");
    if deployed.is_file() {
        return deployed;
    }
    resolve_config_toml_path()
}
fn warn_on_stray_registries(resolved: &str) {
    let canonical = server_registry_path(&resolve_server_config_path());
    let resolved_abs = std::fs::canonicalize(resolved).unwrap_or_else(|_| PathBuf::from(resolved));
    let canonical_abs = std::fs::canonicalize(&canonical).unwrap_or_else(|_| canonical.clone());

    if resolved_abs != canonical_abs {
        eprintln!(
            "⚠  Writing registry '{resolved}', but the running server reads '{}'.\n   \
             This model will NOT appear until you consolidate. Re-run from the deploy\n   \
             directory, or pass --registry '{}'.",
            canonical.display(),
            canonical.display(),
        );
    }

    let mut strays: Vec<String> = Vec::new();
    let mut candidates: Vec<PathBuf> = vec![PathBuf::from("models.toml")];
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(PathBuf::from(&home).join("gguf-switchboard/models.toml"));
        candidates.push(PathBuf::from(&home).join("Projects/gguf-switchboard/models.toml"));
        candidates.push(PathBuf::from(&home).join("models/models.toml"));
    }
    for cand in candidates {
        let Ok(abs) = std::fs::canonicalize(&cand) else {
            continue;
        };
        if abs != canonical_abs
            && abs != resolved_abs
            && !strays.contains(&abs.display().to_string())
        {
            strays.push(abs.display().to_string());
        }
    }
    if !strays.is_empty() {
        eprintln!(
            "⚠  Other stray models.toml files found (not used by the server):\n{}",
            strays
                .iter()
                .map(|s| format!("     {s}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        eprintln!(
            "   Consolidate with: sudo -u ggs ggs discover-models <models-dir> \
             -o {} --merge <stray>",
            canonical.display()
        );
    }
}

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

/// Registry resolution for `models pull` (GGUF and vLLM) writes.
/// Order (option C): an explicit `--registry` always wins; otherwise the
/// canonical deployed registry wins whenever it exists, so pulls from a
/// source checkout land where the running server reads them. A
/// `models.toml` beside the download dir is next (fresh `--dir` pulls
/// keep a local registry until deployed), then `./models.toml` (local
/// dev), then config.toml's `models_file`.
fn resolve_vllm_registry_path(
    explicit: Option<&str>,
    destination: Option<&str>,
    config_path: &Path,
    deployed_registry: &Path,
) -> String {
    if let Some(path) = explicit {
        return path.to_string();
    }
    // Exists but not a readable file (permissions, broken symlink): saying so
    // now beats silently writing pulls into a fallback registry that the
    // server never reads.
    if deployed_registry.symlink_metadata().is_ok() && !deployed_registry.is_file() {
        eprintln!(
            "warning: deployed registry '{}' exists but is not readable; falling back — check group/permissions before pulling",
            deployed_registry.display()
        );
    } else if deployed_registry.is_file() {
        return deployed_registry.to_string_lossy().into_owned();
    }
    if let Some(destination) = destination {
        let candidate = Path::new(destination).join("models.toml");
        if candidate.is_file() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    if Path::new("models.toml").is_file() {
        return "models.toml".to_string();
    }
    if let Ok(content) = std::fs::read_to_string(config_path)
        && let Ok(config) = toml::from_str::<toml::Value>(&content)
        && let Some(path) = config.get("models_file").and_then(toml::Value::as_str)
    {
        let path = PathBuf::from(path);
        let resolved = if path.is_absolute() {
            path
        } else {
            config_path.parent().unwrap_or(Path::new(".")).join(path)
        };
        return resolved.to_string_lossy().into_owned();
    }
    "models.toml".to_string()
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
pub(crate) fn alias_from_repo(repo: &str) -> String {
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
        ModelOption, QuantRecommendation, SearchAssessment, UpdateStatus, assess_repository,
        balanced_quant, chat_url_from_config, classify_gguf_update, complete_model_options,
        extract_quant, extract_speed_stats, filter_hits_by_org, find_renamed_matches,
        format_hardware_summary, is_standalone_model, is_supported, normalize_oid,
        parse_check_updates_args, parse_connections, parse_update_args, refresh_url_from_config,
        render_search_table, resolve_vllm_registry_path, sample_pull_command, select_quant_entry,
        sha256_hex_file,
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
    fn vllm_registry_resolution_uses_deployed_config_when_cwd_registry_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let registry = dir.path().join("models.toml");
        std::fs::write(&registry, "version = 1\n").unwrap();
        let config = dir.path().join("config.toml");
        // Forward slashes: a raw Windows path in a TOML basic string turns
        // `\U`/`\A` into invalid escapes and breaks parsing on Windows.
        let registry_forward = registry.display().to_string().replace('\\', "/");
        std::fs::write(&config, format!("models_file = \"{registry_forward}\"\n")).unwrap();

        assert_eq!(
            resolve_vllm_registry_path(None, None, &config, std::path::Path::new("missing.toml"),)
                .replace('\\', "/"),
            registry_forward
        );
    }

    #[test]
    fn vllm_registry_resolution_preserves_dir_sibling_registry() {
        let dir = tempfile::tempdir().unwrap();
        let registry = dir.path().join("models.toml");
        std::fs::write(&registry, "version = 1\n").unwrap();

        assert_eq!(
            resolve_vllm_registry_path(
                None,
                dir.path().to_str(),
                std::path::Path::new("missing.toml"),
                std::path::Path::new("missing-deployed.toml"),
            ),
            registry.to_string_lossy()
        );
    }

    #[test]
    fn vllm_registry_resolution_uses_canonical_deployed_registry_before_source_config() {
        let dir = tempfile::tempdir().unwrap();
        let deployed_registry = dir.path().join("models.toml");
        std::fs::write(&deployed_registry, "version = 1\n").unwrap();
        let source_config = dir.path().join("source-config.toml");
        std::fs::write(&source_config, "bind = \"127.0.0.1:9090\"\n").unwrap();

        assert_eq!(
            resolve_vllm_registry_path(None, None, &source_config, &deployed_registry),
            deployed_registry.to_string_lossy()
        );
    }

    #[test]
    fn vllm_registry_resolution_prefers_deployed_over_dir_sibling_and_cwd() {
        // Option C: once the deployed registry exists, pulls from a source
        // checkout land where the server reads — no new stray registries.
        let dir = tempfile::tempdir().unwrap();
        let deployed_registry = dir.path().join("deployed-models.toml");
        std::fs::write(&deployed_registry, "version = 1\n").unwrap();
        let sibling_dir = dir.path().join("downloads");
        std::fs::create_dir_all(&sibling_dir).unwrap();
        std::fs::write(sibling_dir.join("models.toml"), "version = 1\n").unwrap();
        let source_config = dir.path().join("source-config.toml");
        std::fs::write(&source_config, "bind = \"127.0.0.1:9090\"\n").unwrap();

        assert_eq!(
            resolve_vllm_registry_path(
                None,
                sibling_dir.to_str(),
                &source_config,
                &deployed_registry,
            ),
            deployed_registry.to_string_lossy()
        );
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
        assert!(header.contains("SUPPORTED"));
        assert!(header.contains("FIT"));
        assert!(header.contains("SPEED"));
        assert!(header.contains("BALANCED"));
        assert!(header.contains("PRECISION"));
        assert!(table.lines().any(|line| line.contains("org/gemma-small")
            && line.contains("Yes")
            && line.contains("95")
            && line.contains("42tok/s")
            && line.contains("96.7%")));
        assert!(table.lines().any(|line| line.contains("org/gemma-large")
            && line.contains("No")
            && line.trim_end().ends_with('-')));
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

    #[test]
    fn normalize_oid_strips_sha256_prefix_and_lowercases() {
        assert_eq!(normalize_oid("sha256:ABC123"), "abc123");
        assert_eq!(normalize_oid("ABC123"), "abc123");
        assert_eq!(normalize_oid("  sha256:def  "), "def");
    }

    #[test]
    fn sha256_hex_file_hashes_without_loading() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, b"gguf-update-check").unwrap();
        let hex = sha256_hex_file(&path).expect("hash");
        assert_eq!(hex.len(), 64);
        // Same content hashes the same; different content differs.
        let other = dir.path().join("other.bin");
        std::fs::write(&other, b"gguf-update-check!").unwrap();
        assert_ne!(hex, sha256_hex_file(&other).expect("hash"));
        assert!(sha256_hex_file(&dir.path().join("missing.bin")).is_err());
    }

    #[test]
    fn classify_gguf_update_detects_size_changes() {
        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join("model-Q4_K_M.gguf");
        std::fs::write(&local, vec![0u8; 1024]).unwrap();

        let smaller = tree_entry("model-Q4_K_M.gguf", 512);
        let (status, remote, detail) = classify_gguf_update(1024, &local, Some(&smaller), false);
        assert_eq!(status, UpdateStatus::UpdateAvailable);
        assert_eq!(remote, Some(512));
        assert!(detail.contains("(1024 B) -> 512 B (512 B)"), "{detail}");
        assert!(detail.contains("diff -512 B"), "{detail}");

        let same = tree_entry("model-Q4_K_M.gguf", 1024);
        let (status, _, _) = classify_gguf_update(1024, &local, Some(&same), false);
        assert_eq!(status, UpdateStatus::UpToDate);

        let (status, remote, _) = classify_gguf_update(1024, &local, None, false);
        assert_eq!(status, UpdateStatus::MissingRemote);
        assert_eq!(remote, None);
    }

    #[test]
    fn classify_gguf_update_verify_catches_same_size_content_change() {
        use crate::config::hf_download::HfLfsMeta;

        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join("model-Q4_K_M.gguf");
        std::fs::write(&local, b"local-bytes!").unwrap();
        let actual = sha256_hex_file(&local).expect("hash");

        // Wrong oid with the same size is an update only under --verify.
        let changed = HfTreeEntry {
            r#type: "file".to_string(),
            path: "model-Q4_K_M.gguf".to_string(),
            size: 12,
            lfs: Some(HfLfsMeta {
                oid: Some(format!("sha256:{}", "0".repeat(64))),
                size: Some(12),
            }),
        };
        let (status, _, _) = classify_gguf_update(12, &local, Some(&changed), false);
        assert_eq!(status, UpdateStatus::UpToDate);
        let (status, _, detail) = classify_gguf_update(12, &local, Some(&changed), true);
        assert_eq!(status, UpdateStatus::UpdateAvailable);
        assert!(detail.contains("hash"), "{detail}");

        // Matching oid stays up-to-date even under --verify.
        let same = HfTreeEntry {
            lfs: Some(HfLfsMeta {
                oid: Some(format!("sha256:{actual}")),
                size: Some(12),
            }),
            ..changed
        };
        let (status, _, _) = classify_gguf_update(12, &local, Some(&same), true);
        assert_eq!(status, UpdateStatus::UpToDate);
    }

    #[test]
    fn check_updates_args_parse_flags_and_target() {
        let args = vec![
            "check-updates".to_string(),
            "--verify".to_string(),
            "--json".to_string(),
        ];
        let opts = parse_check_updates_args(&args).expect("parse");
        assert!(opts.verify && opts.json && opts.target.is_none());

        let args = vec!["check-updates".to_string(), "2".to_string()];
        let opts = parse_check_updates_args(&args).expect("parse");
        assert_eq!(opts.target.as_deref(), Some("2"));

        let args = vec!["check-updates".to_string(), "--bogus".to_string()];
        assert!(parse_check_updates_args(&args).is_err());
    }

    #[test]
    fn find_renamed_matches_unique_same_quant() {
        let tree = [
            tree_entry("Org-Model-Q4_K_M.gguf", 4_000),
            tree_entry("Org-Model-Q8_0.gguf", 8_000),
            tree_entry("Org-Model-mmproj.gguf", 100),
        ];
        // Unique same-quant match despite a different name prefix.
        let one = find_renamed_matches("Old-Name-Q8_0.gguf", &tree);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].path, "Org-Model-Q8_0.gguf");

        // Ambiguous when two files share the quant.
        let tree2 = [
            tree_entry("A-Q4_K_M.gguf", 4_000),
            tree_entry("B-Q4_K_M.gguf", 4_100),
        ];
        assert_eq!(find_renamed_matches("Old-Q4_K_M.gguf", &tree2).len(), 2);

        // No quant in the local name means no match.
        assert!(find_renamed_matches("model.gguf", &tree).is_empty());
        // Auxiliary files never match.
        assert!(find_renamed_matches("old-mmproj.gguf", &tree).is_empty());
    }

    #[test]
    fn org_filter_keeps_only_that_org() {
        let hits = vec![
            serde_json::json!({"id": "Qwen/Qwen3-8B-GGUF"}),
            serde_json::json!({"id": "bartowski/Qwen_Qwen3.5-9B-GGUF"}),
            serde_json::json!({"id": "Qwen/Qwen3-0.6B-GGUF"}),
        ];
        let (kept, note) = filter_hits_by_org("Qwen/Qwen3", hits);
        assert_eq!(kept.len(), 2);
        assert!(
            kept.iter()
                .all(|h| h["id"].as_str().is_some_and(|id| id.starts_with("Qwen/")))
        );
        let note = note.expect("filtering dropped repos, note expected");
        assert!(note.contains("2 of 3"), "{note}");

        // Org match is case-insensitive.
        let hits = vec![serde_json::json!({"id": "QWEN/Qwen3-8B-GGUF"})];
        let (kept, _) = filter_hits_by_org("qwen/qwen3", hits);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn org_filter_falls_back_when_org_has_no_repos() {
        let hits = vec![serde_json::json!({"id": "bartowski/Qwen_Qwen3.5-9B-GGUF"})];
        let (kept, note) = filter_hits_by_org("Qwen/Qwen3", hits);
        assert_eq!(kept.len(), 1);
        let note = note.expect("fallback note expected");
        assert!(note.contains("No 'Qwen/*' repos matched"), "{note}");
    }

    #[test]
    fn org_filter_ignores_plain_and_malformed_queries() {
        let hits = vec![
            serde_json::json!({"id": "Qwen/Qwen3-8B-GGUF"}),
            serde_json::json!({"id": "bartowski/Qwen_Qwen3.5-9B-GGUF"}),
        ];
        // No slash: no filtering.
        let (kept, note) = filter_hits_by_org("Qwen3", hits.clone());
        assert_eq!(kept.len(), 2);
        assert!(note.is_none());
        // Empty org or whitespace in org: no filtering.
        for query in ["/Qwen3", "foo bar/baz", " /x"] {
            let (kept, note) = filter_hits_by_org(query, hits.clone());
            assert_eq!(kept.len(), 2, "{query}");
            assert!(note.is_none(), "{query}");
        }
        // Everything already the org's: no note needed.
        let hits = vec![serde_json::json!({"id": "Qwen/Qwen3-8B-GGUF"})];
        let (kept, note) = filter_hits_by_org("Qwen/Qwen3", hits);
        assert_eq!(kept.len(), 1);
        assert!(note.is_none());
    }

    #[test]
    fn update_args_require_exactly_one_target() {
        let args = vec!["update".to_string(), "--yes".to_string()];
        assert!(parse_update_args(&args).is_err());

        let args = vec!["update".to_string(), "a".to_string(), "b".to_string()];
        assert!(parse_update_args(&args).is_err());

        let args = vec![
            "update".to_string(),
            "all".to_string(),
            "--yes".to_string(),
            "--connections".to_string(),
            "4".to_string(),
        ];
        let opts = parse_update_args(&args).expect("parse");
        assert_eq!(opts.target, "all");
        assert!(opts.yes && opts.connections == 4);
    }
}
