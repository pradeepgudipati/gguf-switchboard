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

const HF_MODELS_API: &str = "https://huggingface.co/api/models";

// ── models search ────────────────────────────────────────────────────────────

/// `gguf-switchboard models search <query> [--limit N]`
pub async fn cmd_search(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut query: Option<String> = None;
    let mut limit: u32 = 10;

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

    let estimates = stream::iter(hits.iter().enumerate().map(|(index, hit)| {
        let client = client.clone();
        let repo = hit.get("id").and_then(Value::as_str).map(str::to_string);
        async move {
            let assessment = match repo {
                Some(repo) => match hf_download::fetch_repo_tree(&client, &repo).await {
                    Ok(entries) => assess_repository(hit, &entries, capacity_bytes),
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
    println!();
    print!("{}", render_search_table(&hits, &assessments));
    println!(
        "Supported requires a standalone GGUF and is estimated from total RAM + NVIDIA VRAM with 20% runtime headroom."
    );
    if let Some(command) = sample_pull_command(&hits, &assessments) {
        println!("{command}");
    }

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
        let quant = assessment.recommended_quant.as_deref()?;
        Some(format!("Try: ggs models pull {repository} --quant {quant}"))
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
                .map(|v| format!("{} tok", v))
                .unwrap_or_default();
            let arch = hit
                .get("gguf")
                .and_then(|g| g.get("architecture"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let assessment = assessments.get(index);
            let support = if assessment.is_some_and(|value| value.supported) {
                "Yes".to_string()
            } else {
                "No".to_string()
            };
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
            (id, siblings, size_mb, support, context, arch, quants)
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
                _ => &row.5,
            };
            width.max(value.len())
        })
    };
    let repo_width = width("REPO", 0);
    let files_width = width("FILES", 1);
    let size_width = width("SIZE", 2);
    let support_width = width("SUPPORTED", 3);
    let context_width = width("CONTEXT", 4);
    let arch_width = width("ARCH", 5);

    let mut output = String::new();
    writeln!(
        output,
        "{:<repo_width$} | {:>files_width$} | {:>size_width$} | {:<support_width$} | {:<context_width$} | {:<arch_width$} | QUANT",
        "REPO", "FILES", "SIZE", "SUPPORTED", "CONTEXT", "ARCH"
    )
    .expect("writing to a String cannot fail");
    for (repo, files, size, support, context, arch, quants) in rows {
        writeln!(
            output,
            "{repo:<repo_width$} | {files:>files_width$} | {size:>size_width$} | {support:<support_width$} | {context:<context_width$} | {arch:<arch_width$} | {quants}"
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
        println!("  {:<44} {:<12} QUANT", "FILENAME", "SIZE");
        for entry in &models {
            let size = format_bytes(entry.size);
            let quant = extract_quant(&entry.path);
            println!("  {:<44} {:<12} {}", entry.path, size, quant);
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
pub async fn cmd_pull(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
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
    match refresh_running_server(&client, Path::new("config.toml")).await {
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
    if kind.eq_ignore_ascii_case("embedding") {
        eprintln!("Speed test skipped: embedding models have no chat generation");
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
    let config_path = Path::new("config.toml");
    let url = chat_url_from_config(config_path).map_err(|e| e.to_string())?;
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct SearchAssessment {
    supported: bool,
    quants: Vec<ModelOption>,
    recommended_quant: Option<String>,
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
) -> SearchAssessment {
    if !is_standalone_model(hit, entries) {
        return SearchAssessment::default();
    }

    let quants = complete_model_options(entries)
        .into_iter()
        .filter(|option| is_supported(Some(option.bytes), capacity_bytes))
        .collect::<Vec<_>>();
    let recommended_quant = quants
        .iter()
        .find(|option| option.quant == "Q4_K_M")
        .or_else(|| quants.last())
        .map(|option| option.quant.clone());

    SearchAssessment {
        supported: !quants.is_empty(),
        quants,
        recommended_quant,
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
    if lower.contains("embed") {
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
        ModelOption, SearchAssessment, assess_repository, chat_url_from_config,
        complete_model_options, extract_quant, extract_speed_stats, format_hardware_summary,
        is_standalone_model, is_supported, parse_connections, refresh_url_from_config,
        render_search_table, sample_pull_command, select_quant_entry,
    };
    use crate::config::hf_download::HfTreeEntry;

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

        let assessment = assess_repository(&hit, &entries, 4_800);

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

        let assessment = assess_repository(&hit, &entries, 7_200);

        assert_eq!(assessment.recommended_quant.as_deref(), Some("Q6_K"));
    }

    #[test]
    fn search_assessment_excludes_auxiliary_repositories() {
        let hit = serde_json::json!({"id": "org/drafter", "tags": ["draft-model"], "gguf": {"architecture": "dflash"}});
        let entries = [tree_entry("model-Q4_K_M.gguf", 4_000)];

        let assessment = assess_repository(&hit, &entries, 4_800);

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
    fn search_table_renders_supported_header_and_values() {
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
            },
            SearchAssessment::default(),
        ];
        let table = render_search_table(&hits, &assessments);

        assert!(table.lines().next().unwrap().contains("SUPPORTED"));
        assert!(
            table
                .lines()
                .any(|line| line.contains("org/gemma-small") && line.contains("Yes"))
        );
        assert!(
            table
                .lines()
                .any(|line| line.contains("org/gemma-large") && line.contains("No"))
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
    fn search_output_builds_q4_pull_command_and_omits_empty_recommendation() {
        let hits = vec![serde_json::json!({"id": "org/model"})];
        let supported = [SearchAssessment {
            supported: true,
            quants: vec![],
            recommended_quant: Some("Q4_K_M".to_string()),
        }];

        assert_eq!(
            sample_pull_command(&hits, &supported).as_deref(),
            Some("Try: ggs models pull org/model --quant Q4_K_M")
        );
        assert_eq!(
            sample_pull_command(&hits, &[SearchAssessment::default()]),
            None
        );
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
