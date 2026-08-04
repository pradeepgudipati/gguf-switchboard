//! CLI handlers for `gguf-switchboard models {search,files,pull}`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

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

    println!("  {:<44} {:<6} {:<10} ARCH", "REPO", "FILES", "CONTEXT");
    for hit in &hits {
        let id = hit.get("id").and_then(|v| v.as_str()).unwrap_or("?");
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
            .unwrap_or(0);
        let context = hit
            .get("gguf")
            .and_then(|g| g.get("context_length"))
            .and_then(|v| v.as_u64())
            .map(|v| v.to_string())
            .unwrap_or_default();
        let arch = hit
            .get("gguf")
            .and_then(|g| g.get("architecture"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!("  {id:<44} {siblings:<6} {context:<10} {arch}");
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

/// `gguf-switchboard models pull <repo-id> [--quant Q4_K_M] [--dir /path]`
pub async fn cmd_pull(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut repo: Option<String> = None;
    let mut quant: Option<String> = None;
    let mut dest_override: Option<String> = None;
    let mut models_file: Option<String> = None;

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
            let matches: Vec<&&HfTreeEntry> = model_entries
                .iter()
                .filter(|e| extract_quant(&e.path).to_uppercase().contains(q))
                .collect();
            match matches.len() {
                0 => {
                    println!("No file matches quant \"{q}\". Available:");
                    for entry in &model_entries {
                        println!("  {} ({})", entry.path, extract_quant(&entry.path));
                    }
                    return Err("models pull: no matching quantization".into());
                }
                1 => matches[0],
                _ => {
                    // Multiple matches — prefer exact quant string.
                    let exact: Vec<&&HfTreeEntry> = matches
                        .iter()
                        .filter(|e| extract_quant(&e.path).to_uppercase() == *q)
                        .copied()
                        .collect();
                    if exact.len() == 1 {
                        exact[0]
                    } else {
                        println!("Multiple files match \"{q}\":");
                        for entry in &matches {
                            println!("  {} ({})", entry.path, extract_quant(&entry.path));
                        }
                        return Err(
                            "models pull: ambiguous quantization; use a more specific --quant"
                                .into(),
                        );
                    }
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
    println!("Destination: {}", dest_dir.join(&selected.path).display());
    let downloaded = hf_download::download_file(&client, &repo, &selected.path, &dest_dir).await?;

    // 5. Validate GGUF.
    match validate_gguf_model(&downloaded) {
        Ok(_) => println!("✓ GGUF metadata validated"),
        Err(reason) => {
            return Err(format!("Downloaded file is not a valid GGUF model: {reason}").into());
        }
    }

    // 6. Generate alias and register.
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

    // Check if file is already registered.
    let already_registered = registry.models.iter().any(|e| e.file == file_ref);
    if already_registered {
        println!("✓ Already registered as: {alias}");
        println!("✓ Available through /v1/models");
        return Ok(());
    }

    let used_aliases: HashSet<String> = registry.models.iter().map(|e| e.alias.clone()).collect();
    let alias = dedupe_alias(&alias, &used_aliases);

    let entry = RegistryEntry {
        alias: alias.clone(),
        file: file_ref,
        display_name: Some(display_name_from_alias(&alias)),
        kind: Some(infer_kind_from_filename(&selected.path)),
        enabled: true,
        hf_repo: Some(repo.clone()),
        min_vram_gb: Some(((selected.size as f64 / 1_000_000_000.0).ceil() as u32).max(1)),
        ..Default::default()
    };
    registry.models.push(entry);

    registry.write(&registry_path)?;
    println!("✓ Registered as: {alias}");
    println!("✓ Run `POST /v1/models/refresh` or restart the server to load");

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

/// Extract a quantization label from a GGUF filename.
fn extract_quant(filename: &str) -> String {
    let stem = Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename);

    // Look for common quant patterns: Q4_K_M, Q5_K_S, IQ4_NL, BF16, etc.
    let parts: Vec<&str> = stem.split('-').collect();
    for part in parts.iter().rev() {
        let upper = part.to_ascii_uppercase();
        if upper.starts_with('Q') && upper.len() >= 3 {
            return upper;
        }
        if upper.starts_with("IQ") && upper.len() >= 4 {
            return upper;
        }
        if matches!(upper.as_str(), "BF16" | "FP16" | "FP32" | "F16" | "F32") {
            return upper;
        }
    }
    String::new()
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
