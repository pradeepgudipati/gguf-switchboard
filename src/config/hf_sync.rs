//! Enrich `models.toml` entries from the Hugging Face Hub GGUF catalog.
//!
//! Offline/CLI only — never called on the request path.

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use super::models_registry::{ModelsRegistry, RegistryEntry, resolve_model_path};
use crate::errors::RuntimeError;

const HF_MODELS_API: &str = "https://huggingface.co/api/models";

#[derive(Debug, Clone)]
pub struct HfEnrichment {
    pub hf_repo: String,
    pub description: Option<String>,
    pub kind: Option<String>,
    pub max_context_length: Option<u32>,
    pub min_vram_gb: Option<u32>,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HfModelHit {
    id: String,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    gguf: Option<HfGgufMeta>,
    #[serde(default)]
    card_data: Option<Value>,
    #[serde(default, rename = "cardData")]
    card_data_camel: Option<Value>,
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
struct HfGgufMeta {
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default, rename = "totalFileSize")]
    total_file_size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    #[serde(default)]
    rfilename: String,
}

/// Map a single HF API model object (JSON) into enrichment fields.
pub fn enrichment_from_hf_json(value: &Value, local_filename: &str) -> Option<HfEnrichment> {
    let hit: HfModelHit = serde_json::from_value(value.clone()).ok()?;
    Some(enrichment_from_hit(&hit, local_filename))
}

fn enrichment_from_hit(hit: &HfModelHit, local_filename: &str) -> HfEnrichment {
    let card = hit.card_data.as_ref().or(hit.card_data_camel.as_ref());
    let description = card
        .and_then(|c| c.get("description"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let size_bytes = hit
        .gguf
        .as_ref()
        .and_then(|g| g.total.or(g.total_file_size))
        .or_else(|| {
            // Prefer matching sibling size is not in list API; fall back to gguf totals only.
            let _ = local_filename;
            None
        });

    let min_vram_gb = size_bytes.map(|bytes| {
        let gb = (bytes as f64 / 1_000_000_000.0).ceil() as u32;
        gb.max(1)
    });

    let max_context_length = hit.gguf.as_ref().and_then(|g| g.context_length);
    let kind = infer_kind_from_hf(hit.pipeline_tag.as_deref(), &hit.tags, local_filename);
    let capabilities = capabilities_from_hf(hit.pipeline_tag.as_deref(), &hit.tags);

    HfEnrichment {
        hf_repo: hit.id.clone(),
        description,
        kind,
        max_context_length,
        min_vram_gb,
        capabilities,
    }
}

fn infer_kind_from_hf(
    pipeline_tag: Option<&str>,
    tags: &[String],
    local_filename: &str,
) -> Option<String> {
    let tag = pipeline_tag.unwrap_or("").to_ascii_lowercase();
    let joined = tags
        .iter()
        .map(|t| t.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    let hay = format!("{tag} {joined} {}", local_filename.to_ascii_lowercase());

    if tag.contains("feature-extraction")
        || tag.contains("sentence-similarity")
        || hay.contains("embed")
    {
        return Some("embedding".to_string());
    }
    if tag.contains("image-text-to-text")
        || tag.contains("any-to-any")
        || hay.contains("vision")
        || hay.contains("-vl")
        || hay.contains("mmproj")
    {
        return Some("vision".to_string());
    }
    if hay.contains("coder") || hay.contains("code") {
        return Some("coder".to_string());
    }
    if !tag.is_empty() || !tags.is_empty() {
        return Some("chat".to_string());
    }
    None
}

fn capabilities_from_hf(pipeline_tag: Option<&str>, tags: &[String]) -> Vec<String> {
    let mut caps = Vec::new();
    let hay = format!("{} {}", pipeline_tag.unwrap_or(""), tags.join(" ")).to_ascii_lowercase();

    if hay.contains("image-text-to-text")
        || hay.contains("any-to-any")
        || hay.contains("vision")
        || hay.contains("vlm")
    {
        caps.push("vision".to_string());
    }
    if hay.contains("tool") || hay.contains("function-calling") {
        caps.push("tools".to_string());
    }
    if hay.contains("reasoning") || hay.contains("think") {
        caps.push("reasoning".to_string());
    }
    caps.sort();
    caps.dedup();
    caps
}

/// Score how well an HF hit matches a local GGUF filename (higher is better).
pub fn score_hf_hit(hit: &Value, local_filename: &str) -> i32 {
    let Ok(parsed): Result<HfModelHit, _> = serde_json::from_value(hit.clone()) else {
        return -1;
    };
    let local_lower = local_filename.to_ascii_lowercase();
    let stem = Path::new(local_filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(local_filename)
        .to_ascii_lowercase();
    let stem_no_quant = strip_quant_suffix(&stem);

    let mut score = 0;
    if parsed
        .siblings
        .iter()
        .any(|s| s.rfilename.to_ascii_lowercase() == local_lower)
    {
        score += 100;
    }
    if parsed
        .siblings
        .iter()
        .any(|s| s.rfilename.to_ascii_lowercase().contains(&stem_no_quant))
    {
        score += 40;
    }
    let id_lower = parsed.id.to_ascii_lowercase();
    if id_lower.starts_with("lmstudio-community/") {
        score += 20;
    }
    if id_lower.contains(&stem_no_quant)
        || stem_no_quant
            .split('-')
            .any(|p| p.len() > 3 && id_lower.contains(p))
    {
        score += 10;
    }
    if parsed.gguf.is_some() {
        score += 5;
    }
    score
}

fn strip_quant_suffix(stem: &str) -> String {
    // e.g. Qwen3.5-9B-Q4_K_M -> Qwen3.5-9B
    let lower = stem.to_ascii_lowercase();
    for sep in [
        "-q4_", "-q5_", "-q6_", "-q8_", "-q3_", "-q2_", "-f16", "-f32", "-iq",
    ] {
        if let Some(idx) = lower.find(sep) {
            return stem[..idx].to_string();
        }
    }
    stem.to_string()
}

fn search_query_for_file(file: &str) -> String {
    let name = Path::new(file)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(file);
    let stem = Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name);
    strip_quant_suffix(stem)
}

/// Apply enrichment only into empty/missing fields (never overwrite explicit pins).
pub fn apply_enrichment(entry: &mut RegistryEntry, enrich: &HfEnrichment) {
    if entry.hf_repo.is_none() {
        entry.hf_repo = Some(enrich.hf_repo.clone());
    }
    if entry.description.is_none() {
        entry.description = enrich.description.clone();
    }
    if entry.kind.is_none() {
        entry.kind = enrich.kind.clone();
    }
    if entry.max_context_length.is_none() {
        entry.max_context_length = enrich.max_context_length;
    }
    if entry.min_vram_gb.is_none() {
        entry.min_vram_gb = enrich.min_vram_gb;
    }
    if entry.capabilities.is_empty() && !enrich.capabilities.is_empty() {
        entry.capabilities = enrich.capabilities.clone();
    }
}

pub struct SyncSummary {
    pub matched: usize,
    pub missed: usize,
    pub skipped: usize,
}

/// Fetch HF metadata for each registry entry and merge overlays into the registry.
pub async fn sync_registry_from_hf(
    registry: &mut ModelsRegistry,
) -> Result<SyncSummary, RuntimeError> {
    let client = reqwest::Client::builder()
        .user_agent(concat!("gguf-switchboard/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| RuntimeError::InternalError(e.to_string()))?;

    let models_dirs = super::models_registry::resolve_models_dirs(&registry.defaults.models_dir)
        .or_else(|_| {
            // Directory may be missing on a pure metadata sync machine; still allow HF match.
            Ok::<Vec<std::path::PathBuf>, RuntimeError>(Vec::new())
        })?;

    let mut matched = 0;
    let mut missed = 0;
    let mut skipped = 0;

    for entry in &mut registry.models {
        if !entry.enabled {
            skipped += 1;
            continue;
        }

        let filename = Path::new(&entry.file)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&entry.file)
            .to_string();

        // Prefer searching by existing hf_repo id when set.
        let hits = if let Some(repo) = entry.hf_repo.clone() {
            fetch_model_by_id(&client, &repo).await?
        } else {
            let query = search_query_for_file(&entry.file);
            search_hf_models(&client, &query).await?
        };

        let best = hits
            .into_iter()
            .map(|hit| {
                let score = score_hf_hit(&hit, &filename);
                (score, hit)
            })
            .filter(|(score, _)| *score > 0)
            .max_by_key(|(score, _)| *score);

        let Some((_score, hit)) = best else {
            missed += 1;
            continue;
        };

        let enrich = enrichment_from_hf_json(&hit, &filename).ok_or_else(|| {
            RuntimeError::InternalError("failed to parse HF model hit".to_string())
        })?;
        apply_enrichment(entry, &enrich);

        // If local file exists, refine min_vram from on-disk size when HF size missing.
        if entry.min_vram_gb.is_none()
            && let Ok(path) = resolve_model_path(&models_dirs, &entry.file)
            && let Ok(meta) = std::fs::metadata(&path)
        {
            let gb = (meta.len() as f64 / 1_000_000_000.0).ceil() as u32;
            entry.min_vram_gb = Some(gb.max(1));
        }

        matched += 1;
    }

    Ok(SyncSummary {
        matched,
        missed,
        skipped,
    })
}

async fn search_hf_models(
    client: &reqwest::Client,
    query: &str,
) -> Result<Vec<Value>, RuntimeError> {
    let url = reqwest::Url::parse_with_params(
        HF_MODELS_API,
        &[
            ("search", query),
            ("filter", "gguf"),
            ("limit", "10"),
            ("expand", "gguf"),
            ("expand", "cardData"),
            ("expand", "pipeline_tag"),
            ("expand", "tags"),
            ("expand", "siblings"),
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

async fn fetch_model_by_id(
    client: &reqwest::Client,
    repo: &str,
) -> Result<Vec<Value>, RuntimeError> {
    let url = format!(
        "{HF_MODELS_API}/{repo}?expand=gguf&expand=cardData&expand=pipeline_tag&expand=tags&expand=siblings"
    );
    let resp = client.get(&url).send().await.map_err(RuntimeError::from)?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(Vec::new());
    }
    if !resp.status().is_success() {
        return Err(RuntimeError::ProxyError(format!(
            "HF API model fetch failed for {repo}: HTTP {}",
            resp.status()
        )));
    }
    let hit: Value = resp.json().await.map_err(RuntimeError::from)?;
    Ok(vec![hit])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn maps_gguf_context_and_vram() {
        let hit = json!({
            "id": "lmstudio-community/Qwen3.5-9B-GGUF",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "text-generation"],
            "gguf": { "context_length": 262144, "total": 5_627_044_256u64 },
            "siblings": [{ "rfilename": "Qwen3.5-9B-Q4_K_M.gguf" }]
        });
        let enrich = enrichment_from_hf_json(&hit, "Qwen3.5-9B-Q4_K_M.gguf").unwrap();
        assert_eq!(enrich.hf_repo, "lmstudio-community/Qwen3.5-9B-GGUF");
        assert_eq!(enrich.max_context_length, Some(262144));
        assert_eq!(enrich.min_vram_gb, Some(6));
        assert_eq!(enrich.kind.as_deref(), Some("chat"));
    }

    #[test]
    fn maps_embedding_pipeline() {
        let hit = json!({
            "id": "ggml-org/embeddinggemma-300M-GGUF",
            "pipeline_tag": "feature-extraction",
            "tags": ["gguf", "feature-extraction"],
            "gguf": { "context_length": 2048, "total": 200_000_000u64 },
            "siblings": []
        });
        let enrich = enrichment_from_hf_json(&hit, "embeddinggemma.gguf").unwrap();
        assert_eq!(enrich.kind.as_deref(), Some("embedding"));
    }

    #[test]
    fn apply_enrichment_does_not_overwrite_kind() {
        let mut entry = RegistryEntry {
            alias: "x".into(),
            file: "x.gguf".into(),
            kind: Some("coder".into()),
            ..Default::default()
        };
        let enrich = HfEnrichment {
            hf_repo: "org/x".into(),
            description: Some("desc".into()),
            kind: Some("chat".into()),
            max_context_length: Some(8192),
            min_vram_gb: Some(4),
            capabilities: vec!["tools".into()],
        };
        apply_enrichment(&mut entry, &enrich);
        assert_eq!(entry.kind.as_deref(), Some("coder"));
        assert_eq!(entry.description.as_deref(), Some("desc"));
        assert_eq!(entry.hf_repo.as_deref(), Some("org/x"));
        assert_eq!(entry.max_context_length, Some(8192));
    }

    #[test]
    fn scores_exact_sibling_highest() {
        let hit = json!({
            "id": "lmstudio-community/Foo-GGUF",
            "siblings": [{ "rfilename": "Foo-Q4_K_M.gguf" }],
            "gguf": { "context_length": 4096 }
        });
        let other = json!({
            "id": "someone/unrelated",
            "siblings": [{ "rfilename": "Bar.gguf" }]
        });
        assert!(score_hf_hit(&hit, "Foo-Q4_K_M.gguf") > score_hf_hit(&other, "Foo-Q4_K_M.gguf"));
    }
}
