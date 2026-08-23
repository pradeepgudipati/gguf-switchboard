pub mod hf_download;
mod hf_sync;
pub mod models_cmd;
mod models_registry;
mod vllm_meta;

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

pub use hf_sync::{SyncSummary, sync_registry_from_hf};
pub use models_cmd::{cmd_files, cmd_pull, cmd_search};
pub use models_registry::{ModelsRegistry, RescanResult, check_vllm_available};

use crate::errors::RuntimeError;
use crate::fit::{FitConfig, FitPlan};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Address to bind the HTTP server (e.g. "0.0.0.0:9090")
    pub bind: String,
    /// Maximum seconds to wait for a backend to become healthy on startup
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout: u64,
    /// Seconds of idle time before the priority model is loaded
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: u64,
    /// Default backend engine (e.g. "llama.cpp")
    #[serde(default = "default_backend")]
    pub default_backend: String,
    /// Path to the token usage SQLite database
    #[serde(default)]
    pub database_path: Option<String>,
    /// GPU VRAM in GB used to size default context windows per model (default: 12).
    #[serde(default = "default_vram_gb")]
    pub vram_gb: u32,
    /// When true, pick `-ngl` at load from free VRAM + GGUF size (see `ngl` module).
    #[serde(default)]
    pub auto_ngl: bool,
    /// Optional path to a simplified models registry (`models.toml` or `models.json`).
    /// When set, model entries are expanded from that file at load time.
    #[serde(default)]
    pub models_file: Option<String>,
    /// Portable JSON export of the model registry (generated at startup).
    #[serde(skip, default)]
    pub registry_json: String,
    /// Model definitions keyed by model id (inline in config, or expanded from `models_file`)
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
    /// Percentage of RAM usage at which a warning is logged (default 85).
    #[serde(default = "default_memory_warning_threshold")]
    pub memory_warning_threshold: u8,
    /// Percentage of RAM usage at which the loaded model is auto-unloaded (default 95).
    #[serde(default = "default_memory_critical_threshold")]
    pub memory_critical_threshold: u8,
    /// Seconds between memory pressure checks (default 30).
    #[serde(default = "default_memory_check_interval_secs")]
    pub memory_check_interval_secs: u64,
    /// Minimum context size (`-c`) when auto-reducing after a failed model load.
    #[serde(default = "default_context_fallback_min")]
    pub context_fallback_min: u32,
    /// Seconds to wait for in-flight requests to finish before switching models.
    #[serde(default = "default_switch_drain_timeout_secs")]
    pub switch_drain_timeout_secs: u64,
    /// Seconds to skip priority-model reload after a failed priority load.
    #[serde(default = "default_priority_load_cooldown_secs")]
    pub priority_load_cooldown_secs: u64,
    /// Seconds between automatic model-directory rescans (default 86400 = 1 day). `0` disables.
    #[serde(default = "default_models_rescan_interval_secs")]
    pub models_rescan_interval_secs: u64,
    /// ModelFitPlanner configuration for hardware-aware load planning.
    #[serde(default)]
    pub fit: FitConfig,
    /// Embedding-specific VRAM budgeting and admission control.
    #[serde(default)]
    pub embedding_fit: EmbeddingFitConfig,
    /// How the scheduler swaps models. `unload_first` (default) stops the resident
    /// `llama-server` before starting the next one so the new model gets the whole GPU.
    /// `load_first` keeps the old model resident until the new one is healthy (instant
    /// rollback, but needs VRAM for both models at once — otherwise the new model OOMs,
    /// walks the context/ngl fallback ladder and ends up partially on the CPU).
    #[serde(default)]
    pub switch_strategy: SwitchStrategy,
    /// Number of recently used models (in addition to the resident one) whose GGUF
    /// files are read back into the OS page cache after every successful load, so the
    /// next switch streams weights from RAM instead of disk. `0` (default) disables.
    /// Only useful when system RAM comfortably exceeds the combined GGUF sizes.
    #[serde(default)]
    pub prewarm_recent_models: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingFitConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_embedding_profile")]
    pub profile: String,
    #[serde(default = "default_embedding_reserve_percent")]
    pub vram_reserve_percent: u8,
    #[serde(default = "default_embedding_reserve_min_mb")]
    pub vram_reserve_min_mb: u32,
    #[serde(default = "default_embedding_queue_timeout_secs")]
    pub queue_timeout_secs: u64,
}

impl Default for EmbeddingFitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            profile: default_embedding_profile(),
            vram_reserve_percent: default_embedding_reserve_percent(),
            vram_reserve_min_mb: default_embedding_reserve_min_mb(),
            queue_timeout_secs: default_embedding_queue_timeout_secs(),
        }
    }
}

/// Ordering of unload/load during a model switch. See [`Config::switch_strategy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwitchStrategy {
    /// Drain → unload previous → load target. Rolls back by re-loading the previous
    /// model if the target fails. Correct default for a single GPU.
    #[default]
    UnloadFirst,
    /// Drain → load target → unload previous. Requires VRAM for both models.
    LoadFirst,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Backend engine type (e.g. "llama.cpp")
    pub backend: String,
    /// Human-readable model name
    pub display_name: String,
    /// Path to the backend server binary
    pub command: String,
    /// Command-line arguments for the backend server
    pub args: Vec<String>,
    /// HTTP URL for the backend's OpenAI-compatible API
    pub backend_url: String,
    /// HTTP URL for the backend's health check endpoint
    pub health_url: String,
    /// If true, this model is loaded automatically after idle timeout
    #[serde(default)]
    pub priority: bool,
    /// Model role: `chat`, `coder`, `vision`, `embedding`, or `reranker`.
    #[serde(default = "default_model_kind")]
    pub kind: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub max_context_length: Option<u32>,
    #[serde(default)]
    pub min_vram_gb: Option<u32>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub hf_repo: Option<String>,
    /// Minimum context size this model must always be served with, even after
    /// VRAM-pressure fallback reduces context (sourced from `RegistryEntry.ctx`).
    #[serde(default)]
    pub ctx_floor: Option<u32>,
    /// GGUF `block_count` when known (used by auto_ngl).
    #[serde(default, skip_serializing)]
    pub block_count: Option<u32>,
    /// When true, auto_ngl must not rewrite `-ngl` (pinned via `models.ngl` or `extra_args`).
    #[serde(default, skip_serializing)]
    pub ngl_pinned: bool,
    /// Fingerprint for profile cache keying (`{filename}:{file_size_mb}`).
    #[serde(default, skip_serializing)]
    pub model_fingerprint: Option<String>,
    /// Maximum context length from GGUF metadata (distinct from serving context_size).
    #[serde(default, skip_serializing)]
    pub max_context_from_gguf: Option<u32>,
    /// The active runtime profile after a successful load.
    #[serde(default, skip_serializing)]
    pub runtime_profile: Option<RuntimeProfile>,
}

/// The effective load profile that was used to successfully start a model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RuntimeProfile {
    pub context_size: u32,
    pub ngl: u32,
    pub split_mode: Option<String>,
    pub tensor_split: Option<Vec<f64>>,
    pub cache_type_k: Option<String>,
    pub cache_type_v: Option<String>,
    pub batch_size: Option<u32>,
    pub ubatch_size: Option<u32>,
    pub embedding_concurrency: Option<u32>,
    pub reason: String,
    pub profile_source: String,
}

impl RuntimeProfile {
    pub fn from_fit_plan(plan: &FitPlan, source: &str) -> Self {
        Self {
            context_size: plan.context_size,
            ngl: plan.ngl,
            split_mode: plan.split_mode.clone(),
            tensor_split: plan.tensor_split.clone(),
            cache_type_k: plan.cache_type_k.clone(),
            cache_type_v: plan.cache_type_v.clone(),
            batch_size: plan.batch_size,
            ubatch_size: plan.ubatch_size,
            embedding_concurrency: plan
                .batch_size
                .map(|batch| crate::embedding_admission::balanced_concurrency(Some(batch)) as u32),
            reason: plan.reason.clone(),
            profile_source: source.to_string(),
        }
    }
}

fn default_model_kind() -> String {
    "chat".to_string()
}

fn default_startup_timeout() -> u64 {
    60
}

fn default_idle_timeout() -> u64 {
    600
}

fn default_backend() -> String {
    "llama.cpp".to_string()
}

fn default_vram_gb() -> u32 {
    12
}

fn default_memory_warning_threshold() -> u8 {
    85
}

fn default_memory_critical_threshold() -> u8 {
    95
}

fn default_memory_check_interval_secs() -> u64 {
    30
}

fn default_context_fallback_min() -> u32 {
    8192
}

fn default_switch_drain_timeout_secs() -> u64 {
    120
}

fn default_priority_load_cooldown_secs() -> u64 {
    300
}

fn default_models_rescan_interval_secs() -> u64 {
    86400
}

fn default_true() -> bool {
    true
}
fn default_embedding_profile() -> String {
    "balanced".to_string()
}
fn default_embedding_reserve_percent() -> u8 {
    15
}
fn default_embedding_reserve_min_mb() -> u32 {
    1536
}
fn default_embedding_queue_timeout_secs() -> u64 {
    30
}

impl Config {
    /// Load configuration from a TOML file.
    pub fn load(path: &str) -> Result<Self, RuntimeError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to read config file '{path}': {e}"))
        })?;
        let mut config: Config = toml::from_str(&content)?;
        config.resolve_models(path)?;
        if config.models.is_empty() {
            return Err(RuntimeError::ConfigError(
                "Configuration must define at least one model (inline [models.*] or models_file)"
                    .to_string(),
            ));
        }
        Ok(config)
    }

    fn resolve_models(&mut self, config_path: &str) -> Result<(), RuntimeError> {
        let models_file = match &self.models_file {
            Some(path) => Some(path.clone()),
            None => {
                let sibling = Path::new(config_path)
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join("models.toml");
                if sibling.is_file() {
                    Some(sibling.to_string_lossy().into_owned())
                } else {
                    None
                }
            }
        };

        if let Some(models_path) = models_file {
            let resolved = resolve_relative_to_config(config_path, &models_path);
            let registry = ModelsRegistry::load(&resolved)?;
            let expanded = registry.expand(&self.default_backend, self.vram_gb)?;
            if !self.models.is_empty() {
                tracing::warn!(
                    models_file = %resolved,
                    "models_file is set; inline [models.*] entries in config.toml are ignored"
                );
            }
            self.models = expanded;
            self.models_file = Some(resolved.clone());
            self.registry_json =
                serde_json::to_string_pretty(&registry.to_json_export()).map_err(|e| {
                    RuntimeError::ConfigError(format!("Failed to serialize models JSON: {e}"))
                })?;
        }

        Ok(())
    }

    /// Enrich `models_file` from Hugging Face Hub and reload expanded models into this config.
    ///
    /// Network failures propagate to the caller so launch/refresh can warn and continue.
    pub async fn sync_hf_metadata(&mut self) -> Result<crate::config::SyncSummary, RuntimeError> {
        let models_path = self.models_file.as_ref().ok_or_else(|| {
            RuntimeError::ConfigError(
                "models_file is not configured; cannot sync HF metadata".to_string(),
            )
        })?;

        let mut registry = ModelsRegistry::load(models_path)?;
        let summary = crate::config::sync_registry_from_hf(&mut registry).await?;
        registry.write(models_path)?;

        let expanded = registry.expand(&self.default_backend, self.vram_gb)?;
        self.models = expanded;
        self.registry_json =
            serde_json::to_string_pretty(&registry.to_json_export()).map_err(|e| {
                RuntimeError::ConfigError(format!("Failed to serialize models JSON: {e}"))
            })?;

        Ok(summary)
    }

    /// Return the model id of the priority model, if one is configured.
    #[allow(dead_code)] // public API; scheduler reads priority from the live model map
    pub fn priority_model_id(&self) -> Option<String> {
        self.models
            .iter()
            .find(|(_, cfg)| cfg.priority)
            .map(|(id, _)| id.clone())
    }
}

fn resolve_relative_to_config(config_path: &str, models_path: &str) -> String {
    let path = Path::new(models_path);
    if path.is_absolute() {
        return models_path.to_string();
    }

    Path::new(config_path)
        .parent()
        .unwrap_or(Path::new("."))
        .join(path)
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_strategy_defaults_to_unload_first() {
        let cfg: Config = toml::from_str(r#"bind = "127.0.0.1:0""#).expect("parse");
        assert_eq!(cfg.switch_strategy, SwitchStrategy::UnloadFirst);
        assert_eq!(cfg.prewarm_recent_models, 0);
        assert!(cfg.embedding_fit.enabled);
        assert_eq!(cfg.embedding_fit.profile, "balanced");
        assert_eq!(cfg.embedding_fit.vram_reserve_percent, 15);
        assert_eq!(cfg.embedding_fit.vram_reserve_min_mb, 1536);
        assert_eq!(cfg.embedding_fit.queue_timeout_secs, 30);
    }

    #[test]
    fn switch_strategy_parses_snake_case() {
        let cfg: Config = toml::from_str(
            r#"
bind = "127.0.0.1:0"
switch_strategy = "load_first"
prewarm_recent_models = 2
"#,
        )
        .expect("parse");
        assert_eq!(cfg.switch_strategy, SwitchStrategy::LoadFirst);
        assert_eq!(cfg.prewarm_recent_models, 2);
    }

    #[test]
    fn switch_strategy_rejects_unknown_value() {
        let err = toml::from_str::<Config>(
            r#"
bind = "127.0.0.1:0"
switch_strategy = "yolo"
"#,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("yolo"), "{msg}");
    }

    #[test]
    fn example_config_keeps_models_file_at_root() {
        let cfg: Config = toml::from_str(include_str!("../../config.example.toml"))
            .expect("config.example.toml must parse");
        assert_eq!(
            cfg.models_file.as_deref(),
            Some("/opt/gguf-switchboard/models.toml")
        );
        assert!(cfg.embedding_fit.enabled);
    }
}
