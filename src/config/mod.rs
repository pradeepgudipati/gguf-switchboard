mod models_registry;

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

pub use models_registry::ModelsRegistry;

use crate::errors::RuntimeError;

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

    /// Return the model id of the priority model, if one is configured.
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
