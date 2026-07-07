use std::collections::HashMap;

use serde::Deserialize;

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
    /// Model definitions keyed by model id
    pub models: HashMap<String, ModelConfig>,
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

impl Config {
    /// Load configuration from a TOML file.
    pub fn load(path: &str) -> Result<Self, RuntimeError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to read config file '{path}': {e}"))
        })?;
        let config: Config = toml::from_str(&content)?;
        if config.models.is_empty() {
            return Err(RuntimeError::ConfigError(
                "Configuration must define at least one model".to_string(),
            ));
        }
        Ok(config)
    }

    /// Return the model id of the priority model, if one is configured.
    pub fn priority_model_id(&self) -> Option<String> {
        self.models
            .iter()
            .find(|(_, cfg)| cfg.priority)
            .map(|(id, _)| id.clone())
    }
}
