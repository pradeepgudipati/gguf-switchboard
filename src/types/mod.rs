pub mod audio;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod models;
pub mod responses;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::config::ModelConfig;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    /// Model creation timestamp as Unix seconds
    pub created: i64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Model role: `chat`, `coder`, `vision`, or `embedding`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Serving context size (`-c`), when known from launch args.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_size: Option<u32>,
    /// Model maximum context length from GGUF/HF metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_vram_gb: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".to_string(),
            created: Utc::now().timestamp(),
            owned_by: "local".to_string(),
            display_name: None,
            kind: None,
            description: None,
            context_size: None,
            max_context_length: None,
            min_vram_gb: None,
            capabilities: Vec::new(),
            hf_repo: None,
        }
    }

    pub fn from_config(id: impl Into<String>, config: &ModelConfig) -> Self {
        let context_size = crate::context::get_context_size(&config.args);
        Self {
            id: id.into(),
            object: "model".to_string(),
            created: Utc::now().timestamp(),
            owned_by: "local".to_string(),
            display_name: Some(config.display_name.clone()),
            kind: Some(config.kind.clone()),
            description: config.description.clone(),
            context_size,
            max_context_length: config.max_context_length,
            min_vram_gb: config.min_vram_gb,
            capabilities: config.capabilities.clone(),
            hf_repo: config.hf_repo.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

impl ListModelsResponse {
    pub fn new(models: Vec<ModelInfo>) -> Self {
        Self {
            object: "list".to_string(),
            data: models,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}
