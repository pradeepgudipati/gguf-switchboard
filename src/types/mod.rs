pub mod anthropic;
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
#[schema(example = json!({
    "id": "qwen2.5-coder-7b",
    "object": "model",
    "created": 1710000000,
    "owned_by": "local",
    "display_name": "Qwen2.5 Coder 7b",
    "kind": "coder",
    "description": "Code-specialized Qwen 2.5 instruct model",
    "context_size": 32768,
    "max_context_length": 32768,
    "min_vram_gb": 8,
    "capabilities": ["tools"],
    "hf_repo": "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF"
}))]
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
    /// The effective runtime profile after the last successful load.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_profile: Option<RuntimeProfileInfo>,
}

/// Runtime profile exposed via the API.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RuntimeProfileInfo {
    pub effective_context: u32,
    pub gpu_layers: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_split: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_k: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_v: Option<String>,
    pub reason: String,
    /// How this profile was determined: "auto-fit", "cached", "manual".
    pub profile_source: String,
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
            runtime_profile: None,
        }
    }

    pub fn from_config(id: impl Into<String>, config: &ModelConfig) -> Self {
        let context_size = crate::context::get_context_size(&config.args);
        let runtime_profile = config
            .runtime_profile
            .as_ref()
            .map(|rp| RuntimeProfileInfo {
                effective_context: rp.context_size,
                gpu_layers: rp.ngl,
                split_mode: rp.split_mode.clone(),
                tensor_split: rp.tensor_split.clone(),
                cache_type_k: rp.cache_type_k.clone(),
                cache_type_v: rp.cache_type_v.clone(),
                reason: rp.reason.clone(),
                profile_source: rp.profile_source.clone(),
            });
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
            runtime_profile,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}
