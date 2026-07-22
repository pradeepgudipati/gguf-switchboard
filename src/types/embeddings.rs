use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::Usage;

fn default_encoding_format() -> String {
    "float".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "model": "gemma-4-e4b",
    "input": "The quick brown fox jumps over the lazy dog."
}))]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    /// Embedding format: `"float"` (default) or `"base64"`.
    /// Always serialized so backends never see null / missing.
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f64>,
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

impl From<Usage> for EmbeddingUsage {
    fn from(usage: Usage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}
