use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "model": "qwen3-reranker-0.6b.q8-0",
    "query": "What is the capital of France?",
    "documents": [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is in Paris."
    ],
    "top_n": 2
}))]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    #[serde(default)]
    pub return_documents: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RerankResult {
    pub index: u32,
    pub relevance_score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RerankResponse {
    pub model: String,
    pub object: String,
    pub results: Vec<RerankResult>,
    pub usage: RerankUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RerankUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}
