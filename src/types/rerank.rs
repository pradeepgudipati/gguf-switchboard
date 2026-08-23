use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use super::Usage;

fn default_return_documents() -> bool {
    false
}

/// Request body for `/v1/rerank`, compatible with the Jina/Cohere rerank
/// convention that llama-server's `--reranking` endpoint implements.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "model": "bge-reranker-v2-m3",
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
    /// The search query to score every document against.
    pub query: String,
    /// Candidate documents to rank against `query`.
    pub documents: Vec<String>,
    /// Only return the top N highest-scoring results. Defaults to all documents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    /// Include the original document text in each result. Defaults to `false`,
    /// mirroring the Jina/Cohere convention of returning indices only.
    #[serde(default = "default_return_documents")]
    pub return_documents: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RerankResponse {
    pub model: String,
    pub object: String,
    pub results: Vec<RerankResult>,
    pub usage: RerankUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RerankResult {
    /// Index of the document in the original `documents` array.
    pub index: u32,
    /// Relevance score for this document against the query (higher is more relevant).
    pub relevance_score: f64,
    /// Present only when the request set `return_documents: true`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RerankUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

impl From<Usage> for RerankUsage {
    fn from(usage: Usage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            total_tokens: usage.total_tokens,
        }
    }
}
