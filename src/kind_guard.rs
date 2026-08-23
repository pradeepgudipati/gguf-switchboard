//! Reject API calls when the model's kind does not match the endpoint.

use crate::config::ModelConfig;
use crate::errors::RuntimeError;

pub const CHAT_KINDS: &[&str] = &["chat", "coder", "vision"];
pub const EMBEDDING_KINDS: &[&str] = &["embedding"];
pub const RERANK_KINDS: &[&str] = &["reranker"];

pub fn require_kind(
    model_id: &str,
    config: &ModelConfig,
    allowed: &[&str],
    endpoint: &str,
) -> Result<(), RuntimeError> {
    let kind = config.kind.as_str();
    if allowed.contains(&kind) {
        return Ok(());
    }
    Err(RuntimeError::InvalidRequest(format!(
        "model '{model_id}' is kind={kind}; {endpoint} allows {}",
        allowed.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(kind: &str) -> ModelConfig {
        ModelConfig {
            backend: "llama.cpp".into(),
            display_name: "t".into(),
            command: "llama-server".into(),
            args: vec![],
            backend_url: "http://127.0.0.1:1/v1".into(),
            health_url: "http://127.0.0.1:1/health".into(),
            priority: false,
            kind: kind.into(),
            description: None,
            max_context_length: None,
            min_vram_gb: None,
            capabilities: vec![],
            hf_repo: None,
            ctx_floor: None,
            block_count: None,
            ngl_pinned: false,
            model_fingerprint: None,
            max_context_from_gguf: None,
            runtime_profile: None,
        }
    }

    #[test]
    fn allows_chat_kinds_on_chat() {
        assert!(require_kind("a", &cfg("chat"), CHAT_KINDS, "/v1/chat/completions").is_ok());
        assert!(require_kind("a", &cfg("vision"), CHAT_KINDS, "/v1/chat/completions").is_ok());
    }

    #[test]
    fn rejects_embedding_on_chat() {
        let err =
            require_kind("emb", &cfg("embedding"), CHAT_KINDS, "/v1/chat/completions").unwrap_err();
        assert!(err.to_string().contains("kind=embedding"));
    }

    #[test]
    fn rejects_chat_on_embeddings() {
        let err = require_kind("c", &cfg("chat"), EMBEDDING_KINDS, "/v1/embeddings").unwrap_err();
        assert!(err.to_string().contains("/v1/embeddings"));
    }

    #[test]
    fn allows_reranker_on_rerank() {
        assert!(require_kind("r", &cfg("reranker"), RERANK_KINDS, "/v1/rerank").is_ok());
    }

    #[test]
    fn rejects_chat_on_rerank() {
        let err = require_kind("c", &cfg("chat"), RERANK_KINDS, "/v1/rerank").unwrap_err();
        assert!(err.to_string().contains("/v1/rerank"));
    }

    #[test]
    fn rejects_reranker_on_embeddings() {
        let err =
            require_kind("r", &cfg("reranker"), EMBEDDING_KINDS, "/v1/embeddings").unwrap_err();
        assert!(err.to_string().contains("kind=reranker"));
    }
}
