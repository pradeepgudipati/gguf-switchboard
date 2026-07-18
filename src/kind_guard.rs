//! Reject API calls when the model's kind does not match the endpoint.

use crate::config::ModelConfig;
use crate::errors::RuntimeError;

pub const CHAT_KINDS: &[&str] = &["chat", "coder", "vision"];
pub const EMBEDDING_KINDS: &[&str] = &["embedding"];

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
            block_count: None,
            ngl_pinned: false,
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
}
