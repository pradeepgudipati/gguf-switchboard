pub mod llama_cpp;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::config::ModelConfig;
use crate::errors::RuntimeError;
use crate::types::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};
use crate::types::completions::{CompletionChunk, CompletionRequest, CompletionResponse};
use crate::types::embeddings::{EmbeddingRequest, EmbeddingResponse};

/// Trait implemented by every inference backend (llama.cpp, vLLM, etc.).
#[async_trait]
pub trait Backend: Send + Sync {
    /// Start the backend process / connect to the remote server.
    async fn load(&self) -> Result<(), RuntimeError>;
    /// Stop the backend and release resources.
    async fn unload(&self) -> Result<(), RuntimeError>;
    /// Return `true` when the backend is ready to accept requests.
    async fn health(&self) -> Result<bool, RuntimeError>;
    /// Non-streaming chat completion.
    async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, RuntimeError>;
    /// Streaming chat completion.
    async fn chat_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, RuntimeError>> + Send>>,
        RuntimeError,
    >;
    /// Non-streaming text completion.
    async fn completions(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, RuntimeError>;
    /// Streaming text completion.
    async fn completions_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, RuntimeError>> + Send>>,
        RuntimeError,
    >;
    /// Embeddings.
    async fn embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, RuntimeError>;
    /// The model id this backend serves.
    fn _name(&self) -> &str;
    /// The engine type (e.g. "llama.cpp").
    fn _backend_type(&self) -> &str;
    /// The base URL for the backend's OpenAI-compatible API.
    fn backend_url(&self) -> &str;
    /// The health-check URL.
    fn _health_url(&self) -> &str;
    /// Return `false` when the backend process has exited unexpectedly.
    async fn process_running(&self) -> bool {
        true
    }
}

/// Create a concrete backend for the given model id and config.
pub fn create_backend(model_id: &str, config: &ModelConfig) -> Box<dyn Backend> {
    match config.backend.as_str() {
        "llama.cpp" => Box::new(llama_cpp::LlamaCppBackend::new(model_id, config)),
        other => {
            tracing::warn!(
                backend = other,
                "Unknown backend type, falling back to llama.cpp"
            );
            Box::new(llama_cpp::LlamaCppBackend::new(model_id, config))
        }
    }
}
