//! A [`Backend`] that forwards to an arbitrary external OpenAI-compatible
//! endpoint (OpenAI, OpenRouter, Together, another llama-server, …).
//!
//! Used only by the conformance console so a user can run the same
//! tool-calling / template diagnostics against a model that is **not** managed
//! by this switchboard. The base URL and (optional) bearer token arrive per
//! request as `X-Conformance-*` headers — nothing is persisted: not to config,
//! not to the history DB, not to logs.
//!
//! Only [`Backend::chat`] is implemented; everything else returns an
//! "unsupported" error, which is what the console's non-chat surfaces
//! (`resolve-template`) already degrade against gracefully.

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;

use crate::errors::RuntimeError;
use crate::types::chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, normalize_chat_response,
};
use crate::types::completions::{CompletionChunk, CompletionRequest, CompletionResponse};
use crate::types::embeddings::{EmbeddingRequest, EmbeddingResponse};
use crate::types::rerank::{RerankRequest, RerankResponse};

use super::Backend;

pub struct ExternalOpenAiBackend {
    /// e.g. `https://api.openai.com/v1` — the `/chat/completions` suffix is added.
    base_url: String,
    api_key: Option<String>,
    client: Client,
}

impl std::fmt::Debug for ExternalOpenAiBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalOpenAiBackend")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

fn unsupported(what: &str) -> RuntimeError {
    RuntimeError::InvalidRequest(format!(
        "{what} is not supported against a custom external endpoint (conformance console only proxies chat completions)"
    ))
}

impl ExternalOpenAiBackend {
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.filter(|k| !k.trim().is_empty()),
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("failed to build reqwest client"),
        }
    }
}

#[async_trait]
impl Backend for ExternalOpenAiBackend {
    async fn load(&self) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn unload(&self) -> Result<(), RuntimeError> {
        Ok(())
    }
    async fn health(&self) -> Result<bool, RuntimeError> {
        Ok(true)
    }

    async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, RuntimeError> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::to_value(&request)?;

        let mut req = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let response = req.send().await.map_err(|e| {
            RuntimeError::ProxyError(format!("Request to external endpoint failed: {e}"))
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            return Err(RuntimeError::BackendError(format!(
                "External endpoint returned {status}: {text}"
            )));
        }

        let mut raw: serde_json::Value = response.json().await.map_err(|e| {
            RuntimeError::BackendError(format!("Failed to parse external endpoint response: {e}"))
        })?;
        super::tool_probe::normalize_tool_call_arguments(&mut raw);
        let parsed: ChatCompletionResponse = serde_json::from_value(raw).map_err(|e| {
            RuntimeError::BackendError(format!("Failed to parse external endpoint response: {e}"))
        })?;
        Ok(normalize_chat_response(parsed))
    }

    async fn chat_stream(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, RuntimeError>> + Send>>,
        RuntimeError,
    > {
        Err(unsupported("streaming chat"))
    }

    async fn completions(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, RuntimeError> {
        Err(unsupported("text completions"))
    }

    async fn completions_stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, RuntimeError>> + Send>>,
        RuntimeError,
    > {
        Err(unsupported("streaming completions"))
    }

    async fn embeddings(
        &self,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, RuntimeError> {
        Err(unsupported("embeddings"))
    }

    async fn rerank(&self, _request: RerankRequest) -> Result<RerankResponse, RuntimeError> {
        Err(unsupported("rerank"))
    }

    fn _name(&self) -> &str {
        "external"
    }
    fn _backend_type(&self) -> &str {
        "external-openai"
    }
    fn backend_url(&self) -> &str {
        &self.base_url
    }
    fn _health_url(&self) -> &str {
        &self.base_url
    }
}
