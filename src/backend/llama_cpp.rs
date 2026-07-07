use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::config::ModelConfig;
use crate::errors::RuntimeError;
use crate::types::chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, normalize_chat_chunk,
    normalize_chat_response,
};
use crate::types::completions::{CompletionChunk, CompletionRequest, CompletionResponse};
use crate::types::embeddings::{EmbeddingRequest, EmbeddingResponse};

use super::Backend;

/// llama.cpp backend: spawns `llama-server` as a child process and proxies
/// OpenAI-compatible requests to it.
pub struct LlamaCppBackend {
    model_id: String,
    config: ModelConfig,
    client: Client,
    process: Arc<Mutex<Option<Child>>>,
    running: AtomicBool,
}

impl LlamaCppBackend {
    pub fn new(model_id: &str, config: &ModelConfig) -> Self {
        Self {
            model_id: model_id.to_string(),
            config: config.clone(),
            client: Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("failed to build reqwest client"),
            process: Arc::new(Mutex::new(None)),
            running: AtomicBool::new(false),
        }
    }

    /// Forward a JSON POST request to the backend and return the raw response.
    async fn forward_json(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<reqwest::Response, RuntimeError> {
        let url = format!("{}{path}", self.config.backend_url);
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RuntimeError::ProxyError(format!("Request to backend failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            return Err(RuntimeError::BackendError(format!(
                "Backend returned {status}: {text}"
            )));
        }
        Ok(response)
    }

    /// Forward a streaming JSON POST request to the backend.
    async fn forward_json_stream(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<reqwest::Response, RuntimeError> {
        let url = format!("{}{path}", self.config.backend_url);
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| RuntimeError::ProxyError(format!("Request to backend failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            return Err(RuntimeError::BackendError(format!(
                "Backend returned {status}: {text}"
            )));
        }
        Ok(response)
    }
}

#[async_trait]
impl Backend for LlamaCppBackend {
    async fn load(&self) -> Result<(), RuntimeError> {
        // Validate backend binary exists
        let cmd_path = std::path::Path::new(&self.config.command);
        if cmd_path.is_absolute() && !cmd_path.exists() {
            return Err(RuntimeError::ModelLoadingFailed(format!(
                "Backend binary not found: '{}'. Install llama.cpp or update the command path in config.toml.",
                self.config.command
            )));
        }

        // Validate GGUF model file exists (look for -m / --model in args)
        if let Some(model_path) = self.find_model_arg() {
            if !std::path::Path::new(model_path).exists() {
                return Err(RuntimeError::ModelNotFound(format!(
                    "Model GGUF file not found: '{}'. Ensure the model file exists and the path is correct in config.toml.",
                    model_path
                )));
            }
        } else {
            tracing::warn!(
                model = %self.model_id,
                "No -m/--model argument found in backend args; skipping GGUF file validation"
            );
        }

        info!(model = %self.model_id, command = %self.command_display(), "Starting backend process");

        let mut cmd = Command::new(&self.config.command);
        cmd.args(&self.config.args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let child = cmd.spawn().map_err(|e| {
            RuntimeError::ModelLoadingFailed(format!("Failed to spawn backend: {e}"))
        })?;

        *self.process.lock().await = Some(child);
        self.running.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn unload(&self) -> Result<(), RuntimeError> {
        info!(model = %self.model_id, "Stopping backend process");
        let mut guard = self.process.lock().await;
        if let Some(mut child) = guard.take() {
            // Send SIGTERM on unix, then wait briefly for graceful exit
            #[cfg(unix)]
            {
                if let Some(id) = child.id() {
                    use nix::sys::signal::{self, Signal};
                    use nix::unistd::Pid;
                    let _ = signal::kill(Pid::from_raw(id as i32), Signal::SIGTERM);
                }
            }
            match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
                Ok(Ok(status)) => {
                    info!(model = %self.model_id, %status, "Backend exited");
                }
                Ok(Err(e)) => {
                    warn!(model = %self.model_id, error = %e, "Error waiting for backend");
                }
                Err(_) => {
                    warn!(model = %self.model_id, "Backend did not exit in time, killing");
                    let _ = child.kill().await;
                }
            }
        }
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }

    async fn health(&self) -> Result<bool, RuntimeError> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(false);
        }
        let url = &self.config.health_url;
        match self.client.get(url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    async fn chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, RuntimeError> {
        let body = serde_json::to_value(&request)?;
        let resp = self.forward_json("/chat/completions", body).await?;
        let response: ChatCompletionResponse = resp.json().await.map_err(|e| {
            RuntimeError::BackendError(format!("Failed to parse backend response: {e}"))
        })?;
        Ok(normalize_chat_response(response))
    }

    async fn chat_stream(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, RuntimeError>> + Send>>,
        RuntimeError,
    > {
        request.stream = Some(true);
        let body = serde_json::to_value(&request)?;
        let resp = self.forward_json_stream("/chat/completions", body).await?;

        let stream = resp.bytes_stream().map(|chunk| {
            chunk.map_err(|e| RuntimeError::ProxyError(format!("Stream read error: {e}")))
        });

        Ok(Box::pin(
            SseLineParser::new(stream).map(|item| item.map(normalize_chat_chunk)),
        ))
    }

    async fn completions(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, RuntimeError> {
        let body = serde_json::to_value(&request)?;
        let resp = self.forward_json("/completions", body).await?;
        let response: CompletionResponse = resp.json().await.map_err(|e| {
            RuntimeError::BackendError(format!("Failed to parse backend response: {e}"))
        })?;
        Ok(response)
    }

    async fn completions_stream(
        &self,
        mut request: CompletionRequest,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<CompletionChunk, RuntimeError>> + Send>>,
        RuntimeError,
    > {
        request.stream = Some(true);
        let body = serde_json::to_value(&request)?;
        let resp = self.forward_json_stream("/completions", body).await?;

        let stream = resp.bytes_stream().map(|chunk| {
            chunk.map_err(|e| RuntimeError::ProxyError(format!("Stream read error: {e}")))
        });

        Ok(Box::pin(SseLineParser::new(stream)))
    }

    async fn embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, RuntimeError> {
        let body = serde_json::to_value(&request)?;
        let resp = self.forward_json("/embeddings", body).await?;
        let response: EmbeddingResponse = resp.json().await.map_err(|e| {
            RuntimeError::BackendError(format!("Failed to parse backend response: {e}"))
        })?;
        Ok(response)
    }

    fn name(&self) -> &str {
        &self.model_id
    }

    fn _backend_type(&self) -> &str {
        &self.config.backend
    }

    fn backend_url(&self) -> &str {
        &self.config.backend_url
    }

    fn _health_url(&self) -> &str {
        &self.config.health_url
    }
}

impl LlamaCppBackend {
    fn command_display(&self) -> String {
        format!("{} {}", self.config.command, self.config.args.join(" "))
    }

    /// Extract the model path from the args list by looking for `-m` or `--model`.
    fn find_model_arg(&self) -> Option<&str> {
        let args = &self.config.args;
        for i in 0..args.len() {
            if (args[i] == "-m" || args[i] == "--model") && i + 1 < args.len() {
                return Some(&args[i + 1]);
            }
        }
        None
    }
}

/// Parses a raw byte SSE stream and yields deserialized typed chunks.
/// Handles partial line buffering, `data: ` prefix stripping, and `[DONE]` termination.
struct SseLineParser<T> {
    inner: Pin<Box<dyn Stream<Item = Result<bytes::Bytes, RuntimeError>> + Send>>,
    buffer: Vec<u8>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> SseLineParser<T> {
    fn new(
        stream: impl Stream<Item = Result<bytes::Bytes, RuntimeError>> + Send + 'static,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            buffer: Vec::with_capacity(4096),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: serde::de::DeserializeOwned + Unpin> futures::Stream for SseLineParser<T> {
    type Item = Result<T, RuntimeError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            // Try to find a complete line in the buffer
            if let Some(pos) = this.buffer.iter().position(|&b| b == b'\n') {
                let line_bytes = this.buffer.drain(..=pos).collect::<Vec<u8>>();
                let line = String::from_utf8_lossy(&line_bytes).trim().to_string();

                if line.is_empty() {
                    continue;
                }

                if line == "data: [DONE]" {
                    return std::task::Poll::Ready(None);
                }

                let json_str = if let Some(rest) = line.strip_prefix("data: ") {
                    rest.trim()
                } else if line.starts_with("data:") {
                    line.strip_prefix("data:").unwrap_or("").trim()
                } else {
                    continue;
                };

                if json_str.is_empty() {
                    continue;
                }

                match serde_json::from_str::<T>(json_str) {
                    Ok(chunk) => return std::task::Poll::Ready(Some(Ok(chunk))),
                    Err(e) => {
                        debug!(error = %e, raw = %json_str, "Failed to parse SSE chunk, skipping");
                        continue;
                    }
                }
            }

            // Need more data from the upstream stream
            match this.inner.as_mut().poll_next(cx) {
                std::task::Poll::Ready(Some(Ok(bytes))) => {
                    this.buffer.extend_from_slice(&bytes);
                }
                std::task::Poll::Ready(Some(Err(e))) => {
                    return std::task::Poll::Ready(Some(Err(e)));
                }
                std::task::Poll::Ready(None) => {
                    // Stream ended without [DONE]
                    if this.buffer.iter().all(|b| b.is_ascii_whitespace()) {
                        return std::task::Poll::Ready(None);
                    }
                    // Try to parse any remaining data
                    let remaining = String::from_utf8_lossy(&this.buffer).trim().to_string();
                    this.buffer.clear();
                    if remaining.is_empty() {
                        return std::task::Poll::Ready(None);
                    }
                    let json_str = if let Some(rest) = remaining.strip_prefix("data: ") {
                        rest.trim()
                    } else {
                        remaining.as_str()
                    };
                    if json_str == "[DONE]" || json_str.is_empty() {
                        return std::task::Poll::Ready(None);
                    }
                    match serde_json::from_str::<T>(json_str) {
                        Ok(chunk) => return std::task::Poll::Ready(Some(Ok(chunk))),
                        Err(_) => return std::task::Poll::Ready(None),
                    }
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}
