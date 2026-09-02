use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use tokio::io::{AsyncBufReadExt, BufReader};
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
use crate::types::rerank::{RerankRequest, RerankResponse};

use super::Backend;

/// llama.cpp backend: spawns `llama-server` as a child process and proxies
/// OpenAI-compatible requests to it.
pub struct LlamaCppBackend {
    model_id: String,
    config: ModelConfig,
    client: Client,
    process: Arc<Mutex<Option<Child>>>,
    running: AtomicBool,
    startup_stderr: Arc<Mutex<String>>,
    server_version: Arc<Mutex<Option<String>>>,
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
            startup_stderr: Arc::new(Mutex::new(String::new())),
            server_version: Arc::new(Mutex::new(None)),
        }
    }

    /// Server root for llama-server's own (non-OpenAI) endpoints like
    /// `/props` and `/apply-template`. `backend_url` is the OpenAI-compatible
    /// base (`http://host:port/v1`); these diagnostics live at the bare root,
    /// so strip a trailing `/v1`.
    fn server_root(&self) -> &str {
        self.config
            .backend_url
            .trim_end_matches('/')
            .trim_end_matches("/v1")
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
                    "Model GGUF file not found: '{model_path}'. Ensure the model file \
                     exists and the path is correct in config.toml."
                )));
            }
        } else {
            tracing::warn!(
                model = %self.model_id,
                "No -m/--model argument found in backend args; skipping GGUF file validation"
            );
        }

        info!(model = %self.model_id, command = %self.command_display(), "Starting backend process");

        // stdout was previously piped but never read: once llama-server wrote >64 KiB to
        // it the pipe filled and the server blocked mid-load. Nothing consumes stdout
        // (startup diagnostics come from stderr), so discard it.
        let mut cmd = Command::new(&self.config.command);
        cmd.args(&self.config.args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().map_err(|e| {
            RuntimeError::ModelLoadingFailed(format!("Failed to spawn backend: {e}"))
        })?;

        if let Some(stderr) = child.stderr.take() {
            let log = Arc::clone(&self.startup_stderr);
            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr).lines();
                let mut combined = String::new();
                while let Ok(Some(line)) = reader.next_line().await {
                    if combined.len() < 65_536 {
                        combined.push_str(&line);
                        combined.push('\n');
                    }
                    let mut guard = log.lock().await;
                    if guard.len() < 65_536 {
                        guard.push_str(&line);
                        guard.push('\n');
                    }
                }
            });
        }

        *self.process.lock().await = Some(child);
        self.running.store(true, Ordering::SeqCst);
        // Resolve `llama-server --version` off the load path: the scheduler starts
        // polling /health as soon as load() returns, and a CUDA build's `--version`
        // initialises the driver, which used to run serially in front of that.
        self.detect_server_version_background();
        Ok(())
    }

    async fn unload(&self) -> Result<(), RuntimeError> {
        info!(model = %self.model_id, "Stopping backend process");
        let mut guard = self.process.lock().await;
        if let Some(mut child) = guard.take() {
            let pid = child.id();
            info!(model = %self.model_id, ?pid, "Sending SIGTERM to backend");

            // Send SIGTERM on unix, then wait briefly for graceful exit
            #[cfg(unix)]
            {
                if let Some(id) = pid {
                    use nix::sys::signal::{self, Signal};
                    use nix::unistd::Pid;
                    if let Err(e) = signal::kill(Pid::from_raw(id as i32), Signal::SIGTERM) {
                        warn!(model = %self.model_id, pid = id, error = %e, "SIGTERM failed");
                    }
                }
            }
            match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
                Ok(Ok(status)) => {
                    info!(model = %self.model_id, %status, "Backend exited after SIGTERM");
                }
                Ok(Err(e)) => {
                    warn!(model = %self.model_id, error = %e, "Error waiting for backend after SIGTERM, sending SIGKILL");
                    let _ = child.kill().await;
                }
                Err(_) => {
                    warn!(model = %self.model_id, "Backend did not exit within 5s, sending SIGKILL");
                    match child.kill().await {
                        Ok(()) => {
                            info!(model = %self.model_id, "SIGKILL sent, waiting for process to exit");
                            // Wait a bit more after SIGKILL to ensure the process is
                            // fully dead and the CUDA driver releases VRAM.
                            match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
                                Ok(Ok(status)) => {
                                    info!(model = %self.model_id, %status, "Backend exited after SIGKILL");
                                }
                                Ok(Err(e)) => {
                                    warn!(model = %self.model_id, error = %e, "Error waiting for backend after SIGKILL");
                                }
                                Err(_) => {
                                    // Even SIGKILL didn't work — the process is likely a
                                    // zombie or stuck in an uninterruptible state.  Report
                                    // the failure so the scheduler can decide whether to
                                    // proceed.
                                    self.running.store(false, Ordering::SeqCst);
                                    return Err(RuntimeError::BackendError(format!(
                                        "Backend process (pid {pid:?}) did not exit after \
                                         SIGTERM and SIGKILL; VRAM may still be in use"
                                    )));
                                }
                            }
                        }
                        Err(e) => {
                            self.running.store(false, Ordering::SeqCst);
                            return Err(RuntimeError::BackendError(format!(
                                "Failed to send SIGKILL to backend process: {e}"
                            )));
                        }
                    }
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
        let mut raw: serde_json::Value = resp.json().await.map_err(|e| {
            RuntimeError::BackendError(format!("Failed to parse backend response: {e}"))
        })?;
        super::tool_probe::normalize_tool_call_arguments(&mut raw);
        let response: ChatCompletionResponse = serde_json::from_value(raw).map_err(|e| {
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

    async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, RuntimeError> {
        let body = serde_json::to_value(&request)?;
        let resp = self.forward_json("/rerank", body).await?;
        let response: RerankResponse = resp.json().await.map_err(|e| {
            RuntimeError::BackendError(format!("Failed to parse backend response: {e}"))
        })?;
        Ok(response)
    }

    fn _name(&self) -> &str {
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

    async fn process_running(&self) -> bool {
        let mut guard = self.process.lock().await;
        let Some(child) = guard.as_mut() else {
            return false;
        };

        match child.try_wait() {
            Ok(Some(status)) => {
                warn!(model = %self.model_id, %status, "Backend process exited");
                self.running.store(false, Ordering::SeqCst);
                false
            }
            Ok(None) => true,
            Err(e) => {
                warn!(model = %self.model_id, error = %e, "Failed to poll backend process");
                false
            }
        }
    }

    async fn take_startup_stderr(&self) -> String {
        let mut guard = self.startup_stderr.lock().await;
        std::mem::take(&mut *guard)
    }

    async fn server_version(&self) -> Option<String> {
        self.server_version.lock().await.clone()
    }

    /// Passthrough for llama-server's own diagnostic endpoints (e.g.
    /// `/props`, which returns the resolved `chat_template` string when the
    /// server was started with `--jinja`). Used by the conformance console,
    /// not part of the OpenAI-compatible surface.
    async fn raw_get(&self, path: &str) -> Result<serde_json::Value, RuntimeError> {
        let url = format!("{}{path}", self.server_root());
        let response = self
            .client
            .get(&url)
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
        response
            .json()
            .await
            .map_err(|e| RuntimeError::ProxyError(format!("Invalid JSON from backend: {e}")))
    }

    /// Passthrough for llama-server's own diagnostic POST endpoints (e.g.
    /// `/apply-template`, which renders `messages`/`tools` into a prompt
    /// string without running inference). See `raw_get`.
    async fn raw_post(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, RuntimeError> {
        let url = format!("{}{path}", self.server_root());
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
        response
            .json()
            .await
            .map_err(|e| RuntimeError::ProxyError(format!("Invalid JSON from backend: {e}")))
    }
}

/// `llama-server --version` output keyed by binary path. A backend instance is
/// recreated on every load (fit planner / context fallback rewrite the args), so
/// caching per instance meant re-running the binary on every model switch.
static SERVER_VERSION_CACHE: LazyLock<StdMutex<HashMap<String, String>>> =
    LazyLock::new(|| StdMutex::new(HashMap::new()));

impl LlamaCppBackend {
    fn detect_server_version_background(&self) {
        if let Some(cached) = SERVER_VERSION_CACHE
            .lock()
            .ok()
            .and_then(|cache| cache.get(&self.config.command).cloned())
        {
            if let Ok(mut slot) = self.server_version.try_lock() {
                *slot = Some(cached);
            } else {
                let slot = Arc::clone(&self.server_version);
                tokio::spawn(async move {
                    *slot.lock().await = Some(cached);
                });
            }
            return;
        }

        let command = self.config.command.clone();
        let slot = Arc::clone(&self.server_version);
        tokio::spawn(async move {
            let output = Command::new(&command).arg("--version").output().await;
            let version = match output {
                Ok(out) if out.status.success() => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    if stdout.is_empty() {
                        String::from_utf8_lossy(&out.stderr).trim().to_string()
                    } else {
                        stdout
                    }
                }
                _ => String::new(),
            };
            if version.is_empty() {
                return;
            }
            if let Ok(mut cache) = SERVER_VERSION_CACHE.lock() {
                cache.insert(command, version.clone());
            }
            *slot.lock().await = Some(version);
        });
    }

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
pub(crate) struct SseLineParser<T> {
    inner: Pin<Box<dyn Stream<Item = Result<bytes::Bytes, RuntimeError>> + Send>>,
    buffer: Vec<u8>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> SseLineParser<T> {
    pub(crate) fn new(
        stream: impl Stream<Item = Result<bytes::Bytes, RuntimeError>> + Send + 'static,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            buffer: Vec::with_capacity(4096),
            _marker: std::marker::PhantomData,
        }
    }
}

/// Parse one SSE data payload into `T`, normalizing any object-valued
/// `function.arguments` to a string first (the shape is generic over `T`,
/// so this is a no-op for response types without `tool_calls`, e.g.
/// `CompletionChunk`).
pub(crate) fn parse_sse_json<T: serde::de::DeserializeOwned>(
    json_str: &str,
) -> Result<T, serde_json::Error> {
    let mut value: serde_json::Value = serde_json::from_str(json_str)?;
    super::tool_probe::normalize_tool_call_arguments(&mut value);
    serde_json::from_value(value)
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

                match parse_sse_json::<T>(json_str) {
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
                    match parse_sse_json::<T>(json_str) {
                        Ok(chunk) => return std::task::Poll::Ready(Some(Ok(chunk))),
                        Err(_) => return std::task::Poll::Ready(None),
                    }
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tool_call_normalization_tests {
    use super::LlamaCppBackend;
    use crate::config::ModelConfig;
    use serde_json::json;

    #[test]
    fn diagnostic_endpoints_use_server_root_without_v1_prefix() {
        let config: ModelConfig = toml::from_str(
            r#"
backend = "llama.cpp"
display_name = "test"
command = "llama-server"
args = []
backend_url = "http://127.0.0.1:8087/v1"
health_url = "http://127.0.0.1:8087/health"
"#,
        )
        .expect("model config");
        let backend = LlamaCppBackend::new("test", &config);

        assert_eq!(backend.server_root(), "http://127.0.0.1:8087");
    }

    // This test exercises the exact call shape `chat()` deserializes into
    // `ChatCompletionResponse`, proving normalization must run before
    // `serde_json::from_value::<ChatCompletionResponse>` or deserialization
    // fails outright (FunctionCall.arguments is typed String).
    #[test]
    fn object_arguments_become_a_parseable_string_before_typed_deserialize() {
        let mut raw = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": {"message": "hello"}}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        crate::backend::tool_probe::normalize_tool_call_arguments(&mut raw);
        let response: crate::types::chat::ChatCompletionResponse =
            serde_json::from_value(raw).expect("must deserialize after normalization");
        let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].function.arguments, "{\"message\":\"hello\"}");
    }
}
