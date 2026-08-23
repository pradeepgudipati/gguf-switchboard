use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::config::{ModelConfig, check_vllm_available};
use crate::errors::RuntimeError;
use crate::types::chat::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, normalize_chat_chunk,
    normalize_chat_response,
};
use crate::types::completions::{CompletionChunk, CompletionRequest, CompletionResponse};
use crate::types::embeddings::{EmbeddingRequest, EmbeddingResponse};
use crate::types::rerank::{RerankRequest, RerankResponse};

use super::Backend;
use super::llama_cpp::SseLineParser;

/// vLLM backend: spawns `<vllm_command> run [--project <dir>] vllm serve ...`
/// as a child process and proxies OpenAI-compatible requests to it. vLLM's
/// server speaks the same `/v1/chat/completions`, `/v1/completions`,
/// `/v1/embeddings`, `/health` surface as llama-server, so no changes to the
/// public API layer are needed — only this trait implementation differs.
pub struct VllmBackend {
    model_id: String,
    config: ModelConfig,
    client: Client,
    process: Arc<Mutex<Option<Child>>>,
    running: AtomicBool,
    startup_stderr: Arc<Mutex<String>>,
    server_version: Arc<Mutex<Option<String>>>,
}

impl VllmBackend {
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

    /// Extract `--project <dir>` from `self.config.args`, if present.
    fn project_dir(&self) -> Option<&str> {
        find_flag_arg(&self.config.args, "--project")
    }

    /// Extract the model path/repo id: the positional argument right after `serve`.
    fn find_model_arg(&self) -> Option<&str> {
        let args = &self.config.args;
        args.iter()
            .position(|a| a == "serve")
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    }

    fn command_display(&self) -> String {
        format!("{} {}", self.config.command, self.config.args.join(" "))
    }
}

fn find_flag_arg<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

#[async_trait]
impl Backend for VllmBackend {
    async fn load(&self) -> Result<(), RuntimeError> {
        // Deploy check: `uv` on PATH and `vllm` importable in the resolved
        // environment. Fails fast with an install nudge instead of letting
        // the spawn below fail opaquely.
        if let Err(reason) = check_vllm_available(&self.config.command, self.project_dir()) {
            return Err(RuntimeError::ModelLoadingFailed(reason));
        }

        // Validate the safetensors model path exists when it's a local dir
        // (an entry served straight from an `hf_repo` has no local path to check).
        if let Some(model_ref) = self.find_model_arg() {
            let looks_local = model_ref.starts_with('/')
                || model_ref.starts_with("./")
                || model_ref.starts_with("../")
                || (model_ref.len() > 1 && model_ref.as_bytes()[1] == b':'); // Windows drive letter
            if looks_local && !std::path::Path::new(model_ref).exists() {
                return Err(RuntimeError::ModelNotFound(format!(
                    "vLLM model path not found: '{model_ref}'. Run `ggs models pull <repo> \
                     --backend vllm` or check the path in models.toml."
                )));
            }
        }

        info!(model = %self.model_id, command = %self.command_display(), "Starting backend process");

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
                while let Ok(Some(line)) = reader.next_line().await {
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
        Ok(())
    }

    async fn unload(&self) -> Result<(), RuntimeError> {
        info!(model = %self.model_id, "Stopping backend process");
        let mut guard = self.process.lock().await;
        if let Some(mut child) = guard.take() {
            let pid = child.id();
            info!(model = %self.model_id, ?pid, "Sending SIGTERM to backend");

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
            match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
                Ok(Ok(status)) => {
                    info!(model = %self.model_id, %status, "Backend exited after SIGTERM");
                }
                Ok(Err(e)) => {
                    warn!(model = %self.model_id, error = %e, "Error waiting for backend after SIGTERM, sending SIGKILL");
                    let _ = child.kill().await;
                }
                Err(_) => {
                    warn!(model = %self.model_id, "Backend did not exit within 10s, sending SIGKILL");
                    match child.kill().await {
                        Ok(()) => {
                            match tokio::time::timeout(Duration::from_secs(10), child.wait()).await
                            {
                                Ok(Ok(status)) => {
                                    info!(model = %self.model_id, %status, "Backend exited after SIGKILL");
                                }
                                Ok(Err(e)) => {
                                    warn!(model = %self.model_id, error = %e, "Error waiting for backend after SIGKILL");
                                }
                                Err(_) => {
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
        let resp = self.forward_json("/chat/completions", body).await?;
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
        let resp = self.forward_json("/completions", body).await?;
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
        // vLLM's OpenAI-compatible server exposes the same Jina/Cohere-style
        // `/rerank` endpoint as llama-server for score/reranking models.
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
}
