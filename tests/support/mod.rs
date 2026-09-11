#![allow(dead_code)]

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::http::{Response, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use gguf_switchboard::config::Config;
use gguf_switchboard::scheduler::Scheduler;
use serde_json::json;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct FakeLlamaServer {
    pub health_url: String,
    pub backend_url: String,
    pub healthy: Arc<AtomicBool>,
    pub requests: Arc<Mutex<Vec<serde_json::Value>>>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl FakeLlamaServer {
    pub async fn start() -> Self {
        let healthy = Arc::new(AtomicBool::new(true));
        let healthy_check = Arc::clone(&healthy);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured_requests = Arc::clone(&requests);
        let app = Router::new()
            .route(
                "/health",
                get(move || {
                    let healthy_check = Arc::clone(&healthy_check);
                    async move {
                        if healthy_check.load(Ordering::SeqCst) {
                            (StatusCode::OK, Json(json!({ "status": "ok" }))).into_response()
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE.into_response()
                        }
                    }
                }),
            )
            .route(
                "/v1/chat/completions",
                post(move |Json(body): Json<serde_json::Value>| {
                    let captured_requests = Arc::clone(&captured_requests);
                    async move {
                        let stream = body["stream"].as_bool() == Some(true);
                        captured_requests.lock().unwrap().push(body);
                        if stream {
                            let chunks = [
                                json!({
                                    "id": "chatcmpl_tool",
                                    "object": "chat.completion.chunk",
                                    "created": 1_700_000_000,
                                    "model": "model-a",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "tool_calls": [{
                                                "index": 0,
                                                "id": "call_weather",
                                                "type": "function",
                                                "function": {
                                                    "name": "get_weather",
                                                    "arguments": "{\"loc"
                                                }
                                            }]
                                        },
                                        "finish_reason": null
                                    }]
                                }),
                                json!({
                                    "id": "chatcmpl_tool",
                                    "object": "chat.completion.chunk",
                                    "created": 1_700_000_000,
                                    "model": "model-a",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": 0,
                                                "id": "",
                                                "type": "",
                                                "function": {
                                                    "name": "",
                                                    "arguments": "ation\":\"Pune\"}"
                                                }
                                            }]
                                        },
                                        "finish_reason": "tool_calls"
                                    }],
                                    "usage": {
                                        "prompt_tokens": 10,
                                        "completion_tokens": 5,
                                        "total_tokens": 15
                                    }
                                }),
                            ];
                            let mut payload = chunks
                                .iter()
                                .map(|chunk| format!("data: {chunk}\n\n"))
                                .collect::<String>();
                            payload.push_str("data: [DONE]\n\n");
                            return Response::builder()
                                .status(StatusCode::OK)
                                .header(header::CONTENT_TYPE, "text/event-stream")
                                .body(Body::from(payload))
                                .unwrap();
                        }
                        Json(json!({
                            "id": "chatcmpl_tool",
                            "object": "chat.completion",
                            "created": 1_700_000_000,
                            "model": "model-a",
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": null,
                                    "tool_calls": [{
                                        "id": "call_weather",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": "{\"location\":\"Pune\"}"
                                        }
                                    }]
                                },
                                "finish_reason": "tool_calls"
                            }],
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15
                            }
                        }))
                        .into_response()
                    }
                }),
            )
            .route(
                "/v1/completions",
                post(move |Json(body): Json<serde_json::Value>| async move {
                    let stream = body["stream"].as_bool() == Some(true);
                    if stream {
                        let chunk = json!({
                            "id": "cmpl-1",
                            "object": "text_completion.chunk",
                            "created": 1_700_000_000,
                            "model": "model-a",
                            "choices": [{
                                "index": 0,
                                "text": " hello",
                                "finish_reason": null
                            }]
                        });
                        let payload = format!("data: {chunk}\n\ndata: [DONE]\n\n");
                        return Response::builder()
                            .status(StatusCode::OK)
                            .header(header::CONTENT_TYPE, "text/event-stream")
                            .body(Body::from(payload))
                            .unwrap();
                    }
                    Json(json!({
                        "id": "cmpl-1",
                        "object": "text_completion",
                        "created": 1_700_000_000,
                        "model": "model-a",
                        "choices": [{
                            "index": 0,
                            "text": " hello world",
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 2,
                            "total_tokens": 7
                        }
                    }))
                    .into_response()
                }),
            )
            .route(
                "/v1/embeddings",
                post(move |Json(body): Json<serde_json::Value>| async move {
                    let count = body["input"]
                        .as_array()
                        .map(|a| a.len())
                        .or_else(|| body["input"].as_str().map(|_| 1))
                        .unwrap_or(1);
                    let data: Vec<serde_json::Value> = (0..count)
                        .map(|i| {
                            json!({
                                "object": "embedding",
                                "index": i,
                                "embedding": [0.1, 0.2, 0.3]
                            })
                        })
                        .collect();
                    Json(json!({
                        "object": "list",
                        "data": data,
                        "model": "model-a",
                        "usage": {
                            "prompt_tokens": 4,
                            "total_tokens": 4
                        }
                    }))
                    .into_response()
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind fake llama-server");
        let addr = listener.local_addr().expect("local addr");
        let health_url = format!("http://{addr}/health");
        let backend_url = format!("http://{addr}/v1");
        let (shutdown, rx) = tokio::sync::oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .expect("serve fake llama-server");
        });
        Self {
            health_url,
            backend_url,
            healthy,
            requests,
            shutdown: Some(shutdown),
            task,
        }
    }

    pub fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::SeqCst);
    }
}

impl Drop for FakeLlamaServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        self.task.abort();
    }
}

/// Handle for a scheduler test config: owns both the directory containing
/// `config.toml` and the parsed path. The config lives in a dedicated temp
/// directory (not the shared temp root) because `Config::resolve_models`
/// auto-loads a sibling `models.toml` — a stray one in the shared temp dir
/// would hijack the inline models below and fail the load.
pub struct SchedulerConfigDir {
    _dir: tempfile::TempDir,
    path: std::path::PathBuf,
}

impl SchedulerConfigDir {
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

pub fn write_scheduler_config(
    fake_a: &FakeLlamaServer,
    fake_b: &FakeLlamaServer,
) -> SchedulerConfigDir {
    write_scheduler_config_with(fake_a, fake_b, "")
}

/// Like [`write_scheduler_config`] but appends `extra_toml` to the top-level section
/// (e.g. `switch_strategy = "load_first"`).
pub fn write_scheduler_config_with(
    fake_a: &FakeLlamaServer,
    fake_b: &FakeLlamaServer,
    extra_toml: &str,
) -> SchedulerConfigDir {
    let dir = tempfile::TempDir::new().expect("temp config dir");
    let path = dir.path().join("config.toml");
    let mut file = std::fs::File::create(&path).expect("create temp config");
    write!(
        file,
        r#"
bind = "127.0.0.1:9090"
startup_timeout = 10
idle_timeout = 600
default_backend = "llama.cpp"
switch_drain_timeout_secs = 2
priority_load_cooldown_secs = 60
{extra_toml}

[models.model-a]
backend = "llama.cpp"
display_name = "Model A"
command = "sleep"
args = ["3600"]
backend_url = "{a_backend}"
health_url = "{a_health}"

[models.model-b]
backend = "llama.cpp"
display_name = "Model B"
command = "/definitely/missing/llama-server"
args = []
backend_url = "{b_backend}"
health_url = "{b_health}"

[models.model-emb]
backend = "llama.cpp"
display_name = "Embedding Model"
command = "sleep"
args = ["3600"]
backend_url = "{a_backend}"
health_url = "{a_health}"
kind = "embedding"
"#,
        extra_toml = extra_toml,
        a_backend = fake_a.backend_url,
        a_health = fake_a.health_url,
        b_backend = fake_b.backend_url,
        b_health = fake_b.health_url,
    )
    .expect("write config");
    SchedulerConfigDir { _dir: dir, path }
}

pub async fn scheduler_from_config(file: &SchedulerConfigDir) -> Scheduler {
    let path = file.path().to_str().expect("utf8 path");
    let config = Config::load(path).expect("load config");
    Scheduler::new(config).await.expect("scheduler")
}
