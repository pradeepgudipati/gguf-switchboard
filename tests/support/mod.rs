use axum::Json;
use axum::Router;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use gguf_switchboard::config::Config;
use gguf_switchboard::scheduler::Scheduler;
use serde_json::json;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tempfile::NamedTempFile;

pub struct FakeLlamaServer {
    pub health_url: String,
    pub backend_url: String,
    pub healthy: Arc<AtomicBool>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl FakeLlamaServer {
    pub async fn start() -> Self {
        let healthy = Arc::new(AtomicBool::new(true));
        let healthy_check = Arc::clone(&healthy);
        let app = Router::new().route(
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

pub fn write_scheduler_config(fake_a: &FakeLlamaServer, fake_b: &FakeLlamaServer) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("temp config");
    write!(
        file,
        r#"
bind = "127.0.0.1:9090"
startup_timeout = 10
idle_timeout = 600
default_backend = "llama.cpp"
switch_drain_timeout_secs = 2
priority_load_cooldown_secs = 60

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
"#,
        a_backend = fake_a.backend_url,
        a_health = fake_a.health_url,
        b_backend = fake_b.backend_url,
        b_health = fake_b.health_url,
    )
    .expect("write config");
    file
}

pub async fn scheduler_from_config(file: &NamedTempFile) -> Scheduler {
    let path = file.path().to_str().expect("utf8 path");
    let config = Config::load(path).expect("load config");
    Scheduler::new(config).await.expect("scheduler")
}
