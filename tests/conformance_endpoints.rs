//! Endpoint conformance: HTTP contracts beyond tool-calling.
//!
//! Covers three areas the existing suites (`integration`, `responses`,
//! `scheduler_switch`) do not pin down end-to-end:
//!
//! 1. **API endpoint contracts** — chat/completions, legacy completions,
//!    embeddings, `/v1/models{,/{id}}`, `/health` + `/status` shapes, and the
//!    OpenAI-style error envelope on 404/400.
//! 2. **Streaming SSE behavior** — `text/event-stream` content type, `data:`
//!    framing, `[DONE]` sentinel, no leakage of the sentinel into Responses
//!    streams (covered in `responses.rs`), and stable per-chunk model naming.
//! 3. **Scheduler semantics observable over HTTP** — unknown model is 404,
//!    drain timeout surfaces as 409 `model_busy`, and a failed switch keeps
//!    the previous model servable.

mod support;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use gguf_switchboard::api;
use gguf_switchboard::config::Config;
use gguf_switchboard::conformance::ConformanceHistory;
use gguf_switchboard::db::TokenDb;
use gguf_switchboard::scheduler::Scheduler;
use gguf_switchboard::state::AppState;
use serde_json::{Value, json};
use support::{FakeLlamaServer, scheduler_from_config, write_scheduler_config};
use tower::ServiceExt;

async fn endpoint_server() -> (Router, Arc<Scheduler>, FakeLlamaServer, FakeLlamaServer) {
    let fake_a = FakeLlamaServer::start().await;
    let fake_b = FakeLlamaServer::start().await;
    let config_file = write_scheduler_config(&fake_a, &fake_b);
    let config = Config::load(config_file.path().to_str().unwrap()).unwrap();
    let scheduler = Arc::new(Scheduler::new(config.clone()).await.unwrap());
    let database = tempfile::NamedTempFile::new().unwrap();
    let token_db = Arc::new(TokenDb::open(database.path()).unwrap());
    let conformance_database = tempfile::NamedTempFile::new().unwrap();
    let conformance_history =
        Arc::new(ConformanceHistory::open(conformance_database.path()).unwrap());
    let state = Arc::new(AppState::new(
        config,
        Arc::clone(&scheduler),
        token_db,
        conformance_history,
    ));
    let server = api::create_router(state);
    (server, scheduler, fake_a, fake_b)
}

async fn post_json(
    server: Router,
    path: &str,
    body: Value,
) -> (StatusCode, Value, Vec<(String, String)>) {
    let response = server
        .oneshot(
            Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response
        .headers()
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    let raw = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
    (status, json, headers)
}

async fn get_json(server: Router, path: &str) -> (StatusCode, Value) {
    let response = server
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let raw = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
    (status, json)
}

// ── 1. API endpoint contracts ────────────────────────────────────────────

#[tokio::test]
async fn chat_nonstreaming_returns_openai_completion_shape() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;
    let (status, body, _) = post_json(
        server,
        "/v1/chat/completions",
        json!({
            "model": "model-a",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": false
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "model-a");
    assert!(body["choices"].as_array().is_some_and(|c| !c.is_empty()));
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["usage"]["total_tokens"].as_u64().is_some());

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn legacy_completions_returns_text_shape() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;
    let (status, body, _) = post_json(
        server,
        "/v1/completions",
        json!({"model": "model-a", "prompt": "Say hello"}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "text_completion");
    assert!(body["choices"][0]["text"].as_str().is_some());

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn embeddings_preserves_input_cardinality() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;
    let (status, body, _) = post_json(
        server,
        "/v1/embeddings",
        json!({"model": "model-emb", "input": ["hello", "world"]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("embedding data array");
    assert_eq!(data.len(), 2, "one embedding per input");
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
    assert!(
        data[0]["embedding"]
            .as_array()
            .is_some_and(|e| !e.is_empty())
    );

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn chat_kind_model_is_rejected_on_embeddings_endpoint() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;
    let (status, body, _) = post_json(
        server,
        "/v1/embeddings",
        json!({"model": "model-a", "input": "hello"}),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "invalid_request");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("kind=chat")
    );

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn models_list_and_get_agree_on_shape() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;

    let (status, list) = get_json(server.clone(), "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list["object"], "list");
    let ids: Vec<&str> = list["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(ids.contains(&"model-a") && ids.contains(&"model-b"));

    let (status, one) = get_json(server, "/v1/models/model-a").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(one["id"], "model-a");
    assert_eq!(one["object"], "model");

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn health_and_status_share_liveness_shape() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;

    let (status, health) = get_json(server.clone(), "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["status"], "ok");

    let (status, reported) = get_json(server, "/status").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(reported["status"], "ok");
    assert!(reported["uptime_secs"].as_u64().is_some());
    assert!(reported["configured_models"].as_array().is_some());

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn unknown_model_returns_openai_error_envelope() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;
    let (status, body, _) = post_json(
        server,
        "/v1/chat/completions",
        json!({
            "model": "no-such-model",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "model_not_found");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no-such-model")
    );

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn malformed_body_is_rejected_before_model_load() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;
    // `messages` is required; axum must 4xx/5xx without ever loading a model.
    let (status, _, _) =
        post_json(server, "/v1/chat/completions", json!({"model": "model-a"})).await;
    assert!(
        status.is_client_error() || status.is_server_error(),
        "unexpected status: {status}"
    );
    assert!(
        fake_a.requests.lock().unwrap().is_empty(),
        "malformed request must not reach the backend"
    );

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

// ── 2. Streaming SSE behavior ────────────────────────────────────────────

async fn post_stream(
    server: Router,
    path: &str,
    body: Value,
) -> (StatusCode, String, Vec<(String, String)>) {
    let response = server
        .oneshot(
            Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response
        .headers()
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    let raw = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(raw.to_vec()).unwrap(), headers)
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> &'a str {
    headers
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
        .unwrap_or("")
}

#[tokio::test]
async fn chat_stream_uses_sse_framing_and_done_sentinel() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;
    let (status, body, headers) = post_stream(
        server,
        "/v1/chat/completions",
        json!({
            "model": "model-a",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(
        header(&headers, "content-type").starts_with("text/event-stream"),
        "headers: {headers:?}"
    );
    assert!(body.contains("data: "), "chunks must be SSE data frames");
    assert!(
        body.trim_end().ends_with("data: [DONE]"),
        "stream must end with [DONE]:\n{body}"
    );

    // Every data frame except the sentinel must be valid JSON with the model set.
    for line in body
        .lines()
        .filter(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
    {
        let chunk: Value =
            serde_json::from_str(line.trim_start_matches("data: ")).expect("chunk JSON");
        assert_eq!(chunk["model"], "model-a");
    }

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn completions_stream_uses_sse_framing_and_done_sentinel() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;
    let (status, body, headers) = post_stream(
        server,
        "/v1/completions",
        json!({"model": "model-a", "prompt": "hi", "stream": true}),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(header(&headers, "content-type").starts_with("text/event-stream"));
    assert!(body.contains("data: "));
    assert!(body.trim_end().ends_with("data: [DONE]"));

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

// ── 3. Scheduler semantics observable over HTTP ──────────────────────────

#[tokio::test]
async fn drain_timeout_surfaces_as_409_model_busy() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;

    // Load model-a directly so the handler path has a resident model.
    scheduler
        .ensure_loaded("model-a")
        .await
        .expect("load model-a");

    // Occupy model-a past the 2s test drain timeout (see support::write_scheduler_config).
    let _guard = scheduler.track_request("model-a");
    let (status, body, _) = post_json(
        server,
        "/v1/chat/completions",
        json!({
            "model": "model-b",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::CONFLICT, "body: {body}");
    assert_eq!(body["error"]["code"], "model_busy");

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn failed_switch_over_http_keeps_previous_model_servable() {
    let (server, scheduler, fake_a, fake_b) = endpoint_server().await;

    // model-a serves fine.
    let (status, _, _) = post_json(
        server.clone(),
        "/v1/chat/completions",
        json!({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // model-b's backend command does not exist: 5xx and rollback to model-a.
    let (failed_status, _, _) = post_json(
        server.clone(),
        "/v1/chat/completions",
        json!({"model": "model-b", "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert!(failed_status.is_server_error(), "status: {failed_status}");

    // model-a still serves after the failed switch.
    let (status, body, _) = post_json(
        server,
        "/v1/chat/completions",
        json!({"model": "model-a", "messages": [{"role": "user", "content": "hi again"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "model-a");

    // Drain the scheduler's queue: model-b's failure must have been rolled back.
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn oom_classification_drives_context_fallback_not_port_conflicts() {
    // Unit-level: the classifier must treat allocation signatures as OOM and
    // everything else as a different kind (so only OOM/health-timeout halves `-c`).
    use gguf_switchboard::load_failure::classify_load_failure;

    for stderr in [
        "CUDA error: out of memory",
        "failed to allocate cuda buffer",
        "ggml_allocr: cannot allocate",
        "VK_ERROR_OUT_OF_DEVICE_MEMORY",
    ] {
        assert!(
            classify_load_failure(stderr, "").is_oom(),
            "should classify as OOM: {stderr}"
        );
    }
    for stderr in [
        "error: address already in use",
        "error: No such file or directory",
        "unexpected token in config",
    ] {
        assert!(
            !classify_load_failure(stderr, "").is_oom(),
            "must NOT classify as OOM: {stderr}"
        );
    }
}

#[tokio::test]
async fn scheduler_from_config_helper_loads_fake_models() {
    // Guards the shared fixture itself: both fake backends must be reachable
    // through a scheduler built by the test helper.
    let fake_a = FakeLlamaServer::start().await;
    let fake_b = FakeLlamaServer::start().await;
    let config = write_scheduler_config(&fake_a, &fake_b);
    let scheduler = scheduler_from_config(&config).await;
    scheduler
        .ensure_loaded("model-a")
        .await
        .expect("model-a loads");
    assert_eq!(scheduler.loaded_model().await.as_deref(), Some("model-a"));
    scheduler.shutdown().await.expect("shutdown");
}
