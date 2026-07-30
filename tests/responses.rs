mod support;

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode, header};
use gguf_switchboard::api;
use gguf_switchboard::config::Config;
use gguf_switchboard::db::TokenDb;
use gguf_switchboard::scheduler::Scheduler;
use gguf_switchboard::state::AppState;
use serde_json::{Value, json};
use support::{FakeLlamaServer, write_scheduler_config};
use tower::ServiceExt;

async fn response_server() -> (Router, Arc<Scheduler>, FakeLlamaServer, FakeLlamaServer) {
    let fake_a = FakeLlamaServer::start().await;
    let fake_b = FakeLlamaServer::start().await;
    let config_file = write_scheduler_config(&fake_a, &fake_b);
    let config = Config::load(config_file.path().to_str().unwrap()).unwrap();
    let scheduler = Arc::new(Scheduler::new(config.clone()).await.unwrap());
    let database = tempfile::NamedTempFile::new().unwrap();
    let token_db = Arc::new(TokenDb::open(database.path()).unwrap());
    let state = Arc::new(AppState::new(config, Arc::clone(&scheduler), token_db));
    let server = api::create_router(state);
    (server, scheduler, fake_a, fake_b)
}

#[tokio::test]
async fn responses_handler_forwards_tools_and_returns_function_call_item() {
    let (server, scheduler, fake_a, fake_b) = response_server().await;

    let request_body = json!({
        "model": "model-a",
        "input": "What is the weather in Pune?",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            },
            "strict": true
        }],
        "tool_choice": "auto"
    });
    let response = server
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await;

    let response = response.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["output"][0]["type"], "function_call");
    assert_eq!(body["output"][0]["call_id"], "call_weather");
    assert_eq!(body["output"][0]["name"], "get_weather");
    assert_eq!(body["output"][0]["arguments"], "{\"location\":\"Pune\"}");
    {
        let requests = fake_a.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0]["tools"][0]["type"], "function");
        assert_eq!(requests[0]["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(requests[0]["tools"][0]["function"]["strict"], true);
        assert_eq!(requests[0]["tool_choice"], "auto");
    }

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn responses_handler_streams_strict_function_call_events() {
    let (server, scheduler, fake_a, fake_b) = response_server().await;
    let request_body = json!({
        "model": "model-a",
        "input": "What is the weather in Pune?",
        "stream": true,
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}}
            }
        }]
    });

    let response = server
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let body = String::from_utf8(
        to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(body.contains("event: response.created"));
    assert!(body.contains("event: response.output_item.added"));
    assert!(body.contains("event: response.function_call_arguments.delta"));
    assert!(body.contains("event: response.function_call_arguments.done"));
    assert!(body.contains("event: response.output_item.done"));
    assert!(body.contains("event: response.completed"));
    assert!(!body.contains("[DONE]"));

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}

#[tokio::test]
async fn responses_handler_rejects_unsupported_tool_with_400() {
    let (server, scheduler, fake_a, fake_b) = response_server().await;
    let request_body = json!({
        "model": "model-a",
        "input": "Search",
        "tools": [{
            "type": "web_search"
        }]
    });

    let response = server
        .oneshot(
            Request::post("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&request_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    scheduler.shutdown().await.unwrap();
    drop((fake_a, fake_b));
}
