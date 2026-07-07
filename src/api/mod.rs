pub mod audio;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod health;
pub mod metrics;
pub mod models;
pub mod responses;

use std::sync::Arc;

use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

/// Build the top-level router with all OpenAI-compatible endpoints.
pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::permissive();

    let trace = TraceLayer::new_for_http()
        .make_span_with(|req: &axum::http::Request<_>| {
            let request_id = req
                .headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown");
            tracing::info_span!(
                "request",
                method = %req.method(),
                uri = %req.uri(),
                request_id = %request_id,
            )
        });

    Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(chat::chat_completions),
        )
        .route(
            "/v1/completions",
            axum::routing::post(completions::completions),
        )
        .route(
            "/v1/embeddings",
            axum::routing::post(embeddings::embeddings),
        )
        .route("/v1/models", axum::routing::get(models::list_models))
        .route(
            "/v1/models/{model_id}",
            axum::routing::get(models::get_model),
        )
        .route(
            "/v1/responses",
            axum::routing::post(responses::responses),
        )
        .route(
            "/v1/audio/transcriptions",
            axum::routing::post(audio::transcriptions),
        )
        .route("/v1/audio/speech", axum::routing::post(audio::speech))
        .route("/health", axum::routing::get(health::health))
        .route("/status", axum::routing::get(health::status))
        .route("/metrics", axum::routing::get(metrics::metrics))
        .layer(cors)
        .layer(trace)
        .with_state(state)
}
