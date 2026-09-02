pub mod anthropic;
pub mod audio;
pub mod chat;
pub mod completions;
pub mod conformance;
pub mod embeddings;
pub mod health;
pub mod metrics;
pub mod models;
pub mod rerank;
pub mod responses;
pub mod usage;

use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{Request, State};
use axum::http::{HeaderValue, header};
use axum::middleware::{self, Next};
use axum::response::Response;
use serde_json::Value;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::{Config, SwaggerUi};

use crate::openapi_models::inject_model_enums;
use crate::state::AppState;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "GGUF Switchboard",
        description = "llama-swap alternative in Rust — OpenAI-compatible GGUF runtime with memory-pressure eviction, OOM context fallback, and usage tracking",
        version = env!("CARGO_PKG_VERSION")
    ),
    paths(
        health::health,
        health::status,
        metrics::metrics,
        models::list_models,
        models::get_model,
        models::get_model_runtime,
        models::registry_json,
        models::refresh_models,
        chat::chat_completions,
        completions::completions,
        embeddings::embeddings,
        rerank::rerank,
        responses::responses,
        audio::transcriptions,
        anthropic::messages,
        audio::speech,
        usage::usage,
        usage::recent_usage,
        conformance::inspect,
        conformance::resolve_template,
        conformance::run_battery,
        conformance::compare,
        conformance::history_list,
        conformance::history_get,
        conformance::history_delete,
        conformance::history_clear,
    ),
    components(schemas(
        health::HealthResponse,
        health::StatusResponse,
        health::GpuInfo,
        health::HostInfo,
        crate::types::ModelInfo,
        crate::types::ListModelsResponse,
        crate::types::RuntimeProfileInfo,
        models::RefreshModelsResponse,
        crate::types::Usage,
        crate::types::FunctionCall,
        crate::types::ToolCall,
        crate::types::chat::ChatCompletionRequest,
        crate::types::chat::ChatCompletionResponse,
        crate::types::chat::ChatMessage,
        crate::types::chat::Role,
        crate::types::chat::ChatChoice,
        crate::types::chat::Tool,
        crate::types::chat::FunctionDefinition,
        crate::types::chat::ImageUrl,
        crate::types::chat::ContentPart,
        crate::types::chat::ChatCompletionChunk,
        crate::types::chat::ChatChunkChoice,
        crate::types::chat::ChatDelta,
        crate::types::completions::CompletionRequest,
        crate::types::completions::CompletionResponse,
        crate::types::completions::Prompt,
        crate::types::completions::CompletionChoice,
        crate::types::completions::CompletionChunk,
        crate::types::completions::CompletionChunkChoice,
        crate::types::embeddings::EmbeddingRequest,
        crate::types::embeddings::EmbeddingResponse,
        crate::types::embeddings::EmbeddingInput,
        crate::types::embeddings::EmbeddingData,
        crate::types::embeddings::EmbeddingUsage,
        crate::types::rerank::RerankRequest,
        crate::types::rerank::RerankResponse,
        crate::types::rerank::RerankResult,
        crate::types::rerank::RerankUsage,
        crate::types::responses::ResponseRequest,
        crate::types::responses::ResponseInput,
        crate::types::responses::ResponseMessage,
        crate::types::responses::ResponseTool,
        crate::types::responses::ResponseFunctionTool,
        crate::types::responses::ResponseResult,
        crate::types::responses::ResponseOutput,
        crate::types::responses::ResponseMessageOutput,
        crate::types::responses::ResponseFunctionCallOutput,
        crate::types::responses::ResponseContent,
        crate::types::responses::ResponseUsage,
        crate::types::audio::TranscriptionRequest,
        crate::types::anthropic::MessagesRequest,
        crate::types::anthropic::MessagesResponse,
        crate::types::anthropic::Message,
        crate::types::anthropic::Role,
        crate::types::anthropic::MessageContent,
        crate::types::anthropic::ContentBlock,
        crate::types::anthropic::ToolDefinition,
        crate::types::anthropic::ToolChoice,
        crate::types::anthropic::ResponseBlock,
        crate::types::anthropic::Usage,
        crate::types::audio::SpeechRequest,
        crate::types::chat::Content,
        conformance::InspectResponse,
        conformance::ResolveTemplateRequest,
        conformance::ResolveTemplateResponse,
        conformance::CompareMode,
        conformance::CompareRequest,
        conformance::CompareResult,
        conformance::CompareReport,
        crate::conformance::classify::ToolCallClassification,
        crate::conformance::classify::ToolCallLocation,
        crate::conformance::battery::BatteryCase,
        crate::conformance::battery::CaseVerdict,
        crate::conformance::battery::BatteryReport,
        crate::conformance::ConformanceRunSummary,
        crate::conformance::ConformanceRunDetail,
        conformance::HistoryClearResponse,
    )),
    tags(
        (name = "health", description = "Health and status endpoints"),
        (name = "models", description = "Model management endpoints"),
        (name = "chat", description = "Chat completion endpoints"),
        (name = "completions", description = "Text completion endpoints"),
        (name = "Anthropic", description = "Anthropic Messages API endpoints"),
        (name = "embeddings", description = "Embedding endpoints"),
        (name = "rerank", description = "Document reranking endpoints"),
        (name = "responses", description = "Responses API endpoints"),
        (name = "audio", description = "Audio transcription and speech endpoints"),
        (name = "usage", description = "Usage statistics endpoints"),
        (name = "metrics", description = "Prometheus metrics endpoint"),
        (name = "conformance", description = "Tool-calling and chat-template conformance diagnostics"),
    )
)]
pub struct ApiDoc;

async fn openapi_json(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut doc = serde_json::to_value(ApiDoc::openapi()).unwrap_or(Value::Null);
    let models: Vec<_> = state
        .scheduler
        .model_ids()
        .into_iter()
        .filter_map(|id| state.scheduler.model_config(&id).map(|cfg| (id, cfg)))
        .collect();
    inject_model_enums(&mut doc, &models);
    Json(doc)
}

/// Swagger UI's static assets (`swagger-initializer.js` and friends) are
/// served with no explicit cache headers, so browsers fall back to
/// heuristic caching and can silently keep serving a stale bundle after a
/// deploy — the exact symptom that made a shipped fix look broken until a
/// hard refresh. Force revalidation on every load for that path prefix.
async fn no_cache_swagger_assets(req: Request, next: Next) -> Response {
    let is_swagger_ui =
        req.uri().path().starts_with("/swagger-ui") || req.uri().path() == "/api-docs/openapi.json";
    let mut response = next.run(req).await;
    if is_swagger_ui {
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    }
    response
}

/// Build the top-level router with all OpenAI-compatible endpoints.
pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::permissive();

    let trace = TraceLayer::new_for_http().make_span_with(|req: &axum::http::Request<_>| {
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

    let swagger_config = Config::new(["/api-docs/openapi.json"])
        .try_it_out_enabled(true)
        .show_mutated_request(true);

    Router::new()
        .route(
            "/",
            axum::routing::get(|| async { axum::response::Redirect::permanent("/swagger-ui/") }),
        )
        .route("/api-docs/openapi.json", axum::routing::get(openapi_json))
        .merge(SwaggerUi::new("/swagger-ui").config(swagger_config))
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
        .route("/v1/rerank", axum::routing::post(rerank::rerank))
        .route("/v1/models", axum::routing::get(models::list_models))
        .route(
            "/v1/models/refresh",
            axum::routing::post(models::refresh_models),
        )
        .route(
            "/v1/models/registry.json",
            axum::routing::get(models::registry_json),
        )
        .route(
            "/v1/models/{model_id}/runtime",
            axum::routing::get(models::get_model_runtime),
        )
        .route(
            "/v1/models/{model_id}",
            axum::routing::get(models::get_model),
        )
        .route("/v1/responses", axum::routing::post(responses::responses))
        .route(
            "/v1/audio/transcriptions",
            axum::routing::post(audio::transcriptions),
        )
        .route("/v1/messages", axum::routing::post(anthropic::messages))
        .route("/v1/audio/speech", axum::routing::post(audio::speech))
        .route("/v1/usage", axum::routing::get(usage::usage))
        .route("/v1/usage/recent", axum::routing::get(usage::recent_usage))
        .route(
            "/v1/conformance/inspect",
            axum::routing::post(conformance::inspect),
        )
        .route(
            "/v1/conformance/resolve-template",
            axum::routing::post(conformance::resolve_template),
        )
        .route(
            "/v1/conformance/battery/{model_id}",
            axum::routing::post(conformance::run_battery),
        )
        .route(
            "/v1/conformance/compare",
            axum::routing::post(conformance::compare),
        )
        .route(
            "/v1/conformance/history",
            axum::routing::get(conformance::history_list).delete(conformance::history_clear),
        )
        .route(
            "/v1/conformance/history/{id}",
            axum::routing::get(conformance::history_get).delete(conformance::history_delete),
        )
        .route("/health", axum::routing::get(health::health))
        .route("/status", axum::routing::get(health::status))
        .route("/metrics", axum::routing::get(metrics::metrics))
        .layer(middleware::from_fn(no_cache_swagger_assets))
        .layer(cors)
        .layer(trace)
        .with_state(state)
}
