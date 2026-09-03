use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Response, StatusCode, header};
use axum::response::{IntoResponse, Json};
use futures::StreamExt;
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::kind_guard::{CHAT_KINDS, require_kind};
use crate::metrics::{ACTIVE_REQUESTS, InferenceTimer, REQUEST_TOTAL, STREAMING_REQUESTS};
use crate::proxy::GuardedStream;
use crate::state::AppState;
use crate::types::completions::{CompletionRequest, CompletionResponse};

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

struct StreamingGuard;
impl Drop for StreamingGuard {
    fn drop(&mut self) {
        STREAMING_REQUESTS.dec();
    }
}

/// Text completions with optional streaming.
#[utoipa::path(
    post,
    path = "/v1/completions",
    tag = "completions",
    request_body(
        content = CompletionRequest,
        example = json!({
            "model": "gemma-4-e4b",
            "prompt": "Say hello in one sentence.",
            "max_tokens": 512
        })
    ),
    responses(
        (status = 200, description = "Text completion response", body = CompletionResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
#[instrument(skip(state, request), fields(model = %request.model, stream = request.stream.unwrap_or(false)))]
pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();
    ACTIVE_REQUESTS.inc();
    // Created immediately so early returns (`?`) below cannot leak the gauge.
    let active_guard = ActiveGuard;

    let cfg = state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    require_kind(&request.model, &cfg, CHAT_KINDS, "/v1/completions")?;
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();
    let request_guard = state.scheduler.track_request(&model_id);
    // Inference timer starts once the model is resident; load/switch time is
    // reported separately via gguf_switchboard_request_model_wait_seconds.
    let inference_timer = InferenceTimer::start();

    if request.stream == Some(true) {
        STREAMING_REQUESTS.inc();
        let streaming_guard = StreamingGuard;

        let stream = backend.completions_stream(request).await?;

        // Record streaming request (token counts not available in stream mode)
        let _ = state
            .token_db
            .record(&model_id, "/v1/completions", 0, 0, 0, None);

        let model_for_stream = model_id.clone();
        let mapped = stream.map(move |chunk| match chunk {
            Ok(mut c) => {
                c.model = model_for_stream.clone();
                let json = serde_json::to_string(&c).unwrap_or_default();
                Ok::<_, std::convert::Infallible>(format!("data: {json}\n\n"))
            }
            Err(e) => {
                let err_json = serde_json::json!({"error": {"message": e.to_string(), "type": "server_error"}});
                Ok::<_, std::convert::Infallible>(format!("data: {err_json}\n\n"))
            }
        });
        let done = futures::stream::once(async {
            Ok::<_, std::convert::Infallible>("data: [DONE]\n\n".to_string())
        });
        let full_stream = mapped.chain(done);

        // Embed guards into the stream so they're dropped when the stream
        // finishes, not when the handler returns.
        let guarded = GuardedStream::new(
            full_stream,
            vec![
                Box::new(request_guard),
                Box::new(active_guard),
                Box::new(streaming_guard),
                // Observes the inference latency histogram when the stream
                // completes (or the client disconnects), not when headers are sent.
                Box::new(inference_timer),
            ],
        );

        let body = Body::from_stream(guarded.map(|s: Result<String, _>| {
            s.map(bytes::Bytes::from)
                .map_err(|e| std::io::Error::other(e.to_string()))
        }));

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .header("x-accel-buffering", "no")
            .body(body)
            .unwrap())
    } else {
        let _guard = active_guard;
        let _request_guard = request_guard;
        let _inference_timer = inference_timer;
        let started = std::time::Instant::now();
        let mut response = backend.completions(request).await?;
        let elapsed = started.elapsed().as_secs_f64();
        response.model = model_id.clone();

        // Record token usage (completions endpoint uses prompt_tokens from usage)
        let _ = state.token_db.record(
            &model_id,
            "/v1/completions",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
            None,
        );
        let _ = state.token_db.record_throughput(
            &model_id,
            "/v1/completions",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            elapsed,
        );

        Ok(Json(response).into_response())
    }
}
