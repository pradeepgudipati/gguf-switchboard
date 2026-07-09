use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Response, StatusCode, header};
use axum::response::{IntoResponse, Json};
use futures::StreamExt;
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::metrics::{ACTIVE_REQUESTS, INFERENCE_LATENCY, REQUEST_TOTAL, STREAMING_REQUESTS};
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

    let start = std::time::Instant::now();
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();

    if request.stream == Some(true) {
        STREAMING_REQUESTS.inc();

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
            vec![Box::new(ActiveGuard), Box::new(StreamingGuard)],
        );

        let body = Body::from_stream(guarded.map(|s: Result<String, _>| {
            s.map(bytes::Bytes::from)
                .map_err(|e| std::io::Error::other(e.to_string()))
        }));

        INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .header("x-accel-buffering", "no")
            .body(body)
            .unwrap())
    } else {
        let _guard = ActiveGuard;
        let mut response = backend.completions(request).await?;
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

        INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());
        Ok(Json(response).into_response())
    }
}
