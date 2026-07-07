use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Response, StatusCode};
use axum::response::{IntoResponse, Json};
use futures::StreamExt;
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::metrics::{ACTIVE_REQUESTS, INFERENCE_LATENCY, REQUEST_TOTAL, STREAMING_REQUESTS};
use crate::state::AppState;
use crate::types::chat::ChatCompletionRequest;

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

/// `POST /v1/chat/completions` — chat completions with optional streaming.
#[instrument(skip(state, request), fields(model = %request.model, stream = request.stream.unwrap_or(false)))]
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();
    ACTIVE_REQUESTS.inc();
    let _guard = ActiveGuard;

    let start = std::time::Instant::now();
    let backend = state.scheduler.ensure_loaded(&request.model).await?;

    if request.stream == Some(true) {
        STREAMING_REQUESTS.inc();
        let stream = backend.chat_stream(request).await?;
        let mapped = stream.map(move |chunk| match chunk {
            Ok(c) => {
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

        let body = Body::from_stream(full_stream.map(|s: Result<String, _>| {
            s.map(bytes::Bytes::from)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
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
        let response = backend.chat(request).await?;
        INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());
        Ok(Json(response).into_response())
    }
}
