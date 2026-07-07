use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json};
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::metrics::{ACTIVE_REQUESTS, INFERENCE_LATENCY, REQUEST_TOTAL};
use crate::state::AppState;
use crate::types::embeddings::EmbeddingRequest;

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

/// `POST /v1/embeddings` — generate embeddings.
#[instrument(skip(state, request), fields(model = %request.model))]
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();
    ACTIVE_REQUESTS.inc();
    let _guard = ActiveGuard;

    let start = std::time::Instant::now();
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let response = backend.embeddings(request).await?;

    // Record token usage
    let _ = state.token_db.record(
        &response.model,
        "/v1/embeddings",
        response.usage.prompt_tokens,
        0,
        response.usage.total_tokens,
        None,
    );

    INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());

    Ok(Json(response))
}
