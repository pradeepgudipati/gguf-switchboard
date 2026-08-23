use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json};
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::kind_guard::{RERANK_KINDS, require_kind};
use crate::metrics::{ACTIVE_REQUESTS, INFERENCE_LATENCY, REQUEST_TOTAL};
use crate::state::AppState;
use crate::types::rerank::{RerankRequest, RerankResponse};

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

/// Score and rank documents against a query using a cross-encoder reranker model.
#[utoipa::path(
    post,
    path = "/v1/rerank",
    tag = "rerank",
    request_body(
        content = RerankRequest,
    ),
    responses(
        (status = 200, description = "Ranked documents", body = RerankResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
#[instrument(skip(state, request), fields(model = %request.model))]
pub async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RerankRequest>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();
    ACTIVE_REQUESTS.inc();
    // Created immediately so early returns (`?`) below cannot leak the gauge.
    let _guard = ActiveGuard;

    if request.documents.is_empty() {
        return Err(RuntimeError::InvalidRequest(
            "'documents' must not be empty".to_string(),
        ));
    }

    let cfg = state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    require_kind(&request.model, &cfg, RERANK_KINDS, "/v1/rerank")?;
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();
    let concurrency = state
        .scheduler
        .model_config(&model_id)
        .and_then(|cfg| cfg.runtime_profile)
        .and_then(|profile| profile.embedding_concurrency)
        .unwrap_or(1) as usize;
    let _admission_permit = state
        .embedding_admission
        .acquire(&model_id, concurrency)
        .await?;
    let _request_guard = state.scheduler.track_request(&model_id);
    // Inference-only timer; model load wait is exported separately.
    let start = std::time::Instant::now();

    let top_n = request.top_n;
    let return_documents = request.return_documents;
    let documents = request.documents.clone();

    let mut response = backend.rerank(request).await?;

    // Rank by relevance (descending) and apply top_n ourselves so behavior is
    // consistent regardless of what the backend already did with these fields.
    response
        .results
        .sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    if let Some(top_n) = top_n {
        response.results.truncate(top_n as usize);
    }
    if return_documents {
        for result in &mut response.results {
            result.document = documents.get(result.index as usize).cloned();
        }
    } else {
        for result in &mut response.results {
            result.document = None;
        }
    }
    response.model = model_id.clone();

    let _ = state.token_db.record(
        &model_id,
        "/v1/rerank",
        response.usage.prompt_tokens,
        0,
        response.usage.total_tokens,
        None,
    );

    INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());

    Ok(Json(response))
}
