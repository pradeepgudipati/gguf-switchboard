//! TypeSafe System One API for decision models (`POST /v1/systemone`).
//!
//! Thin proxy over llama-server's own `/v1/systemone`: the scheduler loads
//! the decision model (like any other switch), the request body is forwarded
//! untouched, and the backend's answers come straight back. No streaming —
//! decision models score in a single forward pass.

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json};
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::kind_guard::{DECISION_KINDS, require_kind};
use crate::metrics::{ACTIVE_REQUESTS, INFERENCE_LATENCY, REQUEST_TOTAL};
use crate::state::AppState;
use crate::types::systemone::{SystemoneRequest, SystemoneResponse};

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

/// Score typed questions against a state with a decision model.
#[utoipa::path(
    post,
    path = "/v1/systemone",
    tag = "systemone",
    request_body(
        content = SystemoneRequest,
    ),
    responses(
        (status = 200, description = "Per-question answers with probabilities", body = SystemoneResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
#[instrument(skip(state, request), fields(model = %request.model))]
pub async fn systemone(
    State(state): State<Arc<AppState>>,
    Json(request): Json<SystemoneRequest>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();
    ACTIVE_REQUESTS.inc();
    // Created immediately so early returns (`?`) below cannot leak the gauge.
    let _guard = ActiveGuard;

    if request.questions.is_empty() {
        return Err(RuntimeError::InvalidRequest(
            "'questions' must not be empty".to_string(),
        ));
    }

    let cfg = state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    require_kind(&request.model, &cfg, DECISION_KINDS, "/v1/systemone")?;
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();
    let _request_guard = state.scheduler.track_request(&model_id);
    // Inference-only timer; model load wait is exported separately.
    let start = std::time::Instant::now();

    let body = serde_json::to_value(&request).map_err(|e| {
        RuntimeError::InternalError(format!("Failed to serialize systemone request: {e}"))
    })?;
    let raw = backend.raw_post("/v1/systemone", body).await?;
    let mut response: SystemoneResponse = serde_json::from_value(raw).map_err(|e| {
        RuntimeError::ProxyError(format!("Invalid systemone response from backend: {e}"))
    })?;
    response.model = model_id.clone();
    // `output_tokens` is always 0 (nothing sampled); total == input.
    let total = response
        .usage
        .input_tokens
        .saturating_add(response.usage.output_tokens);

    let _ = state.token_db.record(
        &model_id,
        "/v1/systemone",
        response.usage.input_tokens,
        response.usage.output_tokens,
        total,
        None,
    );

    INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());

    Ok(Json(response))
}

/// Usage defaults: the backend always reports both counters, but tolerate
/// their absence so a version skew can't turn answers into a 502.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::systemone::SystemoneUsage;

    #[test]
    fn systemone_usage_defaults_to_zero() {
        let usage: SystemoneUsage = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }

    #[test]
    fn systemone_response_round_trips_backend_shape() {
        let raw = serde_json::json!({
            "model": "Kev-4B-Q4_K_M.gguf",
            "answers": {
                "angry": {"type": "noul", "noul": 0.8208},
                "urgency": {
                    "type": "score",
                    "score": 2.282,
                    "legend": {"0": "can wait", "1": "today"},
                    "probabilities": {"0": 0.5, "1": 0.5},
                    "confidence": 0.0
                }
            },
            "usage": {"input_tokens": 130, "output_tokens": 0}
        });
        let response: SystemoneResponse = serde_json::from_value(raw).unwrap();
        assert_eq!(response.answers.len(), 2);
        assert_eq!(response.usage.input_tokens, 130);
        // Unknown top-level fields survive via flatten; typed fields stay typed.
        let back = serde_json::to_value(&response).unwrap();
        assert_eq!(back["answers"]["angry"]["noul"], 0.8208);
    }

    #[test]
    fn systemone_request_keeps_state_and_extra_fields() {
        let raw = serde_json::json!({
            "model": "bespoke-nimble-9b-v3",
            "state": {"ticket": "charged twice", "plan": "pro"},
            "questions": {
                "refund": {"type": "noul", "instructions": "Is a refund requested?"}
            }
        });
        let request: SystemoneRequest = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(request.questions.len(), 1);
        // Re-serialized body is what the backend receives — nothing lost.
        assert_eq!(
            serde_json::to_value(&request).unwrap()["state"],
            raw["state"]
        );
    }
}
