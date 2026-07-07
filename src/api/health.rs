use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::Serialize;
use utoipa::ToSchema;

use crate::state::AppState;

#[derive(Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub loaded_model: Option<String>,
}

#[derive(Serialize, ToSchema)]
pub struct StatusResponse {
    pub status: String,
    pub version: String,
    pub loaded_model: Option<String>,
    pub priority_model: Option<String>,
    pub configured_models: Vec<serde_json::Value>,
    pub uptime_secs: u64,
}

/// Basic liveness probe.
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
pub async fn health(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    let loaded = state.scheduler.loaded_model().await;
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        loaded_model: loaded,
    })
}

/// Detailed status report.
#[utoipa::path(
    get,
    path = "/status",
    tag = "health",
    responses(
        (status = 200, description = "Detailed service status", body = StatusResponse)
    )
)]
pub async fn status(
    State(state): State<Arc<AppState>>,
) -> Json<StatusResponse> {
    let loaded = state.scheduler.loaded_model().await;
    let priority = state.scheduler.priority_model();
    let models = state
        .scheduler
        .model_ids()
        .into_iter()
        .map(|id| {
            let cfg = state.scheduler.model_config(&id);
            serde_json::json!({
                "id": id,
                "display_name": cfg.map(|c| c.display_name.as_str()).unwrap_or(""),
                "backend": cfg.map(|c| c.backend.as_str()).unwrap_or(""),
                "priority": cfg.map(|c| c.priority).unwrap_or(false),
            })
        })
        .collect();

    let uptime_secs = state.started_at.elapsed().as_secs();

    Json(StatusResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        loaded_model: loaded,
        priority_model: priority,
        configured_models: models,
        uptime_secs,
    })
}
