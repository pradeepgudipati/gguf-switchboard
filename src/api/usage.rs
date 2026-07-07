use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Json};
use serde::Deserialize;
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct UsageQuery {
    /// Optional model id to filter by
    pub model: Option<String>,
    /// Number of recent records to return (default 50)
    pub limit: Option<u32>,
}

/// `GET /v1/usage` — token usage statistics per model.
#[instrument(skip(state))]
pub async fn usage(
    State(state): State<Arc<AppState>>,
    Query(query): Query<UsageQuery>,
) -> Result<impl IntoResponse, RuntimeError> {
    if let Some(model) = &query.model {
        let stats = state.token_db.get_model_usage(model)?;
        Ok(Json(serde_json::json!({
            "object": "usage",
            "model": model,
            "data": stats,
        })))
    } else {
        let stats = state.token_db.get_usage_stats()?;
        Ok(Json(serde_json::json!({
            "object": "usage",
            "data": stats,
        })))
    }
}

/// `GET /v1/usage/recent` — recent token usage records.
#[instrument(skip(state))]
pub async fn recent_usage(
    State(state): State<Arc<AppState>>,
    Query(query): Query<UsageQuery>,
) -> Result<impl IntoResponse, RuntimeError> {
    let limit = query.limit.unwrap_or(50);
    let records = state.token_db.get_recent_records(limit)?;
    Ok(Json(serde_json::json!({
        "object": "list",
        "data": records,
    })))
}
