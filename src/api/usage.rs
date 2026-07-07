use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Json};
use serde::Deserialize;
use tracing::instrument;
use utoipa::IntoParams;

use crate::errors::RuntimeError;
use crate::state::AppState;

#[derive(Debug, Deserialize, IntoParams)]
pub struct UsageQuery {
    /// Optional model id to filter by
    pub model: Option<String>,
    /// Number of recent records to return (default 50)
    pub limit: Option<u32>,
}

/// Token usage statistics per model.
#[utoipa::path(
    get,
    path = "/v1/usage",
    tag = "usage",
    params(UsageQuery),
    responses(
        (status = 200, description = "Token usage statistics"),
        (status = 500, description = "Internal server error")
    )
)]
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

/// Recent token usage records.
#[utoipa::path(
    get,
    path = "/v1/usage/recent",
    tag = "usage",
    params(UsageQuery),
    responses(
        (status = 200, description = "Recent usage records"),
        (status = 500, description = "Internal server error")
    )
)]
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
