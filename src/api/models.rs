use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::http::header;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};

use crate::db::ThroughputTrendPoint;
use crate::errors::RuntimeError;
use crate::state::AppState;
use crate::types::{ListModelsResponse, ModelInfo, RuntimeProfileInfo};

/// List all configured models.
#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "models",
    responses(
        (status = 200, description = "List of available models", body = ListModelsResponse),
        (status = 500, description = "Internal server error")
    )
)]
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ListModelsResponse>, RuntimeError> {
    let throughput = state.token_db.throughput_by_model().unwrap_or_default();
    let mut models = Vec::new();
    for id in state.scheduler.model_ids() {
        let Some(cfg) = state.scheduler.model_config(&id) else {
            continue;
        };
        let mut info = ModelInfo::from_config(id.clone(), &cfg);
        info.tools_verified = tools_verified_bool(state.scheduler.tool_capability(&id).await);
        info.tokens_per_sec = throughput.get(&id).map(|s| round1(s.tokens_per_sec));
        models.push(info);
    }

    Ok(Json(ListModelsResponse::new(models)))
}

/// Convert a `ToolCapability` verdict into the API's tri-state boolean:
/// `Verified` -> `Some(true)`, `Failed` -> `Some(false)`, `Skipped` or
/// never-probed -> `None` (field omitted from the response).
fn tools_verified_bool(
    verdict: Option<crate::backend::tool_probe::ToolCapability>,
) -> Option<bool> {
    use crate::backend::tool_probe::ToolCapability;
    match verdict {
        Some(ToolCapability::Verified) => Some(true),
        Some(ToolCapability::Failed(_)) => Some(false),
        Some(ToolCapability::Skipped) | None => None,
    }
}

/// Retrieve a single model by ID.
#[utoipa::path(
    get,
    path = "/v1/models/{model_id}",
    tag = "models",
    params(
        ("model_id" = String, Path, description = "The model ID to retrieve", example = "gemma-4-e4b")
    ),
    responses(
        (status = 200, description = "Model details", body = ModelInfo),
        (status = 404, description = "Model not found")
    )
)]
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelInfo>, RuntimeError> {
    let Some(cfg) = state.scheduler.model_config(&model_id) else {
        return Err(RuntimeError::ModelNotFound(model_id));
    };
    let mut info = ModelInfo::from_config(model_id.clone(), &cfg);
    info.tools_verified = tools_verified_bool(state.scheduler.tool_capability(&model_id).await);
    info.tokens_per_sec = state
        .token_db
        .throughput_by_model()
        .unwrap_or_default()
        .get(&model_id)
        .map(|s| round1(s.tokens_per_sec));
    Ok(Json(info))
}

/// Round to one decimal place — tok/s figures are rough estimates, extra
/// precision is noise.
fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

/// Retrieve the runtime profile for a model.
///
/// Returns the effective load parameters (context, ngl, split, KV cache) that
/// were used to successfully start the model, along with the reason for any
/// degradation from the requested parameters.
#[utoipa::path(
    get,
    path = "/v1/models/{model_id}/runtime",
    tag = "models",
    params(
        ("model_id" = String, Path, description = "The model ID", example = "gemma-4-e4b")
    ),
    responses(
        (status = 200, description = "Runtime profile", body = RuntimeProfileInfo),
        (status = 404, description = "Model not found")
    )
)]
pub async fn get_model_runtime(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<RuntimeProfileInfo>, RuntimeError> {
    let Some(cfg) = state.scheduler.model_config(&model_id) else {
        return Err(RuntimeError::ModelNotFound(model_id));
    };
    let Some(rp) = cfg.runtime_profile else {
        return Err(RuntimeError::ModelNotFound(format!(
            "{model_id} has no runtime profile (model has not been loaded yet)"
        )));
    };
    Ok(Json(RuntimeProfileInfo {
        effective_context: rp.context_size,
        gpu_layers: rp.ngl,
        split_mode: rp.split_mode,
        tensor_split: rp.tensor_split,
        cache_type_k: rp.cache_type_k,
        cache_type_v: rp.cache_type_v,
        batch_size: rp.batch_size,
        ubatch_size: rp.ubatch_size,
        embedding_concurrency: rp.embedding_concurrency,
        reason: rp.reason,
        profile_source: rp.profile_source,
    }))
}

#[derive(Debug, Deserialize, IntoParams)]
#[into_params(parameter_in = Query)]
pub struct ThroughputQuery {
    /// Trailing window in days (default 90, max 90).
    #[param(example = 90)]
    pub days: Option<i64>,
}

/// Measured generation throughput for a model: a current rough tok/s figure
/// plus a per-day trendline.
#[derive(Debug, Serialize, ToSchema)]
pub struct ThroughputResponse {
    pub model: String,
    pub window_days: i64,
    /// Median tok/s over the recent window, `None` if never measured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_sec: Option<f64>,
    pub samples: usize,
    /// Per-day median tok/s, oldest first. Days without samples are omitted.
    pub trend: Vec<ThroughputTrendPoint>,
}

/// Generation throughput history for a model.
///
/// Throughput is a rough estimate: output tokens divided by wall-clock time
/// for each non-streaming request, aggregated as a daily median. Samples are
/// kept for 90 days.
#[utoipa::path(
    get,
    path = "/v1/models/{model_id}/throughput",
    tag = "models",
    params(
        ("model_id" = String, Path, description = "The model ID", example = "gemma-4-e4b"),
        ThroughputQuery
    ),
    responses(
        (status = 200, description = "Throughput trend", body = ThroughputResponse),
        (status = 404, description = "Model not found")
    )
)]
pub async fn get_model_throughput(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Query(query): Query<ThroughputQuery>,
) -> Result<Json<ThroughputResponse>, RuntimeError> {
    if state.scheduler.model_config(&model_id).is_none() {
        return Err(RuntimeError::ModelNotFound(model_id));
    }
    let days = query.days.unwrap_or(90).clamp(1, 90);
    let trend = state.token_db.throughput_trend(&model_id, days)?;
    let stat = state
        .token_db
        .throughput_by_model()
        .unwrap_or_default()
        .remove(&model_id);
    Ok(Json(ThroughputResponse {
        model: model_id,
        window_days: days,
        tokens_per_sec: stat.as_ref().map(|s| round1(s.tokens_per_sec)),
        samples: stat.map(|s| s.samples).unwrap_or(0),
        trend,
    }))
}

/// Download the portable model registry as JSON (shared across local AI tools).
#[utoipa::path(
    get,
    path = "/v1/models/registry.json",
    tag = "models",
    responses(
        (status = 200, description = "Portable model registry JSON", content_type = "application/json")
    )
)]
pub async fn registry_json(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, RuntimeError> {
    let body = state.registry_json.read().await.clone();
    Ok(([(header::CONTENT_TYPE, "application/json")], body))
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RefreshModelsResponse {
    pub added: usize,
    pub removed: usize,
    pub total: usize,
    pub models_dir: String,
}

/// Rescan model directories, persist `models.toml`, and hot-swap the live registry.
#[utoipa::path(
    post,
    path = "/v1/models/refresh",
    tag = "models",
    responses(
        (status = 200, description = "Registry refreshed", body = RefreshModelsResponse),
        (status = 500, description = "Refresh failed")
    )
)]
pub async fn refresh_models(
    State(state): State<Arc<AppState>>,
) -> Result<(StatusCode, Json<RefreshModelsResponse>), RuntimeError> {
    let result = state.refresh_models().await?;
    Ok((
        StatusCode::OK,
        Json(RefreshModelsResponse {
            added: result.added,
            removed: result.removed,
            total: result.total,
            models_dir: result.models_dir,
        }),
    ))
}
