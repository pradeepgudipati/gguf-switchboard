use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::http::header;
use axum::response::IntoResponse;
use serde::Serialize;
use utoipa::ToSchema;

use crate::errors::RuntimeError;
use crate::state::AppState;
use crate::types::{ListModelsResponse, ModelInfo};

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
    let models = state
        .scheduler
        .model_ids()
        .into_iter()
        .filter_map(|id| {
            state
                .scheduler
                .model_config(&id)
                .map(|cfg| ModelInfo::from_config(id, &cfg))
        })
        .collect();

    Ok(Json(ListModelsResponse::new(models)))
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
    Ok(Json(ModelInfo::from_config(model_id, &cfg)))
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
