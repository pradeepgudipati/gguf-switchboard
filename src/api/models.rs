use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use chrono::Utc;

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
        .map(|id| ModelInfo {
            id,
            object: "model".to_string(),
            created: Utc::now().timestamp(),
            owned_by: "local".to_string(),
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
    if state.scheduler.model_config(&model_id).is_none() {
        return Err(RuntimeError::ModelNotFound(model_id));
    }
    Ok(Json(ModelInfo::new(model_id)))
}
