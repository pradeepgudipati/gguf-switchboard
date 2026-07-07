use std::sync::Arc;

use axum::extract::{Path, State};
use axum::Json;
use chrono::Utc;

use crate::errors::RuntimeError;
use crate::state::AppState;
use crate::types::{ListModelsResponse, ModelInfo};

/// `GET /v1/models` — list all configured models.
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ListModelsResponse>, RuntimeError> {
    let models = state
        .scheduler
        .model_ids()
        .into_iter()
        .map(|id| {
            ModelInfo {
                id,
                object: "model".to_string(),
                created: Utc::now(),
                owned_by: "local".to_string(),
            }
        })
        .collect();

    Ok(Json(ListModelsResponse::new(models)))
}

/// `GET /v1/models/{model_id}` — retrieve a single model.
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<ModelInfo>, RuntimeError> {
    if state.scheduler.model_config(&model_id).is_none() {
        return Err(RuntimeError::ModelNotFound(model_id));
    }
    Ok(Json(ModelInfo::new(model_id)))
}
