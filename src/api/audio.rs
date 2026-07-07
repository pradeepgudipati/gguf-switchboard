use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, Response, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::metrics::REQUEST_TOTAL;
use crate::state::AppState;

/// Transcribe audio to text.
///
/// Forwards the request to the loaded backend's native `/audio/transcriptions`
/// endpoint.
#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    tag = "audio",
    request_body = serde_json::Value,
    responses(
        (status = 200, description = "Transcription result"),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
#[instrument(skip(state))]
pub async fn transcriptions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<serde_json::Value>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();

    let model = request
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RuntimeError::InvalidRequest("Missing 'model' field".to_string()))?
        .to_string();

    let backend = state.scheduler.ensure_loaded(&model).await?;

    let url = format!("{}/audio/transcriptions", backend.backend_url());
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| RuntimeError::ProxyError(format!("Audio transcription request failed: {e}")))?;

    if resp.status().is_success() {
        let body = resp.text().await.map_err(|e| {
            RuntimeError::ProxyError(format!("Failed to read transcription response: {e}"))
        })?;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap())
    } else {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Err(RuntimeError::BackendError(format!(
            "Transcription backend returned {status}: {text}"
        )))
    }
}

/// Generate speech from text.
///
/// Forwards the request to the loaded backend's `/audio/speech` endpoint
/// and returns the raw audio bytes.
#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    tag = "audio",
    request_body = serde_json::Value,
    responses(
        (status = 200, description = "Generated audio bytes", content_type = "audio/mpeg"),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
#[instrument(skip(state))]
pub async fn speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<serde_json::Value>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();

    let model = request
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| RuntimeError::InvalidRequest("Missing 'model' field".to_string()))?
        .to_string();

    let backend = state.scheduler.ensure_loaded(&model).await?;

    let url = format!("{}/audio/speech", backend.backend_url());
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .map_err(|e| RuntimeError::ProxyError(format!("Audio speech request failed: {e}")))?;

    if resp.status().is_success() {
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("audio/mpeg")
            .to_string();
        let bytes = resp.bytes().await.map_err(|e| {
            RuntimeError::ProxyError(format!("Failed to read speech response: {e}"))
        })?;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(bytes))
            .unwrap())
    } else {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        Err(RuntimeError::BackendError(format!(
            "Speech backend returned {status}: {text}"
        )))
    }
}
