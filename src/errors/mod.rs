use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Backend not healthy: {0}")]
    _BackendNotHealthy(String),

    #[error("Model loading failed: {0}")]
    ModelLoadingFailed(String),

    #[error("Model loading timeout: {0}")]
    ModelLoadingTimeout(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Proxy error: {0}")]
    ProxyError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Authentication required")]
    _Unauthorized,

    #[error("Rate limit exceeded")]
    _RateLimitExceeded,

    #[error("Service unavailable: {0}")]
    _ServiceUnavailable(String),
}

#[derive(Serialize)]
struct OpenAIErrorResponse {
    error: OpenAIError,
}

#[derive(Serialize)]
struct OpenAIError {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
    param: Option<String>,
    code: Option<String>,
}

impl IntoResponse for RuntimeError {
    fn into_response(self) -> Response {
        let (status, error_type, code) = match &self {
            RuntimeError::ModelNotFound(_) => (StatusCode::NOT_FOUND, "invalid_request_error", "model_not_found"),
            RuntimeError::BackendError(_) => (StatusCode::BAD_GATEWAY, "server_error", "backend_error"),
            RuntimeError::_BackendNotHealthy(_) => (StatusCode::SERVICE_UNAVAILABLE, "server_error", "backend_not_healthy"),
            RuntimeError::ModelLoadingFailed(_) => (StatusCode::SERVICE_UNAVAILABLE, "server_error", "model_loading_failed"),
            RuntimeError::ModelLoadingTimeout(_) => (StatusCode::GATEWAY_TIMEOUT, "server_error", "model_loading_timeout"),
            RuntimeError::ConfigError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", "config_error"),
            RuntimeError::ProxyError(_) => (StatusCode::BAD_GATEWAY, "server_error", "proxy_error"),
            RuntimeError::SerializationError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", "serialization_error"),
            RuntimeError::InternalError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", "internal_error"),
            RuntimeError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error", "invalid_request"),
            RuntimeError::_Unauthorized => (StatusCode::UNAUTHORIZED, "authentication_error", "unauthorized"),
            RuntimeError::_RateLimitExceeded => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", "rate_limit_exceeded"),
            RuntimeError::_ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "server_error", "service_unavailable"),
        };

        let body = OpenAIErrorResponse {
            error: OpenAIError {
                message: self.to_string(),
                error_type: error_type.to_string(),
                param: None,
                code: Some(code.to_string()),
            },
        };

        (status, axum::Json(body)).into_response()
    }
}

impl From<reqwest::Error> for RuntimeError {
    fn from(err: reqwest::Error) -> Self {
        RuntimeError::ProxyError(err.to_string())
    }
}

impl From<serde_json::Error> for RuntimeError {
    fn from(err: serde_json::Error) -> Self {
        RuntimeError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for RuntimeError {
    fn from(err: std::io::Error) -> Self {
        RuntimeError::InternalError(err.to_string())
    }
}

impl From<toml::de::Error> for RuntimeError {
    fn from(err: toml::de::Error) -> Self {
        RuntimeError::ConfigError(err.to_string())
    }
}
