use axum::http::{header, StatusCode};
use axum::response::Response;

use crate::metrics;

/// `GET /metrics` — Prometheus-format metrics endpoint.
pub async fn metrics() -> Response {
    let body = metrics::gather();
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(axum::body::Body::from(body))
        .unwrap_or_else(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(axum::body::Body::empty())
                .unwrap()
        })
}
