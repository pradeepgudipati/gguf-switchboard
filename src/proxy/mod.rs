use axum::body::Body;
use axum::http::{header, Response, StatusCode};
use axum::response::IntoResponse;
use futures::StreamExt;

/// Proxy a raw SSE response from a reqwest response directly to the client.
pub async fn proxy_sse_response(response: reqwest::Response) -> impl IntoResponse {
    let stream = response.bytes_stream().map(|chunk| {
        chunk.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    });

    let body = Body::from_stream(stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .header("x-accel-buffering", "no")
        .body(body)
        .unwrap_or_else(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::empty())
                .unwrap()
        })
}
