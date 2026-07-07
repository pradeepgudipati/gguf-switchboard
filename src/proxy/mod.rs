use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::http::{header, Response, StatusCode};
use axum::response::IntoResponse;
use futures::Stream;
use futures::StreamExt;

/// A stream wrapper that holds guards which are dropped when the stream
/// finishes. This ensures metrics like `ACTIVE_REQUESTS` and
/// `STREAMING_REQUESTS` remain incremented for the full lifetime of the
/// stream, not just the handler function.
pub struct GuardedStream {
    inner: Pin<Box<dyn Stream<Item = Result<String, std::convert::Infallible>> + Send>>,
    _guards: Vec<Box<dyn Send>>,
}

impl GuardedStream {
    pub fn new(
        stream: impl Stream<Item = Result<String, std::convert::Infallible>> + Send + 'static,
        guards: Vec<Box<dyn Send>>,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            _guards: guards,
        }
    }
}

impl Stream for GuardedStream {
    type Item = Result<String, std::convert::Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// Proxy a raw SSE response from a reqwest response directly to the client.
#[allow(dead_code)]
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
