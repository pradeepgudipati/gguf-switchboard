use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Response, StatusCode, header};
use axum::response::{IntoResponse, Json};
use futures::StreamExt;
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::kind_guard::{CHAT_KINDS, require_kind};
use crate::metrics::{ACTIVE_REQUESTS, INFERENCE_LATENCY, REQUEST_TOTAL, STREAMING_REQUESTS};
use crate::proxy::GuardedStream;
use crate::state::AppState;
use crate::types::anthropic::{
    MessagesRequest, MessagesResponse, StreamEvent, Usage, to_anthropic_response, to_openai_request,
};

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

struct StreamingGuard;
impl Drop for StreamingGuard {
    fn drop(&mut self) {
        STREAMING_REQUESTS.dec();
    }
}

/// Anthropic Messages API endpoint.
#[utoipa::path(
    post,
    path = "/v1/messages",
    tag = "Anthropic",
    request_body(
        content = MessagesRequest,
        example = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello, Claude!"}]
        })
    ),
    responses(
        (status = 200, description = "Anthropic message response", body = MessagesResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
#[instrument(skip(state, request), fields(model = %request.model, stream = request.stream.unwrap_or(false)))]
pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MessagesRequest>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();
    ACTIVE_REQUESTS.inc();

    let start = std::time::Instant::now();
    let model_id = request.model.clone();
    let stream = request.stream == Some(true);

    // Validate model exists and is a chat model
    let cfg = state
        .scheduler
        .model_config(&model_id)
        .ok_or_else(|| RuntimeError::ModelNotFound(model_id.clone()))?;
    require_kind(&model_id, &cfg, CHAT_KINDS, "/v1/messages")?;

    let backend = state.scheduler.ensure_loaded(&model_id).await?;
    let request_guard = state.scheduler.track_request(&model_id);

    // Convert Anthropic request → OpenAI request
    let mut openai_req = to_openai_request(&request);
    openai_req.stream = Some(stream);

    if stream {
        STREAMING_REQUESTS.inc();

        let openai_stream = backend.chat_stream(openai_req).await?;

        // Record streaming request
        let _ = state
            .token_db
            .record(&model_id, "/v1/messages", 0, 0, 0, None);

        let model_for_stream = model_id.clone();
        let mapped = openai_stream.map(move |chunk| {
            match chunk {
                Ok(chunk) => {
                    // Convert OpenAI chunk to Anthropic SSE event
                    let events = convert_chunk_to_anthropic_events(&model_for_stream, &chunk);
                    let sse_output = events
                        .into_iter()
                        .map(|ev| {
                            let json = serde_json::to_string(&ev).unwrap_or_default();
                            format!("event: {event_type}\ndata: {json}\n\n", event_type = event_type_name(&ev))
                        })
                        .collect::<String>();
                    Ok::<_, std::convert::Infallible>(sse_output)
                }
                Err(e) => {
                    let err_json = serde_json::json!({"type": "error", "error": {"type": "api_error", "message": e.to_string()}});
                    Ok::<_, std::convert::Infallible>(format!("event: error\ndata: {err_json}\n\n"))
                }
            }
        });

        let done =
            futures::stream::once(async { Ok::<_, std::convert::Infallible>("".to_string()) });
        let full_stream = mapped.chain(done);

        let guarded = GuardedStream::new(
            full_stream,
            vec![
                Box::new(request_guard),
                Box::new(ActiveGuard),
                Box::new(StreamingGuard),
            ],
        );

        let body = Body::from_stream(guarded.map(|s: Result<String, _>| {
            s.map(bytes::Bytes::from)
                .map_err(|e| std::io::Error::other(e.to_string()))
        }));

        INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .header("x-accel-buffering", "no")
            .body(body)
            .unwrap())
    } else {
        let _guard = ActiveGuard;
        let _request_guard = request_guard;

        let openai_response = backend.chat(openai_req).await?;
        let anthropic_response = to_anthropic_response(&model_id, &openai_response);

        // Record token usage
        let _ = state.token_db.record(
            &model_id,
            "/v1/messages",
            anthropic_response.usage.input_tokens,
            anthropic_response.usage.output_tokens,
            anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
            None,
        );

        INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());
        Ok(Json(anthropic_response).into_response())
    }
}

/// Convert an OpenAI streaming chunk to Anthropic SSE events.
fn convert_chunk_to_anthropic_events(
    model: &str,
    chunk: &crate::types::chat::ChatCompletionChunk,
) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    // First chunk with role → message_start
    for choice in &chunk.choices {
        if choice.delta.role.is_some() {
            events.push(StreamEvent::MessageStart {
                message: crate::types::anthropic::MessagesResponse {
                    id: chunk.id.clone(),
                    response_type: "message".to_string(),
                    role: crate::types::anthropic::Role::Assistant,
                    content: Vec::new(),
                    model: model.to_string(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: Usage {
                        input_tokens: 0,
                        output_tokens: 0,
                    },
                },
            });
            events.push(StreamEvent::Ping {});
        }

        // Content text delta
        if let Some(text) = &choice.delta.content
            && !text.is_empty()
        {
            events.push(StreamEvent::ContentBlockDelta {
                index: choice.index as usize,
                delta: crate::types::anthropic::ContentDelta::TextDelta { text: text.clone() },
            });
        }

        // Stop reason → message_delta + message_stop
        if let Some(finish_reason) = &choice.finish_reason {
            let stop_reason = Some(match finish_reason.as_str() {
                "stop" => "end_turn".to_string(),
                "length" => "max_tokens".to_string(),
                "tool_calls" => "tool_use".to_string(),
                other => other.to_string(),
            });

            events.push(StreamEvent::MessageDelta {
                delta: crate::types::anthropic::MessageDeltaBody {
                    stop_reason,
                    stop_sequence: None,
                },
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: chunk
                        .usage
                        .as_ref()
                        .map(|u| u.completion_tokens)
                        .unwrap_or(0),
                },
            });
            events.push(StreamEvent::MessageStop {});
        }
    }

    events
}

/// Get the SSE event type name for an Anthropic stream event.
fn event_type_name(event: &StreamEvent) -> &'static str {
    match event {
        StreamEvent::MessageStart { .. } => "message_start",
        StreamEvent::ContentBlockStart { .. } => "content_block_start",
        StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        StreamEvent::ContentBlockStop { .. } => "content_block_stop",
        StreamEvent::MessageDelta { .. } => "message_delta",
        StreamEvent::MessageStop { .. } => "message_stop",
        StreamEvent::Ping { .. } => "ping",
    }
}
