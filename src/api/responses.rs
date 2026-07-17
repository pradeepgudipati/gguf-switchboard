use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Response, StatusCode, header};
use axum::response::{IntoResponse, Json};
use chrono::Utc;
use futures::StreamExt;
use uuid::Uuid;

use crate::errors::RuntimeError;
use crate::kind_guard::{CHAT_KINDS, require_kind};
use crate::metrics::{ACTIVE_REQUESTS, INFERENCE_LATENCY, REQUEST_TOTAL};
use crate::proxy::GuardedStream;
use crate::state::AppState;
use crate::types::chat::{ChatCompletionRequest, ChatMessage, Content, Role};
use crate::types::responses::{
    ResponseContent, ResponseInput, ResponseOutput, ResponseRequest, ResponseResult, ResponseUsage,
};

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

/// OpenAI Responses API with optional streaming.
///
/// This converts the Responses API request into a Chat Completion request
/// internally so any chat-capable backend can serve it.
#[utoipa::path(
    post,
    path = "/v1/responses",
    tag = "responses",
    request_body(
        content = ResponseRequest,
        example = json!({
            "model": "gemma-4-e4b",
            "input": "What is the capital of France?",
            "instructions": "Answer concisely in one sentence.",
            "max_output_tokens": 512,
            "stream": false
        })
    ),
    responses(
        (status = 200, description = "Response result", body = ResponseResult),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
pub async fn responses(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ResponseRequest>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();
    ACTIVE_REQUESTS.inc();

    let start = std::time::Instant::now();
    let cfg = state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    require_kind(&request.model, &cfg, CHAT_KINDS, "/v1/responses")?;
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();
    let request_guard = state.scheduler.track_request(&model_id);

    // Convert Responses API input to Chat Completion messages
    let mut messages = Vec::new();

    if let Some(ref instructions) = request.instructions {
        messages.push(ChatMessage {
            role: Role::System,
            content: Some(Content::Text(instructions.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    match &request.input {
        ResponseInput::Text(text) => {
            messages.push(ChatMessage {
                role: Role::User,
                content: Some(Content::Text(text.clone())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            });
        }
        ResponseInput::Messages(msgs) => {
            for msg in msgs {
                let role = match msg.role.as_str() {
                    "system" => Role::System,
                    "assistant" => Role::Assistant,
                    "tool" => Role::Tool,
                    _ => Role::User,
                };
                messages.push(ChatMessage {
                    role,
                    content: Some(Content::Text(msg.content.clone())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
        }
    }

    let chat_request = ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_output_tokens,
        stream: request.stream,
        n: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: request.user.clone(),
        tools: None,
        tool_choice: None,
        seed: None,
        response_format: request.response_format.clone(),
        chat_template_kwargs: None,
    };

    if request.stream == Some(true) {
        let stream = backend.chat_stream(chat_request).await?;
        let response_id = format!("resp_{}", Uuid::new_v4().simple());

        // Embed guard into the stream so it's dropped when the stream
        // finishes, not when the handler returns.
        let active_guard = ActiveGuard;

        let model_for_stream = model_id.clone();
        let mapped = stream.map(move |chunk| match chunk {
            Ok(c) => {
                let text = c
                    .choices
                    .first()
                    .and_then(|ch| ch.delta.content.as_deref())
                    .unwrap_or("");
                let status = if c.choices.iter().any(|ch| ch.finish_reason.is_some()) {
                    "completed"
                } else {
                    "in_progress"
                };
                let output = vec![ResponseOutput {
                    r#type: "message".to_string(),
                    id: format!("msg_{}", Uuid::new_v4().simple()),
                    role: "assistant".to_string(),
                    content: vec![ResponseContent {
                        r#type: "output_text".to_string(),
                        text: text.to_string(),
                    }],
                }];
                let chunk_json = serde_json::json!({
                    "id": response_id,
                    "object": "response",
                    "created_at": c.created,
                    "model": model_for_stream.clone(),
                    "output": output,
                    "status": status,
                });
                let json = serde_json::to_string(&chunk_json).unwrap_or_default();
                Ok::<_, std::convert::Infallible>(format!("data: {json}\n\n"))
            }
            Err(e) => {
                let err_json = serde_json::json!({"error": {"message": e.to_string(), "type": "server_error"}});
                Ok::<_, std::convert::Infallible>(format!("data: {err_json}\n\n"))
            }
        });
        let done = futures::stream::once(async {
            Ok::<_, std::convert::Infallible>("data: [DONE]\n\n".to_string())
        });
        let full_stream = mapped.chain(done);

        let guarded = GuardedStream::new(
            full_stream,
            vec![Box::new(request_guard), Box::new(active_guard)],
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
        let chat_response = backend.chat(chat_request).await?;
        let response_id = format!("resp_{}", Uuid::new_v4().simple());

        let text = chat_response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
            .and_then(|c| match c {
                crate::types::chat::Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .unwrap_or("");

        let result = ResponseResult {
            id: response_id,
            object: "response".to_string(),
            created_at: Utc::now().timestamp(),
            model: model_id,
            output: vec![ResponseOutput {
                r#type: "message".to_string(),
                id: format!("msg_{}", Uuid::new_v4().simple()),
                role: "assistant".to_string(),
                content: vec![ResponseContent {
                    r#type: "output_text".to_string(),
                    text: text.to_string(),
                }],
            }],
            usage: ResponseUsage {
                input_tokens: chat_response.usage.prompt_tokens,
                output_tokens: chat_response.usage.completion_tokens,
                total_tokens: chat_response.usage.total_tokens,
            },
            status: "completed".to_string(),
        };

        INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());
        Ok(Json(result).into_response())
    }
}
