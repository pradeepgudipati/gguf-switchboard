use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Response, StatusCode, header};
use axum::response::{IntoResponse, Json};
use futures::StreamExt;
use tracing::instrument;

use crate::errors::RuntimeError;
use crate::kind_guard::{CHAT_KINDS, require_kind};
use crate::metrics::{ACTIVE_REQUESTS, InferenceTimer, REQUEST_TOTAL, STREAMING_REQUESTS};
use crate::proxy::GuardedStream;
use crate::sanitize::sanitize_chat_request;
use crate::state::AppState;
use crate::types::chat::{ChatCompletionRequest, ChatCompletionResponse};

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

/// Returns `Some(error)` when `request` carries a non-empty `tools` array
/// and `verdict` is a known `Failed` outcome for this model. `None`
/// (forward as-is) for chat-only requests, verified models, or models that
/// have never been probed yet (`None` verdict — don't block on ignorance).
fn check_tool_capability_gate(
    request: &ChatCompletionRequest,
    verdict: Option<crate::backend::tool_probe::ToolCapability>,
) -> Option<RuntimeError> {
    use crate::backend::tool_probe::ToolCapability;

    let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
    if !has_tools {
        return None;
    }
    match verdict {
        Some(ToolCapability::Failed(reason)) => Some(RuntimeError::InvalidRequest(format!(
            "Model '{}' failed tool-call verification at load ({reason}); \
             remove `tools` from the request or use a model with verified tool support.",
            request.model
        ))),
        _ => None,
    }
}

/// Chat completions with optional streaming.
#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "chat",
    request_body(
        content = ChatCompletionRequest,
        example = json!({
            "model": "gemma-4-e4b",
            "messages": [{"role": "user", "content": "Is Rust faster than Python for backend services? Explain briefly."}],
            "max_tokens": 2048,
            "stream": false
        })
    ),
    responses(
        (status = 200, description = "Chat completion response", body = ChatCompletionResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
#[instrument(skip(state, request), fields(model = %request.model, stream = request.stream.unwrap_or(false)))]
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();
    ACTIVE_REQUESTS.inc();
    // Created immediately so early returns (`?`) below cannot leak the gauge.
    let active_guard = ActiveGuard;

    let request = sanitize_chat_request(request);
    let cfg = state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    require_kind(&request.model, &cfg, CHAT_KINDS, "/v1/chat/completions")?;
    if let Some(err) = check_tool_capability_gate(
        &request,
        state.scheduler.tool_capability(&request.model).await,
    ) {
        return Err(err);
    }
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();
    let request_guard = state.scheduler.track_request(&model_id);
    // Inference timer starts once the model is resident; load/switch time is
    // reported separately via gguf_switchboard_request_model_wait_seconds.
    let inference_timer = InferenceTimer::start();

    if request.stream == Some(true) {
        STREAMING_REQUESTS.inc();
        let streaming_guard = StreamingGuard;

        let stream = backend.chat_stream(request).await?;

        // Record streaming request (token counts not available in stream mode)
        let _ = state
            .token_db
            .record(&model_id, "/v1/chat/completions", 0, 0, 0, None);

        let model_for_stream = model_id.clone();
        let mapped = stream.map(move |chunk| match chunk {
            Ok(mut c) => {
                c.model = model_for_stream.clone();
                let json = serde_json::to_string(&c).unwrap_or_default();
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

        // Embed guards into the stream so they're dropped when the stream
        // finishes, not when the handler returns.
        let guarded = GuardedStream::new(
            full_stream,
            vec![
                Box::new(request_guard),
                Box::new(active_guard),
                Box::new(streaming_guard),
                // Observes the inference latency histogram when the stream
                // completes (or the client disconnects), not when headers are sent.
                Box::new(inference_timer),
            ],
        );

        let body = Body::from_stream(guarded.map(|s: Result<String, _>| {
            s.map(bytes::Bytes::from)
                .map_err(|e| std::io::Error::other(e.to_string()))
        }));

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .header("x-accel-buffering", "no")
            .body(body)
            .unwrap())
    } else {
        let _guard = active_guard;
        let _request_guard = request_guard;
        let _inference_timer = inference_timer;
        let mut response = backend.chat(request).await?;
        response.model = model_id.clone();

        // Record token usage
        let _ = state.token_db.record(
            &model_id,
            "/v1/chat/completions",
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
            None,
        );

        Ok(Json(response).into_response())
    }
}

#[cfg(test)]
mod tool_gate_tests {
    use super::*;
    use crate::backend::tool_probe::ToolCapability;
    use crate::types::chat::{ChatMessage, Content, FunctionDefinition, Role, Tool};

    fn request_with_tools() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some(Content::Text("hi".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
            temperature: None,
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            tools: Some(vec![Tool {
                r#type: "function".to_string(),
                function: FunctionDefinition {
                    name: "f".to_string(),
                    description: None,
                    parameters: None,
                    strict: None,
                },
            }]),
            tool_choice: None,
            seed: None,
            response_format: None,
            chat_template_kwargs: None,
        }
    }

    fn request_without_tools() -> ChatCompletionRequest {
        let mut req = request_with_tools();
        req.tools = None;
        req
    }

    #[test]
    fn rejects_tools_request_when_verdict_is_failed() {
        let err = check_tool_capability_gate(
            &request_with_tools(),
            Some(ToolCapability::Failed("no_tool_calls")),
        );
        assert!(err.is_some());
    }

    #[test]
    fn allows_tools_request_when_verdict_is_verified() {
        let err = check_tool_capability_gate(&request_with_tools(), Some(ToolCapability::Verified));
        assert!(err.is_none());
    }

    #[test]
    fn allows_tools_request_when_verdict_is_unknown() {
        // Model never loaded/probed yet — don't block; the backend request
        // will trigger a load, which triggers the probe, for next time.
        let err = check_tool_capability_gate(&request_with_tools(), None);
        assert!(err.is_none());
    }

    #[test]
    fn allows_no_tools_request_even_when_verdict_is_failed() {
        let err = check_tool_capability_gate(
            &request_without_tools(),
            Some(ToolCapability::Failed("no_tool_calls")),
        );
        assert!(err.is_none());
    }
}
