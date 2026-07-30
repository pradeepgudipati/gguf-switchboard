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
use crate::types::chat::{
    ChatCompletionRequest, ChatMessage, Content, FunctionDefinition, Role, Tool,
};
use crate::types::responses::{
    ResponseContent, ResponseFunctionCallOutput, ResponseInput, ResponseMessageOutput,
    ResponseOutput, ResponseRequest, ResponseResult, ResponseTool, ResponseUsage,
};

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

fn to_chat_request(request: &ResponseRequest) -> Result<ChatCompletionRequest, RuntimeError> {
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
        ResponseInput::Messages(response_messages) => {
            for message in response_messages {
                let role = match message.role.as_str() {
                    "system" => Role::System,
                    "assistant" => Role::Assistant,
                    "tool" => Role::Tool,
                    _ => Role::User,
                };
                messages.push(ChatMessage {
                    role,
                    content: Some(Content::Text(message.content.clone())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
        }
    }

    let tools = request
        .tools
        .as_ref()
        .map(|response_tools| {
            response_tools
                .iter()
                .map(|tool| match tool {
                    ResponseTool::Function(function) => {
                        if function.name.trim().is_empty() {
                            return Err(RuntimeError::InvalidRequest(
                                "function tool name must not be empty".to_string(),
                            ));
                        }
                        if !function.parameters.is_object() {
                            return Err(RuntimeError::InvalidRequest(
                                "function tool parameters must be a JSON object".to_string(),
                            ));
                        }
                        Ok(Tool {
                            r#type: "function".to_string(),
                            function: FunctionDefinition {
                                name: function.name.clone(),
                                description: function.description.clone(),
                                parameters: Some(function.parameters.clone()),
                                strict: function.strict,
                            },
                        })
                    }
                })
                .collect::<Result<Vec<_>, RuntimeError>>()
        })
        .transpose()?;

    Ok(ChatCompletionRequest {
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
        tools,
        tool_choice: request.tool_choice.clone(),
        seed: None,
        response_format: request.response_format.clone(),
        chat_template_kwargs: None,
    })
}

fn to_response_output(message: &ChatMessage) -> Result<Vec<ResponseOutput>, RuntimeError> {
    let mut output = Vec::new();
    let text = match &message.content {
        Some(Content::Text(text)) => text.clone(),
        Some(Content::Parts(parts)) => parts
            .iter()
            .filter_map(|part| match part {
                crate::types::chat::ContentPart::Text { text } => Some(text.as_str()),
                crate::types::chat::ContentPart::ImageUrl { .. } => None,
            })
            .collect::<String>(),
        None => String::new(),
    };

    if !text.is_empty() {
        output.push(ResponseOutput::Message(ResponseMessageOutput {
            id: format!("msg_{}", Uuid::new_v4().simple()),
            status: "completed".to_string(),
            role: "assistant".to_string(),
            content: vec![ResponseContent::OutputText {
                text,
                annotations: Vec::new(),
            }],
        }));
    }

    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            if tool_call.r#type != "function"
                || tool_call.id.trim().is_empty()
                || tool_call.function.name.trim().is_empty()
            {
                return Err(RuntimeError::BackendError(
                    "backend returned a malformed function tool call".to_string(),
                ));
            }
            output.push(ResponseOutput::FunctionCall(ResponseFunctionCallOutput {
                id: format!("fc_{}", Uuid::new_v4().simple()),
                call_id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone(),
                status: "completed".to_string(),
            }));
        }
    }

    Ok(output)
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

    let start = std::time::Instant::now();
    let chat_request = to_chat_request(&request)?;
    ACTIVE_REQUESTS.inc();
    let active_guard = ActiveGuard;
    let cfg = state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    require_kind(&request.model, &cfg, CHAT_KINDS, "/v1/responses")?;
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();
    let request_guard = state.scheduler.track_request(&model_id);

    if request.stream == Some(true) {
        let stream = backend.chat_stream(chat_request).await?;
        let response_id = format!("resp_{}", Uuid::new_v4().simple());

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
                let output = vec![ResponseOutput::Message(ResponseMessageOutput {
                    id: format!("msg_{}", Uuid::new_v4().simple()),
                    status: status.to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContent::OutputText {
                        text: text.to_string(),
                        annotations: Vec::new(),
                    }],
                })];
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
        let _active_guard = active_guard;
        let _request_guard = request_guard;
        let chat_response = backend.chat(chat_request).await?;
        let response_id = format!("resp_{}", Uuid::new_v4().simple());

        let output = chat_response
            .choices
            .first()
            .map(|choice| to_response_output(&choice.message))
            .transpose()?
            .unwrap_or_default();

        let result = ResponseResult {
            id: response_id,
            object: "response".to_string(),
            created_at: Utc::now().timestamp(),
            model: model_id,
            output,
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::responses::{ResponseFunctionTool, ResponseTool};
    use crate::types::{FunctionCall, ToolCall};

    fn function_request() -> ResponseRequest {
        ResponseRequest {
            model: "test-model".to_string(),
            input: ResponseInput::Text("What is the weather?".to_string()),
            instructions: None,
            temperature: None,
            top_p: None,
            max_output_tokens: Some(128),
            stream: Some(false),
            tools: Some(vec![ResponseTool::Function(ResponseFunctionTool {
                name: "get_weather".to_string(),
                description: Some("Get weather for a location".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }),
                strict: Some(true),
            })]),
            tool_choice: Some(json!("auto")),
            response_format: None,
            user: None,
        }
    }

    #[test]
    fn translates_function_tools_to_chat_request() {
        let chat = to_chat_request(&function_request()).unwrap();
        let tools = chat.tools.as_ref().unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].r#type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(
            tools[0].function.description.as_deref(),
            Some("Get weather for a location")
        );
        assert_eq!(
            tools[0].function.parameters,
            Some(json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }))
        );
        assert_eq!(chat.tool_choice, Some(json!("auto")));
    }

    #[test]
    fn rejects_empty_function_tool_name() {
        let mut request = function_request();
        let Some(ResponseTool::Function(function)) =
            request.tools.as_mut().and_then(|tools| tools.first_mut())
        else {
            panic!("expected function tool")
        };
        function.name.clear();

        assert!(matches!(
            to_chat_request(&request),
            Err(RuntimeError::InvalidRequest(_))
        ));
    }

    #[test]
    fn rejects_non_object_function_parameters() {
        let mut request = function_request();
        let Some(ResponseTool::Function(function)) =
            request.tools.as_mut().and_then(|tools| tools.first_mut())
        else {
            panic!("expected function tool")
        };
        function.parameters = json!("not-an-object");

        assert!(matches!(
            to_chat_request(&request),
            Err(RuntimeError::InvalidRequest(_))
        ));
    }

    fn assistant_message(content: Option<&str>, calls: Vec<(&str, &str, &str)>) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: content.map(|text| Content::Text(text.to_string())),
            name: None,
            tool_calls: (!calls.is_empty()).then(|| {
                calls
                    .into_iter()
                    .map(|(id, name, arguments)| ToolCall {
                        id: id.to_string(),
                        r#type: "function".to_string(),
                        function: FunctionCall {
                            name: name.to_string(),
                            arguments: arguments.to_string(),
                        },
                    })
                    .collect()
            }),
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    #[test]
    fn translates_tool_only_output() {
        let output = to_response_output(&assistant_message(
            None,
            vec![("call_1", "get_weather", "{\"location\":\"Pune\"}")],
        ))
        .unwrap();

        assert_eq!(output.len(), 1);
        let ResponseOutput::FunctionCall(call) = &output[0] else {
            panic!("expected function call")
        };
        assert!(call.id.starts_with("fc_"));
        assert_eq!(call.call_id, "call_1");
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, "{\"location\":\"Pune\"}");
        assert_eq!(call.status, "completed");
    }

    #[test]
    fn translates_text_before_function_calls() {
        let output = to_response_output(&assistant_message(
            Some("I will check."),
            vec![("call_1", "get_weather", "{}")],
        ))
        .unwrap();

        assert!(matches!(output[0], ResponseOutput::Message(_)));
        assert!(matches!(output[1], ResponseOutput::FunctionCall(_)));
    }

    #[test]
    fn translates_multiple_function_calls() {
        let output = to_response_output(&assistant_message(
            None,
            vec![
                ("call_1", "get_weather", "{\"location\":\"Pune\"}"),
                ("call_2", "get_time", "{\"timezone\":\"Asia/Kolkata\"}"),
            ],
        ))
        .unwrap();

        assert_eq!(output.len(), 2);
        let ResponseOutput::FunctionCall(first) = &output[0] else {
            panic!("expected first function call")
        };
        let ResponseOutput::FunctionCall(second) = &output[1] else {
            panic!("expected second function call")
        };
        assert_eq!(first.call_id, "call_1");
        assert_eq!(second.call_id, "call_2");
        assert_ne!(first.id, second.id);
    }

    #[test]
    fn rejects_malformed_backend_function_call() {
        let error = to_response_output(&assistant_message(None, vec![("", "", "{}")])).unwrap_err();

        assert!(matches!(error, RuntimeError::BackendError(_)));
    }
}
