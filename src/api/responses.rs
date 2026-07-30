use std::collections::BTreeMap;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{Response, StatusCode, header};
use axum::response::{IntoResponse, Json};
use chrono::Utc;
use futures::StreamExt;
use serde::Serialize;
use tokio_stream::wrappers::ReceiverStream;
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

#[derive(Debug, Clone, Serialize)]
struct ResponseStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    sequence_number: u64,
    #[serde(flatten)]
    payload: serde_json::Map<String, serde_json::Value>,
}

impl ResponseStreamEvent {
    #[cfg(test)]
    fn event_type(&self) -> &str {
        &self.event_type
    }

    #[cfg(test)]
    fn sequence_number(&self) -> u64 {
        self.sequence_number
    }

    fn to_sse(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        format!(
            "event: {event_type}\ndata: {json}\n\n",
            event_type = self.event_type
        )
    }
}

#[derive(Debug)]
struct TextStreamItem {
    id: String,
    output_index: usize,
    text: String,
}

#[derive(Debug)]
struct ToolStreamItem {
    id: String,
    output_index: usize,
    call_id: String,
    name: String,
    arguments: String,
}

#[derive(Debug)]
struct ResponseStreamState {
    response_id: String,
    model: String,
    created_at: i64,
    next_sequence_number: u64,
    next_output_index: usize,
    text: Option<TextStreamItem>,
    tools: BTreeMap<usize, ToolStreamItem>,
    usage: ResponseUsage,
}

impl ResponseStreamState {
    fn new(response_id: String, model: String, created_at: i64) -> Self {
        Self {
            response_id,
            model,
            created_at,
            next_sequence_number: 0,
            next_output_index: 0,
            text: None,
            tools: BTreeMap::new(),
            usage: ResponseUsage {
                input_tokens: 0,
                output_tokens: 0,
                total_tokens: 0,
            },
        }
    }

    fn event(&mut self, event_type: &str, payload: serde_json::Value) -> ResponseStreamEvent {
        let payload = payload.as_object().cloned().unwrap_or_default();
        let event = ResponseStreamEvent {
            event_type: event_type.to_string(),
            sequence_number: self.next_sequence_number,
            payload,
        };
        self.next_sequence_number += 1;
        event
    }

    fn response_value(&self, status: &str, output: Vec<ResponseOutput>) -> serde_json::Value {
        serde_json::json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "model": self.model,
            "output": output,
            "status": status,
            "usage": self.usage,
            "error": null,
            "incomplete_details": null
        })
    }

    fn start_events(&mut self) -> Vec<ResponseStreamEvent> {
        let created = self.response_value("in_progress", Vec::new());
        let in_progress = created.clone();
        vec![
            self.event("response.created", serde_json::json!({"response": created})),
            self.event(
                "response.in_progress",
                serde_json::json!({"response": in_progress}),
            ),
        ]
    }

    fn apply_chunk(
        &mut self,
        chunk: crate::types::chat::ChatCompletionChunk,
    ) -> Result<Vec<ResponseStreamEvent>, RuntimeError> {
        if let Some(usage) = chunk.usage {
            self.usage = ResponseUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            };
        }

        let mut events = Vec::new();
        for choice in chunk.choices.into_iter().filter(|choice| choice.index == 0) {
            if let Some(delta) = choice.delta.content
                && !delta.is_empty()
            {
                if self.text.is_none() {
                    let item = TextStreamItem {
                        id: format!("msg_{}", Uuid::new_v4().simple()),
                        output_index: self.next_output_index,
                        text: String::new(),
                    };
                    self.next_output_index += 1;
                    let item_value = serde_json::json!({
                        "type": "message",
                        "id": item.id,
                        "status": "in_progress",
                        "role": "assistant",
                        "content": []
                    });
                    events.push(self.event(
                        "response.output_item.added",
                        serde_json::json!({
                            "output_index": item.output_index,
                            "item": item_value
                        }),
                    ));
                    events.push(self.event(
                        "response.content_part.added",
                        serde_json::json!({
                            "item_id": item.id,
                            "output_index": item.output_index,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": []
                            }
                        }),
                    ));
                    self.text = Some(item);
                }

                let text = self.text.as_mut().expect("text stream initialized");
                text.text.push_str(&delta);
                let item_id = text.id.clone();
                let output_index = text.output_index;
                events.push(self.event(
                    "response.output_text.delta",
                    serde_json::json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": delta
                    }),
                ));
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for (position, tool_call) in tool_calls.into_iter().enumerate() {
                    let tool_index = tool_call.index.unwrap_or(position);
                    if !self.tools.contains_key(&tool_index) {
                        let item = ToolStreamItem {
                            id: format!("fc_{}", Uuid::new_v4().simple()),
                            output_index: self.next_output_index,
                            call_id: tool_call.id.clone(),
                            name: tool_call.function.name.clone(),
                            arguments: String::new(),
                        };
                        self.next_output_index += 1;
                        let item_value = serde_json::json!({
                            "type": "function_call",
                            "id": item.id,
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": "",
                            "status": "in_progress"
                        });
                        events.push(self.event(
                            "response.output_item.added",
                            serde_json::json!({
                                "output_index": item.output_index,
                                "item": item_value
                            }),
                        ));
                        self.tools.insert(tool_index, item);
                    }

                    let item = self
                        .tools
                        .get_mut(&tool_index)
                        .expect("tool stream initialized");
                    if !tool_call.id.is_empty() {
                        item.call_id = tool_call.id;
                    }
                    if !tool_call.function.name.is_empty() {
                        item.name = tool_call.function.name;
                    }
                    let delta = tool_call.function.arguments;
                    item.arguments.push_str(&delta);
                    let item_id = item.id.clone();
                    let output_index = item.output_index;
                    if !delta.is_empty() {
                        events.push(self.event(
                            "response.function_call_arguments.delta",
                            serde_json::json!({
                                "item_id": item_id,
                                "output_index": output_index,
                                "delta": delta
                            }),
                        ));
                    }
                }
            }
        }
        Ok(events)
    }

    fn finish_events(&mut self) -> Result<Vec<ResponseStreamEvent>, RuntimeError> {
        let mut events = Vec::new();
        let mut indexed_output = Vec::new();

        if let Some(text) = self.text.take() {
            let content = ResponseContent::OutputText {
                text: text.text.clone(),
                annotations: Vec::new(),
            };
            let item = ResponseOutput::Message(ResponseMessageOutput {
                id: text.id.clone(),
                status: "completed".to_string(),
                role: "assistant".to_string(),
                content: vec![content.clone()],
            });
            events.push(self.event(
                "response.output_text.done",
                serde_json::json!({
                    "item_id": text.id,
                    "output_index": text.output_index,
                    "content_index": 0,
                    "text": text.text
                }),
            ));
            events.push(self.event(
                "response.content_part.done",
                serde_json::json!({
                    "item_id": text.id,
                    "output_index": text.output_index,
                    "content_index": 0,
                    "part": content
                }),
            ));
            events.push(self.event(
                "response.output_item.done",
                serde_json::json!({
                    "output_index": text.output_index,
                    "item": item
                }),
            ));
            indexed_output.push((text.output_index, item));
        }

        for (_, tool) in std::mem::take(&mut self.tools) {
            if tool.call_id.trim().is_empty() || tool.name.trim().is_empty() {
                return Err(RuntimeError::BackendError(
                    "backend returned a malformed streaming function tool call".to_string(),
                ));
            }
            let item = ResponseOutput::FunctionCall(ResponseFunctionCallOutput {
                id: tool.id.clone(),
                call_id: tool.call_id.clone(),
                name: tool.name.clone(),
                arguments: tool.arguments.clone(),
                status: "completed".to_string(),
            });
            events.push(self.event(
                "response.function_call_arguments.done",
                serde_json::json!({
                    "item_id": tool.id,
                    "output_index": tool.output_index,
                    "name": tool.name,
                    "arguments": tool.arguments
                }),
            ));
            events.push(self.event(
                "response.output_item.done",
                serde_json::json!({
                    "output_index": tool.output_index,
                    "item": item
                }),
            ));
            indexed_output.push((tool.output_index, item));
        }

        indexed_output.sort_by_key(|(index, _)| *index);
        let output = indexed_output
            .into_iter()
            .map(|(_, item)| item)
            .collect::<Vec<_>>();
        let response = self.response_value("completed", output);
        events.push(self.event(
            "response.completed",
            serde_json::json!({"response": response}),
        ));
        Ok(events)
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
    request: Result<Json<ResponseRequest>, JsonRejection>,
) -> Result<impl IntoResponse, RuntimeError> {
    let Json(request) = request.map_err(|error| RuntimeError::InvalidRequest(error.body_text()))?;
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
        let mut stream = backend.chat_stream(chat_request).await?;
        let response_id = format!("resp_{}", Uuid::new_v4().simple());
        let mut response_state =
            ResponseStreamState::new(response_id, model_id.clone(), Utc::now().timestamp());
        let (sender, receiver) =
            tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(32);
        tokio::spawn(async move {
            for event in response_state.start_events() {
                if sender.send(Ok(event.to_sse())).await.is_err() {
                    return;
                }
            }

            while let Some(chunk) = stream.next().await {
                let events = match chunk {
                    Ok(chunk) => response_state.apply_chunk(chunk),
                    Err(error) => {
                        let body = serde_json::json!({
                            "type": "error",
                            "code": "server_error",
                            "message": error.to_string(),
                            "param": null
                        });
                        let _ = sender
                            .send(Ok(format!("event: error\ndata: {body}\n\n")))
                            .await;
                        return;
                    }
                };
                match events {
                    Ok(events) => {
                        for event in events {
                            if sender.send(Ok(event.to_sse())).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(error) => {
                        let body = serde_json::json!({
                            "type": "error",
                            "code": "server_error",
                            "message": error.to_string(),
                            "param": null
                        });
                        let _ = sender
                            .send(Ok(format!("event: error\ndata: {body}\n\n")))
                            .await;
                        return;
                    }
                }
            }

            match response_state.finish_events() {
                Ok(events) => {
                    for event in events {
                        if sender.send(Ok(event.to_sse())).await.is_err() {
                            return;
                        }
                    }
                }
                Err(error) => {
                    let body = serde_json::json!({
                        "type": "error",
                        "code": "server_error",
                        "message": error.to_string(),
                        "param": null
                    });
                    let _ = sender
                        .send(Ok(format!("event: error\ndata: {body}\n\n")))
                        .await;
                }
            }
        });

        let guarded = GuardedStream::new(
            ReceiverStream::new(receiver),
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
    use crate::types::Usage;
    use crate::types::chat::{ChatChunkChoice, ChatCompletionChunk, ChatDelta};
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
                        index: None,
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

    fn stream_chunk(content: Option<&str>, finish_reason: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chatcmpl_1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1_700_000_000,
            model: "test-model".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: content.map(str::to_string),
                    tool_calls: None,
                    reasoning_content: None,
                },
                finish_reason: finish_reason.map(str::to_string),
            }],
            system_fingerprint: None,
            usage: finish_reason.map(|_| Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            }),
        }
    }

    #[test]
    fn streams_strict_text_events() {
        let mut state = ResponseStreamState::new(
            "resp_1".to_string(),
            "test-model".to_string(),
            1_700_000_000,
        );
        let mut events = state.start_events();
        events.extend(
            state
                .apply_chunk(stream_chunk(Some("Hello"), None))
                .unwrap(),
        );
        events.extend(state.apply_chunk(stream_chunk(None, Some("stop"))).unwrap());
        events.extend(state.finish_events().unwrap());

        let event_types = events
            .iter()
            .map(ResponseStreamEvent::event_type)
            .collect::<Vec<_>>();
        assert_eq!(
            event_types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let sequence_numbers = events
            .iter()
            .map(ResponseStreamEvent::sequence_number)
            .collect::<Vec<_>>();
        assert_eq!(
            sequence_numbers,
            (0..events.len() as u64).collect::<Vec<_>>()
        );
    }

    fn tool_stream_chunk(
        calls: Vec<(&str, &str, &str)>,
        finish_reason: Option<&str>,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: "chatcmpl_1".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1_700_000_000,
            model: "test-model".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(
                        calls
                            .into_iter()
                            .map(|(id, name, arguments)| ToolCall {
                                index: None,
                                id: id.to_string(),
                                r#type: if id.is_empty() {
                                    String::new()
                                } else {
                                    "function".to_string()
                                },
                                function: FunctionCall {
                                    name: name.to_string(),
                                    arguments: arguments.to_string(),
                                },
                            })
                            .collect(),
                    ),
                    reasoning_content: None,
                },
                finish_reason: finish_reason.map(str::to_string),
            }],
            system_fingerprint: None,
            usage: finish_reason.map(|_| Usage {
                prompt_tokens: 4,
                completion_tokens: 3,
                total_tokens: 7,
            }),
        }
    }

    #[test]
    fn streams_fragmented_function_arguments() {
        let mut state = ResponseStreamState::new(
            "resp_1".to_string(),
            "test-model".to_string(),
            1_700_000_000,
        );
        let mut events = state.start_events();
        events.extend(
            state
                .apply_chunk(tool_stream_chunk(
                    vec![("call_1", "get_weather", "{\"loc")],
                    None,
                ))
                .unwrap(),
        );
        events.extend(
            state
                .apply_chunk(tool_stream_chunk(
                    vec![("", "", "ation\":\"Pune\"}")],
                    Some("tool_calls"),
                ))
                .unwrap(),
        );
        events.extend(state.finish_events().unwrap());

        let values = events
            .iter()
            .map(|event| serde_json::to_value(event).unwrap())
            .collect::<Vec<_>>();
        let added = values
            .iter()
            .filter(|event| event["type"] == "response.output_item.added")
            .collect::<Vec<_>>();
        assert_eq!(added.len(), 1);
        assert_eq!(added[0]["item"]["call_id"], "call_1");
        assert_eq!(added[0]["item"]["name"], "get_weather");

        let deltas = values
            .iter()
            .filter(|event| event["type"] == "response.function_call_arguments.delta")
            .map(|event| event["delta"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(deltas, vec!["{\"loc", "ation\":\"Pune\"}"]);

        let done = values
            .iter()
            .find(|event| event["type"] == "response.function_call_arguments.done")
            .unwrap();
        assert_eq!(done["arguments"], "{\"location\":\"Pune\"}");
        assert_eq!(done["name"], "get_weather");

        let item_done = values
            .iter()
            .find(|event| {
                event["type"] == "response.output_item.done"
                    && event["item"]["type"] == "function_call"
            })
            .unwrap();
        assert_eq!(item_done["item"]["call_id"], "call_1");
        assert_eq!(item_done["item"]["arguments"], "{\"location\":\"Pune\"}");
        assert_eq!(added[0]["output_index"], item_done["output_index"]);
        assert_eq!(added[0]["item"]["id"], item_done["item"]["id"]);
    }

    #[test]
    fn streams_multiple_function_calls_with_distinct_indices() {
        let mut state = ResponseStreamState::new(
            "resp_1".to_string(),
            "test-model".to_string(),
            1_700_000_000,
        );
        let mut events = state.start_events();
        events.extend(
            state
                .apply_chunk(tool_stream_chunk(
                    vec![
                        ("call_1", "get_weather", "{\"location\":\"Pune\"}"),
                        ("call_2", "get_time", "{\"timezone\":\"Asia/Kolkata\"}"),
                    ],
                    Some("tool_calls"),
                ))
                .unwrap(),
        );
        events.extend(state.finish_events().unwrap());

        let calls = events
            .iter()
            .map(|event| serde_json::to_value(event).unwrap())
            .filter(|event| {
                event["type"] == "response.output_item.done"
                    && event["item"]["type"] == "function_call"
            })
            .collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["item"]["call_id"], "call_1");
        assert_eq!(calls[1]["item"]["call_id"], "call_2");
        assert_ne!(calls[0]["item"]["id"], calls[1]["item"]["id"]);
        assert_ne!(calls[0]["output_index"], calls[1]["output_index"]);
    }
}
