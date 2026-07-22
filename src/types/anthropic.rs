use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// ── Request ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MessagesRequest {
    pub model: String,
    /// Max tokens to generate (required by Anthropic API).
    pub max_tokens: u32,
    pub messages: Vec<Message>,
    /// System prompt (string or array of text blocks).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
}

/// System prompt: either a plain string or an array of text blocks.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Message {
    pub role: Role,
    /// Message content: string or array of content blocks.
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Anthropic message content can be a plain string or an array of typed blocks.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Content blocks in a user message.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolResult {
        tool_use_id: String,
        content: Option<String>,
    },
    Image {
        source: ImageSource,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// Tool definition in Anthropic format.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Tool choice configuration.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

// ── Response ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: Role,
    pub content: Vec<ResponseBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ── Streaming events ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: MessagesResponse,
    },
    ContentBlockStart {
        index: usize,
        content_block: ResponseBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDeltaBody,
        usage: Usage,
    },
    MessageStop {},
    Ping {},
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaBody {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

// ── Conversion helpers ───────────────────────────────────────────────────────

use crate::types::StopSequence;
use crate::types::ToolCall;
use crate::types::chat::{
    ChatCompletionRequest, ChatMessage, FunctionDefinition, Role as OpenAIRole, Tool as OpenAITool,
};

/// Extract system prompt text from the Anthropic system field.
fn extract_system_text(system: &SystemPrompt) -> String {
    match system {
        SystemPrompt::Text(s) => s.clone(),
        SystemPrompt::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if b.block_type == "text" {
                    Some(b.text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
    }
}

/// Convert an Anthropic MessagesRequest to an OpenAI ChatCompletionRequest.
pub fn to_openai_request(req: &MessagesRequest) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    // System prompt → system message
    if let Some(system) = &req.system {
        messages.push(ChatMessage {
            role: OpenAIRole::System,
            content: Some(crate::types::chat::Content::Text(extract_system_text(
                system,
            ))),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    for msg in &req.messages {
        match &msg.content {
            MessageContent::Text(text) => {
                messages.push(ChatMessage {
                    role: match msg.role {
                        Role::User => OpenAIRole::User,
                        Role::Assistant => OpenAIRole::Assistant,
                    },
                    content: Some(crate::types::chat::Content::Text(text.clone())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                });
            }
            MessageContent::Blocks(blocks) => {
                match msg.role {
                    Role::User => {
                        // User blocks: text blocks in one message, tool_results each become tool messages
                        let mut text_parts = Vec::new();
                        for block in blocks {
                            match block {
                                ContentBlock::Text { text } => {
                                    text_parts.push(text.as_str());
                                }
                                ContentBlock::ToolResult {
                                    tool_use_id,
                                    content,
                                } => {
                                    messages.push(ChatMessage {
                                        role: OpenAIRole::Tool,
                                        content: Some(crate::types::chat::Content::Text(
                                            content.clone().unwrap_or_default(),
                                        )),
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: Some(tool_use_id.clone()),
                                        reasoning_content: None,
                                    });
                                }
                                ContentBlock::Image { .. } => {}
                            }
                        }
                        if !text_parts.is_empty() {
                            messages.push(ChatMessage {
                                role: OpenAIRole::User,
                                content: Some(crate::types::chat::Content::Text(
                                    text_parts.join("\n"),
                                )),
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                                reasoning_content: None,
                            });
                        }
                    }
                    Role::Assistant => {
                        // Assistant blocks: text in content, tool_uses in tool_calls
                        let mut text_parts = Vec::new();
                        for block in blocks {
                            if let ContentBlock::Text { text } = block {
                                text_parts.push(text.as_str());
                            }
                        }
                        messages.push(ChatMessage {
                            role: OpenAIRole::Assistant,
                            content: if text_parts.is_empty() {
                                None
                            } else {
                                Some(crate::types::chat::Content::Text(text_parts.join("\n")))
                            },
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                            reasoning_content: None,
                        });
                    }
                }
            }
        }
    }

    ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        n: None,
        stream: req.stream,
        stop: req.stop_sequences.as_ref().map(|s| {
            if s.len() == 1 {
                StopSequence::Single(s[0].clone())
            } else {
                StopSequence::Multiple(s.clone())
            }
        }),
        max_tokens: Some(req.max_tokens),
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        tools: req.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| OpenAITool {
                    r#type: "function".to_string(),
                    function: FunctionDefinition {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: Some(t.input_schema.clone()),
                    },
                })
                .collect()
        }),
        tool_choice: req.tool_choice.as_ref().map(|tc| match tc {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::Any => serde_json::json!({ "type": "required" }),
            ToolChoice::Tool { name } => serde_json::json!({
                "type": "function",
                "function": { "name": name }
            }),
        }),
        seed: None,
        response_format: None,
        chat_template_kwargs: None,
    }
}

/// Convert an OpenAI ChatCompletionResponse to an Anthropic MessagesResponse.
pub fn to_anthropic_response(
    model: &str,
    resp: &crate::types::chat::ChatCompletionResponse,
) -> MessagesResponse {
    let choice = resp.choices.first();
    let message = choice.map(|c| &c.message);

    let mut content_blocks = Vec::new();

    if let Some(msg) = message {
        // Text content
        if let Some(text) = msg.content.as_ref() {
            let text_str = match text {
                crate::types::chat::Content::Text(s) => s.clone(),
                crate::types::chat::Content::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| match p {
                        crate::types::chat::ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            };
            if !text_str.is_empty() {
                content_blocks.push(ResponseBlock::Text { text: text_str });
            }
        }

        // Tool calls
        if let Some(tool_calls) = &msg.tool_calls {
            for tc in tool_calls {
                let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                content_blocks.push(ResponseBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    input,
                });
            }
        }
    }

    // Map finish_reason to Anthropic stop_reason
    let stop_reason = choice
        .and_then(|c| c.finish_reason.as_deref())
        .map(map_finish_reason);

    MessagesResponse {
        id: resp.id.clone(),
        response_type: "message".to_string(),
        role: Role::Assistant,
        content: content_blocks,
        model: model.to_string(),
        stop_reason,
        stop_sequence: None,
        usage: Usage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens,
        },
    }
}

fn map_finish_reason(reason: &str) -> String {
    match reason {
        "stop" => "end_turn".to_string(),
        "length" => "max_tokens".to_string(),
        "tool_calls" => "tool_use".to_string(),
        other => other.to_string(),
    }
}

/// Map an Anthropic stop_reason back to OpenAI finish_reason for streaming.
pub fn map_stop_reason_to_finish(reason: &str) -> String {
    match reason {
        "end_turn" => "stop".to_string(),
        "max_tokens" => "length".to_string(),
        "tool_use" => "tool_calls".to_string(),
        other => other.to_string(),
    }
}

/// Build a streaming assistant message for tool calls from OpenAI delta chunks.
pub fn build_tool_use_blocks_from_deltas(tool_call_deltas: &[&ToolCall]) -> Vec<ResponseBlock> {
    tool_call_deltas
        .iter()
        .map(|tc| {
            let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            ResponseBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
            }
        })
        .collect()
}
