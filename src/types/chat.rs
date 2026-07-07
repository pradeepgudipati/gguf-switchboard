use serde::{Deserialize, Serialize};
use serde_json::json;
use utoipa::ToSchema;

use super::{StopSequence, ToolCall, Usage};

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({"role": "user", "content": "Say hello in one sentence."}))]
pub struct ChatMessage {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<String>, example = "Say hello in one sentence.")]
    pub content: Option<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Chain-of-thought from thinking models (llama.cpp `reasoning_content`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Tool {
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "model": "gemma-4-e4b",
    "messages": [{"role": "user", "content": "Is Rust faster than Python for backend services? Explain briefly."}],
    "max_tokens": 2048,
    "stream": false
}))]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatChunkChoice {
    pub index: u32,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

fn content_is_empty(content: &Option<Content>) -> bool {
    match content {
        None => true,
        Some(Content::Text(s)) => s.is_empty(),
        Some(Content::Parts(parts)) => parts.is_empty(),
    }
}

/// When thinking models exhaust `max_tokens` during reasoning, `content` may be
/// empty while `reasoning_content` holds the model output. Promote reasoning into
/// `content` as a fallback so clients still receive a non-empty answer; keep
/// `reasoning_content` so thinking can be shown separately when present.
pub fn normalize_chat_response(mut response: ChatCompletionResponse) -> ChatCompletionResponse {
    for choice in &mut response.choices {
        normalize_message(&mut choice.message);
    }
    response
}

pub fn normalize_chat_chunk(mut chunk: ChatCompletionChunk) -> ChatCompletionChunk {
    for choice in &mut chunk.choices {
        let delta = &mut choice.delta;
        if delta.content.as_ref().is_none_or(|s| s.is_empty()) {
            if let Some(reasoning) = delta.reasoning_content.as_ref() {
                if !reasoning.is_empty() {
                    delta.content = Some(reasoning.clone());
                }
            }
        }
    }
    chunk
}

fn normalize_message(message: &mut ChatMessage) {
    if content_is_empty(&message.content) {
        if let Some(reasoning) = message.reasoning_content.as_ref() {
            if !reasoning.is_empty() {
                message.content = Some(Content::Text(reasoning.clone()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotes_reasoning_content_when_content_empty() {
        let response = ChatCompletionResponse {
            id: "id".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content: Some(Content::Text(String::new())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: Some("Rust is generally faster.".to_string()),
                },
                finish_reason: Some("length".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            system_fingerprint: None,
        };

        let normalized = normalize_chat_response(response);
        let message = &normalized.choices[0].message;
        assert!(matches!(
            &message.content,
            Some(Content::Text(s)) if s == "Rust is generally faster."
        ));
        assert_eq!(
            message.reasoning_content.as_deref(),
            Some("Rust is generally faster.")
        );
    }

    #[test]
    fn promotes_reasoning_in_stream_chunk_without_dropping_reasoning_content() {
        let chunk = ChatCompletionChunk {
            id: "id".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some(String::new()),
                    tool_calls: None,
                    reasoning_content: Some("Thinking...".to_string()),
                },
                finish_reason: None,
            }],
            system_fingerprint: None,
            usage: None,
        };

        let normalized = normalize_chat_chunk(chunk);
        let delta = &normalized.choices[0].delta;
        assert_eq!(delta.content.as_deref(), Some("Thinking..."));
        assert_eq!(delta.reasoning_content.as_deref(), Some("Thinking..."));
    }
}
