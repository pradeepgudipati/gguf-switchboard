use crate::types::chat::{ChatCompletionRequest, ChatMessage, Content, Tool};
use crate::types::ToolCall;

/// Swagger UI defaults `max_tokens` to 2^30; only rewrite those placeholders.
const SWAGGER_PLACEHOLDER_MAX_TOKENS: u32 = 1_000_000_000;
const REASONING_DEFAULT_MAX_TOKENS: u32 = 2048;
const SWAGGER_MAX_INT: i64 = 9_007_199_254_740_991;

/// Strip Swagger UI placeholder values before forwarding to llama-server.
pub fn sanitize_chat_request(mut request: ChatCompletionRequest) -> ChatCompletionRequest {
    request.messages = request
        .messages
        .into_iter()
        .map(sanitize_message)
        .collect();

    if let Some(max_tokens) = request.max_tokens {
        if max_tokens >= SWAGGER_PLACEHOLDER_MAX_TOKENS {
            request.max_tokens = Some(REASONING_DEFAULT_MAX_TOKENS);
        }
    }

    if request.n.is_some_and(|n| n > 128) {
        request.n = None;
    }

    if request.seed == Some(SWAGGER_MAX_INT) {
        request.seed = None;
    }

    if is_placeholder_json(&request.logit_bias) {
        request.logit_bias = None;
    }
    if is_placeholder_json(&request.tool_choice) {
        request.tool_choice = None;
    }
    if is_placeholder_json(&request.response_format) {
        request.response_format = None;
    }
    if request.user.as_deref().is_some_and(is_placeholder_str) {
        request.user = None;
    }
    if request.tools.as_ref().is_some_and(|tools| {
        tools.is_empty() || tools.iter().all(is_placeholder_tool)
    }) {
        request.tools = None;
    }

    request
}

fn sanitize_message(mut message: ChatMessage) -> ChatMessage {
    if message.content.is_none() {
        message.content = Some(default_content_for_role(&message.role));
    }

    if message.name.as_deref().is_some_and(is_placeholder_str) {
        message.name = None;
    }
    if message.tool_call_id.as_deref().is_some_and(is_placeholder_str) {
        message.tool_call_id = None;
    }
    if message
        .tool_calls
        .as_ref()
        .is_some_and(|calls| calls.is_empty() || calls.iter().all(is_placeholder_tool_call))
    {
        message.tool_calls = None;
    }

    message
}

fn default_content_for_role(role: &crate::types::chat::Role) -> Content {
    use crate::types::chat::Role;
    let text = match role {
        Role::System => "You are a helpful assistant.",
        Role::Assistant => "Hello!",
        Role::Tool => "ok",
        Role::User => "Say hello in one sentence.",
    };
    Content::Text(text.to_string())
}

fn is_placeholder_str(value: &str) -> bool {
    value == "string"
}

fn is_placeholder_json(value: &Option<serde_json::Value>) -> bool {
    matches!(value, Some(serde_json::Value::String(s)) if s == "string")
}

fn is_placeholder_tool_call(call: &ToolCall) -> bool {
    call.id == "string"
        || call.r#type == "string"
        || call.function.name == "string"
        || call.function.arguments == "string"
}

fn is_placeholder_tool(tool: &Tool) -> bool {
    tool.r#type == "string" || tool.function.name == "string"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat::{ChatCompletionRequest, ChatMessage, Role};
    use crate::types::{FunctionCall, ToolCall};

    #[test]
    fn strips_swagger_placeholder_tool_calls() {
        let request = ChatCompletionRequest {
            model: "gemma-4-e4b".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some(Content::Text(
                    "is Rust faster than python and js backend ?".to_string(),
                )),
                name: Some("string".to_string()),
                tool_calls: Some(vec![ToolCall {
                    id: "string".to_string(),
                    r#type: "string".to_string(),
                    function: FunctionCall {
                        name: "string".to_string(),
                        arguments: "string".to_string(),
                    },
                }]),
                tool_call_id: Some("string".to_string()),
                reasoning_content: None,
            }],
            temperature: Some(0.1),
            top_p: Some(0.1),
            n: Some(1_073_741_824),
            stream: Some(true),
            stop: None,
            max_tokens: Some(1_073_741_824),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            logit_bias: Some(serde_json::Value::String("string".to_string())),
            user: Some("string".to_string()),
            tools: Some(vec![Tool {
                r#type: "string".to_string(),
                function: crate::types::chat::FunctionDefinition {
                    name: "string".to_string(),
                    description: Some("string".to_string()),
                    parameters: Some(serde_json::Value::String("string".to_string())),
                },
            }]),
            tool_choice: Some(serde_json::Value::String("string".to_string())),
            seed: Some(SWAGGER_MAX_INT),
            response_format: Some(serde_json::Value::String("string".to_string())),
            chat_template_kwargs: None,
        };

        let sanitized = sanitize_chat_request(request);
        let message = &sanitized.messages[0];

        assert!(message.tool_calls.is_none());
        assert!(message.name.is_none());
        assert!(message.tool_call_id.is_none());
        assert_eq!(sanitized.max_tokens, Some(REASONING_DEFAULT_MAX_TOKENS));
        assert!(sanitized.n.is_none());
        assert!(sanitized.tools.is_none());
        assert!(sanitized.logit_bias.is_none());
        assert!(sanitized.user.is_none());
        assert!(sanitized.seed.is_none());
    }

    #[test]
    fn preserves_reasonable_max_tokens() {
        let request = ChatCompletionRequest {
            model: "gemma-4-e4b".to_string(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: Some(Content::Text("hello".to_string())),
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
            max_tokens: Some(50_512),
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            tools: None,
            tool_choice: None,
            seed: None,
            response_format: None,
            chat_template_kwargs: None,
        };

        let sanitized = sanitize_chat_request(request);
        assert_eq!(sanitized.max_tokens, Some(50_512));
    }
}
