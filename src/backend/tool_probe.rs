//! Load-time tool-call capability probe: verifies a model can round-trip an
//! OpenAI-compatible tool call, and normalizes the one known llama.cpp defect
//! where `function.arguments` is emitted as a JSON object instead of the
//! required JSON string.
use serde_json::Value;

use crate::types::chat::{ChatCompletionRequest, FunctionDefinition, Tool};

/// Verdict of the load-time tool-call probe for one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCapability {
    /// The probe's echo tool call round-tripped correctly.
    Verified,
    /// The probe ran and failed; the reason is one of the short static
    /// strings returned by `validate_probe_response`.
    Failed(&'static str),
    /// The model's static `capabilities` never claimed `tools`, so no probe
    /// was run.
    Skipped,
}

/// Build the fixed probe request: ask the model to call a trivial `echo`
/// tool with a known argument.
pub fn probe_request(model: &str) -> ChatCompletionRequest {
    use crate::types::chat::{ChatMessage, Content, Role};

    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: Some(Content::Text(
                "Call the echo tool with message set to \"hello\".".to_string(),
            )),
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
        max_tokens: Some(128),
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        tools: Some(vec![Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "echo".to_string(),
                description: Some("Echo a message back.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"]
                })),
                strict: None,
            },
        }]),
        tool_choice: Some(Value::String("required".to_string())),
        seed: None,
        response_format: None,
        chat_template_kwargs: None,
    }
}

/// Walk both non-streaming (`choices[].message.tool_calls`) and streaming
/// (`choices[].delta.tool_calls`) shapes and rewrite any object-valued
/// `function.arguments` into its JSON string form in place. No-ops on any
/// JSON shape that doesn't match (safe to call on non-chat response bodies).
pub fn normalize_tool_call_arguments(value: &mut Value) {
    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };
    for choice in choices {
        for container_key in ["message", "delta"] {
            if let Some(tool_calls) = choice
                .get_mut(container_key)
                .and_then(|c| c.get_mut("tool_calls"))
                .and_then(Value::as_array_mut)
            {
                for call in tool_calls {
                    normalize_one_call(call);
                }
            }
        }
    }
}

fn normalize_one_call(call: &mut Value) {
    let Some(function) = call.get_mut("function") else {
        return;
    };
    let Some(args) = function.get_mut("arguments") else {
        return;
    };
    if args.is_object() || args.is_array() {
        let as_string = args.to_string();
        *args = Value::String(as_string);
    }
}

/// Validate a (post-normalization) probe response body: exactly one tool
/// call, named `echo`, with arguments `{"message": "hello"}` (case-
/// insensitive on the value).
pub fn validate_probe_response(value: &Value) -> Result<(), &'static str> {
    let tool_calls = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
        .filter(|calls| !calls.is_empty())
        .ok_or("no_tool_calls")?;

    let call = &tool_calls[0];
    let name = call
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .ok_or("no_tool_calls")?;
    if name != "echo" {
        return Err("wrong_function");
    }

    let args_str = call
        .get("function")
        .and_then(|f| f.get("arguments"))
        .and_then(Value::as_str)
        .ok_or("malformed_json")?;
    let parsed: Value = serde_json::from_str(args_str).map_err(|_| "malformed_json")?;
    let message = parsed
        .get("message")
        .and_then(Value::as_str)
        .ok_or("wrong_arguments")?;
    if !message.eq_ignore_ascii_case("hello") {
        return Err("wrong_arguments");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn probe_request_uses_echo_tool_and_requires_a_call() {
        let req = probe_request("test-model");
        assert_eq!(req.model, "test-model");
        let tools = req.tools.expect("probe must declare tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "echo");
        assert_eq!(
            req.tool_choice,
            Some(serde_json::Value::String("required".to_string()))
        );
    }

    #[test]
    fn normalize_converts_object_arguments_to_string_in_message_tool_calls() {
        let mut value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": {"message": "hello"}
                        }
                    }]
                }
            }]
        });
        normalize_tool_call_arguments(&mut value);
        let args = &value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_string());
        let parsed: serde_json::Value = serde_json::from_str(args.as_str().unwrap()).unwrap();
        assert_eq!(parsed, json!({"message": "hello"}));
    }

    #[test]
    fn normalize_leaves_string_arguments_untouched() {
        let mut value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"message\":\"hello\"}"}
                    }]
                }
            }]
        });
        let before = value.clone();
        normalize_tool_call_arguments(&mut value);
        assert_eq!(value, before);
    }

    #[test]
    fn normalize_converts_object_arguments_in_streaming_delta_tool_calls() {
        let mut value = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"name": "echo", "arguments": {"message": "hello"}}
                    }]
                }
            }]
        });
        normalize_tool_call_arguments(&mut value);
        let args = &value["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_string());
    }

    #[test]
    fn normalize_is_a_no_op_on_shapes_without_tool_calls() {
        let mut value = json!({"choices": [{"text": "plain completion"}]});
        let before = value.clone();
        normalize_tool_call_arguments(&mut value);
        assert_eq!(value, before);
    }

    #[test]
    fn validate_accepts_correct_echo_call() {
        let value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"message\":\"hello\"}"}
                    }]
                }
            }]
        });
        assert_eq!(validate_probe_response(&value), Ok(()));
    }

    #[test]
    fn validate_rejects_missing_tool_calls() {
        let value = json!({"choices": [{"message": {"content": "sure, hello!"}}]});
        assert_eq!(validate_probe_response(&value), Err("no_tool_calls"));
    }

    #[test]
    fn validate_rejects_wrong_function_name() {
        let value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "not_echo", "arguments": "{\"message\":\"hello\"}"}
                    }]
                }
            }]
        });
        assert_eq!(validate_probe_response(&value), Err("wrong_function"));
    }

    #[test]
    fn validate_rejects_malformed_arguments_json() {
        let value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "not json"}
                    }]
                }
            }]
        });
        assert_eq!(validate_probe_response(&value), Err("malformed_json"));
    }

    #[test]
    fn validate_rejects_wrong_argument_value() {
        let value = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{\"message\":\"goodbye\"}"}
                    }]
                }
            }]
        });
        assert_eq!(validate_probe_response(&value), Err("wrong_arguments"));
    }
}
