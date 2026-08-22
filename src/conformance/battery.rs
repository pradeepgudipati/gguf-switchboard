//! Fixed conformance battery: four canned tool-calling scenarios run
//! against a model, each scored pass/fail rather than dumped as raw JSON.
//! Generalizes the single-scenario load-time probe in
//! `crate::backend::tool_probe` — case 1 delegates to it directly; cases
//! 2-4 are new, and all four share `classify::classify_message` as the one
//! place that decides where (if anywhere) a tool call ended up.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

use crate::backend::Backend;
use crate::backend::tool_probe;
use crate::conformance::classify::{self, ToolCallClassification, ToolCallLocation};
use crate::types::chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Content, Role,
};
use crate::types::{FunctionCall, ToolCall};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum BatteryCase {
    SingleToolCall,
    ParallelToolCalls,
    ToolCallWithReasoning,
    MultiTurnToolResult,
}

impl BatteryCase {
    pub fn all() -> [BatteryCase; 4] {
        [
            BatteryCase::SingleToolCall,
            BatteryCase::ParallelToolCalls,
            BatteryCase::ToolCallWithReasoning,
            BatteryCase::MultiTurnToolResult,
        ]
    }

    fn build_request(self, model: &str) -> ChatCompletionRequest {
        match self {
            BatteryCase::SingleToolCall => tool_probe::probe_request(model),
            BatteryCase::ParallelToolCalls => parallel_tool_calls_request(model),
            BatteryCase::ToolCallWithReasoning => tool_call_with_reasoning_request(model),
            BatteryCase::MultiTurnToolResult => multi_turn_tool_result_request(model),
        }
    }
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct CaseVerdict {
    pub case: BatteryCase,
    pub pass: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub classification: ToolCallClassification,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct BatteryReport {
    pub model: String,
    pub run_at: String,
    pub cases: Vec<CaseVerdict>,
    pub overall_pass: bool,
}

/// Run all four battery cases against `model` sequentially (each is one
/// chat completion) and return a scored report. No persistence — this is
/// recomputed on demand, same as the load-time tool probe is not cached
/// beyond the single verdict already stored on the scheduler.
pub async fn run_battery(backend: &Arc<dyn Backend>, model: &str) -> BatteryReport {
    let mut cases = Vec::with_capacity(4);
    for case in BatteryCase::all() {
        let request = case.build_request(model);
        let verdict = match backend.chat(request).await {
            Ok(response) => evaluate(case, &response),
            Err(e) => CaseVerdict {
                case,
                pass: false,
                reason: Some(format!("backend_error: {e}")),
                classification: ToolCallClassification {
                    location: ToolCallLocation::NoToolCallDetected,
                    structured_tool_calls: None,
                    detected_json_snippet: None,
                    content_present: false,
                    reasoning_present: false,
                    notes: Vec::new(),
                },
            },
        };
        cases.push(verdict);
    }

    let overall_pass = cases.iter().all(|c| c.pass);
    BatteryReport {
        model: model.to_string(),
        run_at: chrono_like_now(),
        cases,
        overall_pass,
    }
}

fn evaluate(case: BatteryCase, response: &ChatCompletionResponse) -> CaseVerdict {
    let Some(choice) = response.choices.first() else {
        return CaseVerdict {
            case,
            pass: false,
            reason: Some("empty_response".to_string()),
            classification: ToolCallClassification {
                location: ToolCallLocation::NoToolCallDetected,
                structured_tool_calls: None,
                detected_json_snippet: None,
                content_present: false,
                reasoning_present: false,
                notes: Vec::new(),
            },
        };
    };
    let classification = classify::classify_message(&choice.message);

    let (pass, reason) = match case {
        BatteryCase::SingleToolCall => match classification.location {
            ToolCallLocation::StructuredToolCalls => {
                let value = serde_json::to_value(response).unwrap_or(Value::Null);
                match tool_probe::validate_probe_response(&value) {
                    Ok(()) => (true, None),
                    Err(reason) => (false, Some(reason.to_string())),
                }
            }
            _ => (false, Some(format!("{:?}", classification.location))),
        },
        BatteryCase::ParallelToolCalls => {
            let calls = classification
                .structured_tool_calls
                .as_deref()
                .unwrap_or(&[]);
            if calls.len() < 2 {
                (
                    false,
                    Some(format!(
                        "expected >=2 tool calls, got {} ({:?})",
                        calls.len(),
                        classification.location
                    )),
                )
            } else {
                let distinct_args = calls
                    .iter()
                    .map(|c| c.function.arguments.as_str())
                    .collect::<std::collections::HashSet<_>>()
                    .len();
                if distinct_args >= 2 {
                    (true, None)
                } else {
                    (
                        false,
                        Some("parallel calls have identical arguments".to_string()),
                    )
                }
            }
        }
        BatteryCase::ToolCallWithReasoning => {
            if classification.location == ToolCallLocation::LeakedIntoReasoning {
                (
                    false,
                    Some(
                        "tool call leaked into reasoning_content instead of being structured"
                            .to_string(),
                    ),
                )
            } else if classification.location != ToolCallLocation::StructuredToolCalls {
                (false, Some(format!("{:?}", classification.location)))
            } else if !classification.reasoning_present {
                (
                    false,
                    Some("model produced a structured call but no reasoning_content".to_string()),
                )
            } else {
                (true, None)
            }
        }
        BatteryCase::MultiTurnToolResult => {
            let text = match &choice.message.content {
                Some(Content::Text(s)) => s.clone(),
                Some(Content::Parts(_)) | None => String::new(),
            };
            if text.trim().is_empty() {
                (false, Some("empty final content".to_string()))
            } else if serde_json::from_str::<Value>(text.trim()).is_ok() {
                (
                    false,
                    Some("final content is a raw JSON dump, not a summary".to_string()),
                )
            } else {
                (true, None)
            }
        }
    };

    CaseVerdict {
        case,
        pass,
        reason,
        classification,
    }
}

fn parallel_tool_calls_request(model: &str) -> ChatCompletionRequest {
    use crate::types::chat::{FunctionDefinition, Tool};

    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: Some(Content::Text(
                "Call get_weather for both Paris and Tokyo.".to_string(),
            )),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        temperature: None,
        top_p: None,
        n: None,
        stream: Some(false),
        stop: None,
        max_tokens: Some(256),
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        tools: Some(vec![Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get the current weather for a city.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
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

fn tool_call_with_reasoning_request(model: &str) -> ChatCompletionRequest {
    use crate::types::chat::{FunctionDefinition, Tool};

    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: Some(Content::Text(
                "Think step by step about which tool to use, then call the echo tool \
                 with message set to \"hello\"."
                    .to_string(),
            )),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }],
        temperature: None,
        top_p: None,
        n: None,
        stream: Some(false),
        stop: None,
        max_tokens: Some(512),
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

fn multi_turn_tool_result_request(model: &str) -> ChatCompletionRequest {
    use crate::types::chat::{FunctionDefinition, Tool};

    ChatCompletionRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage {
                role: Role::User,
                content: Some(Content::Text("What's the weather in Paris?".to_string())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: Role::Assistant,
                content: None,
                name: None,
                tool_calls: Some(vec![ToolCall {
                    index: None,
                    id: "call_1".to_string(),
                    r#type: "function".to_string(),
                    function: FunctionCall {
                        name: "get_weather".to_string(),
                        arguments: "{\"city\":\"Paris\"}".to_string(),
                    },
                }]),
                tool_call_id: None,
                reasoning_content: None,
            },
            ChatMessage {
                role: Role::Tool,
                content: Some(Content::Text(
                    "{\"temperature_c\": 22, \"conditions\": \"sunny\"}".to_string(),
                )),
                name: Some("get_weather".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            },
        ],
        temperature: None,
        top_p: None,
        n: None,
        stream: Some(false),
        stop: None,
        max_tokens: Some(256),
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        tools: Some(vec![Tool {
            r#type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get the current weather for a city.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                })),
                strict: None,
            },
        }]),
        tool_choice: None,
        seed: None,
        response_format: None,
        chat_template_kwargs: None,
    }
}

/// RFC3339-ish timestamp without pulling in a new `chrono`/`time` dependency
/// beyond what's already in the tree — good enough for a diagnostic report
/// field, not used for any comparison logic.
fn chrono_like_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", now.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Usage;

    fn response_with(message: ChatMessage) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "id".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![crate::types::chat::ChatChoice {
                index: 0,
                message,
                finish_reason: Some("stop".to_string()),
            }],
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            system_fingerprint: None,
        }
    }

    fn assistant_message(
        content: Option<&str>,
        tool_calls: Option<Vec<ToolCall>>,
        reasoning: Option<&str>,
    ) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: content.map(|s| Content::Text(s.to_string())),
            name: None,
            tool_calls,
            tool_call_id: None,
            reasoning_content: reasoning.map(str::to_string),
        }
    }

    #[test]
    fn single_tool_call_passes_on_correct_echo() {
        let response = response_with(assistant_message(
            None,
            Some(vec![ToolCall {
                index: None,
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: "echo".to_string(),
                    arguments: "{\"message\":\"hello\"}".to_string(),
                },
            }]),
            None,
        ));
        let verdict = evaluate(BatteryCase::SingleToolCall, &response);
        assert!(verdict.pass);
    }

    #[test]
    fn parallel_tool_calls_requires_two_distinct_calls() {
        let one_call = response_with(assistant_message(
            None,
            Some(vec![ToolCall {
                index: None,
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: "get_weather".to_string(),
                    arguments: "{\"city\":\"Paris\"}".to_string(),
                },
            }]),
            None,
        ));
        assert!(!evaluate(BatteryCase::ParallelToolCalls, &one_call).pass);

        let two_calls = response_with(assistant_message(
            None,
            Some(vec![
                ToolCall {
                    index: None,
                    id: "call_1".to_string(),
                    r#type: "function".to_string(),
                    function: FunctionCall {
                        name: "get_weather".to_string(),
                        arguments: "{\"city\":\"Paris\"}".to_string(),
                    },
                },
                ToolCall {
                    index: None,
                    id: "call_2".to_string(),
                    r#type: "function".to_string(),
                    function: FunctionCall {
                        name: "get_weather".to_string(),
                        arguments: "{\"city\":\"Tokyo\"}".to_string(),
                    },
                },
            ]),
            None,
        ));
        assert!(evaluate(BatteryCase::ParallelToolCalls, &two_calls).pass);
    }

    #[test]
    fn tool_call_with_reasoning_fails_when_leaked() {
        let leaked = response_with(assistant_message(
            Some("Done."),
            None,
            Some("I should call {\"name\": \"echo\", \"arguments\": {\"message\": \"hi\"}}"),
        ));
        assert!(!evaluate(BatteryCase::ToolCallWithReasoning, &leaked).pass);
    }

    #[test]
    fn tool_call_with_reasoning_fails_without_reasoning_content() {
        let no_reasoning = response_with(assistant_message(
            None,
            Some(vec![ToolCall {
                index: None,
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: "echo".to_string(),
                    arguments: "{\"message\":\"hello\"}".to_string(),
                },
            }]),
            None,
        ));
        assert!(!evaluate(BatteryCase::ToolCallWithReasoning, &no_reasoning).pass);
    }

    #[test]
    fn tool_call_with_reasoning_passes_with_both() {
        let both = response_with(assistant_message(
            None,
            Some(vec![ToolCall {
                index: None,
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: "echo".to_string(),
                    arguments: "{\"message\":\"hello\"}".to_string(),
                },
            }]),
            Some("The user wants me to call echo with hello."),
        ));
        assert!(evaluate(BatteryCase::ToolCallWithReasoning, &both).pass);
    }

    #[test]
    fn multi_turn_tool_result_fails_on_raw_json_dump() {
        let dump = response_with(assistant_message(
            Some("{\"temperature_c\": 22, \"conditions\": \"sunny\"}"),
            None,
            None,
        ));
        assert!(!evaluate(BatteryCase::MultiTurnToolResult, &dump).pass);
    }

    #[test]
    fn multi_turn_tool_result_passes_on_summary() {
        let summary = response_with(assistant_message(
            Some("It's sunny and 22°C in Paris right now."),
            None,
            None,
        ));
        assert!(evaluate(BatteryCase::MultiTurnToolResult, &summary).pass);
    }
}
