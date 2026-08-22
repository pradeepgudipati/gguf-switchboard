//! Classifies where a model actually put its tool call, if anywhere.
//!
//! OpenAI-compatible clients expect `message.tool_calls` to be populated
//! with structured JSON. In practice, local models frequently miss that
//! contract in one of two ways instead of simply omitting the call:
//!   - they dump a JSON blob shaped like a tool call into plain-text
//!     `content` (the model "wanted" to call a tool but the backend/template
//!     never routed it into the structured field), or
//!   - the JSON blob leaks into `reasoning_content` (thinking models that
//!     reason about the call but never emit it structurally).
//!
//! Distinguishing these three outcomes from "no tool call at all" is the
//! whole point of the conformance console's inspect view — a raw JSON dump
//! looks identical to a human skimming it, but only one of these outcomes
//! is usable by a tool-calling harness.

use serde::Serialize;
use serde_json::Value;
use utoipa::ToSchema;

use crate::types::ToolCall;
use crate::types::chat::{ChatMessage, Content};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallLocation {
    /// `message.tool_calls` is populated (well-formed or not — see `notes`).
    StructuredToolCalls,
    /// No structured `tool_calls`, but `content` contains a JSON blob shaped
    /// like a tool call.
    PlainTextJsonDump,
    /// No structured `tool_calls`, but `reasoning_content` contains a JSON
    /// blob shaped like a tool call.
    LeakedIntoReasoning,
    /// Neither structured nor a detectable JSON-shaped blob anywhere.
    NoToolCallDetected,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ToolCallClassification {
    pub location: ToolCallLocation,
    /// Present only when `location == StructuredToolCalls`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_tool_calls: Option<Vec<ToolCall>>,
    /// The raw substring that looked tool-call-shaped, when found via the
    /// plain-text/reasoning heuristic scan (`location` is `PlainTextJsonDump`
    /// or `LeakedIntoReasoning`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_json_snippet: Option<String>,
    pub content_present: bool,
    pub reasoning_present: bool,
    /// Best-effort diagnostic notes, e.g. flagging malformed `arguments`
    /// JSON inside an otherwise-structured tool call. Never fatal to
    /// classification — the console reports what it found, warts and all.
    pub notes: Vec<String>,
}

/// Classify a single response message. Checks `tool_calls` first (even if
/// malformed — that's still "structured", just broken, and is flagged via
/// `notes` rather than reclassified), then heuristically scans `content`,
/// then `reasoning_content`, for a JSON blob shaped like a tool call.
///
/// The heuristic scan is best-effort, not a grammar: it looks for balanced
/// `{...}` spans and `<tool_call>...</tool_call>` / fenced ```json blocks
/// that parse as JSON with plausible tool-call keys. It exists to catch the
/// common local-model failure modes (GLM/DeepSeek-style templates that dump
/// the call as text), not to be a complete parser for every possible model
/// output format.
pub fn classify_message(message: &ChatMessage) -> ToolCallClassification {
    let content_text = extract_text(&message.content);
    let content_present = content_text.as_ref().is_some_and(|s| !s.is_empty());
    let reasoning_present = message
        .reasoning_content
        .as_ref()
        .is_some_and(|s| !s.is_empty());

    if let Some(calls) = message.tool_calls.as_ref().filter(|c| !c.is_empty()) {
        let mut notes = Vec::new();
        for call in calls {
            if serde_json::from_str::<Value>(&call.function.arguments).is_err() {
                notes.push(format!(
                    "tool call '{}' has non-JSON arguments string",
                    call.function.name
                ));
            }
        }
        return ToolCallClassification {
            location: ToolCallLocation::StructuredToolCalls,
            structured_tool_calls: Some(calls.clone()),
            detected_json_snippet: None,
            content_present,
            reasoning_present,
            notes,
        };
    }

    if let Some(text) = content_text.as_deref()
        && let Some(snippet) = find_tool_call_shaped_json(text)
    {
        return ToolCallClassification {
            location: ToolCallLocation::PlainTextJsonDump,
            structured_tool_calls: None,
            detected_json_snippet: Some(snippet),
            content_present,
            reasoning_present,
            notes: vec![
                "model emitted a tool-call-shaped JSON blob as plain content instead of \
                 message.tool_calls; a tool-calling harness will not see this as a call"
                    .to_string(),
            ],
        };
    }

    if let Some(text) = message.reasoning_content.as_deref()
        && let Some(snippet) = find_tool_call_shaped_json(text)
    {
        return ToolCallClassification {
            location: ToolCallLocation::LeakedIntoReasoning,
            structured_tool_calls: None,
            detected_json_snippet: Some(snippet),
            content_present,
            reasoning_present,
            notes: vec![
                "tool-call-shaped JSON found inside reasoning_content; the model reasoned \
                 about calling a tool but never emitted a structured call"
                    .to_string(),
            ],
        };
    }

    ToolCallClassification {
        location: ToolCallLocation::NoToolCallDetected,
        structured_tool_calls: None,
        detected_json_snippet: None,
        content_present,
        reasoning_present,
        notes: Vec::new(),
    }
}

fn extract_text(content: &Option<Content>) -> Option<String> {
    match content {
        None => None,
        Some(Content::Text(s)) => Some(s.clone()),
        Some(Content::Parts(parts)) => {
            let joined = parts
                .iter()
                .filter_map(|p| match p {
                    crate::types::chat::ContentPart::Text { text } => Some(text.as_str()),
                    crate::types::chat::ContentPart::ImageUrl { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        }
    }
}

/// Look for a substring of `text` that parses as JSON and has the shape of
/// a tool call (`name` string key plus `arguments` or `parameters`), first
/// inside `<tool_call>`/```json fences, then as a bare balanced-brace scan.
fn find_tool_call_shaped_json(text: &str) -> Option<String> {
    for candidate in fenced_or_tagged_candidates(text) {
        if is_tool_call_shaped(&candidate) {
            return Some(candidate);
        }
    }
    balanced_brace_spans(text)
        .into_iter()
        .find(|candidate| is_tool_call_shaped(candidate))
}

/// Extract inner text from `<tool_call>...</tool_call>` tags and ```json
/// fenced code blocks — common shapes for models that were trained to talk
/// about a tool call in freeform text rather than the OpenAI schema.
fn fenced_or_tagged_candidates(text: &str) -> Vec<String> {
    let mut out = Vec::new();

    let mut rest = text;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        if let Some(end) = after.find("</tool_call>") {
            out.push(after[..end].trim().to_string());
            rest = &after[end + "</tool_call>".len()..];
        } else {
            break;
        }
    }

    let mut rest = text;
    while let Some(start) = rest.find("```json") {
        let after = &rest[start + "```json".len()..];
        if let Some(end) = after.find("```") {
            out.push(after[..end].trim().to_string());
            rest = &after[end + 3..];
        } else {
            break;
        }
    }

    out
}

/// Scan for balanced `{...}` spans (handles nesting, ignores braces inside
/// JSON string literals) and return each as a candidate substring.
fn balanced_brace_spans(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{'
            && let Some(end) = find_matching_brace(bytes, i)
        {
            out.push(text[i..=end].to_string());
            i = end + 1;
            continue;
        }
        i += 1;
    }
    out
}

fn find_matching_brace(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn is_tool_call_shaped(candidate: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(candidate) else {
        return false;
    };
    let Some(obj) = value.as_object() else {
        return false;
    };
    let has_name = obj.get("name").is_some_and(Value::is_string);
    let has_args = obj.contains_key("arguments") || obj.contains_key("parameters");
    has_name && has_args
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat::Role;
    use crate::types::{FunctionCall, ToolCall};

    fn message(
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
    fn structured_tool_calls_take_precedence() {
        let msg = message(
            Some("ignored"),
            Some(vec![ToolCall {
                index: None,
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: "echo".to_string(),
                    arguments: "{\"message\":\"hi\"}".to_string(),
                },
            }]),
            None,
        );
        let result = classify_message(&msg);
        assert_eq!(result.location, ToolCallLocation::StructuredToolCalls);
        assert!(result.notes.is_empty());
    }

    #[test]
    fn flags_malformed_arguments_but_stays_structured() {
        let msg = message(
            None,
            Some(vec![ToolCall {
                index: None,
                id: "call_1".to_string(),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: "echo".to_string(),
                    arguments: "not json".to_string(),
                },
            }]),
            None,
        );
        let result = classify_message(&msg);
        assert_eq!(result.location, ToolCallLocation::StructuredToolCalls);
        assert_eq!(result.notes.len(), 1);
    }

    #[test]
    fn detects_json_dumped_in_plain_content() {
        let msg = message(
            Some("Sure! {\"name\": \"echo\", \"arguments\": {\"message\": \"hi\"}}"),
            None,
            None,
        );
        let result = classify_message(&msg);
        assert_eq!(result.location, ToolCallLocation::PlainTextJsonDump);
        assert!(result.detected_json_snippet.is_some());
    }

    #[test]
    fn detects_tool_call_tag_in_content() {
        let msg = message(
            Some(
                "<tool_call>{\"name\": \"echo\", \"arguments\": {\"message\": \"hi\"}}</tool_call>",
            ),
            None,
            None,
        );
        let result = classify_message(&msg);
        assert_eq!(result.location, ToolCallLocation::PlainTextJsonDump);
    }

    #[test]
    fn detects_json_leaked_into_reasoning() {
        let msg = message(
            Some("Done."),
            None,
            Some("I should call {\"name\": \"echo\", \"parameters\": {\"message\": \"hi\"}} now"),
        );
        let result = classify_message(&msg);
        assert_eq!(result.location, ToolCallLocation::LeakedIntoReasoning);
    }

    #[test]
    fn plain_content_wins_over_reasoning_when_both_present() {
        let msg = message(
            Some("{\"name\": \"echo\", \"arguments\": {}}"),
            None,
            Some("{\"name\": \"other\", \"arguments\": {}}"),
        );
        let result = classify_message(&msg);
        assert_eq!(result.location, ToolCallLocation::PlainTextJsonDump);
    }

    #[test]
    fn no_call_detected_for_ordinary_prose() {
        let msg = message(Some("Rust is generally faster than Python."), None, None);
        let result = classify_message(&msg);
        assert_eq!(result.location, ToolCallLocation::NoToolCallDetected);
        assert!(result.content_present);
        assert!(!result.reasoning_present);
    }

    #[test]
    fn ordinary_json_object_without_tool_shape_is_not_flagged() {
        // A model returning a plain JSON answer (e.g. "here's the config: {...}")
        // that isn't tool-call-shaped should not be misclassified.
        let msg = message(
            Some("Config: {\"port\": 8080, \"host\": \"localhost\"}"),
            None,
            None,
        );
        let result = classify_message(&msg);
        assert_eq!(result.location, ToolCallLocation::NoToolCallDetected);
    }
}
