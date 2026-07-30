use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "model": "gemma-4-e4b",
    "input": "What is the capital of France?",
    "instructions": "Answer concisely in one sentence.",
    "max_output_tokens": 512,
    "stream": false
}))]
pub struct ResponseRequest {
    pub model: String,
    pub input: ResponseInput,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub tools: Option<Vec<ResponseTool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Messages(Vec<ResponseMessage>),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseTool {
    Function(ResponseFunctionTool),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponseFunctionTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponseResult {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub model: String,
    pub output: Vec<ResponseOutput>,
    pub usage: ResponseUsage,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutput {
    Message(ResponseMessageOutput),
    FunctionCall(ResponseFunctionCallOutput),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponseMessageOutput {
    pub id: String,
    pub status: String,
    pub role: String,
    pub content: Vec<ResponseContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseContent {
    OutputText {
        text: String,
        annotations: Vec<serde_json::Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponseFunctionCallOutput {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct _ResponseStreamChunk {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub model: String,
    pub output: Vec<ResponseOutput>,
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_responses_function_tool() {
        let tool: ResponseTool = serde_json::from_value(serde_json::json!({
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object"},
            "strict": true
        }))
        .unwrap();

        assert!(matches!(tool, ResponseTool::Function(_)));
    }

    #[test]
    fn serializes_function_call_output_item() {
        let item = ResponseOutput::FunctionCall(ResponseFunctionCallOutput {
            id: "fc_1".into(),
            call_id: "call_1".into(),
            name: "get_weather".into(),
            arguments: "{\"city\":\"Pune\"}".into(),
            status: "completed".into(),
        });

        let value = serde_json::to_value(item).unwrap();
        assert_eq!(value["type"], "function_call");
        assert_eq!(value["id"], "fc_1");
        assert_eq!(value["call_id"], "call_1");
        assert_eq!(value["name"], "get_weather");
        assert_eq!(value["arguments"], "{\"city\":\"Pune\"}");
        assert_eq!(value["status"], "completed");
    }
}
