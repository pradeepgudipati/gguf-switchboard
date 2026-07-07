use serde::{Deserialize, Serialize};
use serde_json::json;
use utoipa::ToSchema;

/// JSON body forwarded to the backend transcription endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "model": "gemma-4-e4b",
    "file": "sample.wav",
    "response_format": "json",
    "language": "en"
}))]
pub struct TranscriptionRequest {
    pub model: String,
    /// Audio file name or path understood by the backend.
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// JSON body forwarded to the backend text-to-speech endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "model": "gemma-4-e4b",
    "input": "Hello from the OpenAI Runtime speech API.",
    "voice": "alloy",
    "response_format": "mp3"
}))]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}
