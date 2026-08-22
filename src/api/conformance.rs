//! Tool-calling / template conformance console endpoints.
//!
//! These are switchboard-native diagnostics, not part of the OpenAI-
//! compatible surface — they exist to answer "did this model actually call
//! the tool, or did it just talk about calling the tool" and "what does
//! this model's chat template actually resolve to", the two questions a
//! chat-first UI (Open WebUI, LM Studio, llama-swap's UI) doesn't answer.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use tracing::instrument;
use utoipa::ToSchema;

use crate::conformance::classify::{self, ToolCallClassification};
use crate::errors::RuntimeError;
use crate::kind_guard::{CHAT_KINDS, require_kind};
use crate::sanitize::sanitize_chat_request;
use crate::state::AppState;
use crate::types::chat::{ChatCompletionRequest, ChatCompletionResponse};

#[derive(Debug, Serialize, ToSchema)]
pub struct InspectResponse {
    pub raw_response: ChatCompletionResponse,
    pub classifications: Vec<ToolCallClassification>,
}

/// Run a chat completion and classify where (if anywhere) each choice's
/// tool call actually ended up: structured `tool_calls`, dumped as plain
/// text, leaked into `reasoning_content`, or not present at all.
///
/// Accepts the exact same body shape as `/v1/chat/completions` so a request
/// can be copy-pasted straight from Swagger's try-it-out panel. Always runs
/// non-streaming — classification needs the complete message, and streaming
/// adds nothing for a diagnostic tool.
#[utoipa::path(
    post,
    path = "/v1/conformance/inspect",
    tag = "conformance",
    request_body(
        content = ChatCompletionRequest,
        example = json!({
            "model": "gemma-4-e4b",
            "messages": [{"role": "user", "content": "Call the echo tool with message set to \"hello\"."}],
            "tools": [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}}],
            "tool_choice": "required"
        })
    ),
    responses(
        (status = 200, description = "Raw response plus a per-choice tool-call classification", body = InspectResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
#[instrument(skip(state, request), fields(model = %request.model))]
pub async fn inspect(
    State(state): State<Arc<AppState>>,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Result<Json<InspectResponse>, RuntimeError> {
    request.stream = Some(false);
    let request = sanitize_chat_request(request);

    let cfg = state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    require_kind(&request.model, &cfg, CHAT_KINDS, "/v1/conformance/inspect")?;

    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();
    let mut raw_response = backend.chat(request).await?;
    raw_response.model = model_id;

    let classifications = raw_response
        .choices
        .iter()
        .map(|choice| classify::classify_message(&choice.message))
        .collect();

    Ok(Json(InspectResponse {
        raw_response,
        classifications,
    }))
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct ResolveTemplateRequest {
    pub model: String,
    pub messages: Vec<crate::types::chat::ChatMessage>,
    #[serde(default)]
    pub tools: Option<Vec<crate::types::chat::Tool>>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ResolveTemplateResponse {
    /// `true` when `prompt` is the actual Jinja-resolved prompt string
    /// (from the backend's own template-application endpoint). `false`
    /// means the backend doesn't support live resolution and `template_source`
    /// is the raw (unresolved) template only.
    pub resolved: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Show the actual prompt string a model's chat template resolves to for a
/// given `messages`/`tools` payload — the direct way to catch a broken or
/// missing embedded Jinja template (GLM/DeepSeek-style models are the
/// recurring offenders) instead of guessing from garbled output.
///
/// Tries the backend's own live template-application endpoint first
/// (llama-server's `/apply-template`, when the running build supports it);
/// falls back to reporting the raw (unresolved) template source from
/// `/props` so the console still shows *something* useful instead of a
/// bare failure.
#[utoipa::path(
    post,
    path = "/v1/conformance/resolve-template",
    tag = "conformance",
    request_body = ResolveTemplateRequest,
    responses(
        (status = 200, description = "Resolved prompt, or raw template source if live resolution isn't supported", body = ResolveTemplateResponse),
        (status = 404, description = "Model not found"),
    )
)]
#[instrument(skip(state, request), fields(model = %request.model))]
pub async fn resolve_template(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ResolveTemplateRequest>,
) -> Result<Json<ResolveTemplateResponse>, RuntimeError> {
    state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    let backend = state.scheduler.ensure_loaded(&request.model).await?;

    let body = serde_json::json!({
        "messages": request.messages,
        "tools": request.tools,
    });

    match backend.raw_post("/apply-template", body).await {
        Ok(value) => {
            let prompt = value
                .get("prompt")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            Ok(Json(ResolveTemplateResponse {
                resolved: prompt.is_some(),
                prompt,
                template_source: None,
                error: None,
            }))
        }
        Err(apply_err) => match backend.raw_get("/props").await {
            Ok(props) => {
                let template_source = props
                    .get("chat_template")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                Ok(Json(ResolveTemplateResponse {
                    resolved: false,
                    prompt: None,
                    template_source,
                    error: Some(format!(
                        "backend does not support live template resolution ({apply_err}); \
                         showing raw template source only"
                    )),
                }))
            }
            Err(props_err) => Ok(Json(ResolveTemplateResponse {
                resolved: false,
                prompt: None,
                template_source: None,
                error: Some(format!(
                    "template resolution unavailable: /apply-template failed ({apply_err}), \
                     /props fallback also failed ({props_err})"
                )),
            })),
        },
    }
}
