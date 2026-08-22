//! Tool-calling / template conformance console endpoints.
//!
//! These are switchboard-native diagnostics, not part of the OpenAI-
//! compatible surface — they exist to answer "did this model actually call
//! the tool, or did it just talk about calling the tool" and "what does
//! this model's chat template actually resolve to", the two questions a
//! chat-first UI (Open WebUI, LM Studio, llama-swap's UI) doesn't answer.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use serde::{Deserialize, Serialize};
use tracing::instrument;
use utoipa::ToSchema;

use crate::conformance::battery::{self, BatteryReport};
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

/// Run the fixed 4-case conformance battery (single tool call, parallel
/// tool calls, tool call + reasoning, multi-turn tool result) against one
/// model and return a pass/fail report per case. No persistence — recomputed
/// on every call, matching how the load-time tool probe verdict works.
#[utoipa::path(
    post,
    path = "/v1/conformance/battery/{model_id}",
    tag = "conformance",
    params(("model_id" = String, Path, description = "Model id to run the battery against")),
    responses(
        (status = 200, description = "Pass/fail report for all 4 battery cases", body = BatteryReport),
        (status = 404, description = "Model not found"),
    )
)]
#[instrument(skip(state), fields(model = %model_id))]
pub async fn run_battery(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<BatteryReport>, RuntimeError> {
    state
        .scheduler
        .model_config(&model_id)
        .ok_or_else(|| RuntimeError::ModelNotFound(model_id.clone()))?;
    let backend = state.scheduler.ensure_loaded(&model_id).await?;
    Ok(Json(battery::run_battery(&backend, &model_id).await))
}

#[derive(Debug, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum CompareMode {
    BatteryCase,
    CustomRequest,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct CompareRequest {
    pub model_a: String,
    pub model_b: String,
    pub mode: CompareMode,
    /// Required when `mode == "battery_case"`.
    #[serde(default)]
    pub case: Option<battery::BatteryCase>,
    /// Required when `mode == "custom_request"`; `model` is ignored/overwritten.
    #[serde(default)]
    pub request: Option<ChatCompletionRequest>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CompareResult {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inspect: Option<InspectResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub battery_case: Option<battery::CaseVerdict>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CompareReport {
    pub model_a: String,
    pub model_b: String,
    pub result_a: CompareResult,
    pub result_b: CompareResult,
}

/// Fire the same request (or the same fixed battery case) at two models,
/// sequentially. The scheduler is single-resident-model, so this evicts and
/// reloads between the two runs — expect this to take as long as two model
/// swaps, not an instant response.
#[utoipa::path(
    post,
    path = "/v1/conformance/compare",
    tag = "conformance",
    request_body = CompareRequest,
    responses(
        (status = 200, description = "Both models' results, run sequentially", body = CompareReport),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
    )
)]
#[instrument(skip(state, request), fields(model_a = %request.model_a, model_b = %request.model_b))]
pub async fn compare(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompareRequest>,
) -> Result<Json<CompareReport>, RuntimeError> {
    match request.mode {
        CompareMode::BatteryCase if request.case.is_none() => {
            return Err(RuntimeError::InvalidRequest(
                "mode is 'battery_case' but no case was given".to_string(),
            ));
        }
        CompareMode::CustomRequest if request.request.is_none() => {
            return Err(RuntimeError::InvalidRequest(
                "mode is 'custom_request' but no request body was given".to_string(),
            ));
        }
        _ => {}
    }

    let result_a = run_one(&state, &request.model_a, &request).await;
    let result_b = run_one(&state, &request.model_b, &request).await;

    Ok(Json(CompareReport {
        model_a: request.model_a,
        model_b: request.model_b,
        result_a,
        result_b,
    }))
}

async fn run_one(
    state: &Arc<AppState>,
    model_id: &str,
    request: &CompareRequest,
) -> CompareResult {
    if state.scheduler.model_config(model_id).is_none() {
        return CompareResult {
            model: model_id.to_string(),
            inspect: None,
            battery_case: None,
            error: Some("model not found".to_string()),
        };
    }

    let backend = match state.scheduler.ensure_loaded(model_id).await {
        Ok(backend) => backend,
        Err(e) => {
            return CompareResult {
                model: model_id.to_string(),
                inspect: None,
                battery_case: None,
                error: Some(e.to_string()),
            };
        }
    };

    match request.mode {
        CompareMode::BatteryCase => {
            let case = request.case.expect("checked by caller");
            let full = battery::run_battery(&backend, model_id).await;
            let verdict = full.cases.into_iter().find(|c| c.case == case);
            CompareResult {
                model: model_id.to_string(),
                inspect: None,
                battery_case: verdict,
                error: None,
            }
        }
        CompareMode::CustomRequest => {
            let mut chat_request = request.request.clone().expect("checked by caller");
            chat_request.model = model_id.to_string();
            chat_request.stream = Some(false);
            let chat_request = sanitize_chat_request(chat_request);
            match backend.chat(chat_request).await {
                Ok(mut raw_response) => {
                    raw_response.model = model_id.to_string();
                    let classifications = raw_response
                        .choices
                        .iter()
                        .map(|choice| classify::classify_message(&choice.message))
                        .collect();
                    CompareResult {
                        model: model_id.to_string(),
                        inspect: Some(InspectResponse {
                            raw_response,
                            classifications,
                        }),
                        battery_case: None,
                        error: None,
                    }
                }
                Err(e) => CompareResult {
                    model: model_id.to_string(),
                    inspect: None,
                    battery_case: None,
                    error: Some(e.to_string()),
                },
            }
        }
    }
}
