//! Tool-calling / template conformance console endpoints.
//!
//! These are switchboard-native diagnostics, not part of the OpenAI-
//! compatible surface — they exist to answer "did this model actually call
//! the tool, or did it just talk about calling the tool" and "what does
//! this model's chat template actually resolve to", the two questions a
//! chat-first UI (Open WebUI, LM Studio, llama-swap's UI) doesn't answer.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use serde::{Deserialize, Serialize};
use tracing::{instrument, warn};
use utoipa::ToSchema;

use crate::backend::Backend;
use crate::backend::external_openai::ExternalOpenAiBackend;
use crate::conformance::battery::{self, BatteryReport};
use crate::conformance::classify::{self, ToolCallClassification, ToolCallLocation};
use crate::conformance::{ConformanceRunDetail, ConformanceRunSummary};
use crate::errors::RuntimeError;
use crate::kind_guard::{CHAT_KINDS, require_kind};
use crate::sanitize::sanitize_chat_request;
use crate::scheduler::{RequestGuard, Scheduler};
use crate::state::AppState;
use crate::types::chat::{ChatCompletionRequest, ChatCompletionResponse};

/// Per-request "run this against something other than a managed model" target
/// for the conformance console. Read from `X-Conformance-*` headers (optionally
/// with an `-a` / `-b` suffix for Compare) so no endpoint URL or API key ever
/// touches a request body, a log line, or the history DB.
#[derive(Default)]
struct ConformanceTarget {
    base_url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
}

fn conformance_request_guard(
    scheduler: &Arc<Scheduler>,
    model_id: &str,
    target: &ConformanceTarget,
) -> Option<RequestGuard> {
    target
        .base_url
        .is_none()
        .then(|| scheduler.track_request(model_id))
}

fn target_from_headers(headers: &HeaderMap, suffix: &str) -> ConformanceTarget {
    let get = |name: String| -> Option<String> {
        headers
            .get(&name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    ConformanceTarget {
        base_url: get(format!("x-conformance-base-url{suffix}")),
        api_key: get(format!("x-conformance-api-key{suffix}")),
        model: get(format!("x-conformance-model{suffix}")),
    }
}

/// Resolve the backend to run a conformance case against: either an external
/// OpenAI-compatible endpoint (when `target` carries a base URL) or the
/// switchboard-managed model named by `requested_model`. Returns the backend
/// plus the effective model id to send / record.
async fn resolve_conformance_backend(
    state: &Arc<AppState>,
    requested_model: &str,
    target: &ConformanceTarget,
    allowed_kinds: &[&str],
    endpoint: &str,
) -> Result<(Arc<dyn Backend>, String, Option<RequestGuard>), RuntimeError> {
    if let Some(base) = &target.base_url {
        let model = target
            .model
            .clone()
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| requested_model.to_string());
        let backend: Arc<dyn Backend> =
            Arc::new(ExternalOpenAiBackend::new(base, target.api_key.clone()));
        return Ok((backend, model, None));
    }

    // Acquire the guard before waiting for the load lock. A priority-model
    // switch may already be queued behind this load and must see the request.
    let request_guard = conformance_request_guard(&state.scheduler, requested_model, target);
    let cfg = state
        .scheduler
        .model_config(requested_model)
        .ok_or_else(|| RuntimeError::ModelNotFound(requested_model.to_string()))?;
    require_kind(requested_model, &cfg, allowed_kinds, endpoint)?;
    let backend = state.scheduler.ensure_loaded(requested_model).await?;
    Ok((backend, requested_model.to_string(), request_guard))
}

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
#[instrument(skip(state, request, headers), fields(model = %request.model))]
pub async fn inspect(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut request): Json<ChatCompletionRequest>,
) -> Result<Json<InspectResponse>, RuntimeError> {
    request.stream = Some(false);
    let mut request = sanitize_chat_request(request);

    let target = target_from_headers(&headers, "");
    let requested_model = request.model.clone();
    let (backend, model_id, _request_guard) = match resolve_conformance_backend(
        &state,
        &request.model,
        &target,
        CHAT_KINDS,
        "/v1/conformance/inspect",
    )
    .await
    {
        Ok(pair) => pair,
        Err(e) => {
            record_error(&state, "inspect", Some(&requested_model), None, &e);
            return Err(e);
        }
    };
    request.model = model_id.clone();
    let mut raw_response = match backend.chat(request).await {
        Ok(r) => r,
        Err(e) => {
            record_error(&state, "inspect", Some(&model_id), None, &e);
            return Err(e);
        }
    };
    raw_response.model = model_id.clone();

    let classifications: Vec<ToolCallClassification> = raw_response
        .choices
        .iter()
        .map(|choice| classify::classify_message(&choice.message))
        .collect();

    let response = InspectResponse {
        raw_response,
        classifications,
    };
    let passed = inspect_classifications_passed(&response.classifications);
    record_run(
        &state,
        "inspect",
        Some(&model_id),
        None,
        &summarize_classifications(&response.classifications),
        Some(passed),
        &response,
    );
    Ok(Json(response))
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
#[instrument(skip(state, request, headers), fields(model = %request.model))]
pub async fn resolve_template(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ResolveTemplateRequest>,
) -> Result<Json<ResolveTemplateResponse>, RuntimeError> {
    let target = target_from_headers(&headers, "");
    let requested_model = request.model.clone();
    let (backend, model_id, _request_guard) = match resolve_conformance_backend(
        &state,
        &request.model,
        &target,
        CHAT_KINDS,
        "/v1/conformance/resolve-template",
    )
    .await
    {
        Ok(pair) => pair,
        Err(e) => {
            record_error(&state, "resolve_template", Some(&requested_model), None, &e);
            return Err(e);
        }
    };

    let body = serde_json::json!({
        "messages": request.messages,
        "tools": request.tools,
    });

    let response = match backend.raw_post("/apply-template", body).await {
        Ok(value) => {
            let prompt = value
                .get("prompt")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string);
            ResolveTemplateResponse {
                resolved: prompt.is_some(),
                prompt,
                template_source: None,
                error: None,
            }
        }
        Err(apply_err) => match backend.raw_get("/props").await {
            Ok(props) => {
                let template_source = props
                    .get("chat_template")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                ResolveTemplateResponse {
                    resolved: false,
                    prompt: None,
                    template_source,
                    error: Some(format!(
                        "backend does not support live template resolution ({apply_err}); \
                         showing raw template source only"
                    )),
                }
            }
            Err(props_err) => ResolveTemplateResponse {
                resolved: false,
                prompt: None,
                template_source: None,
                error: Some(format!(
                    "template resolution unavailable: /apply-template failed ({apply_err}), \
                     /props fallback also failed ({props_err})"
                )),
            },
        },
    };

    let summary = if response.resolved {
        "resolved"
    } else if response.template_source.is_some() {
        "template-only"
    } else {
        "error"
    };
    record_run(
        &state,
        "resolve_template",
        Some(&model_id),
        None,
        summary,
        Some(response.resolved),
        &response,
    );
    Ok(Json(response))
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
#[instrument(skip(state, headers), fields(model = %model_id))]
pub async fn run_battery(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Path(model_id): Path<String>,
) -> Result<Json<BatteryReport>, RuntimeError> {
    let target = target_from_headers(&headers, "");
    let requested_model = model_id.clone();
    let (backend, model_id, _request_guard) = match resolve_conformance_backend(
        &state,
        &model_id,
        &target,
        CHAT_KINDS,
        "/v1/conformance/battery",
    )
    .await
    {
        Ok(pair) => pair,
        Err(e) => {
            record_error(&state, "battery", Some(&requested_model), None, &e);
            return Err(e);
        }
    };
    let report = battery::run_battery(&backend, &model_id).await;

    let passed = report.cases.iter().filter(|c| c.pass).count();
    record_run(
        &state,
        "battery",
        Some(&model_id),
        None,
        &format!("{passed}/{} pass", report.cases.len()),
        Some(report.overall_pass),
        &report,
    );
    Ok(Json(report))
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
#[instrument(skip(state, request, headers), fields(model_a = %request.model_a, model_b = %request.model_b))]
pub async fn compare(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
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

    let target_a = target_from_headers(&headers, "-a");
    let target_b = target_from_headers(&headers, "-b");

    let result_a = run_one(&state, &request.model_a, &target_a, &request).await;
    let result_b = run_one(&state, &request.model_b, &target_b, &request).await;

    let report = CompareReport {
        model_a: result_a.model.clone(),
        model_b: result_b.model.clone(),
        result_a,
        result_b,
    };
    let passed = compare_passed(&report.result_a, &report.result_b);
    record_run(
        &state,
        "compare",
        Some(&report.model_a),
        Some(&report.model_b),
        &format!("{} vs {}", report.model_a, report.model_b),
        Some(passed),
        &report,
    );
    Ok(Json(report))
}

async fn run_one(
    state: &Arc<AppState>,
    model_id: &str,
    target: &ConformanceTarget,
    request: &CompareRequest,
) -> CompareResult {
    let (backend, model_id, _request_guard) = match resolve_conformance_backend(
        state,
        model_id,
        target,
        CHAT_KINDS,
        "/v1/conformance/compare",
    )
    .await
    {
        Ok(pair) => pair,
        Err(e) => {
            return CompareResult {
                model: model_id.to_string(),
                inspect: None,
                battery_case: None,
                error: Some(e.to_string()),
            };
        }
    };
    let model_id = model_id.as_str();

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

// ── Run history ──────────────────────────────────────────────────────────────

/// Best-effort persist of a conformance run. Never fails the request — a
/// history write error is logged and swallowed.
fn record_run<T: Serialize>(
    state: &Arc<AppState>,
    kind: &str,
    model: Option<&str>,
    model_b: Option<&str>,
    summary: &str,
    passed: Option<bool>,
    detail: &T,
) {
    let detail = serde_json::to_value(detail).unwrap_or(serde_json::Value::Null);
    if let Err(e) = state
        .conformance_history
        .record(kind, model, model_b, summary, passed, &detail)
    {
        warn!(error = %e, kind, "failed to persist conformance run to history");
    }
}

/// Persist a failed conformance run so the History tab shows it as FAIL
/// instead of the run vanishing on the error path.
fn record_error(
    state: &Arc<AppState>,
    kind: &str,
    model: Option<&str>,
    model_b: Option<&str>,
    err: &RuntimeError,
) {
    let msg = err.to_string();
    record_run(
        state,
        kind,
        model,
        model_b,
        &msg,
        Some(false),
        &serde_json::json!({ "error": msg }),
    );
}

fn inspect_classifications_passed(classifications: &[ToolCallClassification]) -> bool {
    !classifications.is_empty()
        && classifications
            .iter()
            .all(|c| c.location == ToolCallLocation::StructuredToolCalls)
}

fn compare_result_passed(result: &CompareResult) -> bool {
    result.error.is_none()
        && result
            .battery_case
            .as_ref()
            .map(|case| case.pass)
            .or_else(|| {
                result
                    .inspect
                    .as_ref()
                    .map(|inspect| inspect_classifications_passed(&inspect.classifications))
            })
            .unwrap_or(false)
}

fn compare_passed(result_a: &CompareResult, result_b: &CompareResult) -> bool {
    compare_result_passed(result_a) && compare_result_passed(result_b)
}

/// One-line summary of where the tool call(s) landed across all choices.
fn summarize_classifications(classifications: &[ToolCallClassification]) -> String {
    if classifications.is_empty() {
        return "no choices".to_string();
    }
    let loc = |l: &ToolCallLocation| match l {
        ToolCallLocation::StructuredToolCalls => "tool_calls",
        ToolCallLocation::PlainTextJsonDump => "plaintext-json",
        ToolCallLocation::LeakedIntoReasoning => "leaked-reasoning",
        ToolCallLocation::NoToolCallDetected => "none",
    };
    if classifications.len() == 1 {
        return loc(&classifications[0].location).to_string();
    }
    let mut parts: Vec<&str> = classifications.iter().map(|c| loc(&c.location)).collect();
    parts.dedup();
    parts.join(", ")
}

#[derive(Debug, Deserialize, ToSchema, utoipa::IntoParams)]
pub struct HistoryQuery {
    /// Max rows to return (default 50, capped at 500).
    #[serde(default)]
    pub limit: Option<u32>,
    /// Filter by run kind: `battery` | `compare` | `inspect` | `resolve_template`.
    #[serde(default)]
    pub kind: Option<String>,
    /// Filter by (primary) model id.
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HistoryClearResponse {
    pub deleted: u64,
}

/// List recent conformance-console runs (newest first), without the full
/// response payload. Use `GET /v1/conformance/history/{id}` for the detail.
#[utoipa::path(
    get,
    path = "/v1/conformance/history",
    tag = "conformance",
    params(HistoryQuery),
    responses((status = 200, description = "Recent runs", body = [ConformanceRunSummary]))
)]
pub async fn history_list(
    State(state): State<Arc<AppState>>,
    Query(q): Query<HistoryQuery>,
) -> Result<Json<Vec<ConformanceRunSummary>>, RuntimeError> {
    let rows = state.conformance_history.list(
        q.limit.unwrap_or(50),
        q.kind.as_deref(),
        q.model.as_deref(),
    )?;
    Ok(Json(rows))
}

/// Fetch one stored run including its full response payload.
#[utoipa::path(
    get,
    path = "/v1/conformance/history/{id}",
    tag = "conformance",
    params(("id" = i64, Path, description = "History row id")),
    responses(
        (status = 200, description = "The stored run", body = ConformanceRunDetail),
        (status = 404, description = "No such run"),
    )
)]
pub async fn history_get(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Result<Json<ConformanceRunDetail>, StatusCode> {
    match state.conformance_history.get(id) {
        Ok(Some(row)) => Ok(Json(row)),
        Ok(None) => Err(StatusCode::NOT_FOUND),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Delete one stored run.
#[utoipa::path(
    delete,
    path = "/v1/conformance/history/{id}",
    tag = "conformance",
    params(("id" = i64, Path, description = "History row id")),
    responses((status = 204, description = "Deleted"), (status = 404, description = "No such run"))
)]
pub async fn history_delete(State(state): State<Arc<AppState>>, Path(id): Path<i64>) -> StatusCode {
    match state.conformance_history.delete(id) {
        Ok(true) => StatusCode::NO_CONTENT,
        Ok(false) => StatusCode::NOT_FOUND,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Delete the entire conformance run history.
#[utoipa::path(
    delete,
    path = "/v1/conformance/history",
    tag = "conformance",
    responses((status = 200, description = "Cleared", body = HistoryClearResponse))
)]
pub async fn history_clear(
    State(state): State<Arc<AppState>>,
) -> Result<Json<HistoryClearResponse>, RuntimeError> {
    let deleted = state.conformance_history.clear()?;
    Ok(Json(HistoryClearResponse { deleted }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::conformance::classify::ToolCallLocation;
    use crate::scheduler::Scheduler;

    fn classification(location: ToolCallLocation) -> ToolCallClassification {
        ToolCallClassification {
            location,
            structured_tool_calls: None,
            detected_json_snippet: None,
            content_present: false,
            reasoning_present: false,
            notes: Vec::new(),
        }
    }

    #[test]
    fn inspect_passes_only_when_all_choices_have_structured_tool_calls() {
        let structured = vec![classification(ToolCallLocation::StructuredToolCalls)];
        let mixed = vec![
            classification(ToolCallLocation::StructuredToolCalls),
            classification(ToolCallLocation::PlainTextJsonDump),
        ];

        assert!(inspect_classifications_passed(&structured));
        assert!(!inspect_classifications_passed(&mixed));
        assert!(!inspect_classifications_passed(&[]));
    }

    #[test]
    fn compare_passes_only_when_both_results_pass() {
        let passing = CompareResult {
            model: "model-a".to_string(),
            inspect: None,
            battery_case: Some(battery::CaseVerdict {
                case: battery::BatteryCase::SingleToolCall,
                pass: true,
                reason: None,
                classification: classification(ToolCallLocation::StructuredToolCalls),
            }),
            error: None,
        };
        let failing = CompareResult {
            model: "model-b".to_string(),
            inspect: None,
            battery_case: None,
            error: Some("backend failed".to_string()),
        };

        assert!(compare_passed(&passing, &passing));
        assert!(!compare_passed(&passing, &failing));
    }

    #[tokio::test]
    async fn managed_conformance_guard_tracks_request_until_dropped() {
        let config: Config = toml::from_str(r#"bind = "127.0.0.1:0""#).expect("config");
        let scheduler = Arc::new(Scheduler::new(config).await.expect("scheduler"));
        let target = ConformanceTarget::default();

        let guard = conformance_request_guard(&scheduler, "model-a", &target);

        assert_eq!(scheduler.active_requests_for("model-a"), 1);
        drop(guard);
        assert_eq!(scheduler.active_requests_for("model-a"), 0);
    }

    #[tokio::test]
    async fn external_conformance_target_is_not_tracked_by_scheduler() {
        let config: Config = toml::from_str(r#"bind = "127.0.0.1:0""#).expect("config");
        let scheduler = Arc::new(Scheduler::new(config).await.expect("scheduler"));
        let target = ConformanceTarget {
            base_url: Some("https://example.test/v1".to_string()),
            ..Default::default()
        };

        let guard = conformance_request_guard(&scheduler, "external-model", &target);

        assert!(guard.is_none());
        assert_eq!(scheduler.active_requests_for("external-model"), 0);
    }
}
