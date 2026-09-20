use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json};
use tracing::{debug, instrument};

use crate::config::ModelConfig;
use crate::errors::RuntimeError;
use crate::kind_guard::{EMBEDDING_KINDS, require_kind};
use crate::metrics::{ACTIVE_REQUESTS, INFERENCE_LATENCY, REQUEST_TOTAL};
use crate::state::AppState;
use crate::types::embeddings::{EmbeddingInput, EmbeddingRequest, EmbeddingResponse};

struct ActiveGuard;
impl Drop for ActiveGuard {
    fn drop(&mut self) {
        ACTIVE_REQUESTS.dec();
    }
}

/// Estimate token count from text length (rough heuristic: ~4 chars per token).
fn estimate_tokens(text: &str) -> u32 {
    (text.len() as u32).div_ceil(4)
}

/// The live `-b` / `-ub` the model was actually started with.
///
/// The batch sizes that end up on the command line come from the fit planner,
/// not from the registry defaults, so the request path has to read the profile
/// the load settled on. Falls back to the configured args, then to
/// [`crate::batch::EMBEDDING_BATCH_FLOOR`] when neither is known.
fn effective_batch_limits(cfg: &ModelConfig) -> (u32, u32) {
    let floor = crate::batch::EMBEDDING_BATCH_FLOOR;
    let profile = cfg.runtime_profile.as_ref();
    let batch = profile
        .and_then(|p| p.batch_size)
        .or_else(|| crate::batch::get_batch_size(&cfg.args))
        .unwrap_or(floor);
    let ubatch = profile
        .and_then(|p| p.ubatch_size)
        .or_else(|| crate::batch::get_ubatch_size(&cfg.args))
        .unwrap_or(floor)
        .min(batch);
    (batch, ubatch)
}

/// Reject inputs that cannot fit one micro-batch, before they reach the backend.
///
/// A single input is one sequence: it cannot be split without changing the
/// one-input/one-output cardinality, so there is nothing to do but refuse it.
/// Forwarding it instead earns a backend 500 — or, on llama.cpp builds that
/// assert rather than return, a dead llama-server and a reload cycle.
///
/// `estimate_tokens` under-counts (4 bytes per token is generous for English
/// and very generous for CJK), so an estimate already over the limit means the
/// real count is too. That keeps this a guard against certain failures rather
/// than a heuristic that turns servable requests away.
fn reject_oversized_inputs(input: &EmbeddingInput, ubatch: u32) -> Result<(), RuntimeError> {
    let texts: Vec<&String> = match input {
        EmbeddingInput::Single(text) => vec![text],
        EmbeddingInput::Multiple(texts) => texts.iter().collect(),
    };
    for (index, text) in texts.iter().enumerate() {
        let tokens = estimate_tokens(text);
        if tokens > ubatch {
            return Err(RuntimeError::InvalidRequest(format!(
                "input[{index}] is about {tokens} tokens, over the model's physical batch size \
                 of {ubatch}. Split the text into smaller chunks, or raise `ubatch_size` \
                 (and `batch_size`) for this model in models.toml."
            )));
        }
    }
    Ok(())
}

/// Batch distinct embedding inputs without changing one-input/one-output cardinality.
///
/// `max_tokens_per_batch` is the model's logical batch size (`-b`): the cap on
/// the tokens llama-server accepts across one request.
fn chunk_embedding_input(input: &EmbeddingInput, max_tokens_per_batch: u32) -> Vec<EmbeddingInput> {
    match input {
        EmbeddingInput::Single(_) => vec![input.clone()],
        EmbeddingInput::Multiple(texts) => {
            let mut batches = Vec::new();
            let mut current_batch = Vec::new();
            let mut current_tokens = 0;

            for text in texts {
                let tokens = estimate_tokens(text);
                if current_tokens + tokens > max_tokens_per_batch && !current_batch.is_empty() {
                    batches.push(EmbeddingInput::Multiple(current_batch));
                    current_batch = Vec::new();
                    current_tokens = 0;
                }
                current_batch.push(text.clone());
                current_tokens += tokens;
            }

            if !current_batch.is_empty() {
                batches.push(EmbeddingInput::Multiple(current_batch));
            }

            if batches.is_empty() {
                vec![input.clone()]
            } else {
                batches
            }
        }
    }
}

/// Generate embeddings for input text.
#[utoipa::path(
    post,
    path = "/v1/embeddings",
    tag = "embeddings",
    request_body(
        content = EmbeddingRequest,
    ),
    responses(
        (status = 200, description = "Generated embeddings", body = EmbeddingResponse),
        (status = 400, description = "Invalid request"),
        (status = 404, description = "Model not found"),
        (status = 502, description = "Backend error")
    )
)]
#[instrument(skip(state, request), fields(model = %request.model))]
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<impl IntoResponse, RuntimeError> {
    REQUEST_TOTAL.inc();
    ACTIVE_REQUESTS.inc();
    // Created immediately so early returns (`?`) below cannot leak the gauge.
    let _guard = ActiveGuard;

    let cfg = state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    require_kind(&request.model, &cfg, EMBEDDING_KINDS, "/v1/embeddings")?;
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();
    // Re-read after the load: the fit planner writes the runtime profile
    // (batch sizes, concurrency) while starting the backend.
    let loaded_cfg = state.scheduler.model_config(&model_id).unwrap_or(cfg);
    let concurrency = loaded_cfg
        .runtime_profile
        .as_ref()
        .and_then(|profile| profile.embedding_concurrency)
        .unwrap_or(1) as usize;
    let (max_tokens_per_batch, ubatch) = effective_batch_limits(&loaded_cfg);
    reject_oversized_inputs(&request.input, ubatch)?;
    let _admission_permit = state
        .embedding_admission
        .acquire(&model_id, concurrency)
        .await?;
    let _request_guard = state.scheduler.track_request(&model_id);
    // Inference-only timer; model load wait is exported separately.
    let start = std::time::Instant::now();

    // Split large inputs into batches to avoid exceeding the server's physical batch size.
    let batches = chunk_embedding_input(&request.input, max_tokens_per_batch);
    let needs_chunking = batches.len() > 1;

    if needs_chunking {
        debug!(
            model = %request.model,
            original_input = ?format!("{:?}", request.input),
            batch_count = batches.len(),
            "Splitting embedding input into batches"
        );
    }

    let mut all_data = Vec::new();
    let mut total_prompt_tokens = 0u32;
    let mut total_total_tokens = 0u32;

    for (i, batch_input) in batches.into_iter().enumerate() {
        let batch_request = EmbeddingRequest {
            model: request.model.clone(),
            input: batch_input,
            encoding_format: request.encoding_format.clone(),
            dimensions: request.dimensions,
            user: request.user.clone(),
        };

        let response = backend.embeddings(batch_request).await?;

        // Adjust indices for merged response
        let offset = all_data.len() as u32;
        let mut batch_data = response.data;
        for item in &mut batch_data {
            item.index += offset;
        }
        all_data.extend(batch_data);

        total_prompt_tokens += response.usage.prompt_tokens;
        total_total_tokens += response.usage.total_tokens;

        if needs_chunking {
            debug!(
                model = %request.model,
                batch_index = i,
                prompt_tokens = response.usage.prompt_tokens,
                "Completed embedding batch"
            );
        }
    }

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data: all_data,
        model: model_id.clone(),
        usage: crate::types::embeddings::EmbeddingUsage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_total_tokens,
        },
    };

    // Record token usage
    let _ = state.token_db.record(
        &model_id,
        "/v1/embeddings",
        total_prompt_tokens,
        0,
        total_total_tokens,
        None,
    );

    INFERENCE_LATENCY.observe(start.elapsed().as_secs_f64());

    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_short_text() {
        let tokens = estimate_tokens("hello world");
        assert_eq!(tokens, 3); // 11 chars / 4 = 2.75, rounded up to 3
    }

    #[test]
    fn estimate_tokens_empty_text() {
        let tokens = estimate_tokens("");
        assert_eq!(tokens, 0);
    }

    const LIMIT: u32 = crate::batch::EMBEDDING_BATCH_FLOOR;

    fn cfg_with(args: Vec<&str>, profile: Option<(u32, u32)>) -> ModelConfig {
        ModelConfig {
            backend: "llama.cpp".into(),
            display_name: "embed".into(),
            command: "llama-server".into(),
            args: args.into_iter().map(String::from).collect(),
            backend_url: "http://127.0.0.1:1/v1".into(),
            health_url: "http://127.0.0.1:1/health".into(),
            priority: false,
            kind: "embedding".into(),
            description: None,
            max_context_length: None,
            min_vram_gb: None,
            capabilities: vec![],
            hf_repo: None,
            ctx_floor: None,
            block_count: None,
            ngl_pinned: false,
            model_fingerprint: None,
            max_context_from_gguf: None,
            runtime_profile: profile.map(|(batch, ubatch)| crate::config::RuntimeProfile {
                context_size: 8192,
                ngl: 999,
                split_mode: None,
                tensor_split: None,
                cache_type_k: None,
                cache_type_v: None,
                batch_size: Some(batch),
                ubatch_size: Some(ubatch),
                embedding_concurrency: Some(1),
                reason: "test".into(),
                profile_source: "test".into(),
            }),
        }
    }

    #[test]
    fn chunk_single_short_input() {
        let input = EmbeddingInput::Single("short text".to_string());
        let batches = chunk_embedding_input(&input, LIMIT);
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn chunk_single_long_input_preserves_openai_cardinality() {
        // Create a text that exceeds the batch limit (2048 tokens = ~8192 chars)
        let long_text = "word ".repeat(2000); // ~10000 chars, ~2500 tokens
        let input = EmbeddingInput::Single(long_text);
        let batches = chunk_embedding_input(&input, LIMIT);
        assert_eq!(
            batches.len(),
            1,
            "one input must remain one embedding input"
        );
        assert!(matches!(batches[0], EmbeddingInput::Single(_)));
    }

    #[test]
    fn chunk_multiple_inputs() {
        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ];
        let input = EmbeddingInput::Multiple(texts);
        let batches = chunk_embedding_input(&input, LIMIT);
        // All short texts should fit in one batch
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn chunk_multiple_large_inputs() {
        // Create many large texts that exceed the batch limit
        // Each text is ~10000 chars (~2500 tokens), so 3 texts = ~7500 tokens > 2048
        let large_text = "word ".repeat(2000); // ~10000 chars each
        let texts = vec![large_text.clone(), large_text.clone(), large_text.clone()];
        let input = EmbeddingInput::Multiple(texts);
        let batches = chunk_embedding_input(&input, LIMIT);
        // Should be split into multiple batches
        assert!(
            batches.len() > 1,
            "Expected multiple batches, got {}",
            batches.len()
        );
    }

    #[test]
    fn chunk_budget_follows_a_smaller_batch_size() {
        // Two ~600-token texts fit one 2048-token batch but not a 1024 one.
        let text = "word ".repeat(480); // ~2400 chars, ~600 tokens
        let input = EmbeddingInput::Multiple(vec![text.clone(), text]);
        assert_eq!(chunk_embedding_input(&input, 2048).len(), 1);
        assert_eq!(chunk_embedding_input(&input, 1024).len(), 2);
    }

    #[test]
    fn batch_limits_prefer_the_runtime_profile_over_configured_args() {
        // The fit planner rewrites the command line, so the profile is the
        // only place that reflects what llama-server is really running with.
        let cfg = cfg_with(vec!["-b", "2048", "-ub", "2048"], Some((4096, 4096)));
        assert_eq!(effective_batch_limits(&cfg), (4096, 4096));
    }

    #[test]
    fn batch_limits_fall_back_to_args_then_floor() {
        let from_args = cfg_with(vec!["-b", "4096", "-ub", "4096"], None);
        assert_eq!(effective_batch_limits(&from_args), (4096, 4096));

        let bare = cfg_with(vec!["-m", "embed.gguf"], None);
        assert_eq!(effective_batch_limits(&bare), (LIMIT, LIMIT));
    }

    #[test]
    fn batch_limits_never_report_ubatch_above_batch() {
        let cfg = cfg_with(vec![], Some((2048, 8192)));
        let (batch, ubatch) = effective_batch_limits(&cfg);
        assert_eq!((batch, ubatch), (2048, 2048));
    }

    #[test]
    fn oversized_single_input_is_rejected_before_the_backend_sees_it() {
        // The production failure: a chunk longer than the server's -ub. Better
        // a 400 than a backend 500 (or an assert that kills llama-server).
        let input = EmbeddingInput::Single("word ".repeat(2000));
        let error = reject_oversized_inputs(&input, 512).unwrap_err();
        assert!(matches!(error, RuntimeError::InvalidRequest(_)));
        assert!(error.to_string().contains("physical batch size"));
    }

    #[test]
    fn oversized_input_names_its_index() {
        let input = EmbeddingInput::Multiple(vec![
            "short".to_string(),
            "word ".repeat(2000),
            "short".to_string(),
        ]);
        let error = reject_oversized_inputs(&input, 512).unwrap_err();
        assert!(error.to_string().contains("input[1]"));
    }

    #[test]
    fn inputs_within_the_physical_batch_are_admitted() {
        // ~530 tokens — the size that failed in production at -ub 512 and
        // must pass once the floor raises -ub to 2048.
        let input = EmbeddingInput::Single("word ".repeat(424));
        assert!(reject_oversized_inputs(&input, LIMIT).is_ok());
    }
}
