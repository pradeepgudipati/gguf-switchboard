use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json};
use tracing::{debug, instrument};

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

/// Maximum tokens per batch to avoid exceeding server's physical batch size.
/// This is a conservative default; actual limits depend on the model's -b/-ub settings.
const MAX_TOKENS_PER_BATCH: u32 = 2048;

/// Batch distinct embedding inputs without changing one-input/one-output cardinality.
fn chunk_embedding_input(input: &EmbeddingInput) -> Vec<EmbeddingInput> {
    match input {
        EmbeddingInput::Single(_) => vec![input.clone()],
        EmbeddingInput::Multiple(texts) => {
            let mut batches = Vec::new();
            let mut current_batch = Vec::new();
            let mut current_tokens = 0;

            for text in texts {
                let tokens = estimate_tokens(text);
                if current_tokens + tokens > MAX_TOKENS_PER_BATCH && !current_batch.is_empty() {
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
    let _request_guard = state.scheduler.track_request(&model_id);
    // Inference-only timer; model load wait is exported separately.
    let start = std::time::Instant::now();

    // Split large inputs into batches to avoid exceeding the server's physical batch size.
    let batches = chunk_embedding_input(&request.input);
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

    #[test]
    fn chunk_single_short_input() {
        let input = EmbeddingInput::Single("short text".to_string());
        let batches = chunk_embedding_input(&input);
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn chunk_single_long_input_preserves_openai_cardinality() {
        // Create a text that exceeds MAX_TOKENS_PER_BATCH (2048 tokens = ~8192 chars)
        let long_text = "word ".repeat(2000); // ~10000 chars, ~2500 tokens
        let input = EmbeddingInput::Single(long_text);
        let batches = chunk_embedding_input(&input);
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
        let batches = chunk_embedding_input(&input);
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
        let batches = chunk_embedding_input(&input);
        // Should be split into multiple batches
        assert!(
            batches.len() > 1,
            "Expected multiple batches, got {}",
            batches.len()
        );
    }
}
