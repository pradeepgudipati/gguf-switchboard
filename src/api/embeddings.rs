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

/// Split a large embedding input into smaller batches that fit within the server's batch size.
fn chunk_embedding_input(input: &EmbeddingInput) -> Vec<EmbeddingInput> {
    match input {
        EmbeddingInput::Single(text) => {
            let tokens = estimate_tokens(text);
            if tokens <= MAX_TOKENS_PER_BATCH {
                vec![input.clone()]
            } else {
                // Split large text into chunks by sentences or paragraphs
                let chunks = split_text_into_chunks(text, MAX_TOKENS_PER_BATCH);
                chunks.into_iter().map(EmbeddingInput::Single).collect()
            }
        }
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

/// Split text into chunks that fit within the token limit.
/// Tries to split on sentence boundaries, then falls back to word boundaries.
fn split_text_into_chunks(text: &str, max_tokens: u32) -> Vec<String> {
    let max_chars = (max_tokens * 4) as usize; // Rough estimate: 4 chars per token
    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= max_chars {
            chunks.push(remaining.to_string());
            break;
        }

        // Find a good split point (sentence boundary, then word boundary)
        let split_pos = find_split_position(remaining, max_chars);
        let (chunk, rest) = remaining.split_at(split_pos);
        chunks.push(chunk.to_string());
        remaining = rest.trim_start();
    }

    chunks
}

/// Find a good position to split text, preferring sentence boundaries.
fn find_split_position(text: &str, max_chars: usize) -> usize {
    if max_chars >= text.len() {
        return text.len();
    }

    // Try to find sentence boundary
    let search_range = &text[..max_chars];
    if let Some(pos) = search_range.rfind(". ") {
        return pos + 2;
    }
    if let Some(pos) = search_range.rfind(".\n") {
        return pos + 2;
    }
    if let Some(pos) = search_range.rfind("\n\n") {
        return pos + 2;
    }

    // Fall back to word boundary
    if let Some(pos) = search_range.rfind(' ') {
        return pos + 1;
    }

    // Last resort: split at max_chars
    max_chars
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

    let start = std::time::Instant::now();
    let cfg = state
        .scheduler
        .model_config(&request.model)
        .ok_or_else(|| RuntimeError::ModelNotFound(request.model.clone()))?;
    require_kind(&request.model, &cfg, EMBEDDING_KINDS, "/v1/embeddings")?;
    let backend = state.scheduler.ensure_loaded(&request.model).await?;
    let model_id = request.model.clone();
    let _request_guard = state.scheduler.track_request(&model_id);
    let _guard = ActiveGuard;

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
    fn chunk_single_long_input() {
        // Create a text that exceeds MAX_TOKENS_PER_BATCH (2048 tokens = ~8192 chars)
        let long_text = "word ".repeat(2000); // ~10000 chars, ~2500 tokens
        let input = EmbeddingInput::Single(long_text);
        let batches = chunk_embedding_input(&input);
        // Should be split into multiple batches
        assert!(
            batches.len() > 1,
            "Expected multiple batches, got {}",
            batches.len()
        );
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

    #[test]
    fn split_text_into_chunks_sentence_boundary() {
        let text = "First sentence. Second sentence. Third sentence.";
        let chunks = split_text_into_chunks(text, 100); // Large enough to fit all
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn split_text_into_chunks_long_text() {
        let text =
            "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence.";
        let chunks = split_text_into_chunks(text, 3); // 3 tokens ≈ 12 chars, forces splitting
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {}",
            chunks.len()
        );
        // Verify all chunks are non-empty
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
        // Verify original text can be reconstructed from chunks
        let combined: String = chunks.join("");
        // Allow for trimmed whitespace between chunks
        assert!(combined.len() >= text.len() - chunks.len());
    }

    #[test]
    fn find_split_position_sentence_boundary() {
        let text = "Hello world. This is a test.";
        let pos = find_split_position(text, 20);
        assert_eq!(pos, 13); // After ". "
    }

    #[test]
    fn find_split_position_word_boundary() {
        let text = "Hello world this is a test";
        let pos = find_split_position(text, 15);
        assert_eq!(pos, 12); // After "world "
    }

    #[test]
    fn find_split_position_max_chars() {
        let text = "HelloWorldThisIsATest";
        let pos = find_split_position(text, 10);
        assert_eq!(pos, 10); // No good boundary, split at max_chars
    }
}
