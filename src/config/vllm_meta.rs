//! Detect vLLM-relevant serving metadata from a Hugging Face safetensors
//! repo's `config.json` — quantization scheme and Speculators-format
//! speculative-decoding pairing (draft model + token count).
//!
//! CLI only — never called on the request path.

use serde_json::Value;

use super::hf_download::HfTreeEntry;
use crate::errors::RuntimeError;

/// Return true when the repo tree looks like a vLLM-servable safetensors
/// repo (has `*.safetensors` weights and a `config.json`), as opposed to a
/// GGUF-only repo.
pub fn is_safetensors_repo(entries: &[HfTreeEntry]) -> bool {
    let has_safetensors = entries
        .iter()
        .any(|e| e.path.ends_with(".safetensors") || e.path.ends_with(".safetensors.index.json"));
    let has_config = entries.iter().any(|e| e.path == "config.json");
    has_safetensors && has_config
}

/// Files worth pulling for a vLLM safetensors deployment: weight shards plus
/// the small config/tokenizer files vLLM needs to serve the model. Excludes
/// alternate formats (`.bin`, `.gguf`, `.onnx`) some repos ship alongside.
pub fn is_vllm_serving_file(path: &str) -> bool {
    if path.ends_with(".safetensors") || path.ends_with(".safetensors.index.json") {
        return true;
    }
    matches!(
        path,
        "config.json"
            | "generation_config.json"
            | "tokenizer.json"
            | "tokenizer_config.json"
            | "special_tokens_map.json"
            | "vocab.json"
            | "merges.txt"
            | "chat_template.jinja"
    )
}

#[derive(Debug, Clone, Default)]
pub struct VllmMetadata {
    /// vLLM `--quantization` value, from `config.json`'s `quantization_config`.
    pub quantization: Option<String>,
    /// HF repo id or local path of a paired speculative-decoding draft model,
    /// from the Speculators-format `speculators_config` block.
    pub draft_model: Option<String>,
    /// vLLM `--num-speculative-tokens`.
    pub num_speculative_tokens: Option<u32>,
    /// `max_position_embeddings` — used as the max context length.
    pub max_position_embeddings: Option<u32>,
    /// `architectures[0]` (e.g. `Qwen2ForCausalLM`).
    pub architecture: Option<String>,
}

/// Parse a repo's `config.json` text into vLLM serving metadata. Missing or
/// unrecognized fields are left `None` — this is best-effort enrichment, not
/// validation; vLLM will still error clearly at load time if something's off.
pub fn parse_config_json(text: &str) -> VllmMetadata {
    let value: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return VllmMetadata::default(),
    };

    let quantization = value
        .get("quantization_config")
        .and_then(|q| {
            q.get("quant_method")
                .or_else(|| q.get("quantization_method"))
        })
        .and_then(Value::as_str)
        .map(str::to_ascii_lowercase);

    // Speculators format (https://github.com/vllm-project/speculators): a
    // draft model's config.json carries `speculators_config` describing the
    // target model it pairs with and the number of tokens it proposes.
    let speculators = value.get("speculators_config");
    let draft_model = speculators
        .and_then(|s| {
            s.get("target_model")
                .or_else(|| s.get("verifier"))
                .or_else(|| s.get("base_model"))
        })
        .and_then(Value::as_str)
        .map(str::to_string);
    let num_speculative_tokens = speculators
        .and_then(|s| {
            s.get("num_speculative_tokens")
                .or_else(|| s.get("num_lookahead_tokens"))
        })
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok());

    let max_position_embeddings = value
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok());

    let architecture = value
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_str)
        .map(str::to_string);

    VllmMetadata {
        quantization,
        draft_model,
        num_speculative_tokens,
        max_position_embeddings,
        architecture,
    }
}

/// Fetch and parse `config.json` from a repo. Returns default (all-`None`)
/// metadata rather than erroring when the file is missing or unparsable —
/// callers treat this as best-effort enrichment.
pub async fn detect_vllm_metadata(
    client: &reqwest::Client,
    repo: &str,
) -> Result<VllmMetadata, RuntimeError> {
    match super::hf_download::fetch_repo_file_text(client, repo, "config.json").await? {
        Some(text) => Ok(parse_config_json(&text)),
        None => Ok(VllmMetadata::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quantization_config() {
        let meta = parse_config_json(r#"{"quantization_config": {"quant_method": "AWQ"}}"#);
        assert_eq!(meta.quantization.as_deref(), Some("awq"));
    }

    #[test]
    fn parses_speculators_draft_pairing() {
        let meta = parse_config_json(
            r#"{"speculators_config": {"target_model": "org/target-model", "num_speculative_tokens": 4}}"#,
        );
        assert_eq!(meta.draft_model.as_deref(), Some("org/target-model"));
        assert_eq!(meta.num_speculative_tokens, Some(4));
    }

    #[test]
    fn missing_fields_default_to_none() {
        let meta = parse_config_json(r#"{"architectures": ["Qwen2ForCausalLM"]}"#);
        assert_eq!(meta.quantization, None);
        assert_eq!(meta.draft_model, None);
        assert_eq!(meta.architecture.as_deref(), Some("Qwen2ForCausalLM"));
    }

    #[test]
    fn invalid_json_returns_default() {
        let meta = parse_config_json("not json");
        assert!(meta.quantization.is_none());
        assert!(meta.architecture.is_none());
    }
}
