pub mod anthropic;
pub mod audio;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod models;
pub mod rerank;
pub mod responses;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::config::ModelConfig;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[schema(example = json!({
    "id": "qwen2.5-coder-7b",
    "object": "model",
    "created": 1710000000,
    "owned_by": "local",
    "backend": "vllm",
    "display_name": "Qwen2.5 Coder 7b",
    "kind": "coder",
    "description": "Code-specialized Qwen 2.5 instruct model",
    "context_size": 32768,
    "max_context_length": 32768,
    "min_vram_gb": 8,
    "capabilities": ["tools"],
    "hf_repo": "lmstudio-community/Qwen2.5-Coder-7B-Instruct-GGUF",
    "quantization": "awq",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85
}))]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    /// Model creation timestamp as Unix seconds
    pub created: i64,
    pub owned_by: String,
    /// The engine actually serving this model: `"llama.cpp"` or `"vllm"`.
    pub backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Model role: `chat`, `coder`, `vision`, `embedding`, or `reranker`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Serving context size, when known from launch args — `-c`/`--ctx-size`
    /// for llama.cpp, `--max-model-len` for vLLM.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_size: Option<u32>,
    /// Model maximum context length from GGUF/HF metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_vram_gb: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
    /// vLLM `--quantization` (e.g. `awq`, `gptq`, `fp8`). `None` for
    /// llama.cpp models or an unquantized vLLM model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// vLLM `--attention-backend` (e.g. `FLASH_ATTN`, `FLASHINFER`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention_backend: Option<String>,
    /// vLLM `--speculative-model` — the paired dspark draft model, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<String>,
    /// vLLM `--tensor-parallel-size`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<u32>,
    /// vLLM `--gpu-memory-utilization` (0.0-1.0) as configured in
    /// `models.toml`. Note: if this wasn't pinned, the scheduler's
    /// VllmFitPlanner computes and injects the actual value fresh at every
    /// load from live free VRAM — that live-adjusted value isn't reflected
    /// here yet (no vLLM equivalent of `runtime_profile` persists it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory_utilization: Option<f32>,
    /// The effective runtime profile after the last successful load.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_profile: Option<RuntimeProfileInfo>,
    /// Load-time tool-call probe verdict: `Some(true)` verified, `Some(false)`
    /// failed, `None` if the model never claimed `tools` capability or has
    /// never been loaded/probed yet.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_verified: Option<bool>,
    /// Rough measured generation throughput in tokens/second — the median of
    /// recent non-streaming requests (output tokens ÷ wall-clock time).
    /// `None` until the model has served enough requests to estimate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_sec: Option<f64>,
}

/// Runtime profile exposed via the API.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RuntimeProfileInfo {
    pub effective_context: u32,
    pub gpu_layers: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub split_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_split: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_k: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_v: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ubatch_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_concurrency: Option<u32>,
    pub reason: String,
    /// How this profile was determined: "auto-fit", "cached", "manual".
    pub profile_source: String,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".to_string(),
            created: Utc::now().timestamp(),
            owned_by: "local".to_string(),
            backend: "llama.cpp".to_string(),
            display_name: None,
            kind: None,
            description: None,
            context_size: None,
            max_context_length: None,
            min_vram_gb: None,
            capabilities: Vec::new(),
            hf_repo: None,
            quantization: None,
            attention_backend: None,
            draft_model: None,
            tensor_parallel_size: None,
            gpu_memory_utilization: None,
            runtime_profile: None,
            tools_verified: None,
            tokens_per_sec: None,
        }
    }

    pub fn from_config(id: impl Into<String>, config: &ModelConfig) -> Self {
        // llama.cpp uses -c/--ctx-size; vLLM uses --max-model-len. Neither
        // helper matches the other backend's flags, so try both.
        let context_size = crate::context::get_context_size(&config.args)
            .or_else(|| crate::fit::vllm_max_model_len_from_args(&config.args));
        let (
            quantization,
            attention_backend,
            draft_model,
            tensor_parallel_size,
            gpu_memory_utilization,
        ) = if config.backend == "vllm" {
            (
                crate::fit::vllm_quantization_from_args(&config.args),
                crate::fit::vllm_attention_backend_from_args(&config.args),
                crate::fit::vllm_draft_model_from_args(&config.args),
                crate::fit::vllm_tensor_parallel_size_from_args(&config.args),
                crate::fit::vllm_gpu_memory_utilization_from_args(&config.args),
            )
        } else {
            (None, None, None, None, None)
        };
        let runtime_profile = config
            .runtime_profile
            .as_ref()
            .map(|rp| RuntimeProfileInfo {
                effective_context: rp.context_size,
                gpu_layers: rp.ngl,
                split_mode: rp.split_mode.clone(),
                tensor_split: rp.tensor_split.clone(),
                cache_type_k: rp.cache_type_k.clone(),
                cache_type_v: rp.cache_type_v.clone(),
                batch_size: rp.batch_size,
                ubatch_size: rp.ubatch_size,
                embedding_concurrency: rp.embedding_concurrency,
                reason: rp.reason.clone(),
                profile_source: rp.profile_source.clone(),
            });
        Self {
            id: id.into(),
            object: "model".to_string(),
            created: Utc::now().timestamp(),
            owned_by: "local".to_string(),
            backend: config.backend.clone(),
            display_name: Some(config.display_name.clone()),
            kind: Some(config.kind.clone()),
            description: config.description.clone(),
            context_size,
            max_context_length: config.max_context_length,
            min_vram_gb: config.min_vram_gb,
            capabilities: config.capabilities.clone(),
            hf_repo: config.hf_repo.clone(),
            quantization,
            attention_backend,
            draft_model,
            tensor_parallel_size,
            gpu_memory_utilization,
            runtime_profile,
            tools_verified: None,
            tokens_per_sec: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

impl ListModelsResponse {
    pub fn new(models: Vec<ModelInfo>) -> Self {
        Self {
            object: "list".to_string(),
            data: models,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ToolCall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[cfg(test)]
mod tool_verified_tests {
    use super::*;

    #[test]
    fn tools_verified_defaults_to_none() {
        let info = ModelInfo::new("m");
        assert_eq!(info.tools_verified, None);
    }

    #[test]
    fn tools_verified_omitted_from_json_when_none() {
        let info = ModelInfo::new("m");
        let json = serde_json::to_value(&info).unwrap();
        assert!(json.get("tools_verified").is_none());
    }

    #[test]
    fn tools_verified_present_when_set() {
        let mut info = ModelInfo::new("m");
        info.tools_verified = Some(true);
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["tools_verified"], serde_json::json!(true));
    }
}
