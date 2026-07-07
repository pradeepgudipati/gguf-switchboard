use openai_runtime::config::Config;
use openai_runtime::types::chat::{ChatCompletionRequest, ChatMessage, Content, Role};
use openai_runtime::types::completions::{CompletionRequest, Prompt};
use openai_runtime::types::embeddings::{EmbeddingInput, EmbeddingRequest};
use openai_runtime::types::{ListModelsResponse, ModelInfo, Usage};

#[test]
fn test_config_load_from_str() {
    let toml = r#"
bind = "127.0.0.1:8080"
startup_timeout = 30
idle_timeout = 300
default_backend = "llama.cpp"

[models.test-model]
backend = "llama.cpp"
display_name = "Test Model"
command = "/usr/bin/echo"
args = ["hello"]
backend_url = "http://127.0.0.1:9999/v1"
health_url = "http://127.0.0.1:9999/health"
priority = true
"#;

    // Write temp file and load
    let dir = std::env::temp_dir();
    let path = dir.join("test-openai-runtime-config.toml");
    std::fs::write(&path, toml).unwrap();

    let config = Config::load(path.to_str().unwrap()).unwrap();
    assert_eq!(config.bind, "127.0.0.1:8080");
    assert_eq!(config.startup_timeout, 30);
    assert_eq!(config.idle_timeout, 300);
    assert_eq!(config.default_backend, "llama.cpp");
    assert_eq!(config.models.len(), 1);

    let model = config.models.get("test-model").unwrap();
    assert_eq!(model.display_name, "Test Model");
    assert_eq!(model.command, "/usr/bin/echo");
    assert_eq!(model.args, vec!["hello"]);
    assert!(model.priority);

    assert_eq!(config.priority_model_id(), Some("test-model".to_string()));

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_config_rejects_empty_models() {
    let toml = r#"
bind = "127.0.0.1:8080"
default_backend = "llama.cpp"
"#;

    let dir = std::env::temp_dir();
    let path = dir.join("test-openai-runtime-empty.toml");
    std::fs::write(&path, toml).unwrap();

    let result = Config::load(path.to_str().unwrap());
    assert!(result.is_err());

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_chat_request_serialization() {
    let request = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage {
            role: Role::User,
            content: Some(Content::Text("Hello".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        temperature: Some(0.7),
        top_p: None,
        n: None,
        stream: Some(false),
        stop: None,
        max_tokens: Some(100),
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
        tools: None,
        tool_choice: None,
        seed: None,
        response_format: None,
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("\"model\":\"test-model\""));
    assert!(json.contains("\"role\":\"user\""));
    assert!(json.contains("\"content\":\"Hello\""));
    assert!(json.contains("\"max_tokens\":100"));

    // Round-trip
    let deserialized: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.model, "test-model");
    assert_eq!(deserialized.max_tokens, Some(100));
}

#[test]
fn test_completion_request_serialization() {
    let request = CompletionRequest {
        model: "test-model".to_string(),
        prompt: Prompt::Single("fn main() {".to_string()),
        suffix: None,
        max_tokens: Some(256),
        temperature: Some(0.2),
        top_p: None,
        n: None,
        stream: None,
        logprobs: None,
        echo: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        best_of: None,
        logit_bias: None,
        user: None,
        seed: None,
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("\"prompt\":\"fn main() {\""));
    assert!(json.contains("\"max_tokens\":256"));
}

#[test]
fn test_embedding_request_serialization() {
    let request = EmbeddingRequest {
        model: "test-model".to_string(),
        input: EmbeddingInput::Multiple(vec!["hello".to_string(), "world".to_string()]),
        encoding_format: None,
        dimensions: None,
        user: None,
    };

    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("\"input\":[\"hello\",\"world\"]"));
}

#[test]
fn test_model_info_serialization() {
    let model = ModelInfo::new("local-gemma-code");
    assert_eq!(model.id, "local-gemma-code");
    assert_eq!(model.object, "model");
    assert_eq!(model.owned_by, "local");

    let response = ListModelsResponse::new(vec![model]);
    assert_eq!(response.object, "list");
    assert_eq!(response.data.len(), 1);
    assert_eq!(response.data[0].id, "local-gemma-code");
}

#[test]
fn test_usage_serialization() {
    let usage = Usage {
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
    };
    let json = serde_json::to_string(&usage).unwrap();
    assert!(json.contains("\"prompt_tokens\":10"));
    assert!(json.contains("\"total_tokens\":30"));
}
