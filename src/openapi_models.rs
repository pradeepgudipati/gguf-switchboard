//! Inject live model aliases into OpenAPI request schemas for Swagger dropdowns.

use serde_json::{Value, json};

use crate::config::ModelConfig;

/// Mutate an OpenAPI document so `model` fields on request bodies become enums
/// of the currently configured aliases, with a descriptive one-liner.
pub fn inject_model_enums(doc: &mut Value, models: &[(String, ModelConfig)]) {
    let mut aliases: Vec<String> = models.iter().map(|(id, _)| id.clone()).collect();
    aliases.sort();

    let description = if models.is_empty() {
        "Model alias configured in models.toml".to_string()
    } else {
        let mut lines: Vec<String> = models
            .iter()
            .map(|(id, cfg)| {
                let mut parts = vec![format!("{id} ({})", cfg.kind)];
                if let Some(ctx) = cfg
                    .max_context_length
                    .or_else(|| crate::context::get_context_size(&cfg.args))
                {
                    parts.push(format!("ctx={ctx}"));
                }
                if let Some(vram) = cfg.min_vram_gb {
                    parts.push(format!("~{vram}GB"));
                }
                if let Some(desc) = cfg.description.as_deref() {
                    let short = desc.chars().take(80).collect::<String>();
                    parts.push(short);
                }
                parts.join(" — ")
            })
            .collect();
        lines.sort();
        format!(
            "Configured model alias. Available:\n{}",
            lines
                .iter()
                .map(|l| format!("- {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    let schema_names = [
        "ChatCompletionRequest",
        "CompletionRequest",
        "EmbeddingRequest",
        "ResponseRequest",
    ];

    let Some(schemas) = doc
        .pointer_mut("/components/schemas")
        .and_then(|v| v.as_object_mut())
    else {
        return;
    };

    for name in schema_names {
        let Some(schema) = schemas.get_mut(name) else {
            continue;
        };
        let Some(props) = schema.get_mut("properties").and_then(|v| v.as_object_mut()) else {
            continue;
        };
        let Some(model_prop) = props.get_mut("model") else {
            continue;
        };
        if !aliases.is_empty() {
            model_prop["enum"] = Value::Array(aliases.iter().cloned().map(Value::String).collect());
            if let Some(first) = aliases.first() {
                model_prop["example"] = json!(first);
            }
        }
        model_prop["description"] = Value::String(description.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cfg(kind: &str) -> ModelConfig {
        ModelConfig {
            backend: "llama.cpp".into(),
            display_name: "Demo".into(),
            command: "x".into(),
            args: vec!["-c".into(), "4096".into()],
            backend_url: "http://127.0.0.1:1/v1".into(),
            health_url: "http://127.0.0.1:1/health".into(),
            priority: false,
            kind: kind.into(),
            description: Some("A demo model".into()),
            max_context_length: Some(8192),
            min_vram_gb: Some(4),
            capabilities: vec![],
            hf_repo: None,
        }
    }

    #[test]
    fn injects_enum_on_chat_request() {
        let mut doc = json!({
            "components": {
                "schemas": {
                    "ChatCompletionRequest": {
                        "properties": {
                            "model": { "type": "string" }
                        }
                    }
                }
            }
        });
        inject_model_enums(&mut doc, &[("gemma-4-e4b".into(), sample_cfg("chat"))]);
        let model = &doc["components"]["schemas"]["ChatCompletionRequest"]["properties"]["model"];
        assert_eq!(model["enum"], json!(["gemma-4-e4b"]));
        assert!(
            model["description"]
                .as_str()
                .unwrap()
                .contains("gemma-4-e4b")
        );
    }
}
