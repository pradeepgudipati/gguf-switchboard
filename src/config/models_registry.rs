use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::ModelConfig;
use crate::errors::RuntimeError;

/// Simplified model registry — short aliases instead of full GGUF paths in API requests.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsRegistry {
    #[serde(default)]
    pub defaults: RegistryDefaults,
    /// When true, every `.gguf` file under `defaults.models_dir` is registered
    /// unless already listed in `[[models]]`.
    #[serde(default = "default_auto_discover")]
    pub auto_discover: bool,
    #[serde(default)]
    pub models: Vec<RegistryEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegistryDefaults {
    #[serde(default = "default_models_dir")]
    pub models_dir: String,
    #[serde(default = "default_llama_server")]
    pub llama_server: String,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_base_port")]
    pub base_port: u16,
    #[serde(default = "default_context_size")]
    pub context_size: u32,
    #[serde(default = "default_ngl")]
    pub ngl: u32,
    #[serde(default = "default_backend")]
    pub backend: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegistryEntry {
    /// Short model id used in API requests (e.g. `gemma-code`).
    pub alias: String,
    /// GGUF filename relative to `models_dir`, or an absolute path.
    pub file: String,
    /// Human-readable name for `/v1/models`. Defaults to a title-cased alias.
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub priority: bool,
    /// Override the auto-assigned backend port.
    #[serde(default)]
    pub port: Option<u16>,
    /// Override `defaults.context_size` for this model.
    #[serde(default)]
    pub context_size: Option<u32>,
}

impl Default for RegistryDefaults {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            llama_server: default_llama_server(),
            host: default_host(),
            base_port: default_base_port(),
            context_size: default_context_size(),
            ngl: default_ngl(),
            backend: default_backend(),
        }
    }
}

fn default_models_dir() -> String {
    "/models".to_string()
}

fn default_llama_server() -> String {
    "/usr/local/bin/llama-server".to_string()
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_base_port() -> u16 {
    8081
}

fn default_context_size() -> u32 {
    65536
}

fn default_ngl() -> u32 {
    999
}

fn default_backend() -> String {
    "llama.cpp".to_string()
}

fn default_auto_discover() -> bool {
    true
}

/// Resolve `llama-server` from PATH, falling back to the bundled default path.
pub fn detect_llama_server() -> String {
    if Path::new("/usr/local/bin/llama-server").is_file() {
        return "/usr/local/bin/llama-server".to_string();
    }

    if let Ok(output) = std::process::Command::new("sh")
        .arg("-c")
        .arg("command -v llama-server")
        .output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() && Path::new(&path).is_file() {
            return path;
        }
    }

    default_llama_server()
}

fn relative_model_file(models_dir: &Path, path: &Path) -> String {
    if path.starts_with(models_dir) {
        path.strip_prefix(models_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .trim_start_matches('/')
            .to_string()
    } else {
        path.to_string_lossy().into_owned()
    }
}

fn normalize_file_key(models_dir: &str, file: &str) -> String {
    let _ = models_dir;
    let path = Path::new(file);
    let key = if path.is_absolute() {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| file.to_string())
    } else {
        file.trim_start_matches('/').to_string()
    };
    key.to_ascii_lowercase()
}

fn build_existing_file_map(
    existing: &ModelsRegistry,
    models_dir: &str,
) -> HashMap<String, RegistryEntry> {
    existing
        .models
        .iter()
        .map(|entry| (normalize_file_key(models_dir, &entry.file), entry.clone()))
        .collect()
}

impl ModelsRegistry {
    pub fn load(path: &str) -> Result<Self, RuntimeError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to read models file '{path}': {e}"))
        })?;
        toml::from_str(&content).map_err(RuntimeError::from)
    }

    pub fn write(&self, path: &str) -> Result<(), RuntimeError> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to serialize models registry: {e}"))
        })?;
        std::fs::write(path, content).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to write models file '{path}': {e}"))
        })?;
        Ok(())
    }

    /// Scan `dir` for `.gguf` files and build a registry with generated aliases.
    pub fn discover(dir: &str) -> Result<Self, RuntimeError> {
        Self::discover_with_merge(dir, None)
    }

    /// Scan `dir` for `.gguf` files, optionally merging metadata from an existing registry.
    ///
    /// When merging, entries are matched by normalized `file` path. Existing
    /// `alias`, `display_name`, `priority`, `port`, and `context_size` are preserved.
    /// If no `.gguf` files are found but `merge_from` is set, the existing registry
    /// is returned with an updated `defaults.models_dir`.
    pub fn discover_with_merge(dir: &str, merge_from: Option<&Self>) -> Result<Self, RuntimeError> {
        let models_dir = Path::new(dir);
        if !models_dir.is_dir() {
            return Err(RuntimeError::ConfigError(format!(
                "Models directory does not exist: '{dir}'"
            )));
        }

        let mut files = discover_gguf_files(models_dir)?;
        files.sort();

        let base_defaults = merge_from
            .map(|existing| existing.defaults.clone())
            .unwrap_or_default();

        let llama_server = if let Some(existing) = merge_from {
            if existing.defaults.llama_server.is_empty() {
                detect_llama_server()
            } else {
                existing.defaults.llama_server.clone()
            }
        } else {
            detect_llama_server()
        };

        let mut registry = ModelsRegistry {
            defaults: RegistryDefaults {
                models_dir: models_dir.to_string_lossy().into_owned(),
                llama_server,
                ..base_defaults
            },
            auto_discover: merge_from.map(|e| e.auto_discover).unwrap_or(true),
            models: Vec::new(),
        };

        if files.is_empty() {
            if let Some(existing) = merge_from {
                registry.auto_discover = existing.auto_discover;
                registry.models = existing.models.clone();
                return Ok(registry);
            }
            return Err(RuntimeError::ConfigError(format!(
                "No .gguf files found in '{dir}'"
            )));
        }

        let existing_by_file = merge_from
            .map(|existing| build_existing_file_map(existing, &registry.defaults.models_dir))
            .unwrap_or_default();

        let mut used_aliases = HashSet::new();

        for path in files {
            let file = relative_model_file(models_dir, &path);

            if let Some(existing) = existing_by_file.get(&file) {
                let alias = if used_aliases.contains(&existing.alias) {
                    dedupe_alias(&existing.alias, &mut used_aliases)
                } else {
                    used_aliases.insert(existing.alias.clone());
                    existing.alias.clone()
                };

                registry.models.push(RegistryEntry {
                    alias,
                    file,
                    display_name: existing.display_name.clone(),
                    priority: existing.priority,
                    port: existing.port,
                    context_size: existing.context_size,
                });
                continue;
            }

            let mut alias = alias_from_filename(&path);
            alias = dedupe_alias(&alias, &mut used_aliases);

            registry.models.push(RegistryEntry {
                alias: alias.clone(),
                file,
                display_name: Some(display_name_from_alias(&alias)),
                priority: false,
                port: None,
                context_size: None,
            });
        }

        if !registry.models.iter().any(|entry| entry.priority) {
            registry.models[0].priority = true;
        }

        Ok(registry)
    }

    /// Expand registry entries into full `ModelConfig` map keyed by alias.
    pub fn expand(
        &self,
        fallback_backend: &str,
    ) -> Result<HashMap<String, ModelConfig>, RuntimeError> {
        let mut entries = self.models.clone();
        let mut claimed_paths = HashSet::new();

        for entry in &entries {
            let path = self.resolve_model_path(&entry.file)?;
            claimed_paths.insert(path);
        }

        if self.auto_discover {
            let models_dir = Path::new(&self.defaults.models_dir);
            if !models_dir.is_dir() {
                return Err(RuntimeError::ConfigError(format!(
                    "auto_discover is enabled but models_dir does not exist: '{}'",
                    self.defaults.models_dir
                )));
            }

            let mut discovered = discover_gguf_files(models_dir)?;
            discovered.sort();

            let mut used_aliases: HashSet<String> =
                entries.iter().map(|e| e.alias.clone()).collect();

            for path in discovered {
                let canonical = path.to_string_lossy().into_owned();
                if claimed_paths.contains(&canonical) {
                    continue;
                }
                claimed_paths.insert(canonical.clone());

                let mut alias = alias_from_filename(&path);
                alias = dedupe_alias(&alias, &mut used_aliases);

                let file = path
                    .strip_prefix(models_dir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .trim_start_matches('/')
                    .to_string();

                entries.push(RegistryEntry {
                    alias: alias.clone(),
                    file,
                    display_name: Some(display_name_from_alias(&alias)),
                    priority: false,
                    port: None,
                    context_size: None,
                });
            }
        }

        if entries.is_empty() {
            return Err(RuntimeError::ConfigError(
                "Models registry defines no models (add [[models]] entries or enable auto_discover)"
                    .to_string(),
            ));
        }

        validate_unique_aliases(&entries)?;

        let backend = if self.defaults.backend.is_empty() {
            fallback_backend.to_string()
        } else {
            self.defaults.backend.clone()
        };

        let mut models = HashMap::new();
        for (index, entry) in entries.iter().enumerate() {
            let model_path = self.resolve_model_path(&entry.file)?;
            let port = entry
                .port
                .unwrap_or(self.defaults.base_port.saturating_add(index as u16));
            let context_size = entry.context_size.unwrap_or(self.defaults.context_size);

            let config = ModelConfig {
                backend: backend.clone(),
                display_name: entry
                    .display_name
                    .clone()
                    .unwrap_or_else(|| display_name_from_alias(&entry.alias)),
                command: self.defaults.llama_server.clone(),
                args: vec![
                    "-m".to_string(),
                    model_path,
                    "--host".to_string(),
                    self.defaults.host.clone(),
                    "--port".to_string(),
                    port.to_string(),
                    "-c".to_string(),
                    context_size.to_string(),
                    "-ngl".to_string(),
                    self.defaults.ngl.to_string(),
                ],
                backend_url: format!("http://{}:{}/v1", self.defaults.host, port),
                health_url: format!("http://{}:{}/health", self.defaults.host, port),
                priority: entry.priority,
            };
            models.insert(entry.alias.clone(), config);
        }

        Ok(models)
    }

    fn resolve_model_path(&self, file: &str) -> Result<String, RuntimeError> {
        let path = Path::new(file);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            Path::new(&self.defaults.models_dir).join(file)
        };

        let canonical = resolved.to_string_lossy().into_owned();
        if !Path::new(&canonical).exists() {
            return Err(RuntimeError::ConfigError(format!(
                "GGUF file not found for alias: '{canonical}'"
            )));
        }
        Ok(canonical)
    }
}

fn validate_unique_aliases(entries: &[RegistryEntry]) -> Result<(), RuntimeError> {
    let mut seen = HashSet::new();
    for entry in entries {
        if !seen.insert(entry.alias.clone()) {
            return Err(RuntimeError::ConfigError(format!(
                "Duplicate model alias: '{}'",
                entry.alias
            )));
        }
    }
    Ok(())
}

fn dedupe_alias(alias: &str, used: &mut HashSet<String>) -> String {
    if used.insert(alias.to_string()) {
        return alias.to_string();
    }

    let mut counter = 2;
    loop {
        let candidate = format!("{alias}-{counter}");
        if used.insert(candidate.clone()) {
            return candidate;
        }
        counter += 1;
    }
}

/// Generate a short API alias from a GGUF filename or path.
pub fn alias_from_filename(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_default()
        .to_string();

    let mut alias = stem.to_ascii_lowercase();
    alias = strip_quant_suffix(&alias);
    alias = alias.replace('_', "-");
    sanitize_alias(&alias)
}

fn strip_quant_suffix(name: &str) -> String {
    let static_suffixes = ["-instruct", "-it", "-gguf", "-bf16", "-fp16", "-fp32"];

    let mut result = name.to_string();
    loop {
        let before = result.clone();

        for suffix in static_suffixes {
            if let Some(stripped) = result.strip_suffix(suffix) {
                result = stripped.to_string();
            }
        }

        if let Some(stripped) = strip_dynamic_quant_suffix(&result) {
            result = stripped;
        }

        if result == before {
            break;
        }
    }
    result
}

fn strip_dynamic_quant_suffix(name: &str) -> Option<String> {
    let (prefix, suffix) = name.rsplit_once('-')?;
    let first = suffix.chars().next()?;
    if !matches!(first, 'q' | 'Q' | 'i' | 'I' | 'f' | 'F') {
        return None;
    }

    let digit_start = if first.eq_ignore_ascii_case(&'i') {
        suffix.chars().nth(1).filter(|c| c.is_ascii_digit())?;
        1
    } else {
        suffix.chars().nth(1).filter(|c| c.is_ascii_digit())?;
        1
    };

    let rest = &suffix[digit_start..];
    if !rest.is_empty() && !rest.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }

    if prefix.is_empty() {
        None
    } else {
        Some(prefix.to_string())
    }
}

fn sanitize_alias(alias: &str) -> String {
    let mut out = String::with_capacity(alias.len());
    for ch in alias.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '.' {
            out.push(ch);
        } else if ch == ' ' || ch == '_' {
            out.push('-');
        }
    }

    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "model".to_string()
    } else {
        trimmed
    }
}

fn display_name_from_alias(alias: &str) -> String {
    alias
        .split(['-', '_'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => {
                    let rest: String = chars.collect();
                    format!("{}{}", first.to_ascii_uppercase(), rest)
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn discover_gguf_files(dir: &Path) -> Result<Vec<PathBuf>, RuntimeError> {
    let mut files = Vec::new();
    collect_gguf_files(dir, &mut files)?;
    Ok(files)
}

fn collect_gguf_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), RuntimeError> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        RuntimeError::ConfigError(format!("Failed to read directory '{}': {e}", dir.display()))
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to read directory entry: {e}"))
        })?;
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_files(&path, out)?;
        } else if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_strips_quant_suffix() {
        assert_eq!(
            alias_from_filename(Path::new("Qwen3.5-9B-Q4_K_M.gguf")),
            "qwen3.5-9b"
        );
        assert_eq!(
            alias_from_filename(Path::new("gemma-3-4b-it-Q4_K_M.gguf")),
            "gemma-3-4b"
        );
    }

    #[test]
    fn display_name_title_cases_alias() {
        assert_eq!(display_name_from_alias("qwen3.5-9b"), "Qwen3.5 9b");
        assert_eq!(display_name_from_alias("gemma-code"), "Gemma Code");
    }

    #[test]
    fn discover_merge_preserves_customizations() {
        let dir = std::env::temp_dir().join("gguf-switchboard-merge-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("gemma-3-4b.gguf"), b"fake").unwrap();
        std::fs::write(dir.join("qwen2.5-coder-7b.gguf"), b"fake").unwrap();

        let existing = ModelsRegistry {
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                base_port: 9000,
                ..RegistryDefaults::default()
            },
            auto_discover: true,
            models: vec![
                RegistryEntry {
                    alias: "gemma-code".to_string(),
                    file: "gemma-3-4b.gguf".to_string(),
                    display_name: Some("Gemma 3 Coding Model".to_string()),
                    priority: true,
                    port: None,
                    context_size: None,
                },
                RegistryEntry {
                    alias: "legacy-missing".to_string(),
                    file: "removed.gguf".to_string(),
                    display_name: Some("Removed".to_string()),
                    priority: false,
                    port: None,
                    context_size: None,
                },
            ],
        };

        let merged =
            ModelsRegistry::discover_with_merge(dir.to_str().unwrap(), Some(&existing)).unwrap();
        assert_eq!(merged.models.len(), 2);
        let gemma = merged
            .models
            .iter()
            .find(|entry| entry.file == "gemma-3-4b.gguf")
            .unwrap();
        assert_eq!(gemma.alias, "gemma-code");
        assert_eq!(gemma.display_name.as_deref(), Some("Gemma 3 Coding Model"));
        assert!(gemma.priority);
        assert!(
            merged
                .models
                .iter()
                .any(|entry| entry.alias == "qwen2.5-coder-7b")
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_merge_keeps_existing_when_no_gguf_found() {
        let dir = std::env::temp_dir().join("gguf-switchboard-merge-empty-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let existing = ModelsRegistry {
            defaults: RegistryDefaults {
                models_dir: "/old".to_string(),
                ..RegistryDefaults::default()
            },
            auto_discover: false,
            models: vec![RegistryEntry {
                alias: "demo".to_string(),
                file: "demo.gguf".to_string(),
                display_name: Some("Demo".to_string()),
                priority: true,
                port: None,
                context_size: None,
            }],
        };

        let merged =
            ModelsRegistry::discover_with_merge(dir.to_str().unwrap(), Some(&existing)).unwrap();
        assert_eq!(merged.defaults.models_dir, dir.to_string_lossy());
        assert_eq!(merged.models.len(), 1);
        assert_eq!(merged.models[0].alias, "demo");
        assert!(!merged.auto_discover);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expand_builds_backend_args() {
        let dir = std::env::temp_dir().join("gguf-switchboard-registry-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("test-model.gguf");
        std::fs::write(&model_path, b"fake").unwrap();

        let registry = ModelsRegistry {
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                base_port: 9001,
                context_size: 4096,
                ..RegistryDefaults::default()
            },
            auto_discover: false,
            models: vec![RegistryEntry {
                alias: "test".to_string(),
                file: "test-model.gguf".to_string(),
                display_name: Some("Test Model".to_string()),
                priority: true,
                port: None,
                context_size: None,
            }],
        };

        let models = registry.expand("llama.cpp").unwrap();
        let cfg = models.get("test").unwrap();
        assert_eq!(cfg.display_name, "Test Model");
        assert!(cfg.args.contains(&"-m".to_string()));
        assert!(cfg.args.contains(&model_path.to_string_lossy().to_string()));
        assert!(cfg.args.contains(&"9001".to_string()));
        assert!(cfg.args.contains(&"4096".to_string()));
        assert_eq!(cfg.backend_url, "http://127.0.0.1:9001/v1");

        std::fs::remove_dir_all(&dir).ok();
    }
}
