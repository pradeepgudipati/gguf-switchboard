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
    /// Directory or comma-separated directories scanned for llama.cpp-loadable GGUF files.
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
    /// GGUF path relative to `models_dir` (single directory), or an absolute path.
    /// With comma-separated `models_dir` values, discovered files are stored as absolute paths.
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

/// Split `models_dir` on commas (e.g. `"/models,/data/gguf"`).
pub fn parse_models_dirs(configured: &str) -> Vec<PathBuf> {
    configured
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(PathBuf::from)
        .collect()
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

fn model_file_reference(models_dirs: &[PathBuf], path: &Path) -> String {
    if models_dirs.len() == 1 {
        relative_model_file(&models_dirs[0], path)
    } else {
        path.to_string_lossy().into_owned()
    }
}

fn normalize_file_key(models_dirs: &[PathBuf], file: &str) -> String {
    if let Ok(resolved) = resolve_model_path(models_dirs, file) {
        return resolved.to_ascii_lowercase();
    }

    let path = Path::new(file);
    let key = if path.is_absolute() {
        path.to_string_lossy().into_owned()
    } else if models_dirs.len() == 1 {
        file.trim_start_matches('/').to_string()
    } else {
        file.to_string()
    };
    key.to_ascii_lowercase()
}

/// Return true for likely LLM weight filenames; skip sidecars and adapter artifacts.
fn is_discoverable_gguf_name(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    !(lower.starts_with("mmproj")
        || lower.starts_with("mtp-")
        || lower.starts_with("ggml-vocab")
        || lower.contains("-mmproj")
        || lower.contains("-lora")
        || lower.ends_with("-vocab.gguf"))
}

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_METADATA_STRING: u32 = 8;

#[derive(Debug, Default)]
struct GgufMetadata {
    architecture: Option<String>,
    file_type: Option<String>,
}

impl GgufMetadata {
    fn is_llama_cpp_loadable(&self) -> bool {
        let Some(arch) = self.architecture.as_ref() else {
            return false;
        };
        if arch.is_empty() {
            return false;
        }

        let arch_lower = arch.to_ascii_lowercase();
        if matches!(
            arch_lower.as_str(),
            "clip" | "siglip" | "vit" | "wav2vec2" | "whisper" | "encoder"
        ) {
            return false;
        }

        if let Some(file_type) = &self.file_type {
            let file_type_lower = file_type.to_ascii_lowercase();
            if matches!(file_type_lower.as_str(), "lora" | "vocab") {
                return false;
            }
        }

        true
    }
}

fn read_gguf_string(data: &[u8], offset: &mut usize) -> Option<String> {
    if *offset + 8 > data.len() {
        return None;
    }
    let len = u64::from_le_bytes(data[*offset..*offset + 8].try_into().ok()?) as usize;
    *offset += 8;
    if *offset + len > data.len() {
        return None;
    }
    let value = std::str::from_utf8(&data[*offset..*offset + len])
        .ok()?
        .to_string();
    *offset += len;
    Some(value)
}

fn skip_gguf_value(data: &[u8], offset: &mut usize, value_type: u32) -> Option<()> {
    match value_type {
        0 | 1 => *offset += 1,
        2 | 3 => *offset += 2,
        4..=6 => *offset += 4,
        7 => *offset += 1,
        8 => {
            read_gguf_string(data, offset)?;
        }
        9 => {
            if *offset + 12 > data.len() {
                return None;
            }
            let array_type = u32::from_le_bytes(data[*offset..*offset + 4].try_into().ok()?);
            *offset += 4;
            let count = u64::from_le_bytes(data[*offset..*offset + 8].try_into().ok()?) as usize;
            *offset += 8;
            for _ in 0..count {
                skip_gguf_value(data, offset, array_type)?;
            }
        }
        10 | 11 => *offset += 8,
        12 => *offset += 8,
        _ => return None,
    }
    Some(())
}

fn inspect_gguf_metadata(path: &Path) -> Option<GgufMetadata> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 24 {
        return None;
    }

    let magic = u32::from_le_bytes(data[0..4].try_into().ok()?);
    if magic != GGUF_MAGIC {
        return None;
    }

    let kv_count = u64::from_le_bytes(data[16..24].try_into().ok()?) as usize;
    let mut offset = 24usize;
    let mut metadata = GgufMetadata::default();

    for _ in 0..kv_count {
        let key = read_gguf_string(&data, &mut offset)?;
        if offset + 4 > data.len() {
            return None;
        }
        let value_type = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;

        if value_type == GGUF_METADATA_STRING {
            let value = read_gguf_string(&data, &mut offset)?;
            match key.as_str() {
                "general.architecture" => metadata.architecture = Some(value),
                "general.type" => metadata.file_type = Some(value),
                _ => {}
            }
        } else {
            skip_gguf_value(&data, &mut offset, value_type)?;
        }
    }

    Some(metadata)
}

/// Return true when the file looks like a standalone llama.cpp-loadable GGUF model.
fn is_llama_cpp_loadable_gguf(path: &Path) -> bool {
    if !is_discoverable_gguf_name(path) {
        return false;
    }

    match inspect_gguf_metadata(path) {
        Some(metadata) if metadata.is_llama_cpp_loadable() => true,
        Some(_) => false,
        None => {
            tracing::debug!(
                path = %path.display(),
                "Skipping file that is not a valid llama.cpp-loadable GGUF model"
            );
            false
        }
    }
}

/// Resolve every directory listed in `models_dir` (comma-separated).
/// Missing directories are skipped with a warning.
pub fn resolve_models_dirs(configured: &str) -> Result<Vec<PathBuf>, RuntimeError> {
    let dirs = parse_models_dirs(configured);
    if dirs.is_empty() {
        return Err(RuntimeError::ConfigError(
            "models_dir is empty; set defaults.models_dir in models.toml".to_string(),
        ));
    }

    let mut resolved = Vec::new();
    let mut missing = Vec::new();
    for dir in dirs {
        if dir.is_dir() {
            resolved.push(dir);
        } else {
            missing.push(dir.display().to_string());
        }
    }

    for path in &missing {
        tracing::warn!(
            configured = %configured,
            missing = %path,
            "Skipping models_dir entry because the directory does not exist"
        );
    }

    if resolved.is_empty() {
        return Err(RuntimeError::ConfigError(format!(
            "Models directory does not exist: {}",
            missing.join(", ")
        )));
    }

    Ok(resolved)
}

fn is_embedding_like_alias(alias: &str) -> bool {
    let lower = alias.to_ascii_lowercase();
    lower.contains("embed") || lower.contains("granite-embedding")
}

fn priority_preference_score(alias: &str) -> u8 {
    if is_embedding_like_alias(alias) {
        return 0;
    }
    let lower = alias.to_ascii_lowercase();
    if lower.contains("gemma-4-e4b") {
        return 10;
    }
    if lower.contains("gemma") || lower.contains("qwen") || lower.contains("llama") {
        return 5;
    }
    1
}

fn default_priority_index(models: &[RegistryEntry]) -> Option<usize> {
    models
        .iter()
        .enumerate()
        .filter(|(_, entry)| priority_preference_score(&entry.alias) > 0)
        .max_by_key(|(_, entry)| priority_preference_score(&entry.alias))
        .map(|(index, _)| index)
}

fn assign_default_priority(
    registry: &mut ModelsRegistry,
    models_dirs: &[PathBuf],
    merge_from: Option<&ModelsRegistry>,
) {
    if registry.models.iter().any(|entry| entry.priority) {
        return;
    }

    if let Some(existing) = merge_from
        && let Some(prev_priority) = existing.models.iter().find(|entry| entry.priority)
    {
        let prev_key = normalize_file_key(models_dirs, &prev_priority.file);
        if let Some(entry) = registry
            .models
            .iter_mut()
            .find(|entry| normalize_file_key(models_dirs, &entry.file) == prev_key)
        {
            entry.priority = true;
            return;
        }
    }

    if let Some(index) = default_priority_index(&registry.models) {
        registry.models[index].priority = true;
        return;
    }

    if let Some(entry) = registry.models.first_mut() {
        entry.priority = true;
    }
}

fn resolve_model_path(models_dirs: &[PathBuf], file: &str) -> Result<String, RuntimeError> {
    let path = Path::new(file);
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else if models_dirs.len() == 1 {
        models_dirs[0].join(file)
    } else {
        models_dirs
            .iter()
            .map(|dir| dir.join(file))
            .find(|candidate| candidate.exists())
            .ok_or_else(|| {
                RuntimeError::ConfigError(format!(
                    "GGUF file not found for alias: '{file}' (searched {} models_dir entries)",
                    models_dirs.len()
                ))
            })?
    };

    let canonical = resolved.to_string_lossy().into_owned();
    if !Path::new(&canonical).exists() {
        return Err(RuntimeError::ConfigError(format!(
            "GGUF file not found for alias: '{canonical}'"
        )));
    }
    Ok(canonical)
}

fn build_existing_file_map(
    existing: &ModelsRegistry,
    models_dirs: &[PathBuf],
) -> HashMap<String, RegistryEntry> {
    existing
        .models
        .iter()
        .map(|entry| (normalize_file_key(models_dirs, &entry.file), entry.clone()))
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

    /// Scan configured model directories for `.gguf` files and build a registry with generated aliases.
    pub fn discover(dirs: &str) -> Result<Self, RuntimeError> {
        Self::discover_with_merge(dirs, None)
    }

    /// Scan configured model directories for `.gguf` files, optionally merging metadata from an existing registry.
    ///
    /// `dirs` may be a single path or comma-separated list (e.g. `"/models,/data/gguf"`).
    /// When merging, entries are matched by normalized `file` path. Existing
    /// `alias`, `display_name`, `priority`, `port`, and `context_size` are preserved.
    /// If no `.gguf` files are found but `merge_from` is set, the existing registry
    /// is returned with an updated `defaults.models_dir`.
    pub fn discover_with_merge(
        dirs: &str,
        merge_from: Option<&Self>,
    ) -> Result<Self, RuntimeError> {
        let models_dirs = resolve_models_dirs(dirs)?;

        let mut files = discover_gguf_files(&models_dirs)?;
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
            defaults: {
                let mut defaults = base_defaults;
                defaults.models_dir = dirs.trim().to_string();
                defaults.llama_server = llama_server;
                defaults
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
                "No llama.cpp-loadable .gguf files found in '{dirs}'"
            )));
        }

        let existing_by_file = merge_from
            .map(|existing| build_existing_file_map(existing, &models_dirs))
            .unwrap_or_default();

        let mut used_aliases = HashSet::new();

        for path in files {
            let file = model_file_reference(&models_dirs, &path);
            let file_key = normalize_file_key(&models_dirs, &file);

            if let Some(existing) = existing_by_file.get(&file_key) {
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

        assign_default_priority(&mut registry, &models_dirs, merge_from);

        Ok(registry)
    }

    /// Expand registry entries into full `ModelConfig` map keyed by alias.
    pub fn expand(
        &self,
        fallback_backend: &str,
    ) -> Result<HashMap<String, ModelConfig>, RuntimeError> {
        let models_dirs = resolve_models_dirs(&self.defaults.models_dir)?;

        let mut entries = Vec::new();
        let mut claimed_files = HashSet::new();

        for entry in &self.models {
            match resolve_model_path(&models_dirs, &entry.file) {
                Ok(path) => {
                    if !is_llama_cpp_loadable_gguf(Path::new(&path)) {
                        tracing::warn!(
                            alias = %entry.alias,
                            file = %entry.file,
                            "Skipping explicit model entry because the GGUF file is not llama.cpp-loadable"
                        );
                        continue;
                    }
                    claimed_files.insert(normalize_file_key(&models_dirs, &entry.file));
                    entries.push(entry.clone());
                }
                Err(err) => {
                    tracing::warn!(
                        alias = %entry.alias,
                        file = %entry.file,
                        error = %err,
                        "Skipping explicit model entry because the GGUF file was not found"
                    );
                }
            }
        }

        if self.auto_discover {
            let mut discovered = discover_gguf_files(&models_dirs)?;
            discovered.sort();

            let mut used_aliases: HashSet<String> =
                entries.iter().map(|e| e.alias.clone()).collect();

            for path in discovered {
                let file = model_file_reference(&models_dirs, &path);
                let file_key = normalize_file_key(&models_dirs, &file);
                if claimed_files.contains(&file_key) {
                    continue;
                }
                claimed_files.insert(file_key);

                let mut alias = alias_from_filename(&path);
                alias = dedupe_alias(&alias, &mut used_aliases);

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
            let model_path = resolve_model_path(&models_dirs, &entry.file)?;
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

fn discover_gguf_files(dirs: &[PathBuf]) -> Result<Vec<PathBuf>, RuntimeError> {
    let mut files = Vec::new();
    for dir in dirs {
        collect_gguf_files(dir, &mut files)?;
    }
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
            && is_llama_cpp_loadable_gguf(&path)
        {
            out.push(path);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_gguf_string(buf: &mut Vec<u8>, value: &str) {
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value.as_bytes());
    }

    fn write_minimal_gguf(path: &Path, architecture: &str) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        write_gguf_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&GGUF_METADATA_STRING.to_le_bytes());
        write_gguf_string(&mut buf, architecture);
        std::fs::write(path, buf).unwrap();
    }

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
        write_minimal_gguf(&dir.join("gemma-3-4b.gguf"), "gemma");
        write_minimal_gguf(&dir.join("qwen2.5-coder-7b.gguf"), "qwen2");

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
        write_minimal_gguf(&model_path, "llama");

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

    #[test]
    fn discover_skips_mmproj_and_vocab_artifacts() {
        let dir = std::env::temp_dir().join("gguf-switchboard-skip-artifacts-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_minimal_gguf(&dir.join("Qwen3.5-9B-Q4_K_M.gguf"), "qwen2");
        std::fs::write(dir.join("mmproj-Qwen3.5-9B.gguf"), b"fake").unwrap();
        std::fs::write(dir.join("ggml-vocab-qwen3.gguf"), b"fake").unwrap();
        std::fs::write(dir.join("Gemma-4-12B-mmproj-bf16.gguf"), b"fake").unwrap();

        let registry = ModelsRegistry::discover(dir.to_str().unwrap()).unwrap();
        assert_eq!(registry.models.len(), 1);
        assert_eq!(registry.models[0].alias, "qwen3.5-9b");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_registers_nested_relative_paths() {
        let dir = std::env::temp_dir().join("gguf-switchboard-nested-test");
        let nested = dir.join("lmstudio-community/Qwen3.5-9B-GGUF");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&nested).unwrap();
        write_minimal_gguf(&nested.join("Qwen3.5-9B-Q4_K_M.gguf"), "qwen2");

        let registry = ModelsRegistry::discover(dir.to_str().unwrap()).unwrap();
        assert_eq!(registry.models.len(), 1);
        assert_eq!(
            registry.models[0].file,
            "lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expand_errors_when_configured_models_dir_is_missing() {
        let registry = ModelsRegistry {
            defaults: RegistryDefaults {
                models_dir: "/nonexistent-models-root".to_string(),
                ..RegistryDefaults::default()
            },
            auto_discover: true,
            models: Vec::new(),
        };

        let err = registry.expand("llama.cpp").unwrap_err().to_string();
        assert!(err.contains("does not exist"));
    }

    #[test]
    fn expand_skips_missing_explicit_entries_and_auto_discovers() {
        let dir = std::env::temp_dir().join("gguf-switchboard-skip-missing-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_minimal_gguf(&dir.join("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), "llama");

        let registry = ModelsRegistry {
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                ..RegistryDefaults::default()
            },
            auto_discover: true,
            models: vec![RegistryEntry {
                alias: "llama-3".to_string(),
                file: "llama-3.2-3b.gguf".to_string(),
                display_name: Some("Llama 3.2 3B".to_string()),
                priority: true,
                port: None,
                context_size: None,
            }],
        };

        let models = registry.expand("llama.cpp").unwrap();
        assert_eq!(models.len(), 1);
        assert!(models.contains_key("meta-llama-3.1-8b"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_models_dirs_skips_missing_entries() {
        let home = std::env::temp_dir().join("gguf-switchboard-partial-models-dir");
        let existing = home.join("exists");
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&existing).unwrap();

        let configured = format!("/missing-models-dir,{}", existing.display());
        let resolved = resolve_models_dirs(&configured).unwrap();
        assert_eq!(resolved, vec![existing]);

        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn resolve_models_dirs_requires_existing_paths() {
        let home = std::env::temp_dir().join("gguf-switchboard-configured-dir");
        let configured = home.join("configured");
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&configured).unwrap();

        let resolved = resolve_models_dirs(configured.to_str().unwrap()).unwrap();
        assert_eq!(resolved, vec![configured]);

        let err = resolve_models_dirs("/definitely-missing-models-dir").unwrap_err();
        assert!(err.to_string().contains("does not exist"));

        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn discover_loads_from_comma_separated_models_dirs() {
        let root = std::env::temp_dir().join("gguf-switchboard-multi-dir-test");
        let first = root.join("first");
        let second = root.join("second");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        write_minimal_gguf(&first.join("alpha.gguf"), "llama");
        write_minimal_gguf(&second.join("beta.gguf"), "llama");

        let configured = format!("{},{}", first.display(), second.display());
        let registry = ModelsRegistry::discover(&configured).unwrap();
        assert_eq!(registry.models.len(), 2);
        assert!(registry.models.iter().any(|entry| entry.alias == "alpha"));
        assert!(registry.models.iter().any(|entry| entry.alias == "beta"));
        assert!(
            registry
                .models
                .iter()
                .all(|entry| Path::new(&entry.file).is_absolute())
        );

        let expanded = registry.expand("llama.cpp").unwrap();
        assert_eq!(expanded.len(), 2);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn expand_auto_discover_adds_unlisted_nested_models() {
        let dir = std::env::temp_dir().join("gguf-switchboard-expand-nested-test");
        let nested = dir.join("nested");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&nested).unwrap();
        write_minimal_gguf(&dir.join("listed.gguf"), "llama");
        write_minimal_gguf(&nested.join("extra-model.gguf"), "llama");

        let registry = ModelsRegistry {
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                ..RegistryDefaults::default()
            },
            auto_discover: true,
            models: vec![RegistryEntry {
                alias: "listed".to_string(),
                file: "listed.gguf".to_string(),
                display_name: Some("Listed".to_string()),
                priority: true,
                port: None,
                context_size: None,
            }],
        };

        let models = registry.expand("llama.cpp").unwrap();
        assert_eq!(models.len(), 2);
        assert!(models.contains_key("listed"));
        assert!(models.contains_key("extra-model"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
