use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::ModelConfig;
use crate::errors::RuntimeError;

fn default_registry_version() -> u32 {
    1
}

/// Simplified model registry — short aliases instead of full GGUF paths in API requests.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsRegistry {
    #[serde(default = "default_registry_version")]
    pub version: u32,
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

fn default_enabled() -> bool {
    true
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
    /// Model role: `chat`, `coder`, `vision`, or `embedding`. Inferred from alias/file when omitted.
    #[serde(default)]
    pub kind: Option<String>,
    /// When false, the model is omitted from scheduling and `/v1/models`.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub priority: bool,
    /// Override the auto-assigned backend port.
    #[serde(default)]
    pub port: Option<u16>,
    /// Override `defaults.context_size` for this model.
    #[serde(default)]
    pub context_size: Option<u32>,
    /// Override `defaults.ngl` for this model (also pins against auto_ngl).
    #[serde(default)]
    pub ngl: Option<u32>,
    /// Extra `llama-server` flags appended after the default args (e.g. `--jinja`).
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Short description for `/v1/models` / Swagger (often filled by `sync-hf-metadata`).
    #[serde(default)]
    pub description: Option<String>,
    /// Model max context from GGUF/HF metadata (distinct from serving `context_size`).
    #[serde(default)]
    pub max_context_length: Option<u32>,
    /// Approximate minimum VRAM in GB (weights floor; filled by HF sync).
    #[serde(default)]
    pub min_vram_gb: Option<u32>,
    /// Capability tags (e.g. `tools`, `vision`, `reasoning`).
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Matched Hugging Face repo id (e.g. `lmstudio-community/Qwen3.5-9B-GGUF`).
    #[serde(default)]
    pub hf_repo: Option<String>,
}

impl Default for RegistryEntry {
    fn default() -> Self {
        Self {
            alias: String::new(),
            file: String::new(),
            display_name: None,
            kind: None,
            enabled: true,
            priority: false,
            port: None,
            context_size: None,
            ngl: None,
            extra_args: Vec::new(),
            description: None,
            max_context_length: None,
            min_vram_gb: None,
            capabilities: Vec::new(),
            hf_repo: None,
        }
    }
}

impl RegistryEntry {
    pub fn effective_kind(&self) -> String {
        self.kind
            .clone()
            .unwrap_or_else(|| infer_kind(&self.alias, &self.file))
    }
}

/// Portable JSON export shared across tools (Open WebUI, scripts, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsJsonExport {
    pub version: u32,
    pub models_dir: String,
    pub models: Vec<ModelsJsonEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsJsonEntry {
    pub id: String,
    pub file: String,
    pub display_name: String,
    pub kind: String,
    pub enabled: bool,
    pub priority: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_vram_gb: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
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
    16384
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

/// Join model directories into the comma-separated `models_dir` form.
pub fn format_models_dirs(dirs: &[PathBuf]) -> String {
    dirs.iter()
        .map(|d| d.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(",")
}

fn dir_identity_key(dir: &Path) -> String {
    dir.canonicalize()
        .unwrap_or_else(|_| dir.to_path_buf())
        .to_string_lossy()
        .to_ascii_lowercase()
}

/// Well-known locations scanned after configured `models_dir` entries.
pub fn common_models_dir_candidates() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        dirs.push(home.join("models"));
        dirs.push(home.join(".lmstudio").join("models"));
    }
    dirs.push(PathBuf::from("/models"));
    dirs.push(PathBuf::from("/var/lib/gguf-switchboard/models"));
    dirs
}

/// Resolve scan roots: existing configured dirs first (even if empty), then common
/// dirs that contain at least one llama.cpp-loadable GGUF. Dedupes by canonical path.
pub fn resolve_models_dirs_with_fallback(configured: &str) -> Result<Vec<PathBuf>, RuntimeError> {
    let mut resolved = Vec::new();
    let mut seen = HashSet::new();

    for dir in parse_models_dirs(configured) {
        if !dir.is_dir() {
            tracing::warn!(
                path = %dir.display(),
                "Skipping configured models_dir entry because the directory does not exist"
            );
            continue;
        }
        let key = dir_identity_key(&dir);
        if seen.insert(key) {
            resolved.push(dir);
        }
    }

    for candidate in common_models_dir_candidates() {
        if !candidate.is_dir() {
            continue;
        }
        let key = dir_identity_key(&candidate);
        if seen.contains(&key) {
            continue;
        }
        match discover_gguf_files(std::slice::from_ref(&candidate)) {
            Ok(files) if !files.is_empty() => {
                seen.insert(key);
                resolved.push(candidate);
            }
            Ok(_) => {}
            Err(err) => {
                tracing::debug!(
                    path = %candidate.display(),
                    error = %err,
                    "Skipping common models_dir candidate"
                );
            }
        }
    }

    if resolved.is_empty() {
        return Err(RuntimeError::ConfigError(
            "No model directories found; set defaults.models_dir, MODELS_DIR, or place GGUFs under a common path (e.g. ~/models)".to_string(),
        ));
    }

    Ok(resolved)
}

/// Result of a disk rescan that is ready to persist and hot-swap.
#[derive(Debug, Clone)]
pub struct RescanResult {
    pub registry: ModelsRegistry,
    pub models: HashMap<String, ModelConfig>,
    pub registry_json: String,
    pub added: usize,
    pub removed: usize,
    pub total: usize,
    pub models_dir: String,
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
        || lower.contains("-projector")
        || lower.contains("projector")
        || lower.contains("-adapter")
        || lower.contains("adapter")
        || lower.contains("tokenizer")
        || lower.ends_with("-vocab.gguf"))
}

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_METADATA_STRING: u32 = 8;
const GGUF_METADATA_UINT32: u32 = 4;
const GGUF_METADATA_INT32: u32 = 5;
const GGUF_METADATA_UINT64: u32 = 10;
const GGUF_METADATA_INT64: u32 = 11;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GgufSkipReason {
    BadName,
    BadMagic,
    BadVersion,
    NoTensors,
    BadArch,
    BadType,
    ZeroBlockCount,
    Unreadable,
}

impl std::fmt::Display for GgufSkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadName => write!(f, "filename looks like a sidecar or adapter"),
            Self::BadMagic => write!(f, "missing GGUF magic"),
            Self::BadVersion => write!(f, "unsupported GGUF version (need 2 or 3)"),
            Self::NoTensors => write!(f, "tensor_count is 0"),
            Self::BadArch => write!(f, "architecture is not a standalone llama.cpp model"),
            Self::BadType => write!(f, "general.type is lora/vocab (not a full model)"),
            Self::ZeroBlockCount => write!(f, "architecture block_count is 0"),
            Self::Unreadable => write!(f, "could not read GGUF metadata prefix"),
        }
    }
}

#[derive(Debug, Default)]
struct GgufMetadata {
    version: u32,
    tensor_count: u64,
    architecture: Option<String>,
    file_type: Option<String>,
    block_count: Option<u64>,
}

impl GgufMetadata {
    fn loadable_skip_reason(&self) -> Option<GgufSkipReason> {
        if !(self.version == 2 || self.version == 3) {
            return Some(GgufSkipReason::BadVersion);
        }
        if self.tensor_count == 0 {
            return Some(GgufSkipReason::NoTensors);
        }
        let Some(arch) = self.architecture.as_ref() else {
            return Some(GgufSkipReason::BadArch);
        };
        if arch.is_empty() {
            return Some(GgufSkipReason::BadArch);
        }

        let arch_lower = arch.to_ascii_lowercase();
        if matches!(
            arch_lower.as_str(),
            "clip" | "siglip" | "vit" | "wav2vec2" | "whisper" | "encoder"
        ) {
            return Some(GgufSkipReason::BadArch);
        }

        if let Some(file_type) = &self.file_type {
            let file_type_lower = file_type.to_ascii_lowercase();
            if matches!(file_type_lower.as_str(), "lora" | "vocab") {
                return Some(GgufSkipReason::BadType);
            }
        }

        if self.block_count == Some(0) {
            return Some(GgufSkipReason::ZeroBlockCount);
        }

        None
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

fn read_gguf_u64(data: &[u8], offset: &mut usize, value_type: u32) -> Option<u64> {
    match value_type {
        GGUF_METADATA_UINT32 | GGUF_METADATA_INT32 => {
            if *offset + 4 > data.len() {
                return None;
            }
            let value = u32::from_le_bytes(data[*offset..*offset + 4].try_into().ok()?) as u64;
            *offset += 4;
            Some(value)
        }
        GGUF_METADATA_UINT64 | GGUF_METADATA_INT64 => {
            if *offset + 8 > data.len() {
                return None;
            }
            let value = u64::from_le_bytes(data[*offset..*offset + 8].try_into().ok()?);
            *offset += 8;
            Some(value)
        }
        _ => None,
    }
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

fn inspect_gguf_metadata(path: &Path) -> Result<GgufMetadata, GgufSkipReason> {
    use std::io::Read;

    // Metadata is a prefix before tensor weights. Never load multi-GB GGUFs into RAM.
    // ponytail: 32 MiB ceiling — raise only if a model ships absurdly large KV headers.
    const MAX_METADATA_BYTES: u64 = 32 * 1024 * 1024;

    let mut file = std::fs::File::open(path).map_err(|_| GgufSkipReason::Unreadable)?;
    let file_len = file
        .metadata()
        .map_err(|_| GgufSkipReason::Unreadable)?
        .len();
    let to_read = file_len.min(MAX_METADATA_BYTES) as usize;
    let mut data = vec![0u8; to_read];
    file.read_exact(&mut data)
        .map_err(|_| GgufSkipReason::Unreadable)?;
    if data.len() < 24 {
        return Err(GgufSkipReason::Unreadable);
    }

    let magic = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| GgufSkipReason::Unreadable)?,
    );
    if magic != GGUF_MAGIC {
        return Err(GgufSkipReason::BadMagic);
    }

    let version = u32::from_le_bytes(
        data[4..8]
            .try_into()
            .map_err(|_| GgufSkipReason::Unreadable)?,
    );
    let tensor_count = u64::from_le_bytes(
        data[8..16]
            .try_into()
            .map_err(|_| GgufSkipReason::Unreadable)?,
    );
    let kv_count = u64::from_le_bytes(
        data[16..24]
            .try_into()
            .map_err(|_| GgufSkipReason::Unreadable)?,
    ) as usize;
    let mut offset = 24usize;
    let mut metadata = GgufMetadata {
        version,
        tensor_count,
        ..GgufMetadata::default()
    };

    for _ in 0..kv_count {
        let key = read_gguf_string(&data, &mut offset).ok_or(GgufSkipReason::Unreadable)?;
        if offset + 4 > data.len() {
            return Err(GgufSkipReason::Unreadable);
        }
        let value_type = u32::from_le_bytes(
            data[offset..offset + 4]
                .try_into()
                .map_err(|_| GgufSkipReason::Unreadable)?,
        );
        offset += 4;

        if value_type == GGUF_METADATA_STRING {
            let value = read_gguf_string(&data, &mut offset).ok_or(GgufSkipReason::Unreadable)?;
            match key.as_str() {
                "general.architecture" => metadata.architecture = Some(value),
                "general.type" => metadata.file_type = Some(value),
                _ => {}
            }
        } else if key.ends_with(".block_count")
            && matches!(
                value_type,
                GGUF_METADATA_UINT32
                    | GGUF_METADATA_INT32
                    | GGUF_METADATA_UINT64
                    | GGUF_METADATA_INT64
            )
        {
            metadata.block_count = Some(
                read_gguf_u64(&data, &mut offset, value_type).ok_or(GgufSkipReason::Unreadable)?,
            );
        } else {
            skip_gguf_value(&data, &mut offset, value_type).ok_or(GgufSkipReason::Unreadable)?;
        }
    }

    Ok(metadata)
}

/// Cheap prefix-only validation: filename → header → architecture/type metadata.
fn validate_gguf_model(path: &Path) -> Result<GgufMetadata, GgufSkipReason> {
    if !is_discoverable_gguf_name(path) {
        return Err(GgufSkipReason::BadName);
    }
    let metadata = inspect_gguf_metadata(path)?;
    if let Some(reason) = metadata.loadable_skip_reason() {
        return Err(reason);
    }
    Ok(metadata)
}

/// Return true when the file looks like a standalone llama.cpp-loadable GGUF model.
fn is_llama_cpp_loadable_gguf(path: &Path) -> bool {
    match validate_gguf_model(path) {
        Ok(_) => true,
        Err(reason) => {
            tracing::debug!(
                path = %path.display(),
                reason = %reason,
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

fn infer_kind(alias: &str, file: &str) -> String {
    let combined = format!("{alias} {file}").to_ascii_lowercase();
    if combined.contains("embed") {
        "embedding".to_string()
    } else if combined.contains("-vl") || combined.contains("vision") || combined.contains("mmproj")
    {
        "vision".to_string()
    } else if combined.contains("coder") || combined.contains("-code") {
        "coder".to_string()
    } else {
        "chat".to_string()
    }
}

fn dedupe_registry_entries(entries: &mut Vec<RegistryEntry>) {
    let mut by_file = HashMap::new();
    for entry in entries.drain(..) {
        let key = entry.file.clone();
        by_file
            .entry(key)
            .and_modify(|existing: &mut RegistryEntry| {
                merge_registry_entry(existing, &entry);
            })
            .or_insert(entry);
    }

    let mut by_alias = HashMap::new();
    for entry in by_file.into_values() {
        let key = entry.alias.clone();
        by_alias
            .entry(key)
            .and_modify(|existing: &mut RegistryEntry| {
                merge_registry_entry(existing, &entry);
            })
            .or_insert(entry);
    }

    let mut deduped: Vec<RegistryEntry> = by_alias.into_values().collect();
    deduped.sort_by(|a, b| a.alias.cmp(&b.alias));
    *entries = deduped;
}

fn merge_registry_entry(target: &mut RegistryEntry, incoming: &RegistryEntry) {
    if target.display_name.is_none() {
        target.display_name = incoming.display_name.clone();
    }
    if target.kind.is_none() {
        target.kind = incoming.kind.clone();
    }
    if target.port.is_none() {
        target.port = incoming.port;
    }
    if target.context_size.is_none() {
        target.context_size = incoming.context_size;
    }
    if target.ngl.is_none() {
        target.ngl = incoming.ngl;
    }
    if incoming.priority {
        target.priority = true;
    }
    if !incoming.enabled {
        target.enabled = false;
    }
    if target.extra_args.is_empty() {
        target.extra_args = incoming.extra_args.clone();
    }
    if target.description.is_none() {
        target.description = incoming.description.clone();
    }
    if target.max_context_length.is_none() {
        target.max_context_length = incoming.max_context_length;
    }
    if target.min_vram_gb.is_none() {
        target.min_vram_gb = incoming.min_vram_gb;
    }
    if target.capabilities.is_empty() {
        target.capabilities = incoming.capabilities.clone();
    }
    if target.hf_repo.is_none() {
        target.hf_repo = incoming.hf_repo.clone();
    }
}

fn normalize_priority_entries(entries: &mut [RegistryEntry]) {
    let priority_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.priority)
        .map(|(index, _)| index)
        .collect();

    if priority_indices.len() <= 1 {
        return;
    }

    tracing::warn!(
        count = priority_indices.len(),
        kept = priority_indices[0],
        "Multiple priority models configured; keeping only the first"
    );

    for (index, entry) in entries.iter_mut().enumerate() {
        if index != priority_indices[0] {
            entry.priority = false;
        }
    }
}

fn file_size_gb(path: &str) -> Option<f64> {
    std::fs::metadata(path)
        .ok()
        .map(|meta| meta.len() as f64 / 1_073_741_824.0)
}

/// Suggest a context window (`-c`) from available VRAM and model file size.
pub fn suggest_context_size(
    vram_gb: u32,
    entry: &RegistryEntry,
    model_path: &str,
    default_context: u32,
) -> u32 {
    if let Some(context_size) = entry.context_size {
        return context_size;
    }

    let kind = entry.effective_kind();
    if kind == "embedding" {
        return 8192.min(default_context);
    }

    let size_gb = file_size_gb(model_path).unwrap_or(5.0);
    let vram = f64::from(vram_gb.max(1));

    if size_gb >= 12.0 || entry.alias.contains("30b") || entry.alias.contains("70b") {
        return if vram <= 12.0 { 16384 } else { 32768 };
    }
    if size_gb >= 8.0 {
        return if vram <= 12.0 { 16384 } else { 32768 };
    }
    if vram <= 8.0 {
        return 16384;
    }
    if vram <= 12.0 {
        return 32768;
    }
    default_context.min(65536)
}

fn json_sibling_path(toml_path: &str) -> String {
    if let Some(idx) = toml_path.rfind(".toml") {
        format!("{}json{}", &toml_path[..idx], &toml_path[idx + 5..])
    } else {
        format!("{toml_path}.json")
    }
}

fn tags_for_entry(entry: &RegistryEntry) -> Vec<String> {
    let mut tags = vec![entry.effective_kind()];
    if entry.priority {
        tags.push("priority".to_string());
    }
    if !entry.enabled {
        tags.push("disabled".to_string());
    }
    tags
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

pub(crate) fn resolve_model_path(
    models_dirs: &[PathBuf],
    file: &str,
) -> Result<String, RuntimeError> {
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
        if path.ends_with(".json") {
            return Self::load_json(path);
        }
        Self::load_toml(path)
    }

    fn load_toml(path: &str) -> Result<Self, RuntimeError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to read models file '{path}': {e}"))
        })?;
        let mut registry: ModelsRegistry = toml::from_str(&content).map_err(RuntimeError::from)?;
        dedupe_registry_entries(&mut registry.models);
        normalize_priority_entries(&mut registry.models);
        Ok(registry)
    }

    fn load_json(path: &str) -> Result<Self, RuntimeError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to read models file '{path}': {e}"))
        })?;
        let export: ModelsJsonExport =
            serde_json::from_str(&content).map_err(RuntimeError::from)?;
        Self::from_json_export(export)
    }

    pub fn from_json_export(export: ModelsJsonExport) -> Result<Self, RuntimeError> {
        let mut registry = ModelsRegistry {
            version: export.version,
            defaults: RegistryDefaults {
                models_dir: export.models_dir,
                ..RegistryDefaults::default()
            },
            auto_discover: false,
            models: export
                .models
                .into_iter()
                .map(|entry| RegistryEntry {
                    alias: entry.id,
                    file: entry.file,
                    display_name: Some(entry.display_name),
                    kind: Some(entry.kind),
                    enabled: entry.enabled,
                    priority: entry.priority,
                    port: None,
                    context_size: entry.context_size,
                    ngl: None,
                    extra_args: Vec::new(),
                    description: entry.description,
                    max_context_length: entry.max_context_length,
                    min_vram_gb: entry.min_vram_gb,
                    capabilities: entry.capabilities,
                    hf_repo: entry.hf_repo,
                })
                .collect(),
        };
        dedupe_registry_entries(&mut registry.models);
        normalize_priority_entries(&mut registry.models);
        Ok(registry)
    }

    pub fn to_json_export(&self) -> ModelsJsonExport {
        ModelsJsonExport {
            version: self.version,
            models_dir: self.defaults.models_dir.clone(),
            models: self
                .models
                .iter()
                .map(|entry| ModelsJsonEntry {
                    id: entry.alias.clone(),
                    file: entry.file.clone(),
                    display_name: entry
                        .display_name
                        .clone()
                        .unwrap_or_else(|| display_name_from_alias(&entry.alias)),
                    kind: entry.effective_kind(),
                    enabled: entry.enabled,
                    priority: entry.priority,
                    context_size: entry.context_size,
                    description: entry.description.clone(),
                    max_context_length: entry.max_context_length,
                    min_vram_gb: entry.min_vram_gb,
                    capabilities: entry.capabilities.clone(),
                    hf_repo: entry.hf_repo.clone(),
                    tags: tags_for_entry(entry),
                })
                .collect(),
        }
    }

    pub fn write(&self, path: &str) -> Result<(), RuntimeError> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to serialize models registry: {e}"))
        })?;
        std::fs::write(path, content).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to write models file '{path}': {e}"))
        })?;
        self.write_json(&json_sibling_path(path))?;
        Ok(())
    }

    pub fn write_json(&self, path: &str) -> Result<(), RuntimeError> {
        let content = serde_json::to_string_pretty(&self.to_json_export()).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to serialize models JSON: {e}"))
        })?;
        std::fs::write(path, content).map_err(|e| {
            RuntimeError::ConfigError(format!("Failed to write models JSON '{path}': {e}"))
        })?;
        Ok(())
    }

    /// Scan configured model directories for `.gguf` files and build a registry with generated aliases.
    #[allow(dead_code)] // thin wrapper; used by tests and as the no-merge discover entry point
    pub fn discover(dirs: &str) -> Result<Self, RuntimeError> {
        Self::discover_with_merge(dirs, None)
    }

    /// Scan configured model directories for `.gguf` files, optionally merging metadata from an existing registry.
    ///
    /// `dirs` may be a single path or comma-separated list (e.g. `"/models,/data/gguf"`).
    /// When merging, entries are matched by normalized `file` path. Existing
    /// `alias`, `display_name`, `priority`, `port`, and `context_size` are preserved.
    /// Entries whose files are gone are dropped (including when the scan finds zero GGUFs).
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
                defaults.models_dir = format_models_dirs(&models_dirs);
                defaults.llama_server = llama_server;
                defaults
            },
            version: merge_from.map(|e| e.version).unwrap_or(1),
            auto_discover: merge_from.map(|e| e.auto_discover).unwrap_or(true),
            models: Vec::new(),
        };

        if files.is_empty() {
            if merge_from.is_some() {
                // Drop missing: keep defaults/settings, clear models whose files are gone.
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
                    kind: existing.kind.clone(),
                    enabled: existing.enabled,
                    priority: existing.priority,
                    port: existing.port,
                    context_size: existing.context_size,
                    ngl: existing.ngl,
                    extra_args: existing.extra_args.clone(),
                    description: existing.description.clone(),
                    max_context_length: existing.max_context_length,
                    min_vram_gb: existing.min_vram_gb,
                    capabilities: existing.capabilities.clone(),
                    hf_repo: existing.hf_repo.clone(),
                });
                continue;
            }

            let mut alias = alias_from_filename(&path);
            alias = dedupe_alias(&alias, &mut used_aliases);
            let kind = infer_kind(&alias, &file);

            registry.models.push(RegistryEntry {
                alias: alias.clone(),
                file,
                display_name: Some(display_name_from_alias(&alias)),
                kind: Some(kind),
                enabled: true,
                priority: false,
                port: None,
                context_size: None,
                ngl: None,
                extra_args: Vec::new(),
                ..Default::default()
            });
        }

        dedupe_registry_entries(&mut registry.models);
        assign_default_priority(&mut registry, &models_dirs, merge_from);
        normalize_priority_entries(&mut registry.models);

        Ok(registry)
    }

    /// Resolve dirs (config/`MODELS_DIR` first, then common), discover, drop missing, merge, expand.
    ///
    /// When `override_dirs` is set, only those directories are scanned (no common-dir fallback).
    /// When unset, uses `merge_from.defaults.models_dir`, then `MODELS_DIR`, then common dirs.
    pub fn rescan(
        override_dirs: Option<&str>,
        merge_from: Option<&Self>,
        fallback_backend: &str,
        vram_gb: u32,
    ) -> Result<RescanResult, RuntimeError> {
        let previous_aliases: HashSet<String> = merge_from
            .map(|existing| existing.models.iter().map(|e| e.alias.clone()).collect())
            .unwrap_or_default();

        let dirs_str = if let Some(dirs) = override_dirs {
            let resolved = resolve_models_dirs(dirs)?;
            format_models_dirs(&resolved)
        } else {
            let mut configured = merge_from
                .map(|m| m.defaults.models_dir.clone())
                .unwrap_or_default();
            if let Ok(env_dirs) = std::env::var("MODELS_DIR") {
                let env_dirs = env_dirs.trim();
                if !env_dirs.is_empty() {
                    configured = if configured.is_empty() {
                        env_dirs.to_string()
                    } else {
                        format!("{env_dirs},{configured}")
                    };
                }
            }
            let resolved = resolve_models_dirs_with_fallback(&configured)?;
            format_models_dirs(&resolved)
        };

        let registry = Self::discover_with_merge(&dirs_str, merge_from)?;
        let models = registry.expand(fallback_backend, vram_gb)?;
        let registry_json =
            serde_json::to_string_pretty(&registry.to_json_export()).map_err(|e| {
                RuntimeError::ConfigError(format!("Failed to serialize models JSON: {e}"))
            })?;

        let new_aliases: HashSet<String> =
            registry.models.iter().map(|e| e.alias.clone()).collect();
        let added = new_aliases.difference(&previous_aliases).count();
        let removed = previous_aliases.difference(&new_aliases).count();
        let total = registry.models.len();
        let models_dir = registry.defaults.models_dir.clone();

        Ok(RescanResult {
            registry,
            models,
            registry_json,
            added,
            removed,
            total,
            models_dir,
        })
    }

    /// Rescan, write `models_file` (+ JSON sibling), return the applied result.
    /// Fails before writing nothing on discover/expand errors; caller must not hot-swap on write failure.
    pub fn rescan_and_write(
        models_file: &str,
        override_dirs: Option<&str>,
        fallback_backend: &str,
        vram_gb: u32,
    ) -> Result<RescanResult, RuntimeError> {
        let merge_from = if Path::new(models_file).is_file() {
            Some(Self::load(models_file)?)
        } else {
            None
        };
        let result = Self::rescan(
            override_dirs,
            merge_from.as_ref(),
            fallback_backend,
            vram_gb,
        )?;
        result.registry.write(models_file)?;
        Ok(result)
    }

    /// Expand registry entries into full `ModelConfig` map keyed by alias.
    pub fn expand(
        &self,
        fallback_backend: &str,
        vram_gb: u32,
    ) -> Result<HashMap<String, ModelConfig>, RuntimeError> {
        let models_dirs = resolve_models_dirs_with_fallback(&self.defaults.models_dir)?;

        let mut entries = Vec::new();
        let mut claimed_files = HashSet::new();

        for entry in &self.models {
            match resolve_model_path(&models_dirs, &entry.file) {
                Ok(path) => {
                    if !Path::new(&path).is_file() {
                        tracing::warn!(
                            alias = %entry.alias,
                            file = %entry.file,
                            "Skipping explicit model entry because the GGUF path is not a file"
                        );
                        continue;
                    }
                    if let Err(reason) = validate_gguf_model(Path::new(&path)) {
                        tracing::warn!(
                            alias = %entry.alias,
                            file = %entry.file,
                            reason = %reason,
                            "Skipping explicit model entry because the GGUF file failed validation"
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
                let kind = infer_kind(&alias, &file);

                entries.push(RegistryEntry {
                    alias: alias.clone(),
                    file,
                    display_name: Some(display_name_from_alias(&alias)),
                    kind: Some(kind),
                    enabled: true,
                    priority: false,
                    port: None,
                    context_size: None,
                    ngl: None,
                    extra_args: Vec::new(),
                    ..Default::default()
                });
            }
        }

        entries.retain(|entry| entry.enabled);
        dedupe_registry_entries(&mut entries);
        normalize_priority_entries(&mut entries);

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
            let context_size =
                suggest_context_size(vram_gb, entry, &model_path, self.defaults.context_size);
            let ngl = entry.ngl.unwrap_or(self.defaults.ngl);
            let extra = effective_extra_args(entry);
            let ngl_pinned = entry.ngl.is_some() || extra_args_pin_ngl(&extra);
            let block_count = inspect_gguf_metadata(Path::new(&model_path))
                .ok()
                .and_then(|m| m.block_count)
                .and_then(|n| u32::try_from(n).ok());

            let mut args = vec![
                "-m".to_string(),
                model_path,
                "--host".to_string(),
                self.defaults.host.clone(),
                "--port".to_string(),
                port.to_string(),
                "-c".to_string(),
                context_size.to_string(),
                "-ngl".to_string(),
                ngl.to_string(),
            ];
            if entry.effective_kind() == "embedding" {
                args.push("--embeddings".to_string());
            }
            args.extend(extra);

            let config = ModelConfig {
                backend: backend.clone(),
                display_name: entry
                    .display_name
                    .clone()
                    .unwrap_or_else(|| display_name_from_alias(&entry.alias)),
                command: self.defaults.llama_server.clone(),
                args,
                backend_url: format!("http://{}:{}/v1", self.defaults.host, port),
                health_url: format!("http://{}:{}/health", self.defaults.host, port),
                priority: entry.priority,
                kind: entry.effective_kind(),
                description: entry.description.clone(),
                max_context_length: entry.max_context_length,
                min_vram_gb: entry.min_vram_gb,
                capabilities: entry.capabilities.clone(),
                hf_repo: entry.hf_repo.clone(),
                block_count,
                ngl_pinned,
            };
            models.insert(entry.alias.clone(), config);
        }

        Ok(models)
    }
}

fn extra_args_pin_ngl(extra_args: &[String]) -> bool {
    extra_args
        .iter()
        .any(|arg| matches!(arg.as_str(), "-ngl" | "--n-gpu-layers" | "--gpu-layers"))
}

/// Llama 3.1 GGUFs often ship a tool-use chat template that makes the model
/// emit `{"name":...,"parameters":...}` even when the request has no tools.
/// Force llama.cpp's built-in `llama3` template unless the user already set one.
fn effective_extra_args(entry: &RegistryEntry) -> Vec<String> {
    let mut args = entry.extra_args.clone();
    if !should_force_llama3_chat_template(entry, &args) {
        return args;
    }
    tracing::info!(
        alias = %entry.alias,
        "Applying --chat-template llama3 so Llama 3.1 answers in plain text (override via extra_args)"
    );
    args.push("--chat-template".to_string());
    args.push("llama3".to_string());
    args
}

fn should_force_llama3_chat_template(entry: &RegistryEntry, extra_args: &[String]) -> bool {
    if !is_llama_3_1_model(&entry.alias, &entry.file) {
        return false;
    }
    !extra_args.iter().any(|arg| {
        let lower = arg.to_ascii_lowercase();
        lower == "--chat-template"
            || lower == "--chat-template-file"
            || lower.starts_with("--chat-template=")
            || lower == "--jinja"
    })
}

fn is_llama_3_1_model(alias: &str, file: &str) -> bool {
    let hay = format!("{alias} {file}").to_ascii_lowercase();
    hay.contains("llama-3.1") || hay.contains("llama3.1") || hay.contains("meta-llama-3.1")
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
        write_gguf_fixture(path, 2, 1, architecture, None, None);
    }

    fn write_gguf_fixture(
        path: &Path,
        version: u32,
        tensor_count: u64,
        architecture: &str,
        file_type: Option<&str>,
        block_count: Option<u32>,
    ) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        let mut kv_count = 1u64;
        if file_type.is_some() {
            kv_count += 1;
        }
        if block_count.is_some() {
            kv_count += 1;
        }
        buf.extend_from_slice(&kv_count.to_le_bytes());
        write_gguf_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&GGUF_METADATA_STRING.to_le_bytes());
        write_gguf_string(&mut buf, architecture);
        if let Some(file_type) = file_type {
            write_gguf_string(&mut buf, "general.type");
            buf.extend_from_slice(&GGUF_METADATA_STRING.to_le_bytes());
            write_gguf_string(&mut buf, file_type);
        }
        if let Some(block_count) = block_count {
            let key = format!("{architecture}.block_count");
            write_gguf_string(&mut buf, &key);
            buf.extend_from_slice(&GGUF_METADATA_UINT32.to_le_bytes());
            buf.extend_from_slice(&block_count.to_le_bytes());
        }
        std::fs::write(path, buf).unwrap();
    }

    #[test]
    fn inspect_gguf_metadata_reads_prefix_only() {
        let dir = std::env::temp_dir().join("gguf-switchboard-inspect-prefix-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("huge-but-header-only.gguf");

        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        write_gguf_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&GGUF_METADATA_STRING.to_le_bytes());
        write_gguf_string(&mut buf, "llama");
        // Simulate multi-GB weights without writing multi-GB: sparse-ish padding.
        buf.resize(buf.len() + 8 * 1024 * 1024, 0);
        std::fs::write(&path, &buf).unwrap();

        let meta = inspect_gguf_metadata(&path).expect("prefix metadata");
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.tensor_count, 1);
        assert!(is_llama_cpp_loadable_gguf(&path));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_gguf_model_accepts_chat_and_embedding_arches() {
        let dir = std::env::temp_dir().join("gguf-switchboard-validate-accept-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let chat = dir.join("chat-model.gguf");
        write_minimal_gguf(&chat, "llama");
        assert!(validate_gguf_model(&chat).is_ok());

        let embed = dir.join("nomic-embed.gguf");
        write_minimal_gguf(&embed, "nomic-bert");
        assert!(validate_gguf_model(&embed).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_gguf_model_rejects_bad_name_magic_version_tensors_arch() {
        let dir = std::env::temp_dir().join("gguf-switchboard-validate-reject-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mmproj = dir.join("mmproj-model.gguf");
        write_minimal_gguf(&mmproj, "llama");
        assert_eq!(
            validate_gguf_model(&mmproj).unwrap_err(),
            GgufSkipReason::BadName
        );

        let projector = dir.join("vision-projector.gguf");
        write_minimal_gguf(&projector, "llama");
        assert_eq!(
            validate_gguf_model(&projector).unwrap_err(),
            GgufSkipReason::BadName
        );

        let bad_magic = dir.join("bad-magic.gguf");
        let mut bad = b"NOTG".to_vec();
        bad.extend_from_slice(&[0u8; 20]);
        std::fs::write(&bad_magic, bad).unwrap();
        assert_eq!(
            validate_gguf_model(&bad_magic).unwrap_err(),
            GgufSkipReason::BadMagic
        );

        let bad_version = dir.join("bad-version.gguf");
        write_gguf_fixture(&bad_version, 99, 1, "llama", None, None);
        assert_eq!(
            validate_gguf_model(&bad_version).unwrap_err(),
            GgufSkipReason::BadVersion
        );

        let no_tensors = dir.join("no-tensors.gguf");
        write_gguf_fixture(&no_tensors, 2, 0, "llama", None, None);
        assert_eq!(
            validate_gguf_model(&no_tensors).unwrap_err(),
            GgufSkipReason::NoTensors
        );

        let clip = dir.join("clip-encoder.gguf");
        write_minimal_gguf(&clip, "clip");
        assert_eq!(
            validate_gguf_model(&clip).unwrap_err(),
            GgufSkipReason::BadArch
        );

        let zero_blocks = dir.join("zero-blocks.gguf");
        write_gguf_fixture(&zero_blocks, 2, 1, "llama", None, Some(0));
        assert_eq!(
            validate_gguf_model(&zero_blocks).unwrap_err(),
            GgufSkipReason::ZeroBlockCount
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn expand_skips_explicit_pin_that_fails_validation() {
        let dir = std::env::temp_dir().join("gguf-switchboard-explicit-skip-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let good = dir.join("good.gguf");
        let bad = dir.join("bad.gguf");
        write_minimal_gguf(&good, "llama");
        write_minimal_gguf(&bad, "clip");

        let registry = ModelsRegistry {
            version: 1,
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                ..RegistryDefaults::default()
            },
            auto_discover: false,
            models: vec![
                RegistryEntry {
                    alias: "good".to_string(),
                    file: "good.gguf".to_string(),
                    display_name: None,
                    kind: None,
                    enabled: true,
                    priority: false,
                    port: None,
                    context_size: None,
                    ngl: None,
                    extra_args: Vec::new(),
                    ..Default::default()
                },
                RegistryEntry {
                    alias: "bad".to_string(),
                    file: "bad.gguf".to_string(),
                    display_name: None,
                    kind: None,
                    enabled: true,
                    priority: false,
                    port: None,
                    context_size: None,
                    ngl: None,
                    extra_args: Vec::new(),
                    ..Default::default()
                },
            ],
        };

        let models = registry.expand("llama.cpp", 12).unwrap();
        assert!(models.contains_key("good"));
        assert!(!models.contains_key("bad"));

        let _ = std::fs::remove_dir_all(&dir);
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
            version: 1,
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
                    kind: None,
                    enabled: true,
                    priority: true,
                    port: None,
                    context_size: None,
                    ngl: None,
                    extra_args: Vec::new(),
                    ..Default::default()
                },
                RegistryEntry {
                    alias: "legacy-missing".to_string(),
                    file: "removed.gguf".to_string(),
                    display_name: Some("Removed".to_string()),
                    kind: None,
                    enabled: true,
                    priority: false,
                    port: None,
                    context_size: None,
                    ngl: None,
                    extra_args: Vec::new(),
                    ..Default::default()
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
    fn discover_merge_drops_entries_when_no_gguf_found() {
        let dir = std::env::temp_dir().join("gguf-switchboard-merge-empty-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let existing = ModelsRegistry {
            version: 1,
            defaults: RegistryDefaults {
                models_dir: "/old".to_string(),
                ..RegistryDefaults::default()
            },
            auto_discover: false,
            models: vec![RegistryEntry {
                alias: "demo".to_string(),
                file: "demo.gguf".to_string(),
                display_name: Some("Demo".to_string()),
                kind: None,
                enabled: true,
                priority: true,
                port: None,
                context_size: None,
                ngl: None,
                extra_args: Vec::new(),
                ..Default::default()
            }],
        };

        let merged =
            ModelsRegistry::discover_with_merge(dir.to_str().unwrap(), Some(&existing)).unwrap();
        assert_eq!(merged.defaults.models_dir, dir.to_string_lossy());
        assert!(merged.models.is_empty());
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
            version: 1,
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
                kind: None,
                enabled: true,
                priority: true,
                port: None,
                context_size: Some(4096),
                ngl: None,
                extra_args: Vec::new(),
                ..Default::default()
            }],
        };

        let models = registry.expand("llama.cpp", 12).unwrap();
        let cfg = models.get("test").unwrap();
        assert_eq!(cfg.display_name, "Test Model");
        assert!(cfg.args.contains(&"-m".to_string()));
        assert!(cfg.args.contains(&model_path.to_string_lossy().to_string()));
        assert!(cfg.args.contains(&"9001".to_string()));
        assert!(cfg.args.contains(&"4096".to_string()));
        assert_eq!(cfg.backend_url, "http://127.0.0.1:9001/v1");
        assert!(!cfg.ngl_pinned);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expand_pins_ngl_when_model_sets_ngl() {
        let dir = std::env::temp_dir().join("gguf-switchboard-ngl-pin-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("test-model.gguf");
        write_minimal_gguf(&model_path, "llama");

        let registry = ModelsRegistry {
            version: 1,
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                base_port: 9001,
                ..RegistryDefaults::default()
            },
            auto_discover: false,
            models: vec![RegistryEntry {
                alias: "test".to_string(),
                file: "test-model.gguf".to_string(),
                ngl: Some(24),
                ..Default::default()
            }],
        };

        let models = registry.expand("llama.cpp", 12).unwrap();
        let cfg = models.get("test").unwrap();
        assert!(cfg.ngl_pinned);
        assert!(cfg.args.windows(2).any(|w| w[0] == "-ngl" && w[1] == "24"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expand_adds_embeddings_flag_for_embedding_models() {
        let dir = std::env::temp_dir().join("gguf-switchboard-embeddings-flag-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let embed_path = dir.join("nomic-embed-text-v1.5.Q4_K_M.gguf");
        let chat_path = dir.join("chat-model.gguf");
        write_minimal_gguf(&embed_path, "nomic-bert");
        write_minimal_gguf(&chat_path, "llama");

        let registry = ModelsRegistry {
            version: 1,
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                base_port: 9100,
                ..RegistryDefaults::default()
            },
            auto_discover: false,
            models: vec![
                RegistryEntry {
                    alias: "nomic-embed-text-v1.5".to_string(),
                    file: "nomic-embed-text-v1.5.Q4_K_M.gguf".to_string(),
                    kind: Some("embedding".to_string()),
                    ..Default::default()
                },
                RegistryEntry {
                    alias: "chat-model".to_string(),
                    file: "chat-model.gguf".to_string(),
                    kind: Some("chat".to_string()),
                    ..Default::default()
                },
            ],
        };

        let models = registry.expand("llama.cpp", 12).unwrap();
        let embed_cfg = models.get("nomic-embed-text-v1.5").unwrap();
        assert!(
            embed_cfg.args.contains(&"--embeddings".to_string()),
            "Embedding model must have --embeddings flag"
        );

        let chat_cfg = models.get("chat-model").unwrap();
        assert!(
            !chat_cfg.args.contains(&"--embeddings".to_string()),
            "Chat model must NOT have --embeddings flag"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expand_forces_llama3_chat_template_for_llama_3_1() {
        let dir = std::env::temp_dir().join("gguf-switchboard-llama31-template-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf");
        write_minimal_gguf(&model_path, "llama");

        let registry = ModelsRegistry {
            version: 1,
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                ..RegistryDefaults::default()
            },
            auto_discover: false,
            models: vec![RegistryEntry {
                alias: "meta-llama-3.1-8b".to_string(),
                file: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".to_string(),
                display_name: None,
                kind: None,
                enabled: true,
                priority: true,
                port: None,
                context_size: None,
                ngl: None,
                extra_args: Vec::new(),
                ..Default::default()
            }],
        };

        let models = registry.expand("llama.cpp", 0).unwrap();
        let cfg = models.get("meta-llama-3.1-8b").unwrap();
        assert!(
            cfg.args
                .windows(2)
                .any(|w| { w[0] == "--chat-template" && w[1] == "llama3" })
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expand_respects_explicit_jinja_for_llama_3_1() {
        let dir = std::env::temp_dir().join("gguf-switchboard-llama31-jinja-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model_path = dir.join("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf");
        write_minimal_gguf(&model_path, "llama");

        let registry = ModelsRegistry {
            version: 1,
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                ..RegistryDefaults::default()
            },
            auto_discover: false,
            models: vec![RegistryEntry {
                alias: "meta-llama-3.1-8b".to_string(),
                file: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".to_string(),
                display_name: None,
                kind: None,
                enabled: true,
                priority: true,
                port: None,
                context_size: None,
                ngl: None,
                extra_args: vec!["--jinja".to_string()],
                ..Default::default()
            }],
        };

        let models = registry.expand("llama.cpp", 0).unwrap();
        let cfg = models.get("meta-llama-3.1-8b").unwrap();
        assert!(cfg.args.contains(&"--jinja".to_string()));
        assert!(!cfg.args.iter().any(|a| a == "--chat-template"));

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
    fn expand_errors_when_no_model_directories_found() {
        let registry = ModelsRegistry {
            version: 1,
            defaults: RegistryDefaults {
                models_dir: "/nonexistent-models-root".to_string(),
                ..RegistryDefaults::default()
            },
            auto_discover: true,
            models: Vec::new(),
        };

        let err = registry.expand("llama.cpp", 12).unwrap_err().to_string();
        assert!(
            err.contains("No model directories found"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn expand_skips_missing_explicit_entries_and_auto_discovers() {
        let dir = std::env::temp_dir().join("gguf-switchboard-skip-missing-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_minimal_gguf(&dir.join("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), "llama");

        let registry = ModelsRegistry {
            version: 1,
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                ..RegistryDefaults::default()
            },
            auto_discover: true,
            models: vec![RegistryEntry {
                alias: "llama-3".to_string(),
                file: "llama-3.2-3b.gguf".to_string(),
                display_name: Some("Llama 3.2 3B".to_string()),
                kind: None,
                enabled: true,
                priority: true,
                port: None,
                context_size: None,
                ngl: None,
                extra_args: Vec::new(),
                ..Default::default()
            }],
        };

        let models = registry.expand("llama.cpp", 12).unwrap();
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

        let expanded = registry.expand("llama.cpp", 12).unwrap();
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
            version: 1,
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                ..RegistryDefaults::default()
            },
            auto_discover: true,
            models: vec![RegistryEntry {
                alias: "listed".to_string(),
                file: "listed.gguf".to_string(),
                display_name: Some("Listed".to_string()),
                kind: None,
                enabled: true,
                priority: true,
                port: None,
                context_size: None,
                ngl: None,
                extra_args: Vec::new(),
                ..Default::default()
            }],
        };

        let models = registry.expand("llama.cpp", 12).unwrap();
        assert_eq!(models.len(), 2);
        assert!(models.contains_key("listed"));
        assert!(models.contains_key("extra-model"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dedupe_registry_entries_by_alias_and_file() {
        let mut entries = vec![
            RegistryEntry {
                alias: "qwen3-vl-8b".to_string(),
                file: "Qwen3-VL-8B-Instruct-Q4_K_M.gguf".to_string(),
                display_name: Some("First".to_string()),
                kind: None,
                enabled: true,
                priority: false,
                port: None,
                context_size: None,
                ngl: None,
                extra_args: Vec::new(),
                ..Default::default()
            },
            RegistryEntry {
                alias: "qwen3-vl-8b".to_string(),
                file: "Qwen3-VL-8B-Instruct-Q4_K_M.gguf".to_string(),
                display_name: Some("Duplicate".to_string()),
                kind: None,
                enabled: true,
                priority: true,
                port: None,
                context_size: None,
                ngl: None,
                extra_args: Vec::new(),
                ..Default::default()
            },
        ];

        dedupe_registry_entries(&mut entries);
        normalize_priority_entries(&mut entries);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].display_name.as_deref(), Some("First"));
        assert!(entries[0].priority);
    }

    #[test]
    fn suggest_context_size_uses_vram_for_small_models() {
        let entry = RegistryEntry {
            alias: "gemma-4-e4b".to_string(),
            file: "gemma-4-E4B-it-Q4_K_M.gguf".to_string(),
            display_name: None,
            kind: Some("chat".to_string()),
            enabled: true,
            priority: false,
            port: None,
            context_size: None,
            ngl: None,
            extra_args: Vec::new(),
            ..Default::default()
        };

        assert_eq!(
            suggest_context_size(12, &entry, "/tmp/gemma-4-E4B-it-Q4_K_M.gguf", 65536),
            32768
        );
    }

    #[test]
    fn resolve_models_dirs_with_fallback_prefers_configured_then_common() {
        let root = std::env::temp_dir().join("gguf-switchboard-fallback-order-test");
        let configured = root.join("configured");
        let common_home = root.join("home-models");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&configured).unwrap();
        std::fs::create_dir_all(&common_home).unwrap();
        write_minimal_gguf(&common_home.join("only-in-common.gguf"), "llama");

        // Empty configured dir still comes first; common with GGUF is appended.
        let resolved = resolve_models_dirs_with_fallback(configured.to_str().unwrap()).unwrap();
        assert_eq!(resolved[0], configured);
        // common_models_dir_candidates uses $HOME/models — not our temp path.
        // Verify configured-only when no GGUFs in configured and no overlap with real home.
        assert!(resolved.iter().any(|d| d == &configured));

        write_minimal_gguf(&configured.join("cfg.gguf"), "llama");
        let with_files = resolve_models_dirs_with_fallback(configured.to_str().unwrap()).unwrap();
        assert_eq!(with_files[0], configured);

        let joined = format_models_dirs(&with_files);
        assert!(joined.starts_with(&configured.to_string_lossy().into_owned()));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolve_models_dirs_with_fallback_dedupes_configured() {
        let dir = std::env::temp_dir().join("gguf-switchboard-fallback-dedupe-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_minimal_gguf(&dir.join("a.gguf"), "llama");

        let path = dir.to_string_lossy().into_owned();
        let configured = format!("{path},{path}");
        let resolved = resolve_models_dirs_with_fallback(&configured).unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0], dir);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rescan_appends_models_dir_and_reports_added_removed() {
        let dir = std::env::temp_dir().join("gguf-switchboard-rescan-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_minimal_gguf(&dir.join("alpha.gguf"), "llama");
        write_minimal_gguf(&dir.join("beta.gguf"), "llama");

        let existing = ModelsRegistry {
            version: 1,
            defaults: RegistryDefaults {
                models_dir: dir.to_string_lossy().into_owned(),
                ..RegistryDefaults::default()
            },
            auto_discover: true,
            models: vec![
                RegistryEntry {
                    alias: "alpha-custom".to_string(),
                    file: "alpha.gguf".to_string(),
                    display_name: Some("Alpha Custom".to_string()),
                    kind: None,
                    enabled: true,
                    priority: true,
                    port: None,
                    context_size: None,
                    ngl: None,
                    extra_args: Vec::new(),
                    ..Default::default()
                },
                RegistryEntry {
                    alias: "gone".to_string(),
                    file: "missing.gguf".to_string(),
                    display_name: Some("Gone".to_string()),
                    kind: None,
                    enabled: true,
                    priority: false,
                    port: None,
                    context_size: None,
                    ngl: None,
                    extra_args: Vec::new(),
                    ..Default::default()
                },
            ],
        };

        let result = ModelsRegistry::rescan(
            Some(dir.to_str().unwrap()),
            Some(&existing),
            "llama.cpp",
            12,
        )
        .unwrap();

        assert_eq!(result.removed, 1);
        assert_eq!(result.added, 1); // beta newly discovered
        assert_eq!(result.total, 2);
        assert!(result.registry.models.iter().any(
            |e| e.alias == "alpha-custom" && e.display_name.as_deref() == Some("Alpha Custom")
        ));
        assert!(!result.registry.models.iter().any(|e| e.alias == "gone"));
        assert_eq!(result.models_dir, dir.to_string_lossy());

        std::fs::remove_dir_all(&dir).ok();
    }
}
