//! `ggs models list` / `ggs models delete` — filesystem-level model inventory.
//!
//! `discover-models` scans the model directories only as a side effect of
//! writing a registry file. These two commands expose the scan directly:
//! `list` prints a numbered inventory of every GGUF file and safetensors model
//! directory found on disk (with the registered alias, if any), and `delete`
//! removes one of them — the file/dir **and** the matching `models.toml`
//! entry — targeted by that number or by name.

use std::error::Error;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use serde_json::json;

use super::hf_download::format_bytes;
use super::models_registry::{
    ModelsRegistry, discover_gguf_files, parse_models_dirs, resolve_model_path,
    resolve_models_dirs_with_fallback,
};

/// One model on disk: a single `.gguf` file, or a safetensors model directory.
#[derive(Debug, Clone)]
pub(crate) struct LocalModelEntry {
    /// 1-based position in the scan — stable within a single invocation and
    /// what `ggs models delete <n>` consumes.
    pub index: usize,
    /// Absolute path to the `.gguf` file or the safetensors directory.
    pub path: PathBuf,
    pub size_bytes: u64,
    pub kind: String,
    /// Registered alias in `models.toml`, when the file is registered.
    pub alias: Option<String>,
    pub is_safetensors_dir: bool,
}

impl LocalModelEntry {
    fn display_name(&self) -> String {
        if let Some(alias) = &self.alias {
            return alias.clone();
        }
        if self.is_safetensors_dir {
            self.path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| self.path.display().to_string())
        } else {
            self.path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| self.path.display().to_string())
        }
    }
}

fn infer_kind(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    if lower.contains("rerank") {
        "reranker"
    } else if lower.contains("embed") {
        "embedding"
    } else if lower.contains("-vl") || lower.contains("vision") || lower.contains("mmproj") {
        "vision"
    } else if lower.contains("coder") || lower.contains("-code") {
        "coder"
    } else {
        "chat"
    }
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// A safetensors model dir = contains `config.json` and at least one
/// `*.safetensors` file.
fn is_safetensors_model_dir(dir: &Path) -> bool {
    if !dir.join("config.json").is_file() {
        return false;
    }
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        })
}

/// Recursively look for safetensors model dirs, up to `depth` levels below each
/// root (HF-style layouts nest as `<models_dir>/<org>/<model>/`).
fn collect_safetensors_dirs(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if is_safetensors_model_dir(dir) {
        out.push(dir.to_path_buf());
        return; // don't descend into a model dir's shards
    }
    if depth == 0 {
        return;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_safetensors_dirs(&path, depth - 1, out);
            }
        }
    }
}

fn dir_size_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum()
}

/// Map every registered model's resolved file path → its alias.
fn registered_alias_map(
    registry: Option<&ModelsRegistry>,
    dirs: &[PathBuf],
) -> Vec<(PathBuf, String)> {
    let Some(registry) = registry else {
        return Vec::new();
    };
    let mut map = Vec::new();
    for entry in &registry.models {
        for file in [Some(entry.file.as_str()), entry.vllm_file.as_deref()]
            .into_iter()
            .flatten()
            .filter(|f| !f.is_empty())
        {
            if let Ok(resolved) = resolve_model_path(dirs, file) {
                map.push((canonical(Path::new(&resolved)), entry.alias.clone()));
            }
        }
    }
    map
}

/// Build the on-disk model list. `list` and `delete` MUST call this identically
/// so the `#` index means the same thing to both.
pub(crate) fn scan_local_models(
    dirs: &[PathBuf],
    registry: Option<&ModelsRegistry>,
) -> Vec<LocalModelEntry> {
    let alias_map = registered_alias_map(registry, dirs);
    let alias_for = |path: &Path| -> Option<String> {
        let c = canonical(path);
        alias_map
            .iter()
            .find(|(p, _)| *p == c)
            .map(|(_, a)| a.clone())
    };

    let mut raw: Vec<(PathBuf, u64, bool)> = Vec::new();

    for gguf in discover_gguf_files(dirs).unwrap_or_default() {
        let size = std::fs::metadata(&gguf).map(|m| m.len()).unwrap_or(0);
        raw.push((gguf, size, false));
    }

    let mut st_dirs = Vec::new();
    for root in dirs {
        collect_safetensors_dirs(root, 3, &mut st_dirs);
    }
    st_dirs.sort();
    st_dirs.dedup();
    for st in st_dirs {
        let size = dir_size_bytes(&st);
        raw.push((st, size, true));
    }

    raw.sort_by(|a, b| a.0.cmp(&b.0));
    raw.dedup_by(|a, b| a.0 == b.0);

    raw.into_iter()
        .enumerate()
        .map(|(i, (path, size_bytes, is_safetensors_dir))| {
            let alias = alias_for(&path);
            let name_for_kind = alias.clone().unwrap_or_else(|| {
                path.file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            LocalModelEntry {
                index: i + 1,
                kind: infer_kind(&name_for_kind).to_string(),
                alias,
                path,
                size_bytes,
                is_safetensors_dir,
            }
        })
        .collect()
}

// ── shared arg parsing ───────────────────────────────────────────────────────

struct CommonArgs {
    dir_overrides: Vec<String>,
    registry_path: Option<String>,
    json: bool,
    /// Remaining positional args (e.g. the delete target).
    positional: Vec<String>,
    yes: bool,
}

fn parse_common(args: &[String], allow_yes: bool) -> Result<CommonArgs, String> {
    let mut out = CommonArgs {
        dir_overrides: Vec::new(),
        registry_path: None,
        json: false,
        positional: Vec::new(),
        yes: false,
    };
    let mut i = 1; // args[0] is the subcommand ("list" / "delete")
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| format!("models {}: missing value for --dir", args[0]))?;
                out.dir_overrides.push(v.clone());
                i += 2;
            }
            "--registry" => {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| format!("models {}: missing value for --registry", args[0]))?;
                out.registry_path = Some(v.clone());
                i += 2;
            }
            "--json" => {
                out.json = true;
                i += 1;
            }
            "--yes" | "-y" if allow_yes => {
                out.yes = true;
                i += 1;
            }
            flag if flag.starts_with('-') => {
                return Err(format!("models {}: unknown flag '{flag}'", args[0]));
            }
            other => {
                out.positional.push(other.to_string());
                i += 1;
            }
        }
    }
    Ok(out)
}

fn resolve_registry(explicit: Option<&str>) -> Option<(String, ModelsRegistry)> {
    let path = match explicit {
        Some(p) => p.to_string(),
        None if Path::new("models.toml").is_file() => "models.toml".to_string(),
        None => return None,
    };
    match ModelsRegistry::load(&path) {
        Ok(reg) => Some((path, reg)),
        Err(e) => {
            eprintln!("warning: could not load registry '{path}': {e}");
            None
        }
    }
}

fn resolve_scan_dirs(
    overrides: &[String],
    registry: Option<&ModelsRegistry>,
) -> Result<Vec<PathBuf>, String> {
    if !overrides.is_empty() {
        let dirs: Vec<PathBuf> = overrides
            .iter()
            .flat_map(|s| parse_models_dirs(s))
            .filter(|p| p.is_dir())
            .collect();
        if dirs.is_empty() {
            return Err("models list: no existing directory among --dir values".to_string());
        }
        return Ok(dirs);
    }
    let configured = registry
        .map(|r| r.defaults.models_dir.clone())
        .unwrap_or_default();
    resolve_models_dirs_with_fallback(&configured)
        .map_err(|e| format!("models list: could not resolve a model directory ({e}); pass --dir"))
}

fn json_sibling(toml_path: &str) -> String {
    match toml_path.rfind(".toml") {
        Some(idx) => format!("{}.json{}", &toml_path[..idx], &toml_path[idx + 5..]),
        None => format!("{toml_path}.json"),
    }
}

// ── ggs models list ──────────────────────────────────────────────────────────

pub async fn cmd_list_local(args: &[String]) -> Result<(), Box<dyn Error>> {
    let parsed = parse_common(args, false)?;
    let registry = resolve_registry(parsed.registry_path.as_deref());
    let dirs = resolve_scan_dirs(&parsed.dir_overrides, registry.as_ref().map(|(_, r)| r))?;
    let entries = scan_local_models(&dirs, registry.as_ref().map(|(_, r)| r));

    if parsed.json {
        let arr: Vec<_> = entries
            .iter()
            .map(|e| {
                json!({
                    "index": e.index,
                    "path": e.path.to_string_lossy(),
                    "size_bytes": e.size_bytes,
                    "kind": e.kind,
                    "alias": e.alias,
                    "registered": e.alias.is_some(),
                    "safetensors_dir": e.is_safetensors_dir,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr)?);
        return Ok(());
    }

    println!(
        "Scanned: {}",
        dirs.iter()
            .map(|d| d.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if entries.is_empty() {
        println!("No GGUF files or safetensors model directories found.");
        return Ok(());
    }

    println!();
    println!(
        "  {:>3}  {:>10}  {:<9}  {:<28}  PATH",
        "#", "SIZE", "KIND", "ALIAS / NAME"
    );
    for e in &entries {
        let name = e
            .alias
            .clone()
            .unwrap_or_else(|| "(unregistered)".to_string());
        let shown_path = shorten_path(&e.path, &dirs);
        println!(
            "  {:>3}  {:>10}  {:<9}  {:<28}  {}",
            e.index,
            format_bytes(e.size_bytes),
            e.kind,
            truncate(&name, 28),
            shown_path
        );
    }
    println!();
    println!("Delete one with:  ggs models delete <#|name> [--yes]");
    Ok(())
}

fn shorten_path(path: &Path, dirs: &[PathBuf]) -> String {
    for dir in dirs {
        if let Ok(rel) = path.strip_prefix(dir) {
            return rel.to_string_lossy().into_owned();
        }
    }
    path.display().to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

// ── ggs models delete ────────────────────────────────────────────────────────

pub async fn cmd_delete_local(args: &[String]) -> Result<(), Box<dyn Error>> {
    let parsed = parse_common(args, true)?;
    let target = parsed.positional.first().cloned().ok_or(
        "models delete: missing target — pass a name or the number from `ggs models list`",
    )?;
    if parsed.positional.len() > 1 {
        return Err("models delete: expected exactly one target".into());
    }

    let registry = resolve_registry(parsed.registry_path.as_deref());
    let dirs = resolve_scan_dirs(&parsed.dir_overrides, registry.as_ref().map(|(_, r)| r))?;
    let entries = scan_local_models(&dirs, registry.as_ref().map(|(_, r)| r));
    if entries.is_empty() {
        return Err("models delete: no models found on disk to delete".into());
    }

    let entry = resolve_target(&target, &entries)?;

    // Path-traversal / escape guard: the resolved path must live inside one of
    // the scanned directories.
    let canon = canonical(&entry.path);
    let inside = dirs.iter().any(|d| canon.starts_with(canonical(d)));
    if !inside {
        return Err(format!(
            "models delete: refusing to delete '{}' — it is outside the scanned model directories",
            entry.path.display()
        )
        .into());
    }

    println!("Target:      {}", entry.display_name());
    println!("Path:        {}", entry.path.display());
    println!("Size:        {}", format_bytes(entry.size_bytes));
    println!(
        "Kind:        {}{}",
        entry.kind,
        if entry.is_safetensors_dir {
            " (safetensors directory)"
        } else {
            ""
        }
    );
    match &entry.alias {
        Some(a) => println!("Registered:  yes (alias '{a}')"),
        None => println!("Registered:  no"),
    }

    if !parsed.yes {
        if !std::io::stdin().is_terminal() {
            return Err(
                "models delete: refusing to delete without --yes in a non-interactive shell".into(),
            );
        }
        print!("\nDelete this model from disk? [y/N] ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        if !matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            println!("Aborted; nothing was deleted.");
            return Ok(());
        }
    }

    if entry.is_safetensors_dir {
        // Re-verify right before the recursive delete.
        if !is_safetensors_model_dir(&entry.path) {
            return Err(format!(
                "models delete: '{}' no longer looks like a safetensors model directory; not deleting",
                entry.path.display()
            )
            .into());
        }
        std::fs::remove_dir_all(&entry.path)?;
        println!("Deleted directory {}", entry.path.display());
    } else {
        std::fs::remove_file(&entry.path)?;
        println!("Deleted {}", entry.path.display());
        report_sidecars(&entry.path);
    }

    // Registry cleanup. `entry.alias` was resolved during the scan, while the
    // file still existed, so match on that rather than re-resolving a path that
    // no longer points at anything.
    if let (Some((path, mut reg)), Some(alias)) = (registry, entry.alias.clone()) {
        let before = reg.models.len();
        reg.models.retain(|m| !m.alias.eq_ignore_ascii_case(&alias));
        if reg.models.len() != before {
            reg.write(&path)?;
            let json_path = json_sibling(&path);
            let _ = reg.write_json(&json_path);
            println!("Unregistered from {path} (and {json_path})");
        }
    }

    println!("\nRestart the service or POST /v1/models/refresh to drop it from the live registry.");
    Ok(())
}

fn resolve_target<'a>(
    target: &str,
    entries: &'a [LocalModelEntry],
) -> Result<&'a LocalModelEntry, String> {
    // Numeric index from `ggs models list`.
    if let Ok(n) = target.parse::<usize>() {
        return entries.iter().find(|e| e.index == n).ok_or_else(|| {
            format!(
                "models delete: index {n} out of range (1..={})",
                entries.len()
            )
        });
    }

    let want = target.to_ascii_lowercase();
    let matches: Vec<&LocalModelEntry> = entries
        .iter()
        .filter(|e| {
            e.alias
                .as_deref()
                .is_some_and(|a| a.eq_ignore_ascii_case(target))
                || e.path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_ascii_lowercase())
                    .as_deref()
                    == Some(want.as_str())
                || (e.is_safetensors_dir
                    && e.path
                        .file_name()
                        .map(|s| s.to_string_lossy().to_ascii_lowercase())
                        .as_deref()
                        == Some(want.as_str()))
        })
        .collect();

    match matches.as_slice() {
        [] => Err(format!(
            "models delete: no model matches '{target}' — run `ggs models list` to see names and numbers"
        )),
        [one] => Ok(one),
        many => Err(format!(
            "models delete: '{target}' is ambiguous ({} matches: {}); use the number instead",
            many.len(),
            many.iter()
                .map(|e| format!("#{}", e.index))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// Print (but never auto-delete) files that commonly ride alongside a GGUF.
fn report_sidecars(gguf: &Path) {
    let (Some(parent), Some(stem)) = (gguf.parent(), gguf.file_stem()) else {
        return;
    };
    let stem = stem.to_string_lossy();
    let mut related = Vec::new();
    if let Ok(entries) = std::fs::read_dir(parent) {
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if e.path() == gguf {
                continue;
            }
            let lower = name.to_ascii_lowercase();
            if lower.contains("mmproj") || (name.starts_with(&*stem) && lower.ends_with(".json")) {
                related.push(name);
            }
        }
    }
    if !related.is_empty() {
        println!(
            "note: left {} related file(s) in place: {}",
            related.len(),
            related.join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_safetensors_dir(root: &Path, name: &str) -> PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors"), vec![0u8; 2048]).unwrap();
        dir
    }

    #[test]
    fn infer_kind_matches_common_names() {
        assert_eq!(infer_kind("qwen3-embedding-4b"), "embedding");
        assert_eq!(infer_kind("bge-reranker-v2"), "reranker");
        assert_eq!(infer_kind("Qwen2.5-VL-7B"), "vision");
        assert_eq!(infer_kind("qwen2.5-coder-7b"), "coder");
        assert_eq!(infer_kind("gemma-3-12b-it"), "chat");
    }

    #[test]
    fn scan_finds_safetensors_dirs_nested() {
        let tmp = tempfile::tempdir().unwrap();
        make_safetensors_dir(tmp.path(), "DirectModel");
        make_safetensors_dir(&tmp.path().join("org"), "NestedModel");
        std::fs::create_dir_all(tmp.path().join("empty")).unwrap();

        let entries = scan_local_models(&[tmp.path().to_path_buf()], None);
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|e| e.is_safetensors_dir));
        assert_eq!(entries[0].index, 1);
        assert_eq!(entries[1].index, 2);
        assert!(entries[0].size_bytes > 0);
    }

    #[test]
    fn resolve_target_by_index_name_and_ambiguity() {
        let tmp = tempfile::tempdir().unwrap();
        make_safetensors_dir(tmp.path(), "alpha-embed");
        make_safetensors_dir(tmp.path(), "beta-chat");
        let entries = scan_local_models(&[tmp.path().to_path_buf()], None);

        assert_eq!(resolve_target("1", &entries).unwrap().index, 1);
        assert!(resolve_target("99", &entries).is_err());
        assert_eq!(
            resolve_target("beta-chat", &entries)
                .unwrap()
                .path
                .file_name()
                .unwrap(),
            "beta-chat"
        );
        assert!(resolve_target("nope", &entries).is_err());
    }

    #[test]
    fn delete_target_must_be_inside_scan_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = make_safetensors_dir(tmp.path(), "m");
        let outside = tmp.path().join("m");
        assert!(canonical(&dir).starts_with(canonical(tmp.path())));
        let other = tempfile::tempdir().unwrap();
        assert!(!canonical(&outside).starts_with(canonical(other.path())));
    }
}
