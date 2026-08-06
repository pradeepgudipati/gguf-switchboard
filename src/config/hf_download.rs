//! Hugging Face Hub file listing and download.
//!
//! CLI only — never called on the request path.

use std::path::{Path, PathBuf};

use futures::StreamExt;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use crate::errors::RuntimeError;

const HF_MODELS_API: &str = "https://huggingface.co/api/models";
const HF_BASE_URL: &str = "https://huggingface.co";

fn download_url(repo: &str, filename: &str) -> String {
    format!("{HF_BASE_URL}/{repo}/resolve/main/{filename}")
}

/// A single file entry from the HF repo tree API.
#[derive(Debug, Clone, Deserialize)]
pub struct HfTreeEntry {
    pub r#type: String,
    pub path: String,
    pub size: u64,
    #[serde(default)]
    pub lfs: Option<HfLfsMeta>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfLfsMeta {
    #[serde(default)]
    pub oid: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
}

/// Build a reqwest client with the standard user-agent and optional HF token.
pub fn build_hf_client() -> Result<reqwest::Client, RuntimeError> {
    let mut builder = reqwest::Client::builder()
        .user_agent(concat!("gguf-switchboard/", env!("CARGO_PKG_VERSION")));

    if let Ok(token) = std::env::var("HF_TOKEN")
        && !token.is_empty()
    {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|e| RuntimeError::InternalError(format!("Invalid HF_TOKEN: {e}")))?,
        );
        builder = builder.default_headers(headers);
    }

    builder
        .build()
        .map_err(|e| RuntimeError::InternalError(e.to_string()))
}

/// Fetch the file tree for a repo, returning only `.gguf` entries.
pub async fn fetch_repo_tree(
    client: &reqwest::Client,
    repo: &str,
) -> Result<Vec<HfTreeEntry>, RuntimeError> {
    let url = format!("{HF_MODELS_API}/{repo}/tree/main");
    let resp = client.get(&url).send().await.map_err(RuntimeError::from)?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(RuntimeError::ConfigError(format!(
            "Repository not found: {repo}"
        )));
    }
    if !resp.status().is_success() {
        return Err(RuntimeError::ProxyError(format!(
            "HF tree API failed for {repo}: HTTP {}",
            resp.status()
        )));
    }

    let entries: Vec<HfTreeEntry> = resp.json().await.map_err(RuntimeError::from)?;
    Ok(entries
        .into_iter()
        .filter(|e| {
            e.r#type == "file"
                && Path::new(&e.path)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .collect())
}

/// Stream-download a file from HF into `dest_dir`, printing progress to stdout.
///
/// Returns the path to the downloaded file.
pub async fn download_file(
    client: &reqwest::Client,
    repo: &str,
    filename: &str,
    dest_dir: &Path,
) -> Result<PathBuf, RuntimeError> {
    let url = download_url(repo, filename);
    let resp = client.get(&url).send().await.map_err(RuntimeError::from)?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(RuntimeError::ConfigError(format!(
            "File not found: {repo}/{filename}"
        )));
    }
    if !resp.status().is_success() {
        return Err(RuntimeError::ProxyError(format!(
            "HF download failed for {repo}/{filename}: HTTP {}",
            resp.status()
        )));
    }

    let total = resp.content_length().unwrap_or(0);

    tokio::fs::create_dir_all(dest_dir).await.map_err(|e| {
        RuntimeError::ConfigError(format!(
            "Failed to create directory '{}': {e}",
            dest_dir.display()
        ))
    })?;

    let dest_path = dest_dir.join(filename);
    let file = tokio::fs::File::create(&dest_path).await.map_err(|e| {
        RuntimeError::ConfigError(format!(
            "Failed to create file '{}': {e}",
            dest_path.display()
        ))
    })?;
    let mut writer = tokio::io::BufWriter::new(file);

    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(RuntimeError::from)?;
        writer
            .write_all(&chunk)
            .await
            .map_err(|e| RuntimeError::ConfigError(format!("Write error: {e}")))?;
        downloaded += chunk.len() as u64;

        if total > 0 {
            let pct = (downloaded as f64 / total as f64 * 100.0) as u64;
            print!("\rDownloading... {}% [{}]", pct, format_bytes(downloaded));
        } else {
            print!("\rDownloading... {}", format_bytes(downloaded));
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    writer
        .flush()
        .await
        .map_err(|e| RuntimeError::ConfigError(format!("Flush error: {e}")))?;

    // Final progress line.
    if total > 0 {
        println!(
            "\rDownloading... 100% [{}]",
            format_bytes(total.max(downloaded))
        );
    } else {
        println!("\rDownloading... {} complete", format_bytes(downloaded));
    }

    Ok(dest_path)
}

/// Format bytes as a human-readable string (e.g. "5.4 GB").
pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;

    let b = bytes as f64;
    if b >= TB {
        format!("{:.1} TB", b / TB)
    } else if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::download_url;

    #[test]
    fn download_url_uses_hugging_face_repository_route() {
        assert_eq!(
            download_url(
                "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
                "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
            ),
            "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
        );
    }
}
