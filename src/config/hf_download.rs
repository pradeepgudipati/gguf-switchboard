//! Hugging Face Hub file listing and download.
//!
//! CLI only — never called on the request path.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use futures::StreamExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;

use crate::errors::RuntimeError;

const HF_MODELS_API: &str = "https://huggingface.co/api/models";
const HF_BASE_URL: &str = "https://huggingface.co";

fn download_url(repo: &str, filename: &str) -> String {
    format!("{HF_BASE_URL}/{repo}/resolve/main/{filename}")
}

fn aria2_args(
    url: &str,
    filename: &str,
    dest_dir: &Path,
    connections: u16,
    checksum: Option<&str>,
) -> Vec<OsString> {
    let mut args = vec![
        OsString::from("--continue=true"),
        OsString::from(format!("--split={connections}")),
        OsString::from(format!("--max-connection-per-server={connections}")),
        OsString::from("--min-split-size=64M"),
        OsString::from("--auto-file-renaming=false"),
        OsString::from(format!("--dir={}", dest_dir.display())),
        OsString::from(format!("--out={filename}")),
    ];
    args.push(OsString::from("--file-allocation=none"));
    if let Some(checksum) = checksum {
        args.push(OsString::from(format!("--checksum=sha-256={checksum}")));
    }
    args.push(OsString::from(url));
    args
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

/// Fetch the full file tree for a repo (no extension filter) — used for
/// safetensors/vLLM repos, which ship many files (shards, `config.json`,
/// tokenizer files) rather than a single `.gguf`.
pub async fn fetch_repo_tree_all(
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
    Ok(entries.into_iter().filter(|e| e.r#type == "file").collect())
}

/// Fetch a single file's raw text content from a repo (e.g. `config.json`).
/// Returns `Ok(None)` when the file doesn't exist rather than erroring, since
/// callers use this for best-effort metadata detection.
pub async fn fetch_repo_file_text(
    client: &reqwest::Client,
    repo: &str,
    filename: &str,
) -> Result<Option<String>, RuntimeError> {
    let url = download_url(repo, filename);
    let resp = client.get(&url).send().await.map_err(RuntimeError::from)?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        return Ok(None);
    }
    let text = resp.text().await.map_err(RuntimeError::from)?;
    Ok(Some(text))
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
    tokio::fs::create_dir_all(dest_dir).await.map_err(|e| {
        RuntimeError::ConfigError(format!(
            "Failed to create directory '{}': {e}",
            dest_dir.display()
        ))
    })?;

    let dest_path = dest_dir.join(filename);
    download_url_to_path(client, &url, &dest_path).await?;
    Ok(dest_path)
}

async fn download_url_to_path(
    client: &reqwest::Client,
    url: &str,
    dest_path: &Path,
) -> Result<(), RuntimeError> {
    let existing = tokio::fs::metadata(dest_path)
        .await
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let mut request = client.get(url);
    if existing > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={existing}-"));
        println!("Resuming native download at {}", format_bytes(existing));
    }
    let resp = request.send().await.map_err(RuntimeError::from)?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(RuntimeError::ConfigError(format!("File not found: {url}")));
    }
    if resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && existing > 0 {
        return Ok(());
    }
    if !resp.status().is_success() {
        return Err(RuntimeError::ProxyError(format!(
            "Download failed for {url}: HTTP {}",
            resp.status()
        )));
    }

    let resumed = existing > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let starting_size = if resumed { existing } else { 0 };
    let total = resp
        .content_length()
        .map(|length| length + starting_size)
        .unwrap_or(0);
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(dest_path)
        .await
        .map_err(|e| {
            RuntimeError::ConfigError(format!(
                "Failed to create file '{}': {e}",
                dest_path.display()
            ))
        })?;
    let mut writer = tokio::io::BufWriter::new(file);

    let mut stream = resp.bytes_stream();
    let mut downloaded = starting_size;

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

    Ok(())
}

/// Download through aria2c when it is available and safe, otherwise use reqwest.
pub async fn download_file_auto(
    client: &reqwest::Client,
    repo: &str,
    entry: &HfTreeEntry,
    dest_dir: &Path,
    connections: u16,
) -> Result<PathBuf, RuntimeError> {
    let sha256 = entry
        .lfs
        .as_ref()
        .and_then(|lfs| lfs.oid.as_deref())
        .map(|oid| oid.strip_prefix("sha256:").unwrap_or(oid));
    let filename = Path::new(&entry.path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| RuntimeError::ConfigError("Invalid Hugging Face filename".to_string()))?;

    let authenticated = std::env::var_os("HF_TOKEN").is_some_and(|token| !token.is_empty());
    let aria2_available = if authenticated {
        false
    } else {
        Command::new("aria2c")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_ok_and(|status| status.success())
    };

    let downloaded = if aria2_available {
        tokio::fs::create_dir_all(dest_dir).await.map_err(|e| {
            RuntimeError::ConfigError(format!(
                "Failed to create directory '{}': {e}",
                dest_dir.display()
            ))
        })?;
        println!("Using aria2c with {connections} connections");
        let status = Command::new("aria2c")
            .args(aria2_args(
                &download_url(repo, &entry.path),
                filename,
                dest_dir,
                connections,
                sha256,
            ))
            .status()
            .await
            .map_err(|e| RuntimeError::ConfigError(format!("Failed to start aria2c: {e}")))?;
        if !status.success() {
            eprintln!(
                "Warning: aria2c failed with status {status}; resuming with native downloader"
            );
            let downloaded = download_file(client, repo, &entry.path, dest_dir).await?;
            let control_file = PathBuf::from(format!("{}.aria2", downloaded.display()));
            let _ = tokio::fs::remove_file(control_file).await;
            downloaded
        } else {
            dest_dir.join(filename)
        }
    } else {
        if authenticated {
            println!("HF_TOKEN detected; using secure native downloader");
        } else {
            println!("aria2c not found; using native downloader");
        }
        download_file(client, repo, &entry.path, dest_dir).await?
    };

    verify_download(&downloaded, entry.size, sha256).await?;
    Ok(downloaded)
}

async fn verify_download(
    path: &Path,
    expected_size: u64,
    sha256: Option<&str>,
) -> Result<(), RuntimeError> {
    let actual_size = tokio::fs::metadata(path)
        .await
        .map_err(|e| {
            RuntimeError::ConfigError(format!("Cannot inspect '{}': {e}", path.display()))
        })?
        .len();
    if actual_size != expected_size {
        return Err(RuntimeError::ConfigError(format!(
            "Downloaded file size mismatch: expected {expected_size} bytes, got {actual_size}"
        )));
    }

    if let Some(expected) = sha256 {
        let mut file = tokio::fs::File::open(path).await.map_err(|e| {
            RuntimeError::ConfigError(format!("Cannot verify '{}': {e}", path.display()))
        })?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .await
                .map_err(|e| RuntimeError::ConfigError(format!("Checksum read failed: {e}")))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(RuntimeError::ConfigError(format!(
                "Downloaded file SHA-256 mismatch: expected {expected}, got {actual}"
            )));
        }
    }
    Ok(())
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
    use std::ffi::OsString;

    use axum::Router;
    use axum::http::{HeaderMap, StatusCode, header};
    use axum::routing::get;

    use super::{aria2_args, download_url, download_url_to_path, verify_download};

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

    #[test]
    fn aria2_args_enable_parallel_resume_and_checksum() {
        let dir = std::path::Path::new("/home/pradeep/models");
        let args = aria2_args(
            "https://huggingface.co/example/model/resolve/main/model.gguf",
            "model.gguf",
            dir,
            8,
            Some("abc123"),
        );

        for expected in [
            "--continue=true",
            "--split=8",
            "--max-connection-per-server=8",
            "--min-split-size=64M",
            "--file-allocation=none",
            "--dir=/home/pradeep/models",
            "--out=model.gguf",
            "--checksum=sha-256=abc123",
        ] {
            assert!(
                args.contains(&OsString::from(expected)),
                "missing {expected}"
            );
        }
        assert_eq!(
            args.last(),
            Some(&OsString::from(
                "https://huggingface.co/example/model/resolve/main/model.gguf"
            ))
        );
    }

    #[tokio::test]
    async fn verify_download_rejects_wrong_size() {
        let file = tempfile::NamedTempFile::new().unwrap();
        tokio::fs::write(file.path(), b"gguf").await.unwrap();

        let err = verify_download(file.path(), 5, None).await.unwrap_err();
        assert!(err.to_string().contains("size mismatch"));
    }

    #[tokio::test]
    async fn verify_download_rejects_wrong_sha256() {
        let file = tempfile::NamedTempFile::new().unwrap();
        tokio::fs::write(file.path(), b"gguf").await.unwrap();

        let err = verify_download(file.path(), 4, Some(&"0".repeat(64)))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("SHA-256 mismatch"));
    }

    #[tokio::test]
    async fn native_download_resumes_partial_file() {
        async fn ranged(headers: HeaderMap) -> (StatusCode, HeaderMap, &'static [u8]) {
            assert_eq!(headers.get(header::RANGE).unwrap(), "bytes=2-");
            let mut response_headers = HeaderMap::new();
            response_headers.insert(header::CONTENT_RANGE, "bytes 2-3/4".parse().unwrap());
            (StatusCode::PARTIAL_CONTENT, response_headers, b"uf")
        }

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/model.gguf", get(ranged)))
                .await
                .unwrap();
        });

        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("model.gguf");
        tokio::fs::write(&destination, b"gg").await.unwrap();
        let client = reqwest::Client::new();

        download_url_to_path(
            &client,
            &format!("http://{address}/model.gguf"),
            &destination,
        )
        .await
        .unwrap();

        assert_eq!(tokio::fs::read(destination).await.unwrap(), b"gguf");
    }
}
