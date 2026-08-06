# aria2c Model Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gguf-switchboard models pull` automatically accelerate public GGUF downloads with aria2c, fall back securely to the native downloader, validate completed files, refresh a running server, and install aria2 through `deploy.sh`.

**Architecture:** Keep Hugging Face URL construction and download execution in `src/config/hf_download.rs`. The CLI handler in `src/config/models_cmd.rs` parses user options, selects the destination, validates and registers the file, and requests a live registry refresh. `deploy.sh` supplies aria2 on apt-based installations while runtime detection preserves portability.

**Tech Stack:** Rust 2024, Tokio process execution, reqwest, sha2, aria2c, Bash deployment tests.

## Global Constraints

- Preserve the existing `models pull <repo-id> --quant <quant>` syntax and destination resolution.
- Default to 8 aria2 connections and accept `--connections <N>`.
- Use aria2 only for public downloads when `HF_TOKEN` is absent; authenticated downloads use the native reqwest path.
- Invoke aria2 directly without a shell and never expose or print `HF_TOKEN`.
- Do not fall back to a second download after aria2 starts and fails.
- Register only after size, SHA-256 when available, and GGUF validation succeed.
- Refresh the running server after registration; refresh failure is a warning, not a failed pull.
- Preserve the deployment alias `ggs='gguf-switchboard'`.

---

### Task 1: Accelerated and verified download boundary

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `src/config/hf_download.rs`

**Interfaces:**
- Produces: `pub async fn download_file_auto(client: &reqwest::Client, repo: &str, entry: &HfTreeEntry, dest_dir: &Path, connections: u16) -> Result<PathBuf, RuntimeError>`
- Produces: `fn aria2_args(url: &str, filename: &str, dest_dir: &Path, connections: u16, checksum: Option<&str>) -> Vec<OsString>`
- Produces: `async fn verify_download(path: &Path, expected_size: u64, sha256: Option<&str>) -> Result<(), RuntimeError>`

- [ ] **Step 1: Add failing unit tests for aria2 arguments and validation**

Add tests in `src/config/hf_download.rs` that assert the argument vector contains `--continue=true`, `--split=8`, `--max-connection-per-server=8`, `--min-split-size=64M`, the exact destination and filename, Linux `--file-allocation=falloc`, and `--checksum=sha-256=<oid>` when an LFS digest exists. Add asynchronous tests where a temporary file with the wrong size and wrong SHA-256 returns `ConfigError`.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `cargo test --locked config::hf_download::tests -- --nocapture`

Expected: compilation fails because `aria2_args` and `verify_download` do not exist.

- [ ] **Step 3: Implement downloader selection and verification**

Add `sha2 = "0.10"` to dependencies. Build aria2 arguments as `OsString` values and invoke `tokio::process::Command::new("aria2c")` directly. Detect aria2 availability before download. Use aria2 only when it exists and `HF_TOKEN` is absent; otherwise call `download_file`. Return an error on non-zero aria2 status. Verify the final byte length and stream SHA-256 comparison after either path.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `cargo test --locked config::hf_download::tests -- --nocapture`

Expected: all Hugging Face download tests pass.

- [ ] **Step 5: Commit the download boundary**

```bash
git add Cargo.toml Cargo.lock src/config/hf_download.rs
git commit -m "feat: accelerate model downloads with aria2"
```

### Task 2: Pull options and live refresh

**Files:**
- Modify: `src/config/models_cmd.rs`
- Modify: `src/main.rs`

**Interfaces:**
- Consumes: `hf_download::download_file_auto(...)`
- Produces: `fn parse_connections(value: &str) -> Result<u16, &'static str>` accepting values from 1 through 16.
- Produces: `async fn refresh_running_server(client: &reqwest::Client, config_path: &Path) -> Result<(), RuntimeError>`.

- [ ] **Step 1: Add failing tests for connection parsing and refresh address construction**

Add unit tests in `src/config/models_cmd.rs` that accept `1`, `8`, and `16`, reject `0`, `17`, and non-numeric values, and convert `bind = "0.0.0.0:9090"` into `http://127.0.0.1:9090/v1/models/refresh` for the client request.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `cargo test --locked config::models_cmd::tests -- --nocapture`

Expected: compilation fails because connection parsing and refresh URL construction are absent.

- [ ] **Step 3: Implement CLI parsing, automatic download, and refresh**

Parse `--connections N` with a default of 8 and update the usage string in `src/main.rs`. Replace the direct `download_file` call with `download_file_auto`. After registry persistence, read `bind` from `config.toml`, normalize unspecified bind addresses to loopback, and POST `/v1/models/refresh`. Print a success line for HTTP success. Print a warning with a restart instruction for missing configuration, connection failure, or non-success status, then return success from the completed pull.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `cargo test --locked config::models_cmd::tests -- --nocapture`

Expected: all model command tests pass.

- [ ] **Step 5: Commit CLI and refresh behavior**

```bash
git add src/config/models_cmd.rs src/main.rs
git commit -m "feat: refresh models after pull"
```

### Task 3: Deployment dependency and regression gate

**Files:**
- Modify: `deploy.sh`
- Modify: `scripts/test-deploy-models.sh`
- Modify: `README.md`

**Interfaces:**
- Consumes: apt package list in `deploy.sh`.
- Produces: apt-based installations with `aria2c` available and documentation for `--connections` plus native fallback.

- [ ] **Step 1: Extend the failing deployment regression test**

Add assertions to `scripts/test-deploy-models.sh` that `APT_PKGS` contains `aria2`, the `ggs` alias remains present, and the legacy `gs` alias remains absent.

- [ ] **Step 2: Run the deployment test and confirm RED**

Run: `bash scripts/test-deploy-models.sh`

Expected: failure because `deploy.sh` does not include the `aria2` package.

- [ ] **Step 3: Install aria2 and document accelerated pulls**

Add `aria2` to `APT_PKGS`. Update the model-pull README example to show the unchanged default command and an optional `--connections 8` example. State that public downloads use aria2 when installed and authenticated or aria2-unavailable downloads use the native path.

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
bash scripts/test-deploy-models.sh
cargo fmt --check
./precommit.sh
git diff --check
```

Expected: deployment validation passes, formatting passes, all pre-commit checks pass, and the diff has no whitespace errors.

- [ ] **Step 5: Commit deployment and documentation**

```bash
git add deploy.sh scripts/test-deploy-models.sh README.md
git commit -m "build: install aria2 for model downloads"
```
