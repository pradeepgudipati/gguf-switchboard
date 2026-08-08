# Deploy Runtime Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `./deploy.sh` create and preserve user-owned runtime configuration plus a default `~/models` directory.

**Architecture:** Track configuration examples in Git and ignore runtime copies. Add testable deployment helper functions that initialize the model directory and runtime files before systemd installation, while keeping `--refresh-models` as the only model-registry regeneration path.

**Tech Stack:** Bash, Git, TOML, Rust deployment regression scripts, Markdown

## Global Constraints

- Runtime `config.toml`, `models.toml`, and `models.json` are user-owned and ignored by Git.
- Tracked defaults are `config.example.toml` and `models.example.toml`.
- Existing runtime configuration is never overwritten during ordinary deployment.
- `--refresh-models` may regenerate and merge `models.toml`, but never replaces `config.toml`.
- When `MODELS_DIR` is unset, deployment creates and uses `~/models`.
- Preserve `GGUF_SWITCHBOARD_CONFIG_DIR` and comma-separated model-directory behavior.

---

### Task 1: Add failing deployment ownership regression

**Files:**
- Modify: `scripts/test-deploy-models.sh`

**Interfaces:**
- Consumes: `deploy.sh` sourceable helper functions.
- Produces: Regression assertions for ignored runtime files, tracked examples, default model-directory creation, first-run config creation, and second-run preservation.

- [x] Add a shell test that sources `deploy.sh` in library mode, initializes a temporary home and config directory, and asserts the desired first-run and second-run behavior.
- [x] Run `bash scripts/test-deploy-models.sh` and confirm it fails because the example files and initialization helper do not exist.

### Task 2: Implement runtime initialization

**Files:**
- Rename: `config.toml` to `config.example.toml`
- Rename: `models.toml` to `models.example.toml`
- Modify: `.gitignore`
- Modify: `deploy.sh`

**Interfaces:**
- Produces: `initialize_runtime_config`, which resolves/creates the default model directory and copies missing runtime files without replacing existing ones.

- [x] Rename tracked configuration defaults and ignore runtime outputs.
- [x] Add library-mode protection so the regression test can source deployment helpers without running a deployment.
- [x] Implement first-run directory and runtime-file creation using the tracked examples.
- [x] Update model generation fallbacks and runtime references to use `.example.toml` sources.
- [x] Run `bash scripts/test-deploy-models.sh` and confirm it passes.

### Task 3: Update documentation and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/USAGE.md`
- Modify: deployment references found by exact search

**Interfaces:**
- Produces: Documentation that distinguishes tracked examples from generated runtime configuration.

- [x] Replace stale commands and descriptions that treat runtime config as tracked templates.
- [x] Run exact searches for stale references, `git diff --check`, the deployment regression script, and `./precommit.sh`.
- [x] Commit the implementation to `main` with message `fix: preserve deploy runtime configuration`.
