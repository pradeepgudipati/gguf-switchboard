# Empty First Install and Download Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Complete installation cleanly without models and recover from Hugging Face aria2c range failures.

**Architecture:** Detect whether the runtime registry or configured directories contain model candidates before starting systemd. When none exist, leave the service disabled and print concrete setup commands. Retain aria2c acceleration, but avoid preallocation and resume partial downloads through the native HTTP client when aria2c fails.

**Tech Stack:** Bash, Rust, reqwest, aria2c, Axum test server

## Global Constraints

- An empty first installation exits successfully without starting a restart-looping service.
- Helper text includes concrete `models search`, `models pull`, and `--refresh-models` commands.
- aria2c failures fall back to the native downloader.
- Native fallback resumes partial data with an HTTP Range request.
- Download size, checksum, and GGUF validation remain mandatory.

---

### Task 1: Empty first-install completion

- [x] Add failing regression assertions for empty and populated model directories plus helper output.
- [x] Add model-candidate detection and an empty-install helper path before systemd startup.
- [x] Verify the deployment regression passes.

### Task 2: aria2c fallback

- [x] Add a failing HTTP Range resume test using a local Axum server.
- [x] Add resumable native downloading and aria2c failure fallback.
- [x] Disable aria2c file preallocation so partial file length remains resumable.
- [x] Verify focused download tests pass.

### Task 3: Documentation and completion

- [x] Document empty-install behavior and aria2c fallback.
- [x] Run the full precommit gate.
- [x] Commit the combined fix to `main`.
