# Deploy First-Install Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee runtime files and `~/models` exist before fallible installation work, and configure a detected llama-server executable without replacing custom settings.

**Architecture:** Move existing runtime initialization immediately after repository update and legacy migration. Add sourceable helpers that resolve llama-server from `PATH` and known installation locations, then update only an empty or untouched example value in `models.toml`.

**Tech Stack:** Bash, TOML, shell regression tests

## Global Constraints

- Create runtime files and the default model directory before dependency installation, compilation, or systemd operations.
- Preserve existing `config.toml`, `models.toml`, and custom llama-server paths.
- Store the llama-server executable in `models.toml`, which owns the supported `defaults.llama_server` setting.
- Search `PATH`, `/usr/local/bin`, `/usr/bin`, `~/llama.cpp/build/bin`, and `/opt/llama.cpp/build/bin`.

---

### Task 1: Reproduce missing first-install behavior

- [x] Extend `scripts/test-deploy-models.sh` with ordering, executable discovery, default replacement, and custom-path preservation assertions.
- [x] Run the regression and confirm it fails because llama-server resolution is missing and initialization occurs too late.

### Task 2: Implement the correction

- [x] Add `resolve_llama_server` and `configure_llama_server` helpers to `deploy.sh`.
- [x] Move legacy migration and runtime initialization immediately after Git update.
- [x] Configure the resolved executable before model discovery while retaining warnings when none is found.
- [x] Run the focused regression and confirm it passes.

### Task 3: Verify and commit

- [x] Update README and configuration documentation with discovery locations and ownership behavior.
- [x] Run shell syntax checks, `git diff --check`, the deployment regression, and `./precommit.sh`.
- [x] Commit to `main` as `fix: initialize deploy runtime before installation`.
