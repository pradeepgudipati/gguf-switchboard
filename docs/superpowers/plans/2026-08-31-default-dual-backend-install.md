# Default Dual-Backend Installation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `./deploy.sh` install llama.cpp, vLLM, and gguf-switchboard by default, with a three-command README installation path and accurate dual-backend documentation.

**Architecture:** Keep backend installation in focused scripts: the existing llama.cpp updater remains authoritative, while a new idempotent vLLM setup script owns a system-visible uv project. `deploy.sh` orchestrates both by default, supports explicit skip flags, preserves user registry values, and accepts either GGUF or vLLM sources before starting the service.

**Tech Stack:** Bash, systemd, uv, vLLM, Rust 2024, Markdown

**Spec:** `docs/superpowers/specs/2026-08-31-default-dual-backend-install-design.md`

## Global Constraints

- Plain `./deploy.sh` installs or updates llama.cpp and vLLM.
- The README installation block contains exactly clone, `cd`, and deploy commands.
- `--skip-llama-cpp` and `--skip-vllm` are opt-outs, not defaults.
- Runtime configuration remains user-owned; only missing deployment-managed paths are added.
- Automated Hugging Face setup accepts Safetensors and never enables `trust_remote_code`.
- Tests must not download or compile llama.cpp or vLLM.

---

### Task 1: Test and implement default backend setup

**Files:**
- Create: `scripts/setup-vllm.sh`
- Create: `vllm-runtime/pyproject.toml`
- Modify: `deploy.sh`
- Modify: `scripts/test-deploy-models.sh`

**Interfaces:**
- Produces: `setup_vllm <project_dir>` and `vllm_ready <project_dir>` shell functions.
- Produces: deploy flags `--skip-llama-cpp` and `--skip-vllm`.
- Produces: registry defaults `vllm_command = "/usr/local/bin/uv"` and `vllm_project = "/opt/gguf-switchboard/vllm-runtime"` when absent.

- [ ] **Step 1: Add failing deployment contract assertions**

Add assertions that `deploy.sh` invokes both setup scripts by default, exposes both skip flags, sources `scripts/setup-vllm.sh`, preserves existing vLLM defaults, and treats a `vllm_file` or `vllm_hf_repo` registry entry as a model candidate. Assert that the empty-install help prints both pull commands.

- [ ] **Step 2: Run the deployment test and confirm RED**

Run: `./scripts/test-deploy-models.sh`

Expected: FAIL because `scripts/setup-vllm.sh`, skip flags, and vLLM help do not exist.

- [ ] **Step 3: Add the isolated vLLM runtime project**

Create `vllm-runtime/pyproject.toml` with Python `>=3.10,<3.15` and `vllm>=0.28,<0.29`, matching the reviewed current release lane.

- [ ] **Step 4: Implement the idempotent vLLM setup helper**

The sourceable helper must use overridable `UV_BIN`, `VLLM_PROJECT_DIR`, and `UV_INSTALL_URL`, install uv only when absent, run `uv sync --project "$project_dir"`, and verify `uv run --project "$project_dir" vllm --version`. Its executable entry point calls `setup_vllm`.

- [ ] **Step 5: Wire both backends into deploy**

Parse both skip flags. After prerequisites and project sync, invoke `SKIP_SERVICE=1 scripts/update-llama-cpp.sh` unless skipped and `setup_vllm "$VLLM_PROJECT_DIR"` unless skipped. Add missing registry defaults without overwriting existing values. Validate only the backend executables required by registered sources.

- [ ] **Step 6: Run deployment tests and confirm GREEN**

Run: `./scripts/test-deploy-models.sh`

Expected: PASS without network downloads or backend compilation.

### Task 2: Reorganize and correct product documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/USAGE.md`
- Modify: `docs/COMPATIBILITY.md`
- Modify: `docs/COMPARISON.md`

**Interfaces:**
- Consumes: dual-backend deployment and registry behavior from Task 1.
- Produces: a README ordered as top features, installation, and details.

- [ ] **Step 1: Add failing documentation assertions**

Extend `scripts/test-deploy-models.sh` to assert that the README's Top features heading precedes Installation, Installation precedes Details, the installation code block has the three required commands, and both `models pull` forms appear.

- [ ] **Step 2: Run the deployment test and confirm RED**

Run: `./scripts/test-deploy-models.sh`

Expected: FAIL on the current heading order and four-line installation path.

- [ ] **Step 3: Rewrite README entry path**

Preserve the user's concurrent dual-backend copy while consolidating the opening into `Top features`, `Installation`, and `Details`. Move lengthy prerequisites and alternatives below Details. State default dual installation and skip flags accurately.

- [ ] **Step 4: Correct detailed docs**

Describe the backend trait, vLLM fields, Safetensors pull flow, backend-dependent compatibility, one-slot selection/fallback, and comparison boundaries. Remove claims that the current product is llama.cpp-only.

- [ ] **Step 5: Run documentation and deployment checks**

Run: `./scripts/test-deploy-models.sh && git diff --check`

Expected: PASS.

### Task 3: Full verification and durable-state review

**Files:**
- Modify only files required to fix implementation-caused failures.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: verified branch ready for review.

- [ ] **Step 1: Run the mandatory completion gate**

Run:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
./scripts/test-deploy-models.sh
```

Expected: all commands PASS.

- [ ] **Step 2: Inspect the final diff and repository status**

Run: `git diff --check && git status --short && git diff --stat`

Expected: no whitespace errors and only scoped files changed.

- [ ] **Step 3: Review durable project memory**

Search Context Harbor project `GGUF-Switchboard` for the deployment decision. Record the default dual-backend installation contract if not already covered; otherwise report it as already recorded. If the MCP tools remain unavailable, report that status without substituting another memory provider.
