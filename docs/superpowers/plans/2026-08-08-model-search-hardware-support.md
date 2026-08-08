# Hardware-Aware Model Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an estimated `SUPPORTED` Yes/No column to `ggs models search` based on the smallest complete GGUF option and total host RAM plus NVIDIA VRAM.

**Architecture:** Extend the existing hardware probes and keep sizing decisions in pure helpers. Enrich Hugging Face search hits with bounded concurrent repository-tree requests, then render the existing rows with the support estimate while isolating per-repository failures.

**Tech Stack:** Rust, Tokio, futures, reqwest, Hugging Face model/tree APIs, existing platform memory probes.

## Global Constraints

- Capacity uses total system RAM plus total NVIDIA VRAM, not currently free memory.
- Reserve exactly 20 percent runtime headroom.
- `SUPPORTED` contains only `Yes` or `No`.
- CPU-only systems use system RAM alone.
- A repository-tree failure does not fail the complete search.
- Add no CLI flag or configuration field.

---

### Task 1: Total NVIDIA VRAM probe

**Files:**
- Modify: `src/gpu.rs`

**Interfaces:**
- Produces: `pub fn total_vram_mb() -> Option<u64>` and `pub fn parse_nvidia_smi_total_mb(stdout: &str) -> Option<u64>`.

- [ ] **Step 1: Write failing parser tests**

Add tests asserting `parse_nvidia_smi_total_mb("8192\n") == Some(8192)`, `parse_nvidia_smi_total_mb("8192\n24576\n") == Some(32768)`, malformed lines are ignored when a valid line exists, and entirely unusable output returns `None`.

- [ ] **Step 2: Verify RED**

Run `cargo test gpu::tests::parse_nvidia_smi_total --lib` and confirm compilation fails because the parser does not exist.

- [ ] **Step 3: Implement the probe**

Run `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`, reject a failing process, parse every positive MiB line, and return their checked sum. Keep the existing first-GPU free-memory probe unchanged.

- [ ] **Step 4: Verify GREEN**

Run `cargo test gpu::tests --lib` and confirm all GPU tests pass.

### Task 2: Complete model-option sizing and support estimate

**Files:**
- Modify: `src/config/models_cmd.rs`

**Interfaces:**
- Consumes: `HfTreeEntry { path, size, .. }`.
- Produces: `fn smallest_complete_model_bytes(entries: &[HfTreeEntry]) -> Option<u64>` and `fn is_supported(model_bytes: Option<u64>, capacity_bytes: u64) -> bool`.

- [ ] **Step 1: Write failing sizing tests**

Use literal `HfTreeEntry` fixtures to prove that the helper selects a 4 GiB normal model over an 8 GiB model, excludes a smaller `mmproj`, sums `model-00001-of-00002.gguf` and `model-00002-of-00002.gguf`, and rejects an incomplete shard group.

- [ ] **Step 2: Verify RED**

Run `cargo test config::models_cmd::tests::smallest_complete --lib` and confirm compilation fails because the sizing helper does not exist.

- [ ] **Step 3: Implement minimal option grouping**

Classify normal GGUFs with `is_model_gguf`. Parse the terminal `-NNNNN-of-NNNNN.gguf` shard marker, group by the preceding path, retain only groups containing every index from 1 through the declared count, sum with checked arithmetic, and return the smallest normal or complete grouped option.

- [ ] **Step 4: Verify sizing GREEN**

Run `cargo test config::models_cmd::tests::smallest_complete --lib` and confirm the sizing tests pass.

- [ ] **Step 5: Write failing support-boundary tests**

Assert that a 100-byte model fits capacity 120, fails capacity 119, `None` is `No`, and a CPU-only nonzero capacity follows the same rule.

- [ ] **Step 6: Verify support RED**

Run `cargo test config::models_cmd::tests::support_estimate --lib` and confirm compilation fails because the decision helper does not exist.

- [ ] **Step 7: Implement support arithmetic**

Implement `model_bytes * 120 <= capacity_bytes * 100` with `u128` operands so valid `u64` metadata cannot overflow.

- [ ] **Step 8: Verify support GREEN**

Run `cargo test config::models_cmd::tests --lib` and confirm all model-command unit tests pass.

### Task 3: Search enrichment and rendering

**Files:**
- Modify: `src/config/models_cmd.rs`
- Modify: `docs/USAGE.md`

**Interfaces:**
- Consumes: `memory::check_memory()`, `gpu::total_vram_mb()`, `hf_download::fetch_repo_tree()` and the Task 2 helpers.
- Produces: search rows containing `SUPPORTED` and a concise estimation note.

- [ ] **Step 1: Write a failing render test**

Extract row rendering into a function that accepts search hits and their Boolean estimates. Assert the output header contains `SUPPORTED`, a fitting row contains `Yes`, and a failed-size row contains `No`.

- [ ] **Step 2: Verify render RED**

Run `cargo test config::models_cmd::tests::search_table --lib` and confirm compilation fails because the renderer does not exist.

- [ ] **Step 3: Implement bounded enrichment and rendering**

Calculate bytes as `(total_ram_mb + total_vram_mb) * 1024 * 1024` with saturating arithmetic. Use `futures::stream::iter(...).buffer_unordered(4)` to fetch trees. Preserve original result ordering by carrying each index, convert each tree into `smallest_complete_model_bytes`, map failures to `false`, and render `SUPPORTED` after `SIZE`.

- [ ] **Step 4: Add the estimation note and documentation**

Print `Supported is estimated from the smallest complete GGUF, total RAM + NVIDIA VRAM, and 20% runtime headroom.` after the table. Update the model-search section of `docs/USAGE.md` with the same semantics and note that context/runtime settings can still prevent loading.

- [ ] **Step 5: Verify focused GREEN**

Run `cargo test config::models_cmd::tests gpu::tests --lib` and confirm all focused tests pass.

### Task 4: Full verification and live search

**Files:**
- Verify all modified files.

**Interfaces:**
- Produces: validated CLI behavior against live Hugging Face data.

- [ ] **Step 1: Run the repository gate**

Run `./precommit.sh` and require formatting, Clippy, build, unit, integration, response, scheduler, and doc tests to pass.

- [ ] **Step 2: Run the real command**

Run `cargo run -- models search "gemma" --limit 3` and verify every row retains the existing metadata columns and includes `SUPPORTED=Yes` or `No`, followed by the estimation note.

- [ ] **Step 3: Inspect the final diff**

Run `git diff --check`, `git status --short`, and `git diff --stat`. Confirm `.zcode/` remains untouched and only the plan, hardware probe, model command, tests, and usage documentation changed.
