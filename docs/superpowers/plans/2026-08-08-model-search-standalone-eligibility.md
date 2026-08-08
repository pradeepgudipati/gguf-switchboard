# Model Search Standalone Eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require standalone model eligibility in addition to memory fit before `ggs models search` reports `SUPPORTED=Yes`.

**Architecture:** Add a pure classifier over Hugging Face hit metadata and GGUF filenames, then compose it with the existing capacity decision in the bounded search-enrichment path. Preserve existing result ordering and failure behavior.

**Tech Stack:** Rust, serde_json, existing Hugging Face search/tree clients and Rust unit tests.

## Global Constraints

- Reject explicit auxiliary tags and architectures from the approved design.
- Use only `drafter` and `speculator` as fallback repository/filename tokens.
- Do not reject a model merely for `dspark`, `speculative-decoding`, `mtp`, `draft`, or `support` in its repository name.
- Keep `SUPPORTED` as Yes/No and retain the existing memory calculation.

---

### Task 1: Standalone eligibility classifier

**Files:**
- Modify: `src/config/models_cmd.rs`

**Interfaces:**
- Produces: `fn is_standalone_model(hit: &Value, entries: &[HfTreeEntry]) -> bool`.

- [ ] Write failing table-driven tests with literal JSON hits for `draft-model`, `auxiliary-model`, `dflash`, `deepseek4-dspark`, architecture strings containing `draft` or `speculator`, and strong repository/file tokens. Include a full `org/DeepSeek-V4-Flash-DSpark-GGUF` target fixture that remains eligible.
- [ ] Run `cargo test config::models_cmd::tests::standalone_eligibility --lib` and confirm RED because the classifier does not exist.
- [ ] Implement normalized exact tag checks, architecture checks, and tokenized fallback matching. Tokenization splits on non-alphanumeric characters and compares whole lowercase tokens.
- [ ] Run the focused test and confirm GREEN.

### Task 2: Compose eligibility with hardware support

**Files:**
- Modify: `src/config/models_cmd.rs`
- Modify: `docs/USAGE.md`

**Interfaces:**
- Consumes: `is_standalone_model(hit, entries)`, `smallest_complete_model_bytes(entries)`, and `is_supported(size, capacity)`.
- Produces: final repository support Boolean requiring both eligibility and capacity.

- [ ] Write a failing pure decision test proving an auxiliary model that fits memory returns `No` while a standalone model with the same size returns `Yes`.
- [ ] Run `cargo test config::models_cmd::tests::repository_support --lib` and confirm RED because the composed helper does not exist.
- [ ] Add `fn repository_is_supported(hit: &Value, entries: &[HfTreeEntry], capacity_bytes: u64) -> bool` and call it from the bounded enrichment closure.
- [ ] Update the CLI note and usage documentation to say `Yes` requires a standalone GGUF and sufficient capacity.
- [ ] Run `cargo test config::models_cmd::tests --lib` and confirm GREEN.

### Task 3: Verification

**Files:**
- Verify all changed files.

**Interfaces:**
- Produces: validated local and live behavior.

- [ ] Run `./precommit.sh` and require all formatting, lint, build, and test checks to pass.
- [ ] Run `cargo run -- models search "DSpark Drafter" --limit 3` and verify auxiliary results show `No`.
- [ ] Run `cargo run -- models search "gemma" --limit 3` and verify normal standalone results retain hardware-based Yes/No values.
- [ ] Run `git diff --check` and inspect `git status --short`, preserving `.zcode/`.
