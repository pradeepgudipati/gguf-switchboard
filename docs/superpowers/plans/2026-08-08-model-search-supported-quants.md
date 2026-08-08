# Model Search Supported Quants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show fitting quantizations in an aligned search table and print a balanced sample pull command.

**Architecture:** Extract complete named model options once from each repository tree, filter them by standalone eligibility and capacity, and carry a row assessment into a dynamic-width renderer. Derive the recommendation from the same assessment so support, quants, and the command cannot disagree.

**Tech Stack:** Rust, serde_json, futures, existing Hugging Face tree API and unit tests.

## Global Constraints

- `SUPPORTED=Yes` exactly when at least one named standalone quant fits.
- Prefer `Q4_K_M`; otherwise recommend the largest fitting option.
- Preserve complete split-file sizing, auxiliary exclusion, 20 percent headroom, bounded concurrency, and result order.
- Render visible ` | ` delimiters with `QUANT` last and no truncation.
- Print total system RAM, NVIDIA VRAM, and combined capacity before the table from the same MiB inputs used for support decisions.

---

### Task 1: Complete named model options

**Files:** Modify and test `src/config/models_cmd.rs`.

**Interfaces:** Produce `ModelOption { quant: String, bytes: u64 }` and `complete_model_options(entries: &[HfTreeEntry]) -> Vec<ModelOption>`.

- [ ] Add failing tests proving hyphen- and dot-separated quant extraction, strict quant-token validation, auxiliary DSpark/MTP file exclusion, complete split aggregation, incomplete split rejection, unnamed exclusion, and smallest-size deduplication.
- [ ] Run `cargo test config::models_cmd::tests::complete_model_options --lib` and confirm RED.
- [ ] Implement option extraction and sort by bytes then quant.
- [ ] Run the focused tests and confirm GREEN.

### Task 2: Per-row assessment and recommendation

**Files:** Modify and test `src/config/models_cmd.rs`.

**Interfaces:** Produce `SearchAssessment { supported: bool, quants: Vec<ModelOption>, recommended_quant: Option<String> }` and `assess_repository(hit, entries, capacity_bytes)`.

- [ ] Add failing tests for capacity filtering, auxiliary exclusion, `Q4_K_M` preference, and largest-fitting fallback.
- [ ] Run `cargo test config::models_cmd::tests::search_assessment --lib` and confirm RED.
- [ ] Implement assessment and replace the bounded Boolean enrichment with ordered assessments.
- [ ] Run focused model-command tests and confirm GREEN.

### Task 3: Hardware header, aligned rendering, and sample command

**Files:** Modify and test `src/config/models_cmd.rs`; modify `docs/USAGE.md`.

**Interfaces:** Render dynamic core-column widths and produce `sample_pull_command(hits, assessments) -> Option<String>`.

- [ ] Add failing tests proving the hardware header renders one-decimal binary GiB values from RAM and VRAM MiB inputs, long and short repository rows share identical delimiter positions, `QUANT` renders last, unsupported rows show `-`, Q4 recommendation renders correctly, and all-auxiliary results omit the command.
- [ ] Run `cargo test config::models_cmd::tests::search_output --lib` and confirm RED.
- [ ] Implement the hardware header, dynamic-width row rendering, and the `Try: ggs models pull ...` line.
- [ ] Update usage documentation for fitting quants, clean columns, and recommendation selection.
- [ ] Run all model-command tests and confirm GREEN.

### Task 4: Verification

**Files:** Verify all changed files.

- [ ] Run `./precommit.sh` and require all checks to pass.
- [ ] Run `cargo run -- models search "deepseek" --limit 10` and verify the hardware header, clean aligned columns, fitting quants, and a pull command.
- [ ] Run `cargo run -- models search "DSpark Drafter" --limit 5` and verify `No`, `-`, and no pull command when all rows are auxiliary.
- [ ] Run `cargo run -- models files "nomic-ai/nomic-embed-text-v1.5-GGUF"` and verify dot-separated quant labels, including `Q4_K_M`, are recognized.
- [ ] Run `git diff --check` and inspect status while preserving `.zcode/`.
