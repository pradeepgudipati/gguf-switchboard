# Quant Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement exact, family, alias, and hardware-aware automatic quant selection for `models pull`.

**Architecture:** Extract candidate resolution into a pure selector in `src/config/models_cmd.rs`. Reuse filename quant parsing, complete-model filtering, system RAM detection, NVIDIA VRAM detection, and the existing 20 percent headroom calculation.

**Tech Stack:** Rust, Cargo test framework, existing Hugging Face tree-entry and hardware helpers.

## Global Constraints

- Exact quant matching is case-insensitive.
- `Q4` preference order is `Q4_K_M`, `Q4_K_S`, `Q4_0`, `Q4_1`.
- `K_M` means `Q4_K_M`; it does not silently select another quality level.
- `auto` selects the largest complete standalone quant that fits with 20 percent runtime headroom.
- Add no dependencies.

---

### Task 1: Pure quant selector

**Files:**
- Modify: `src/config/models_cmd.rs`
- Test: inline `tests` module in `src/config/models_cmd.rs`

**Interfaces:**
- Consumes: `&[&HfTreeEntry]`, selector text, available memory in MB.
- Produces: a selected `&HfTreeEntry` or a structured selection error.

- [ ] Write failing tests for exact `Q4_K_M`, family `Q4`, alias `K_M`, automatic largest-fitting selection, and no-fitting automatic selection.
- [ ] Run the focused tests and confirm each fails because the selector does not exist.
- [ ] Implement normalized candidate extraction and deterministic selector resolution.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Pull-command integration

**Files:**
- Modify: `src/config/models_cmd.rs`

**Interfaces:**
- Consumes: parsed `--quant` and detected hardware capacity.
- Produces: the existing selected-file download flow or actionable CLI errors.

- [ ] Replace substring matching in `cmd_pull` with the pure selector.
- [ ] Preserve available-file diagnostics for invalid and ambiguous selectors.
- [ ] Run formatting, the focused test suite, and the repository validation gate.
- [ ] Commit and push the verified changes to `main`.
