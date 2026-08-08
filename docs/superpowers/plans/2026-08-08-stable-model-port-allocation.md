# Stable Model Port Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assign every registered model a deterministic consecutive backend port beginning at the configured base, whose new default is `18081`.

**Architecture:** Add one registry-local allocator that overwrites model entry ports in normalized alias order using checked `u16` arithmetic. Call it when discovery produces a registry and when expansion produces active runtime entries, so refresh artifacts and runtime backend URLs use the same deterministic contract.

**Tech Stack:** Rust, serde/TOML registry configuration, built-in unit tests, repository `precommit.sh` gate.

## Global Constraints

- The implicit default backend base port is exactly `18081`.
- Registry models receive consecutive ports in deterministic normalized order.
- Existing per-model `port` values do not override normalized allocation.
- A configured `defaults.base_port` remains the sequence starting point.
- Port overflow returns `RuntimeError::ConfigError`; it never saturates or duplicates a port.
- Preserve unrelated working-tree changes.

---

### Task 1: Deterministic registry port allocation

**Files:**
- Modify: `src/config/models_registry.rs`
- Test: `src/config/models_registry.rs`

**Interfaces:**
- Consumes: `RegistryEntry`, `RegistryDefaults::base_port`, `RuntimeError::ConfigError`.
- Produces: `assign_consecutive_ports(entries: &mut [RegistryEntry], base_port: u16) -> Result<(), RuntimeError>`.

- [ ] **Step 1: Write failing tests for the allocation contract**

Add unit tests proving that `RegistryDefaults::default().base_port == 18081`; entries sorted as `alpha`, `beta`, `gamma` receive `18081`, `18082`, `18083`; existing entry ports are overwritten; repeated assignment is stable; and assigning two entries from base `u16::MAX` returns `RuntimeError::ConfigError`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cargo test --locked config::models_registry::tests::model_ports -- --nocapture`

Expected: FAIL because the default remains `8081` and `assign_consecutive_ports` does not exist.

- [ ] **Step 3: Implement checked consecutive allocation**

Change `default_base_port()` to return `18081`. Add `assign_consecutive_ports`, using `u16::try_from(index)` and `base_port.checked_add(offset)`; return a configuration error containing the base and model count when the range is exhausted. Set every entry's `port` to the computed value.

Call the allocator after `dedupe_registry_entries` in `discover_with_merge`, and after disabled entries are removed and entries are deduplicated in `expand`. In `expand`, consume the normalized `entry.port` rather than using saturating index arithmetic.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `cargo test --locked config::models_registry::tests::model_ports -- --nocapture`

Expected: all model-port tests PASS.

- [ ] **Step 5: Run the complete model-registry test module**

Run: `cargo test --locked config::models_registry::tests -- --nocapture`

Expected: all model-registry tests PASS.

### Task 2: Configuration examples, documentation, and final verification

**Files:**
- Modify: `models.example.toml`
- Modify: `docs/CONFIGURATION.md`
- Modify: `src/config/models_registry.rs`

**Interfaces:**
- Consumes: the `18081` default and normalization behavior from Task 1.
- Produces: accurate operator guidance for discovery, refresh, and runtime port assignment.

- [ ] **Step 1: Update configuration comments and documentation**

Change the example `base_port` from `8081` to `18081`. Replace the port table with `18081`, `18082`, and `18083`. Document that discovery and refresh rewrite existing model ports and that `defaults.base_port`, rather than per-model pins, controls the sequence.

Update the `RegistryEntry::port` Rust doc comment and `discover_with_merge` method documentation so neither claims per-model ports are preserved or override allocation.

- [ ] **Step 2: Check formatting and documentation consistency**

Run: `rg -n "base 8081|base_port = 8081|Override the auto-assigned backend port|port.*preserved" models.example.toml docs/CONFIGURATION.md src/config/models_registry.rs`

Expected: no stale port-contract matches.

- [ ] **Step 3: Run the full repository gate**

Run: `./precommit.sh`

Expected: formatting, Clippy, build, unit, integration, Responses, scheduler, and doc tests all PASS.

- [ ] **Step 4: Review the scoped diff**

Run: `git diff --check && git diff -- src/config/models_registry.rs models.example.toml docs/CONFIGURATION.md`

Expected: no whitespace errors; diff contains only the port allocator, regression tests, new default, and matching documentation.

- [ ] **Step 5: Commit the implementation**

Run: `git add src/config/models_registry.rs models.example.toml docs/CONFIGURATION.md docs/superpowers/plans/2026-08-08-stable-model-port-allocation.md && git commit -m "fix: stabilize registered model ports"`

Expected: commit succeeds without staging unrelated working-tree changes.
