# Release Notes Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fail tag workflows before platform builds when release notes or the crate version do not match the tag, and preserve the published v0.1.4 notes in the repository.

**Architecture:** A standalone POSIX shell validator owns release metadata checks. A shell regression test executes that validator against isolated fixture repositories, while a tag-only GitHub Actions preflight job gates every release build.

**Tech Stack:** POSIX shell, GitHub Actions YAML, Cargo manifest metadata, Markdown

## Global Constraints

- Do not rewrite the existing `v0.1.4` tag or alter the historical failed workflow run.
- Resolve validator inputs relative to a supplied repository root so tests never modify the checkout.
- Keep the existing release action and `body_path: releases/${{ github.ref_name }}.md` contract.
- Run focused red-green tests, workflow structural validation, `git diff --check`, and `./precommit.sh`.

---

### Task 1: Release metadata validator

**Files:**
- Create: `scripts/test-release-preflight.sh`
- Create: `scripts/check-release-preflight.sh`

**Interfaces:**
- Consumes: `scripts/check-release-preflight.sh <tag> [repository-root]`
- Produces: exit zero for valid metadata; a specific stderr message and nonzero exit for malformed tags, missing or empty notes, and version mismatches

- [ ] **Step 1: Write the failing regression test**

Create an isolated fixture with `Cargo.toml` and `releases/`, invoke the validator for valid and invalid cases, and assert the expected status and message for each case.

- [ ] **Step 2: Run the focused test and verify red**

Run: `./scripts/test-release-preflight.sh`
Expected: FAIL because `scripts/check-release-preflight.sh` does not exist.

- [ ] **Step 3: Implement the minimal validator**

Validate the `vMAJOR.MINOR.PATCH` tag form, locate `releases/<tag>.md`, reject missing and whitespace-only notes, extract the first package `version = "..."` from `Cargo.toml`, and compare it to the tag without `v`.

- [ ] **Step 4: Run the focused test and verify green**

Run: `./scripts/test-release-preflight.sh`
Expected: PASS for all five scenarios.

### Task 2: Notes and workflow gate

**Files:**
- Create: `releases/v0.1.4.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `scripts/check-release-preflight.sh "${GITHUB_REF_NAME}" "${GITHUB_WORKSPACE}"`
- Produces: a successful `release-preflight` job required by `build-release`

- [ ] **Step 1: Add the published v0.1.4 release body**

Copy the current GitHub release body exactly into `releases/v0.1.4.md` and confirm it is non-empty.

- [ ] **Step 2: Add the tag-only preflight job**

Checkout the tag, run the validator, and change `build-release.needs` from `ci` to `[ci, release-preflight]`.

- [ ] **Step 3: Validate workflow structure**

Parse `.github/workflows/ci.yml` with an available YAML parser and assert that `build-release` requires both jobs and the preflight invokes the validator.

- [ ] **Step 4: Verify current tag metadata through the validator**

Run: `./scripts/check-release-preflight.sh v0.1.4 .`
Expected: PASS and a message identifying version `0.1.4` and `releases/v0.1.4.md`.

### Task 3: Full verification and delivery

**Files:**
- Verify all files changed in Tasks 1 and 2

**Interfaces:**
- Consumes: repository quality gate and Git history
- Produces: committed and pushed release-preflight fix on `main`

- [ ] **Step 1: Run focused and static checks**

Run `./scripts/test-release-preflight.sh`, workflow structural assertions, `git diff --check`, and inspect `git diff`.

- [ ] **Step 2: Run the full repository gate**

Run: `./precommit.sh`
Expected: formatting, clippy, build, all tests, and doc tests pass.

- [ ] **Step 3: Commit implementation**

Stage only the release notes, validator, regression test, workflow, and implementation plan. Commit as `ci: validate release notes before builds`.

- [ ] **Step 4: Push and verify remote state**

Push `main`, confirm `main` equals `origin/main`, and inspect the new GitHub Actions run without claiming remote completion while it remains pending.
