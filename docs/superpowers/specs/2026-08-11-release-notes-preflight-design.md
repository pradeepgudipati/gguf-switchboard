# Release Notes Preflight Design

## Goal

Prevent a version tag from consuming all four release build jobs and then failing during publication because its release-note file is missing, empty, or inconsistent with the crate version.

## Scope

Add the published `v0.1.4` notes to `releases/v0.1.4.md`. Add one tag-only GitHub Actions preflight job and a locally executable regression test. Do not rewrite the existing `v0.1.4` tag or attempt to change the historical failed workflow run.

## Workflow design

Introduce a `release-preflight` job in `.github/workflows/ci.yml`. It runs only for `refs/tags/v*`, checks out the tagged commit, and invokes a repository script with the tag name. The script verifies that:

1. The tag has the `vMAJOR.MINOR.PATCH` form.
2. `releases/<tag>.md` exists and is non-empty.
3. The package version in `Cargo.toml` equals the tag version without the leading `v`.

`build-release` will depend on both `ci` and `release-preflight`. A metadata failure therefore stops all platform builds. The existing release job and its `body_path` remain unchanged.

## Validation script

Add `scripts/check-release-preflight.sh`. It accepts one tag argument, resolves files relative to the repository root rather than the caller's working directory, emits a specific error for each invalid condition, and exits nonzero on failure. It must not call GitHub APIs or require network access.

## Regression test

Add `scripts/test-release-preflight.sh`. The test creates an isolated temporary fixture and exercises the real validation script against:

- matching version and non-empty notes, which succeeds;
- missing notes, which fails;
- empty notes, which fails;
- a tag and `Cargo.toml` version mismatch, which fails;
- a malformed tag, which fails.

The test must clean up its fixture and must not modify the working checkout. Implementation follows red-green TDD: add the regression test first, confirm that it fails because the validator is absent, then add the minimal validator and workflow wiring.

## Verification

Run the focused regression test, validate the workflow syntax structurally, run `git diff --check`, and run the full `./precommit.sh` gate. Confirm that `releases/v0.1.4.md` is non-empty and matches the published release body.
