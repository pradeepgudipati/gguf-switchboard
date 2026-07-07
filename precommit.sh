#!/usr/bin/env bash
# Pre-commit / CI checks: format, clippy (deny warnings), build, test.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "→ cargo fmt --check"
cargo fmt --all -- --check

echo "→ cargo clippy (deny warnings)"
cargo clippy --all-targets --locked -- -D warnings

echo "→ cargo build"
cargo build --locked

echo "→ cargo test"
cargo test --locked

echo "All pre-commit checks passed."
