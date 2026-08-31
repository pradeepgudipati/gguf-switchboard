#!/usr/bin/env bash
# Hosted CI checks. The full local gate remains in precommit.sh.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "→ cargo fmt --check"
cargo fmt --all -- --check

echo "→ cargo clippy (deny warnings)"
cargo clippy --all-targets --locked -- -D warnings

echo "→ cargo test"
cargo test --locked

echo "All hosted CI checks passed."
