#!/usr/bin/env bash
# Pre-commit / CI checks: format, clippy (deny warnings), build, test.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

if ! command -v cargo >/dev/null 2>&1; then
    # Git hooks can run in a non-login shell with a minimal PATH.
    if [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
    fi
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo: command not found. Install Rust/Cargo or add it to PATH." >&2
    exit 127
fi

echo "→ cargo fmt --check"
cargo fmt --all -- --check

echo "→ cargo clippy (deny warnings)"
cargo clippy --all-targets --locked -- -D warnings

echo "→ cargo build"
cargo build --locked

echo "→ cargo test"
cargo test --locked

echo "All pre-commit checks passed."
