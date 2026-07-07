#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git config core.hooksPath .githooks
chmod +x .githooks/pre-commit precommit.sh

echo "Git hooks installed (core.hooksPath=.githooks)"
echo "Pre-commit runs: fmt check, clippy, build, test"
