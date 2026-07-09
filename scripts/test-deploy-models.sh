#!/usr/bin/env bash
# Validate discover-models output used by deploy.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/models/nested"
touch "$TMP/models/gemma-3-4b.gguf" "$TMP/models/nested/qwen2.5-coder-7b.gguf"

discover() {
    cargo run -q -- discover-models "$@"
}

out="$TMP/models.toml"
discover "$TMP/models" -o "$out"

grep -q 'alias = "gemma-3-4b"' "$out"
grep -q 'alias = "qwen2.5-coder-7b"' "$out"
grep -q 'auto_discover = true' "$out"
grep -q "models_dir = \"$TMP/models\"" "$out"

merge="$TMP/merge.toml"
cat >"$merge" <<EOF
[defaults]
models_dir = "$TMP/models"
base_port = 9000

[[models]]
alias = "gemma-code"
file = "gemma-3-4b.gguf"
display_name = "Custom Gemma"
priority = true
EOF

merged_out="$TMP/merged.toml"
discover "$TMP/models" -o "$merged_out" --merge "$merge"

grep -q 'alias = "gemma-code"' "$merged_out"
grep -q 'display_name = "Custom Gemma"' "$merged_out"
grep -q 'priority = true' "$merged_out"

echo "deploy models generation validation passed"
