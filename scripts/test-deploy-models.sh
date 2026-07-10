#!/usr/bin/env bash
# Validate discover-models output used by deploy.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

write_minimal_gguf() {
    python3 - "$1" "$2" <<'PY'
import struct, sys
path, arch = sys.argv[1], sys.argv[2]
buf = bytearray()
buf += struct.pack('<I', 0x46554747)
buf += struct.pack('<I', 2)
buf += struct.pack('<Q', 0)
buf += struct.pack('<Q', 1)
key = b'general.architecture'
buf += struct.pack('<Q', len(key)) + key
buf += struct.pack('<I', 8)
buf += struct.pack('<Q', len(arch)) + arch.encode()
open(path, 'wb').write(buf)
PY
}

mkdir -p "$TMP/models/nested"
write_minimal_gguf "$TMP/models/gemma-3-4b.gguf" gemma
write_minimal_gguf "$TMP/models/nested/qwen2.5-coder-7b.gguf" qwen2

discover() {
    cargo run -q -- discover-models "$@"
}

out="$TMP/models.toml"
discover "$TMP/models" -o "$out"

grep -q 'alias = "gemma-3-4b"' "$out"
grep -q 'alias = "qwen2.5-coder-7b"' "$out"
grep -q 'auto_discover = true' "$out"
grep -q "models_dir = \"$TMP/models\"" "$out"
grep -q 'nested/qwen2.5-coder-7b.gguf' "$out"

touch "$TMP/models/mmproj-test.gguf" "$TMP/models/ggml-vocab-test.gguf"
discover "$TMP/models" -o "$TMP/no-artifacts.toml"
! grep -q 'mmproj-test' "$TMP/no-artifacts.toml"
! grep -q 'ggml-vocab-test' "$TMP/no-artifacts.toml"

mkdir -p "$TMP/extra"
write_minimal_gguf "$TMP/extra/beta.gguf" llama
multi_out="$TMP/multi.toml"
discover "$TMP/models,$TMP/extra" -o "$multi_out"
grep -q 'alias = "beta"' "$multi_out"

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
