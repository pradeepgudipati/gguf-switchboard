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
buf += struct.pack('<Q', 1)  # tensor_count > 0
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

grep -q "alias ggs='gguf-switchboard'" deploy.sh
! grep -q "alias gs='gguf-switchboard'" deploy.sh
grep -Eq 'APT_PKGS=.*\baria2\b' deploy.sh

# Deploy keeps tracked examples separate from user-owned runtime configuration.
test -f config.example.toml
test -f models.example.toml
grep -qx '/config.toml' .gitignore
grep -qx '/models.toml' .gitignore
grep -qx '/models.json' .gitignore

runtime_home="$TMP/home"
runtime_config="$TMP/config"
mkdir -p "$runtime_home"

GGUF_SWITCHBOARD_DEPLOY_LIB=1 source ./deploy.sh
HOME="$runtime_home"
CONFIG_DIR="$runtime_config"
CONFIG_FILE="$CONFIG_DIR/config.toml"
MODELS_FILE="$CONFIG_DIR/models.toml"
unset MODELS_DIR

initialize_runtime_config

test -d "$runtime_home/models"
test -f "$CONFIG_FILE"
test -f "$MODELS_FILE"
grep -q 'models_file = "models.toml"' "$CONFIG_FILE"
grep -q "models_dir = \"$runtime_home/models\"" "$MODELS_FILE"

printf '\n# user setting\n' >> "$CONFIG_FILE"
printf '\n# user registry\n' >> "$MODELS_FILE"
initialize_runtime_config
grep -q '^# user setting$' "$CONFIG_FILE"
grep -q '^# user registry$' "$MODELS_FILE"

# Runtime initialization must happen before fallible build and systemd work.
init_line="$(grep -n '^initialize_runtime_config$' deploy.sh | tail -1 | cut -d: -f1)"
build_line="$(grep -n '^cargo build --release$' deploy.sh | cut -d: -f1)"
service_line="$(grep -n '^sudo tee "\$SERVICE_FILE"' deploy.sh | cut -d: -f1)"
test "$init_line" -lt "$build_line"
test "$init_line" -lt "$service_line"

fake_llama="$runtime_home/llama.cpp/build/bin/llama-server"
mkdir -p "$(dirname "$fake_llama")"
cat >"$fake_llama" <<'EOF'
#!/bin/sh
# Minimal stand-in for deploy llama-server readiness checks.
if [ "${1:-}" = "--version" ]; then
  echo "fake llama-server 0.0.0"
  exit 0
fi
exit 0
EOF
chmod +x "$fake_llama"
fake_llama="$(realpath "$fake_llama")"

resolved_llama="$(resolve_llama_server)"
test "$resolved_llama" = "$fake_llama"

configure_llama_server "$resolved_llama"
grep -q "llama_server = \"$fake_llama\"" "$MODELS_FILE"

llama_server_ready
test "$(effective_llama_server)" = "$fake_llama"

custom_llama="$TMP/custom/llama-server"
sed -i.bak -e "s|^llama_server = .*|llama_server = \"$custom_llama\"|" "$MODELS_FILE"
rm -f "$MODELS_FILE.bak"
configure_llama_server /usr/bin/llama-server
grep -q "llama_server = \"$custom_llama\"" "$MODELS_FILE"
! llama_server_ready

mkdir -p "$(dirname "$custom_llama")"
cp "$fake_llama" "$custom_llama"
chmod +x "$custom_llama"
llama_server_ready

empty_models="$TMP/empty-models"
mkdir -p "$empty_models"
sed -i.bak -e "s|^models_dir = .*|models_dir = \"$empty_models\"|" "$MODELS_FILE"
# Point at a working binary so readiness tests stay focused on models next.
sed -i.bak -e "s|^llama_server = .*|llama_server = \"$fake_llama\"|" "$MODELS_FILE"
rm -f "$MODELS_FILE.bak"
! registry_has_model_candidates "$MODELS_FILE"

write_minimal_gguf "$empty_models/starter.gguf" llama
registry_has_model_candidates "$MODELS_FILE"

model_help="$(model_setup_help "$empty_models")"
grep -q 'ggs models search "Qwen3.5 9B"' <<<"$model_help"
grep -q 'ggs models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M' <<<"$model_help"
grep -q './deploy.sh --refresh-models' <<<"$model_help"

llama_help="$(llama_setup_help)"
grep -q './scripts/update-llama-cpp.sh' <<<"$llama_help"
grep -q 'llama_server = "/path/to/llama-server"' <<<"$llama_help"

llama_check_line="$(grep -n '^if ! llama_server_ready; then$' deploy.sh | cut -d: -f1)"
empty_check_line="$(grep -n 'registry_has_model_candidates "\$MODELS_FILE"' deploy.sh | tail -1 | cut -d: -f1)"
start_line="$(grep -n '^sudo systemctl start gguf-switchboard$' deploy.sh | cut -d: -f1)"
test "$llama_check_line" -lt "$empty_check_line"
test "$empty_check_line" -lt "$start_line"

echo "deploy models generation validation passed"
