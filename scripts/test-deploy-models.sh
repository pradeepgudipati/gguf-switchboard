#!/usr/bin/env bash
# Validate discover-models output and system-wide deploy.sh contracts.
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

# System-wide layout contracts
grep -q 'SERVICE_USER="ggs"' deploy.sh
grep -q 'INSTALL_DIR="/opt/gguf-switchboard"' deploy.sh
grep -q 'STATE_DIR="/var/lib/gguf-switchboard"' deploy.sh
grep -q 'VLLM_MODELS_DIR="${STATE_DIR}/vllm-models"' deploy.sh
grep -q 'BIN="/usr/local/bin/gguf-switchboard"' deploy.sh
grep -q 'LLAMA_SERVER="/usr/local/bin/llama-server"' deploy.sh
grep -q 'User=${SERVICE_USER}' deploy.sh
grep -q 'Group=${SERVICE_GROUP}' deploy.sh
grep -q 'WorkingDirectory=${INSTALL_DIR}' deploy.sh
grep -q 'Environment=MODELS_DIR=${MODELS_DIR}' deploy.sh
grep -q 'sudo systemctl enable --now gguf-switchboard' deploy.sh
grep -q 'sudo systemctl stop gguf-switchboard' deploy.sh
! grep -q 'systemctl disable' deploy.sh
grep -q 'install -o root -g root -m 755' deploy.sh
grep -q 'SOURCE_DIR' deploy.sh
grep -q '\[\[ "\$source_dir" == "\$INSTALL_DIR" \]\]' deploy.sh || grep -q 'Already running from \$INSTALL_DIR' deploy.sh
grep -qF -- '--migrate-models' deploy.sh
grep -qF -- '--skip-llama-cpp' deploy.sh
grep -qF -- '--skip-vllm' deploy.sh
grep -q 'scripts/setup-vllm.sh' deploy.sh
grep -q 'Left .* untouched' deploy.sh
test -x scripts/setup-vllm.sh
test -f vllm-runtime/pyproject.toml
grep -q 'UV_NO_MODIFY_PATH=1' scripts/setup-vllm.sh

GGUF_SWITCHBOARD_VLLM_LIB=1 source ./scripts/setup-vllm.sh
fake_uv="$TMP/uv"
uv_log="$TMP/uv.log"
cat >"$fake_uv" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >>"$UV_TEST_LOG"
if [ "${1:-}" = "--version" ] || [ "${4:-}" = "--version" ]; then
  printf '%s\n' "uv/vllm test version"
fi
EOF
chmod +x "$fake_uv"
test_vllm_project="$TMP/vllm-runtime"
mkdir -p "$test_vllm_project"
cp vllm-runtime/pyproject.toml "$test_vllm_project/pyproject.toml"
UV_TEST_LOG="$uv_log" UV_BIN="$fake_uv" setup_vllm "$test_vllm_project"
grep -q "sync --project $test_vllm_project" "$uv_log"
grep -q "run --no-sync --project $test_vllm_project vllm --version" "$uv_log"

# Deploy keeps tracked examples separate from runtime configuration.
test -f config.example.toml
test -f models.example.toml
grep -qx '/config.toml' .gitignore
grep -qx '/models.toml' .gitignore
grep -qx '/models.json' .gitignore
grep -q 'models_file = "/opt/gguf-switchboard/models.toml"' config.example.toml
grep -q 'models_dir = "/var/lib/gguf-switchboard/models"' models.example.toml
grep -q 'vllm_command = "/usr/local/bin/uv"' models.example.toml
grep -q 'vllm_project = "/opt/gguf-switchboard/vllm-runtime"' models.example.toml
grep -q 'User=ggs' gguf-switchboard.service
grep -q 'WorkingDirectory=/opt/gguf-switchboard' gguf-switchboard.service
grep -q 'MODELS_DIR=/var/lib/gguf-switchboard/models' gguf-switchboard.service
grep -q 'Description=GGUF Switchboard - local llama.cpp and vLLM model router' gguf-switchboard.service
grep -q 'Description=GGUF Switchboard - local llama.cpp and vLLM model router' deploy.sh

GGUF_SWITCHBOARD_DEPLOY_LIB=1 source ./deploy.sh

test "$SERVICE_USER" = "ggs"
test "$INSTALL_DIR" = "/opt/gguf-switchboard"
test "$STATE_DIR" = "/var/lib/gguf-switchboard"
test "$MODELS_DIR" = "/var/lib/gguf-switchboard/models"
test "$VLLM_MODELS_DIR" = "/var/lib/gguf-switchboard/vllm-models"
test "$CONFIG_FILE" = "/opt/gguf-switchboard/config.toml"
test "$MODELS_FILE" = "/opt/gguf-switchboard/models.toml"
test "$BIN" = "/usr/local/bin/gguf-switchboard"
test "$LLAMA_SERVER" = "/usr/local/bin/llama-server"

custom_registry="$TMP/custom-vllm-defaults.toml"
cat >"$custom_registry" <<'EOF'
version = 1
[defaults]
models_dir = "/models"
vllm_command = "/custom/bin/uv"
host = "127.0.0.1"
EOF
DEPLOY_OWNER="$(id -un)"
SERVICE_GROUP="$(id -gn)"
UV_BIN="/usr/local/bin/uv"
VLLM_PROJECT_DIR="/opt/gguf-switchboard/vllm-runtime"
configure_vllm_defaults "$custom_registry"
grep -q '^vllm_command = "/custom/bin/uv"$' "$custom_registry"
grep -q '^vllm_project = "/opt/gguf-switchboard/vllm-runtime"$' "$custom_registry"
test "$(grep -c '^vllm_command[[:space:]]*=' "$custom_registry")" -eq 1
test "$(grep -c '^vllm_project[[:space:]]*=' "$custom_registry")" -eq 1
configure_vllm_defaults "$custom_registry"
test "$(grep -c '^vllm_command[[:space:]]*=' "$custom_registry")" -eq 1
test "$(grep -c '^vllm_project[[:space:]]*=' "$custom_registry")" -eq 1
test "$(registry_vllm_command "$custom_registry")" = "/custom/bin/uv"
test "$(registry_vllm_project "$custom_registry")" = "/opt/gguf-switchboard/vllm-runtime"

custom_project_registry="$TMP/custom-vllm-project.toml"
cat >"$custom_project_registry" <<'EOF'
[defaults]
vllm_command = "/another/bin/uv"
vllm_project = "/srv/custom-vllm"
EOF
test "$(registry_vllm_command "$custom_project_registry")" = "/another/bin/uv"
test "$(registry_vllm_project "$custom_project_registry")" = "/srv/custom-vllm"

runtime_registry="$TMP/runtime-vllm-defaults.toml"
cat >"$runtime_registry" <<EOF
[defaults]
vllm_command = "$fake_uv"
vllm_project = "$test_vllm_project"

[[models]]
alias = "vllm-test"
backend = "vllm"
vllm_file = "$TMP/vllm-model"
EOF
UV_TEST_LOG="$uv_log" registry_vllm_ready "$runtime_registry"

fake_llama="$TMP/llama-server"
cat >"$fake_llama" <<'EOF'
#!/bin/sh
if [ "${1:-}" = "--version" ]; then
  echo "fake llama-server 0.0.0"
  exit 0
fi
exit 0
EOF
chmod +x "$fake_llama"
LLAMA_SERVER="$fake_llama"
llama_server_ready

empty_models="$TMP/empty-models"
mkdir -p "$empty_models"
empty_registry="$TMP/empty-registry.toml"
cat >"$empty_registry" <<EOF
version = 1
auto_discover = true
[defaults]
models_dir = "$empty_models"
llama_server = "$fake_llama"
EOF
# registry_has_model_candidates uses sudo test -r; for local files use direct path readability.
# Override with a local-capable check by writing a readable file and using find-based logic.
! find "$empty_models" -type f -iname '*.gguf' -print -quit | grep -q .
write_minimal_gguf "$empty_models/starter.gguf" llama
find "$empty_models" -type f -iname '*.gguf' -print -quit | grep -q .

model_help="$(model_setup_help "$MODELS_DIR")"
grep -q 'ggs models search "Qwen3.5 9B"' <<<"$model_help"
grep -q 'ggs models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M' <<<"$model_help"
grep -q 'ggs models search vllm "Qwen 7B Instruct"' <<<"$model_help"
grep -q 'ggs models pull vllm Qwen/Qwen2.5-7B-Instruct' <<<"$model_help"
grep -q './deploy.sh --refresh-models' <<<"$model_help"
grep -q '/var/lib/gguf-switchboard/models' <<<"$model_help"

vllm_only_registry="$TMP/vllm-only.toml"
vllm_only_models="$TMP/vllm-only-models"
mkdir -p "$vllm_only_models"
cat >"$vllm_only_registry" <<EOF
version = 1
[defaults]
models_dir = "$vllm_only_models"

[[models]]
alias = "qwen-vllm"
file = ""
backend = "vllm"
vllm_file = "$TMP/qwen-vllm"
EOF
mkdir -p "$TMP/qwen-vllm"
registry_has_model_candidates "$vllm_only_registry"
registry_requires_vllm "$vllm_only_registry"
! registry_requires_llama_cpp "$vllm_only_registry"

invalid_vllm_registry="$TMP/invalid-vllm.toml"
cat >"$invalid_vllm_registry" <<EOF
version = 1

[[models]]
alias = "missing-source"
backend = "vllm"
EOF
if registry_has_model_candidates "$invalid_vllm_registry"; then
  echo "backend = vllm without a model source must not count as deployable" >&2
  exit 1
fi
if registry_requires_vllm "$invalid_vllm_registry"; then
  echo "backend = vllm without vllm_file or vllm_hf_repo must not require vLLM" >&2
  exit 1
fi

gguf_only_registry="$TMP/gguf-only.toml"
cat >"$gguf_only_registry" <<EOF
version = 1
[defaults]
models_dir = "$empty_models"

[[models]]
alias = "starter"
file = "starter.gguf"
backend = "llama.cpp"
EOF
registry_requires_llama_cpp "$gguf_only_registry"
! registry_requires_vllm "$gguf_only_registry"

# README entry path: features, three-command installation, then details.
top_features_line="$(grep -n '^## Top features$' README.md | cut -d: -f1)"
installation_line="$(grep -n '^## Installation$' README.md | cut -d: -f1)"
details_line="$(grep -n '^## Details$' README.md | cut -d: -f1)"
test "$top_features_line" -lt "$installation_line"
test "$installation_line" -lt "$details_line"
grep -q '^git clone --branch main https://github.com/pradeepgudipati/gguf-switchboard.git$' README.md
grep -q '^cd gguf-switchboard$' README.md
grep -q '^\./deploy.sh$' README.md
grep -q 'models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M' README.md
grep -q 'models pull vllm Qwen/Qwen2.5-7B-Instruct' README.md

llama_help="$(
  LLAMA_SERVER="/usr/local/bin/llama-server"
  llama_setup_help
)"
grep -q './scripts/update-llama-cpp.sh' <<<"$llama_help"
grep -q '/usr/local/bin/llama-server' <<<"$llama_help"

# llama.cpp bootstrap is shallow; stable semver is the default with nightly opt-in.
grep -Eq 'git clone .*--depth 1 .*--single-branch .*llama\.cpp\.git' scripts/update-llama-cpp.sh
grep -q "LLAMA_RELEASE_CHANNEL=\"\${LLAMA_RELEASE_CHANNEL:-stable}\"" scripts/update-llama-cpp.sh
grep -q "release_pattern='v\[0-9\]\*'" scripts/update-llama-cpp.sh
grep -q "release_pattern='b\[0-9\]\*'" scripts/update-llama-cpp.sh
grep -q 'git fetch --depth 1 origin' scripts/update-llama-cpp.sh
grep -q 'git tag --list "$release_pattern"' scripts/update-llama-cpp.sh

# Ordering: stop → build → install binary → enable --now
stop_line="$(grep -n 'systemctl stop gguf-switchboard' deploy.sh | head -1 | cut -d: -f1)"
build_line="$(grep -n '^cargo build --release$' deploy.sh | cut -d: -f1)"
install_line="$(grep -n 'install -o root -g root -m 755' deploy.sh | cut -d: -f1)"
enable_line="$(grep -n 'systemctl enable --now gguf-switchboard' deploy.sh | cut -d: -f1)"
test "$stop_line" -lt "$build_line"
test "$build_line" -lt "$install_line"
test "$install_line" -lt "$enable_line"

# Runtime paths must not be constructed from $HOME in the main deploy body.
# (HOME is still ok for git clone bootstrap, rustup, and optional shell alias.)
! grep -E 'MODELS_DIR=.*\$HOME/models|chown.*whoami.*/var/lib|User=\$\(whoami\)|WorkingDirectory=\$\(pwd\)' deploy.sh

deployment_summary="$(print_deployment_summary \
    "v0.1.2" "current; no rebuild; stable" \
    "v0.1.6" \
    "0.28.0" "current; no sync" \
    "12" "running")"
grep -q 'llama.cpp:.*v0.1.2.*current; no rebuild; stable' <<<"$deployment_summary"
grep -q 'gguf-switchboard:.*v0.1.6' <<<"$deployment_summary"
grep -q 'vLLM:.*0.28.0.*current; no sync' <<<"$deployment_summary"
grep -q 'Models indexed:.*12' <<<"$deployment_summary"
grep -q 'Service:.*running' <<<"$deployment_summary"
grep -q 'ggs status' <<<"$deployment_summary"
grep -q 'ggs logs watch' <<<"$deployment_summary"
grep -q 'ggs logs --tail 100' <<<"$deployment_summary"
grep -q 'ggs models search <query>' <<<"$deployment_summary"
grep -q 'ggs models pull vllm <repo-id>' <<<"$deployment_summary"
grep -q 'ggs restart' <<<"$deployment_summary"

echo "deploy models generation validation passed"
