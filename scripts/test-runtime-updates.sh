#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=runtime-update-lib.sh
source "$ROOT/scripts/runtime-update-lib.sh"

assert_eq() {
    local expected="$1"
    local actual="$2"
    local label="$3"
    if [[ "$actual" != "$expected" ]]; then
        echo "$label: expected '$expected', got '$actual'" >&2
        exit 1
    fi
}

latest_llama="$({
    printf '%s\n' b999 b1001 b1000 v0.1.2 invalid b1001
} | latest_numbered_llama_tag)"
assert_eq "b1001" "$latest_llama" "latest numbered llama.cpp tag"

llama_update_required "b1000" "b1001" true
! llama_update_required "b1001" "b1001" true
llama_update_required "b1001" "b1001" false
llama_update_required "" "b1001" true

vllm_json='{
  "releases": {
    "0.27.4": [{"yanked": false}],
    "0.28.0": [{"yanked": false}],
    "0.28.1": [{"yanked": true}],
    "0.28.2": [{"yanked": false}],
    "0.29.0": [{"yanked": false}],
    "0.28.3rc1": [{"yanked": false}]
  }
}'
latest_vllm="$(printf '%s\n' "$vllm_json" | latest_stable_vllm_version "0.28" "0.29")"
assert_eq "0.28.2" "$latest_vllm" "latest compatible vLLM release"

vllm_update_required "0.28.0" "0.28.2" true
! vllm_update_required "0.28.2" "0.28.2" true
vllm_update_required "0.28.2" "0.28.2" false
vllm_update_required "" "0.28.2" true

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/project"
printf '%s\n' '[project]' >"$TMP/project/pyproject.toml"

cat >"$TMP/bin/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output=""
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        output="$2"
        shift 2
    else
        shift
    fi
done
cat >"$output" <<'INSTALLER'
#!/usr/bin/env sh
cat >"$UV_INSTALL_DIR/uv" <<'UV'
#!/usr/bin/env sh
if [ "${1:-}" = "run" ]; then
    echo "0.28.2"
fi
exit 0
UV
chmod +x "$UV_INSTALL_DIR/uv"
INSTALLER
EOF
chmod +x "$TMP/bin/curl"

cat >"$TMP/bin/sudo" <<'EOF'
#!/usr/bin/env sh
exec "$@"
EOF
chmod +x "$TMP/bin/sudo"

GGUF_SWITCHBOARD_VLLM_LIB=1 source "$ROOT/scripts/setup-vllm.sh"
PATH="$TMP/bin:$PATH"
UV_BIN="$TMP/bin/uv"
UV_INSTALL_URL="https://example.invalid/uv-installer.sh"
setup_vllm "$TMP/project" >/dev/null

cat >"$TMP/project/pyproject.toml" <<'EOF'
[project]
dependencies = ["vllm>=0.28,<0.29"]
EOF
cat >"$TMP/bin/curl" <<EOF
#!/usr/bin/env sh
printf '%s\n' '$vllm_json'
EOF
chmod +x "$TMP/bin/curl"
cat >"$TMP/bin/uv" <<'EOF'
#!/usr/bin/env sh
printf '%s\n' "$*" >>"$UV_TEST_LOG"
if [ "${1:-}" = "run" ]; then
    echo "0.28.2"
fi
exit 0
EOF
chmod +x "$TMP/bin/uv"
: >"$TMP/current-uv.log"
current_vllm_output="$(
    PATH="$TMP/bin:$PATH" \
    UV_BIN="$TMP/bin/uv" \
    UV_TEST_LOG="$TMP/current-uv.log" \
    ensure_vllm_current "$TMP/project"
)"
grep -q 'already current (0.28.2); skipping sync' <<<"$current_vllm_output"
! grep -q '^sync ' "$TMP/current-uv.log"

mkdir -p "$TMP/llama-source/.git" "$TMP/llama-prefix/bin" \
    "$TMP/llama-prefix/share/gguf-switchboard"
cat >"$TMP/bin/git" <<'EOF'
#!/usr/bin/env sh
if [ "${1:-}" = "ls-remote" ]; then
    printf '%s\trefs/tags/b1000\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    printf '%s\trefs/tags/b1001\n' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    exit 0
fi
exit 1
EOF
chmod +x "$TMP/bin/git"
cat >"$TMP/llama-prefix/bin/llama-server" <<'EOF'
#!/usr/bin/env sh
if [ "${1:-}" = "--version" ]; then
    echo "version: 1001 (test)"
fi
exit 0
EOF
chmod +x "$TMP/llama-prefix/bin/llama-server"
printf '%s\n' b1001 >"$TMP/llama-prefix/share/gguf-switchboard/llama-cpp-release"

llama_output="$(
    PATH="$TMP/bin:$PATH" \
    LLAMA_DIR="$TMP/llama-source" \
    PREFIX="$TMP/llama-prefix" \
    SKIP_SERVICE=1 \
    "$ROOT/scripts/update-llama-cpp.sh"
)"
grep -q 'already current (b1001); skipping rebuild' <<<"$llama_output"
! grep -q 'Configuring CUDA build' <<<"$llama_output"

echo "runtime update decision validation passed"
