#!/usr/bin/env bash
# Install and verify the isolated vLLM runtime used by gguf-switchboard.
set -euo pipefail

UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
UV_INSTALL_URL="${UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
VLLM_PROJECT_DIR="${VLLM_PROJECT_DIR:-/opt/gguf-switchboard/vllm-runtime}"

install_uv() {
    if [[ -x "$UV_BIN" ]]; then
        return 0
    fi

    echo "==> Installing uv → $UV_BIN..."
    local installer
    installer="$(mktemp)"
    trap 'rm -f "$installer"' RETURN
    curl --proto '=https' --tlsv1.2 -fsSL "$UV_INSTALL_URL" -o "$installer"
    sudo env \
        UV_INSTALL_DIR="$(dirname "$UV_BIN")" \
        UV_NO_MODIFY_PATH=1 \
        sh "$installer"
    [[ -x "$UV_BIN" ]] || {
        echo "ERROR: uv installer did not create $UV_BIN" >&2
        return 1
    }
}

vllm_ready() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    [[ -f "$project_dir/pyproject.toml" ]] || return 1
    "$UV_BIN" run --project "$project_dir" vllm --version >/dev/null 2>&1
}

setup_vllm() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    [[ -f "$project_dir/pyproject.toml" ]] || {
        echo "ERROR: vLLM runtime project missing: $project_dir/pyproject.toml" >&2
        return 1
    }

    install_uv
    echo "==> Installing vLLM in $project_dir..."
    "$UV_BIN" sync --project "$project_dir"
    "$UV_BIN" run --project "$project_dir" vllm --version
    echo "==> vLLM runtime ready."
}

if [[ "${GGUF_SWITCHBOARD_VLLM_LIB:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

setup_vllm "$VLLM_PROJECT_DIR"
