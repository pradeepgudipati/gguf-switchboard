#!/usr/bin/env bash
# Install and verify the isolated vLLM runtime used by gguf-switchboard.
set -euo pipefail

UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
UV_INSTALL_URL="${UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
VLLM_PROJECT_DIR="${VLLM_PROJECT_DIR:-/opt/gguf-switchboard/vllm-runtime}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime-update-lib.sh
source "$SCRIPT_DIR/runtime-update-lib.sh"

install_uv() {
    if [[ -x "$UV_BIN" ]]; then
        return 0
    fi

    echo "==> Installing uv → $UV_BIN..."
    local installer
    installer="$(mktemp)"
    if ! curl --proto '=https' --tlsv1.2 -fsSL "$UV_INSTALL_URL" -o "$installer"; then
        rm -f "$installer"
        return 1
    fi
    if ! sudo env \
        UV_INSTALL_DIR="$(dirname "$UV_BIN")" \
        UV_NO_MODIFY_PATH=1 \
        sh "$installer"; then
        rm -f "$installer"
        return 1
    fi
    rm -f "$installer"
    [[ -x "$UV_BIN" ]] || {
        echo "ERROR: uv installer did not create $UV_BIN" >&2
        return 1
    }
}

vllm_ready() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    [[ -f "$project_dir/pyproject.toml" ]] || return 1
    env -u VIRTUAL_ENV "$UV_BIN" run --no-sync --project "$project_dir" vllm --version >/dev/null 2>&1
}

installed_vllm_version() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    env -u VIRTUAL_ENV "$UV_BIN" run --no-sync --project "$project_dir" \
        vllm --version 2>/dev/null | tail -n 1 | tr -d '[:space:]'
}

vllm_version_bounds() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    sed -nE 's/.*"vllm>=([0-9]+\.[0-9]+),<([0-9]+\.[0-9]+)".*/\1 \2/p' \
        "$project_dir/pyproject.toml" | head -n 1
}

latest_allowed_vllm_release() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    local minimum_minor maximum_minor
    read -r minimum_minor maximum_minor < <(vllm_version_bounds "$project_dir")
    [[ -n "${minimum_minor:-}" && -n "${maximum_minor:-}" ]] || return 1
    curl -fsSL https://pypi.org/pypi/vllm/json \
        | latest_stable_vllm_version "$minimum_minor" "$maximum_minor"
}

setup_vllm() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    [[ -f "$project_dir/pyproject.toml" ]] || {
        echo "ERROR: vLLM runtime project missing: $project_dir/pyproject.toml" >&2
        return 1
    }

    install_uv
    echo "==> Installing vLLM in $project_dir..."
    env -u VIRTUAL_ENV "$UV_BIN" sync --project "$project_dir"
    env -u VIRTUAL_ENV "$UV_BIN" run --no-sync --project "$project_dir" vllm --version
    echo "==> vLLM runtime ready."
}

ensure_vllm_current() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    [[ -f "$project_dir/pyproject.toml" ]] || {
        echo "ERROR: vLLM runtime project missing: $project_dir/pyproject.toml" >&2
        return 1
    }

    install_uv
    local runtime_ready=false installed_version="" latest_version=""
    if vllm_ready "$project_dir"; then
        runtime_ready=true
        installed_version="$(installed_vllm_version "$project_dir")"
    fi

    if ! latest_version="$(latest_allowed_vllm_release "$project_dir")" || [[ -z "$latest_version" ]]; then
        if [[ "$runtime_ready" == "true" ]]; then
            echo "WARNING: vLLM release check failed; keeping installed vLLM $installed_version." >&2
            return 0
        fi
        echo "WARNING: vLLM release check failed and no working runtime exists; attempting installation." >&2
        setup_vllm "$project_dir"
        return
    fi

    if [[ "${FORCE_VLLM_SYNC:-0}" != "1" ]] \
        && ! vllm_update_required "$installed_version" "$latest_version" "$runtime_ready"; then
        echo "==> vLLM already current ($installed_version); skipping sync."
        return 0
    fi

    echo "==> Updating vLLM ${installed_version:-not installed} → $latest_version"
    setup_vllm "$project_dir"
}

if [[ "${GGUF_SWITCHBOARD_VLLM_LIB:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

ensure_vllm_current "$VLLM_PROJECT_DIR"
