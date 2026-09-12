#!/usr/bin/env bash
# Install and verify the isolated vLLM runtime used by gguf-switchboard.
set -euo pipefail

UV_BIN="${UV_BIN:-/usr/local/bin/uv}"
UV_INSTALL_URL="${UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
VLLM_PROJECT_DIR="${VLLM_PROJECT_DIR:-/opt/gguf-switchboard/vllm-runtime}"
# Release channel, mirroring LLAMA_RELEASE_CHANNEL semantics:
#   stable  Track the locked/stable vLLM release in uv.lock (default).
#           Deploys sync --frozen and never chase PyPI.
#   nightly Re-resolve the allowed vLLM range against PyPI on every deploy
#           (still capped by vllm-runtime/pyproject.toml bounds) — opt in via
#           VLLM_RELEASE_CHANNEL=nightly ./deploy.sh
VLLM_RELEASE_CHANNEL="${VLLM_RELEASE_CHANNEL:-stable}"
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

# Probe the installed vLLM version without importing the vLLM CLI: `vllm
# --version` parses engine args (and infers the device type) at startup, so
# on GPU-less or driver-broken hosts it crashes with `Can't initialize
# NVML` / `Failed to infer device type` even when the install is fine.
vllm_probe_script() {
    printf '%s' "import importlib.metadata, sys; print(importlib.metadata.version('vllm'))"
}

vllm_ready() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    [[ -f "$project_dir/pyproject.toml" ]] || return 1
    # UV_TEST_LOG marker: shell regression tests assert the import probe
    # (not `vllm --version`) drives readiness — keep the token stable.
    if [[ -n "${UV_TEST_LOG:-}" ]]; then
        printf '%s\n' "vllm_ready probe" >>"$UV_TEST_LOG"
    fi
    env -u VIRTUAL_ENV "$UV_BIN" run --no-sync --project "$project_dir" \
        python -c "$(vllm_probe_script)" >/dev/null 2>&1
}

installed_vllm_version() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    env -u VIRTUAL_ENV "$UV_BIN" run --no-sync --project "$project_dir" \
        python -c "$(vllm_probe_script)" 2>/dev/null | tail -n 1 | tr -d '[:space:]'
}

# Warn (never fail) when no usable NVIDIA GPU is visible. A missing driver
# is the root cause of `Can't initialize NVML` / `Failed to infer device
# type` at `vllm serve` time; surfacing it here turns a cryptic serve crash
# into an actionable preflight message.
warn_if_no_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        if ! nvidia-smi -L >/dev/null 2>&1; then
            echo "WARNING: nvidia-smi exists but lists no GPUs; vLLM serve will likely fail device detection." >&2
        fi
    else
        echo "WARNING: nvidia-smi not found; vLLM requires a CUDA GPU + driver (NVML errors expected otherwise)." >&2
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && [[ "${CUDA_VISIBLE_DEVICES//[ ,]/}" == "" ]]; then
        echo "WARNING: CUDA_VISIBLE_DEVICES is set but empty; unset it or list GPU ids (hides all GPUs from vLLM)." >&2
    fi
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
    warn_if_no_gpu
    echo "==> Installing vLLM in $project_dir..."
    local sync_flags=(--project "$project_dir")
    if [[ -f "$project_dir/uv.lock" ]]; then
        sync_flags+=(--frozen)
    fi
    env -u VIRTUAL_ENV "$UV_BIN" sync "${sync_flags[@]}"
    env -u VIRTUAL_ENV "$UV_BIN" run --no-sync --project "$project_dir" \
        python -c "$(vllm_probe_script)"
    echo "==> vLLM runtime ready."
}

ensure_vllm_current() {
    local project_dir="${1:-$VLLM_PROJECT_DIR}"
    [[ -f "$project_dir/pyproject.toml" ]] || {
        echo "ERROR: vLLM runtime project missing: $project_dir/pyproject.toml" >&2
        return 1
    }

    install_uv
    warn_if_no_gpu
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

    case "$VLLM_RELEASE_CHANNEL" in
        stable | nightly) ;;
        *)
            echo "ERROR: VLLM_RELEASE_CHANNEL must be 'stable' or 'nightly'." >&2
            return 2
            ;;
    esac

    # Stable channel (default): when the deployed tree ships a uv.lock,
    # drift is measured against the lockfile, not PyPI. This keeps deploys
    # reproducible when the network is down and stops surprise transitive
    # upgrades (tokenspeed-triton post-releases, xgrammar bumps) unless the
    # lock itself was refreshed in the repo. Nightly skips this fast path
    # and always re-resolves against PyPI below.
    if [[ "$VLLM_RELEASE_CHANNEL" == "stable" && -f "$project_dir/uv.lock" ]]; then
        if [[ "$runtime_ready" == "true" ]] \
            && env -u VIRTUAL_ENV "$UV_BIN" sync --locked --check --project "$project_dir" >/dev/null 2>&1; then
            echo "==> vLLM already current ($installed_version, locked, ${VLLM_RELEASE_CHANNEL} channel); skipping sync."
            return 0
        fi
        if [[ "${FORCE_VLLM_SYNC:-0}" != "1" ]] && [[ "$runtime_ready" == "true" ]]; then
            echo "==> vLLM drifted from uv.lock; re-syncing to the locked set."
            setup_vllm "$project_dir"
            return 0
        fi
    fi

    if [[ "${FORCE_VLLM_SYNC:-0}" != "1" ]] \
        && ! vllm_update_required "$installed_version" "$latest_version" "$runtime_ready"; then
        echo "==> vLLM already current ($installed_version, ${VLLM_RELEASE_CHANNEL} channel); skipping sync."
        return 0
    fi

    echo "==> Updating vLLM ${installed_version:-not installed} → $latest_version (${VLLM_RELEASE_CHANNEL} channel)"
    setup_vllm "$project_dir"
}

if [[ "${GGUF_SWITCHBOARD_VLLM_LIB:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

ensure_vllm_current "$VLLM_PROJECT_DIR"
