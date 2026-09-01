#!/usr/bin/env bash
# Idempotent CUDA build + install of llama.cpp's llama-server into PREFIX.
# Restarts gguf-switchboard when the systemd unit exists; otherwise ignores it.
#
# Usage (as your normal user — script sudo's only where needed):
#   ./scripts/update-llama-cpp.sh
#
# Env overrides:
#   LLAMA_DIR   Source tree (default: $HOME/llama.cpp)
#   PREFIX      Install prefix (default: /usr/local)
#   SERVICE     systemd unit name (default: gguf-switchboard)
#   SKIP_PULL=1 Skip git pull
#   SKIP_SERVICE=1  Never touch systemd
#   FORCE_REBUILD=1 Rebuild even when the installed release is current
#   LLAMA_RELEASE_CHANNEL=stable  Track manual vMAJOR.MINOR.PATCH releases (default)
#   LLAMA_RELEASE_CHANNEL=nightly Track automated bNNNNN snapshots
set -euo pipefail

# If invoked via sudo, keep the invoking user's home for LLAMA_DIR defaults.
if [[ -n "${SUDO_USER:-}" && "${EUID}" -eq 0 ]]; then
  _home="$(getent passwd "$SUDO_USER" | cut -d: -f6 || true)"
  if [[ -n "${_home}" ]]; then
    HOME="${_home}"
  fi
fi

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
PREFIX="${PREFIX:-/usr/local}"
SERVICE="${SERVICE:-gguf-switchboard}"
INSTALLED_BIN="${PREFIX}/bin/llama-server"
RELEASE_MARKER="${LLAMA_RELEASE_MARKER:-${PREFIX}/share/gguf-switchboard/llama-cpp-release}"
LLAMA_RELEASE_CHANNEL="${LLAMA_RELEASE_CHANNEL:-stable}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime-update-lib.sh
source "$SCRIPT_DIR/runtime-update-lib.sh"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing required command: $1" >&2
    exit 1
  }
}

service_installed() {
  systemctl cat "${SERVICE}.service" >/dev/null 2>&1
}

need git

case "$LLAMA_RELEASE_CHANNEL" in
  stable)
    release_pattern='v[0-9]*'
    ;;
  nightly)
    release_pattern='b[0-9]*'
    ;;
  *)
    echo "ERROR: LLAMA_RELEASE_CHANNEL must be 'stable' or 'nightly'." >&2
    exit 2
    ;;
esac

select_latest_release() {
  if [[ "$LLAMA_RELEASE_CHANNEL" == "stable" ]]; then
    latest_semver_llama_tag
  else
    latest_numbered_llama_tag
  fi
}

echo "==> Checking llama.cpp ${LLAMA_RELEASE_CHANNEL} release at ${LLAMA_DIR}"
if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_DIR")"
  git clone --depth 1 --single-branch https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
release_check_failed=false
if [[ "${SKIP_PULL:-0}" != "1" ]]; then
  if ! remote_tags="$(git ls-remote --tags --refs origin "$release_pattern")"; then
    release_check_failed=true
    latest_release=""
  else
    latest_release="$(printf '%s\n' "$remote_tags" | awk -F/ '{ print $3 }' | select_latest_release)"
  fi
else
  latest_release="$(git tag --list "$release_pattern" | select_latest_release)"
fi

runtime_ready=false
if [[ -x "$INSTALLED_BIN" ]] && "$INSTALLED_BIN" --list-devices >/dev/null 2>&1; then
  runtime_ready=true
fi

installed_release=""
if [[ -r "$RELEASE_MARKER" ]]; then
  installed_release="$(tr -d '[:space:]' <"$RELEASE_MARKER")"
elif [[ "$runtime_ready" == "true" ]]; then
  installed_build="$("$INSTALLED_BIN" --version 2>&1 | sed -nE 's/.*version:[[:space:]]*([0-9]+).*/\1/p' | head -n 1)"
  [[ -n "$installed_build" ]] && installed_release="b${installed_build}"
fi

if [[ "$release_check_failed" == "true" ]]; then
  if [[ "$runtime_ready" == "true" ]]; then
    echo "WARNING: llama.cpp release check failed; keeping ${installed_release:-installed runtime}." >&2
    exit 0
  fi
  echo "ERROR: llama.cpp release check failed and no working runtime is installed." >&2
  exit 1
fi

if [[ -z "$latest_release" ]]; then
  if [[ "$runtime_ready" == "true" ]]; then
    echo "WARNING: Could not determine the latest llama.cpp ${LLAMA_RELEASE_CHANNEL} release; keeping ${installed_release:-installed runtime}." >&2
    exit 0
  fi
  echo "ERROR: Could not determine the latest llama.cpp ${LLAMA_RELEASE_CHANNEL} release and no working runtime is installed." >&2
  exit 1
fi

if [[ "${FORCE_REBUILD:-0}" != "1" ]] && ! llama_update_required "$installed_release" "$latest_release" "$runtime_ready"; then
  if [[ ! -r "$RELEASE_MARKER" ]]; then
    marker_tmp="$(mktemp)"
    printf '%s\n' "$latest_release" >"$marker_tmp"
    sudo mkdir -p "$(dirname "$RELEASE_MARKER")"
    sudo install -o root -g root -m 644 "$marker_tmp" "$RELEASE_MARKER"
    rm -f "$marker_tmp"
  fi
  echo "==> llama.cpp already current ($latest_release, ${LLAMA_RELEASE_CHANNEL} channel); skipping rebuild."
  exit 0
fi

echo "==> Updating llama.cpp ${installed_release:-not installed} → $latest_release"
if ! git rev-parse --verify --quiet "refs/tags/$latest_release" >/dev/null; then
  git fetch --depth 1 origin "refs/tags/$latest_release:refs/tags/$latest_release"
fi
git checkout --detach "$latest_release"

echo "==> Checking CUDA toolchain"
need nvcc
need nvidia-smi
need cmake
need nproc
need readelf
need ldd
nvcc --version
nvidia-smi

echo "==> Configuring CUDA build"
rm -rf build
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

echo "==> Building"
cmake --build build -j"$(nproc)"

echo "==> Verifying CUDA backend (build tree)"
./build/bin/llama-server --list-devices

service_was_active=0
if [[ "${SKIP_SERVICE:-0}" != "1" ]] && service_installed; then
  echo "==> Stopping ${SERVICE}"
  if systemctl is-active --quiet "$SERVICE"; then
    service_was_active=1
  fi
  sudo systemctl stop "$SERVICE" || true
elif [[ "${SKIP_SERVICE:-0}" != "1" ]]; then
  echo "==> ${SERVICE} not installed — skipping stop/start"
fi

echo "==> Installing to ${PREFIX}"
sudo cmake --install build --prefix "$PREFIX"

echo "==> Fixing llama-server RUNPATH if needed"
if readelf -d "$INSTALLED_BIN" | grep -qE 'RPATH|RUNPATH'; then
  if command -v patchelf >/dev/null 2>&1; then
    sudo patchelf --remove-rpath "$INSTALLED_BIN"
  else
    echo "ERROR: patchelf missing (needed to strip RPATH/RUNPATH from ${INSTALLED_BIN})" >&2
    echo "Install with: sudo apt install patchelf" >&2
    exit 1
  fi
fi

echo "==> Refreshing linker cache"
sudo ldconfig

echo "==> Verifying installed libraries"
ldd "$INSTALLED_BIN" | grep -E 'ggml|llama|mtmd' || {
  echo "ERROR: installed llama-server does not link expected ggml/llama libs" >&2
  exit 1
}

# Fail if the installed binary still resolves libs from the source build tree.
if ldd "$INSTALLED_BIN" | grep -F "${LLAMA_DIR}/build"; then
  echo "ERROR: llama-server still resolves libraries from build directory (${LLAMA_DIR}/build)" >&2
  exit 1
fi

echo "==> Verifying installed CUDA backend"
"$INSTALLED_BIN" --list-devices

marker_tmp="$(mktemp)"
printf '%s\n' "$latest_release" >"$marker_tmp"
sudo mkdir -p "$(dirname "$RELEASE_MARKER")"
sudo install -o root -g root -m 644 "$marker_tmp" "$RELEASE_MARKER"
rm -f "$marker_tmp"

if [[ "${SKIP_SERVICE:-0}" != "1" ]] && service_installed; then
  echo "==> Starting ${SERVICE}"
  sudo systemctl start "$SERVICE"
  echo "==> Service status"
  sudo systemctl --no-pager --full status "$SERVICE" || true
elif [[ "${service_was_active}" -eq 1 ]]; then
  echo "WARNING: ${SERVICE} was active before upgrade but unit disappeared" >&2
fi

echo
echo "DONE: ${INSTALLED_BIN} refreshed (CUDA)"
