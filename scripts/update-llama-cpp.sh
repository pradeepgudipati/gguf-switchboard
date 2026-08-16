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

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: missing required command: $1" >&2
    exit 1
  }
}

service_installed() {
  systemctl cat "${SERVICE}.service" >/dev/null 2>&1
}

echo "==> Checking CUDA toolchain"
need nvcc
need nvidia-smi
need cmake
need git
need nproc
need readelf
need ldd
nvcc --version
nvidia-smi

echo "==> Updating llama.cpp at ${LLAMA_DIR}"
if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  mkdir -p "$(dirname "$LLAMA_DIR")"
  git clone --depth 1 --single-branch https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
if [[ "${SKIP_PULL:-0}" != "1" ]]; then
  git pull --ff-only
fi

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
