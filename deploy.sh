#!/usr/bin/env bash
# Deploy gguf-switchboard as a system-wide systemd service.
#
# Runtime layout (not under the invoking user's $HOME):
#   /opt/gguf-switchboard/          project + config.toml + models.toml
#   /var/lib/gguf-switchboard/      models/ + usage.db
#   /usr/local/bin/gguf-switchboard
#   /usr/local/bin/llama-server     (required; install via scripts/update-llama-cpp.sh)
#   /etc/systemd/system/gguf-switchboard.service  (User=ggs)
#
# Flags:
#   --refresh-models   Regenerate models.toml from disk (merge existing pins)
#   --skip-pull        Rebuild + reinstall without git fetch/pull
#   --migrate-models   Copy legacy ~/models into /var/lib/gguf-switchboard/models
#
set -euo pipefail

# Group-writable by default so files created during deploy stay rw for ggs.
umask 0002

# ─── Fixed system paths (never derived from $HOME for runtime) ────────────────
SERVICE_USER="ggs"
SERVICE_GROUP="ggs"
INSTALL_DIR="/opt/gguf-switchboard"
STATE_DIR="/var/lib/gguf-switchboard"
# Canonical runtime models dir (never $HOME). Capture env override for discover only.
DISCOVER_MODELS_DIR="${MODELS_DIR:-}"
MODELS_DIR="${STATE_DIR}/models"
CONFIG_FILE="${INSTALL_DIR}/config.toml"
MODELS_FILE="${INSTALL_DIR}/models.toml"
BIN="/usr/local/bin/gguf-switchboard"
LLAMA_SERVER="/usr/local/bin/llama-server"
SERVICE_FILE="/etc/systemd/system/gguf-switchboard.service"
ETC_DIR="/etc/gguf-switchboard"

REPO_URL="https://github.com/pradeepgudipati/gguf-switchboard.git"
# Source checkout only (build/rsync). Runtime never lives here.
REPO_DIR="${GGUF_SWITCHBOARD_DIR:-$HOME/gguf-switchboard}"
BRANCH="main"
CONFIG_TEMPLATE="config.example.toml"
MODELS_TEMPLATE="models.example.toml"

DEPLOY_OWNER="${SUDO_USER:-${USER:-$(id -un)}}"

read_config() {
    if [[ -r "$1" ]]; then
        cat "$1"
    else
        sudo cat "$1"
    fi
}

print_models_from_config() {
    local file="$1"
    local config
    config="$(read_config "$file")"

    local models_path=""
    models_path="$(printf '%s\n' "$config" | awk -F'"' '/^models_file\s*=/ { print $2; exit }')"
    if [[ -z "$models_path" ]]; then
        local config_dir
        config_dir="$(dirname "$file")"
        if [[ -f "$config_dir/models.toml" ]]; then
            models_path="$config_dir/models.toml"
        fi
    elif [[ "$models_path" != /* ]]; then
        models_path="$(dirname "$file")/$models_path"
    fi

    echo "==> Available models:"
    echo ""
    printf "  %-24s %-30s %s\n" "MODEL ID" "DISPLAY NAME" "STATE"

    if [[ -n "$models_path" ]] && { [[ -r "$models_path" ]] || sudo test -r "$models_path"; }; then
        local models_content
        models_content="$(read_config "$models_path")"
        while IFS= read -r block; do
            [[ -z "$block" ]] && continue
            local alias display_name priority state=""
            alias="$(printf '%s\n' "$block" | awk -F'"' '/^alias = / { print $2; exit }')"
            display_name="$(printf '%s\n' "$block" | awk -F'"' '/^display_name = / { print $2; exit }')"
            priority="$(printf '%s\n' "$block" | awk '/^priority = / { print $3; exit }' | tr -d ' ')"
            [[ "$priority" == "true" ]] && state="priority"
            [[ -n "$alias" ]] && printf "  %-24s %-30s %s\n" "$alias" "${display_name:-—}" "$state"
        done < <(printf '%s\n' "$models_content" | awk '
            BEGIN { block = "" }
            /^\[\[models\]\]/ {
                if (block != "") { print block; block = "" }
                next
            }
            /^alias = / || /^display_name = / || /^priority = / {
                block = block $0 "\n"
            }
            END { if (block != "") print block }
        ')
        echo ""
        return
    fi

    while IFS= read -r section; do
        local id="${section#[models.}"
        id="${id%]}"
        local display_name priority state=""
        display_name="$(printf '%s\n' "$config" | awk -v section="$section" '
            $0 == section { found=1; next }
            found && /^display_name/ { gsub(/^[^"]*"/, ""); gsub(/".*/, ""); print; exit }
            found && /^\[/ { exit }
        ')"
        priority="$(printf '%s\n' "$config" | awk -v section="$section" '
            $0 == section { found=1; next }
            found && /^priority/ { gsub(/^[^=]*=\s*/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print; exit }
            found && /^\[/ { exit }
        ')"
        [[ "$priority" == "true" ]] && state="priority"
        printf "  %-24s %-30s %s\n" "$id" "${display_name:-—}" "$state"
    done < <(printf '%s\n' "$config" | grep -E '^\[models\.')
    echo ""
}

read_models_dir_from_toml() {
    local file="$1"
    [[ -r "$file" ]] || sudo test -r "$file" || return 1
    read_config "$file" | awk -F'"' '/^models_dir[[:space:]]*=/ { print $2; exit }'
}

print_models_dir_hints() {
    echo "    Canonical models_dir: $MODELS_DIR"
    echo "    To regenerate models.toml from disk:"
    echo "    ./deploy.sh --refresh-models"
}

ensure_service_account() {
    if ! id "$SERVICE_USER" >/dev/null 2>&1; then
        echo "==> Creating system user $SERVICE_USER..."
        sudo useradd \
            --system \
            --home "$STATE_DIR" \
            --shell /usr/sbin/nologin \
            "$SERVICE_USER"
    else
        echo "==> Service account $SERVICE_USER already exists."
    fi
    # NVIDIA / DRM device access (ignore if groups absent)
    sudo usermod -aG video,render "$SERVICE_USER" 2>/dev/null || true
}

ensure_system_directories() {
    echo "==> Ensuring system directories..."
    sudo mkdir -p \
        "$INSTALL_DIR" \
        "$MODELS_DIR" \
        "$ETC_DIR"

    # STATE_DIR (usage.db): owned by service user, writable by owner only.
    sudo chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$STATE_DIR"
    sudo chmod 755 "$STATE_DIR"

    # MODELS_DIR: shared between service user and deploy user via ggs group.
    #   2775 = rwxrwsr-x  (setgid ensures new files inherit the ggs group)
    #   664 for files     (group-writable so deploy user can download models)
    sudo chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$MODELS_DIR"
    sudo find "$MODELS_DIR" -type d -exec chmod 2775 {} \;
    sudo find "$MODELS_DIR" -type f -exec chmod 664 {} \;
}

# Shared project tree: deploy owner + ggs group (setgid dirs).
fix_install_ownership() {
    sudo chown -R "${DEPLOY_OWNER}:${SERVICE_GROUP}" "$INSTALL_DIR"
    sudo chmod -R g+rwX "$INSTALL_DIR"
    sudo find "$INSTALL_DIR" -type d -exec chmod g+s {} \;
}

sync_project_to_install() {
    local source_dir="$1"
    if [[ "$source_dir" == "$INSTALL_DIR" ]]; then
        echo "==> Already running from $INSTALL_DIR; skipping rsync."
        fix_install_ownership
        return 0
    fi

    echo "==> Syncing project → $INSTALL_DIR..."
    sudo rsync -aH \
        --exclude target/ \
        --exclude .git/ \
        --exclude '/config.toml' \
        --exclude '/models.toml' \
        --exclude '/models.json' \
        "${source_dir}/" \
        "${INSTALL_DIR}/"

    # First system install: seed live config from the checkout if /opt lacks it.
    if [[ ! -f "$CONFIG_FILE" && -f "${source_dir}/config.toml" ]]; then
        echo "==> Seeding $CONFIG_FILE from checkout..."
        sudo cp "${source_dir}/config.toml" "$CONFIG_FILE"
    fi
    if [[ ! -f "$MODELS_FILE" && -f "${source_dir}/models.toml" ]]; then
        echo "==> Seeding $MODELS_FILE from checkout..."
        sudo cp "${source_dir}/models.toml" "$MODELS_FILE"
        if [[ -f "${source_dir}/models.json" ]]; then
            sudo cp "${source_dir}/models.json" "${MODELS_FILE%.toml}.json"
        fi
    fi

    fix_install_ownership
}

write_system_config() {
    local tmp
    tmp="$(mktemp)"
    cat >"$tmp" <<EOF
# GGUF Switchboard configuration (system install)
bind = "0.0.0.0:9090"
startup_timeout = 60
idle_timeout = 600
default_backend = "llama.cpp"
vram_gb = 12
auto_ngl = false
switch_drain_timeout_secs = 120
priority_load_cooldown_secs = 300
models_rescan_interval_secs = 86400
database_path = "${STATE_DIR}/usage.db"
models_file = "${MODELS_FILE}"
EOF

    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "==> Writing $CONFIG_FILE..."
        sudo install -o "$DEPLOY_OWNER" -g "$SERVICE_GROUP" -m 664 "$tmp" "$CONFIG_FILE"
        CONFIG_CREATED=true
    else
        echo "==> Keeping existing $CONFIG_FILE (ensuring absolute paths)..."
        # Absolute paths so the unit does not depend on WorkingDirectory for models_file.
        if sudo grep -q '^models_file[[:space:]]*=' "$CONFIG_FILE"; then
            sudo sed -i.bak -e "s|^models_file[[:space:]]*=.*|models_file = \"${MODELS_FILE}\"|" "$CONFIG_FILE"
        else
            echo "models_file = \"${MODELS_FILE}\"" | sudo tee -a "$CONFIG_FILE" >/dev/null
        fi
        if sudo grep -q '^database_path[[:space:]]*=' "$CONFIG_FILE"; then
            sudo sed -i.bak -e "s|^database_path[[:space:]]*=.*|database_path = \"${STATE_DIR}/usage.db\"|" "$CONFIG_FILE"
        else
            echo "database_path = \"${STATE_DIR}/usage.db\"" | sudo tee -a "$CONFIG_FILE" >/dev/null
        fi
        sudo rm -f "${CONFIG_FILE}.bak"
        sudo chown "${DEPLOY_OWNER}:${SERVICE_GROUP}" "$CONFIG_FILE"
        sudo chmod 664 "$CONFIG_FILE"
    fi
    rm -f "$tmp"
}

legacy_models_dir() {
    if [[ -n "${SUDO_USER:-}" && -d "/home/${SUDO_USER}/models" ]]; then
        printf '%s\n' "/home/${SUDO_USER}/models"
        return 0
    fi
    if [[ -n "${HOME:-}" && -d "${HOME}/models" && "${HOME}/models" != "$MODELS_DIR" ]]; then
        printf '%s\n' "${HOME}/models"
        return 0
    fi
    return 1
}

legacy_has_gguf() {
    local dir="$1"
    [[ -d "$dir" ]] && find "$dir" -type f -iname '*.gguf' -print -quit 2>/dev/null | grep -q .
}

maybe_migrate_legacy_models() {
    local force="${1:-false}"
    local legacy=""
    legacy="$(legacy_models_dir || true)"
    [[ -n "$legacy" ]] || return 0
    legacy_has_gguf "$legacy" || return 0

    if [[ "$force" != "true" ]] && legacy_has_gguf "$MODELS_DIR"; then
        echo "==> Legacy models at $legacy (canonical dir already has GGUFs; skipping copy)."
        echo "    To merge anyway: ./deploy.sh --migrate-models"
        return 0
    fi

    if [[ "$force" != "true" ]] && ! legacy_has_gguf "$MODELS_DIR"; then
        echo "==> Legacy models detected at $legacy."
        echo "    Migrating (copy) to $MODELS_DIR..."
    else
        echo "==> Migrating legacy models from $legacy → $MODELS_DIR (copy only)..."
    fi

    sudo rsync -aH "${legacy}/" "${MODELS_DIR}/"
    sudo chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$STATE_DIR"
    echo "==> Copied. Left $legacy untouched (not deleted)."
}

maybe_migrate_legacy_etc_config() {
    [[ -d "$ETC_DIR" ]] || return 0
    if [[ ! -f "$MODELS_FILE" && -r "$ETC_DIR/models.toml" ]]; then
        echo "==> Migrating $ETC_DIR/models.toml → $MODELS_FILE"
        sudo cp "$ETC_DIR/models.toml" "$MODELS_FILE"
        sudo chown "${DEPLOY_OWNER}:${SERVICE_GROUP}" "$MODELS_FILE"
        sudo chmod 664 "$MODELS_FILE"
    fi
    if [[ ! -f "$CONFIG_FILE" && -r "$ETC_DIR/config.toml" ]]; then
        echo "==> Migrating $ETC_DIR/config.toml → $CONFIG_FILE"
        sudo cp "$ETC_DIR/config.toml" "$CONFIG_FILE"
        sudo chown "${DEPLOY_OWNER}:${SERVICE_GROUP}" "$CONFIG_FILE"
        sudo chmod 664 "$CONFIG_FILE"
    fi
}

llama_server_ready() {
    [[ -x "$LLAMA_SERVER" ]] || return 1
    "$LLAMA_SERVER" --version >/dev/null 2>&1
}

registry_has_model_candidates() {
    local registry="$1" dirs dir
    sudo test -r "$registry" || return 1

    if read_config "$registry" | grep -q '^\[\[models\]\]'; then
        return 0
    fi

    dirs="$(read_models_dir_from_toml "$registry" 2>/dev/null || true)"
    [[ -n "$dirs" ]] || return 1
    IFS=',' read -r -a model_dirs <<<"$dirs"
    for dir in "${model_dirs[@]}"; do
        if [[ -d "$dir" ]] && find "$dir" -type f -iname '*.gguf' -print -quit | grep -q .; then
            return 0
        fi
    done
    return 1
}

model_setup_help() {
    local models_dir="$1"
    cat <<EOF
==> gguf-switchboard installed. No GGUF models are available yet.

Search for a model:
  ggs models search "Qwen3.5 9B"

Download a recommended quantization:
  ggs models pull lmstudio-community/Qwen3.5-9B-GGUF --quant Q4_K_M --dir "$models_dir"

Then finish service setup:
  ./deploy.sh --refresh-models
EOF
}

llama_setup_help() {
    cat <<EOF
==> gguf-switchboard installed, but $LLAMA_SERVER was not found (or is broken).

gguf-switchboard is a swap proxy — it needs llama.cpp's llama-server before models can load.

Recommended (NVIDIA / CUDA Linux):
  ./scripts/update-llama-cpp.sh

That installs llama-server to $LLAMA_SERVER, then re-run:
  ./deploy.sh
EOF
}

generate_models_toml() {
    local refresh="${1:-false}"
    local merge_source=""
    local -a discover_cmd
    local discover_models_dir="${DISCOVER_MODELS_DIR:-$MODELS_DIR}"

    if [[ "$refresh" != "true" ]] && sudo test -f "$MODELS_FILE"; then
        echo "==> Keeping existing $MODELS_FILE (pass --refresh-models to regenerate from disk)."
        print_models_dir_hints
        return 0
    fi

    if sudo test -f "$MODELS_FILE"; then
        merge_source="$MODELS_FILE"
    elif [[ -f "$INSTALL_DIR/$MODELS_TEMPLATE" ]]; then
        merge_source="$INSTALL_DIR/$MODELS_TEMPLATE"
    elif [[ -f "$MODELS_TEMPLATE" ]]; then
        merge_source="$MODELS_TEMPLATE"
    fi

    echo "==> Generating $MODELS_FILE via discover-models..."
    discover_cmd=(sudo -u "$SERVICE_USER" "$BIN" discover-models "$discover_models_dir" -o "$MODELS_FILE")
    if [[ -n "$merge_source" ]]; then
        discover_cmd+=(--merge "$merge_source")
    fi

    if "${discover_cmd[@]}"; then
        sudo chown "${DEPLOY_OWNER}:${SERVICE_GROUP}" "$MODELS_FILE"
        sudo chmod 664 "$MODELS_FILE"
        if sudo test -f "${MODELS_FILE%.toml}.json"; then
            sudo chown "${DEPLOY_OWNER}:${SERVICE_GROUP}" "${MODELS_FILE%.toml}.json"
            sudo chmod 664 "${MODELS_FILE%.toml}.json"
        fi
        # Ensure defaults point at the system layout.
        sudo sed -i.bak \
            -e "s|^models_dir[[:space:]]*=.*|models_dir = \"${MODELS_DIR}\"|" \
            -e "s|^llama_server[[:space:]]*=.*|llama_server = \"${LLAMA_SERVER}\"|" \
            "$MODELS_FILE"
        sudo rm -f "${MODELS_FILE}.bak"
        echo "==> Installed $MODELS_FILE"
        return 0
    fi

    echo "==> Warning: discover-models failed; seeding defaults if missing."
    if ! sudo test -f "$MODELS_FILE"; then
        local tmp
        tmp="$(mktemp)"
        cat >"$tmp" <<EOF
version = 1
auto_discover = true

[defaults]
models_dir = "${MODELS_DIR}"
llama_server = "${LLAMA_SERVER}"
host = "127.0.0.1"
base_port = 18081
context_size = 16384
ngl = 999
backend = "llama.cpp"
EOF
        sudo install -o "$DEPLOY_OWNER" -g "$SERVICE_GROUP" -m 664 "$tmp" "$MODELS_FILE"
        rm -f "$tmp"
        MODELS_CREATED=true
    fi
}

write_systemd_unit() {
    echo "==> Installing systemd unit ($SERVICE_FILE)..."
    sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=GGUF Switchboard - GPU-aware local GGUF model scheduler
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_DIR}
Environment=RUST_LOG=info
Environment=MODELS_DIR=${MODELS_DIR}
ExecStart=${BIN} ${CONFIG_FILE}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
}

validate_runtime_access() {
    echo "==> Validating runtime access as $SERVICE_USER..."
    local failed=0
    sudo -u "$SERVICE_USER" test -r "$CONFIG_FILE" || {
        echo "ERROR: $SERVICE_USER cannot read $CONFIG_FILE" >&2
        failed=1
    }
    sudo -u "$SERVICE_USER" test -r "$MODELS_FILE" || {
        echo "ERROR: $SERVICE_USER cannot read $MODELS_FILE" >&2
        failed=1
    }
    sudo -u "$SERVICE_USER" test -x "$BIN" || {
        echo "ERROR: $SERVICE_USER cannot execute $BIN" >&2
        failed=1
    }
    sudo -u "$SERVICE_USER" test -x "$LLAMA_SERVER" || {
        echo "ERROR: $SERVICE_USER cannot execute $LLAMA_SERVER" >&2
        failed=1
    }
    sudo -u "$SERVICE_USER" test -r "$MODELS_DIR" || {
        echo "ERROR: $SERVICE_USER cannot read $MODELS_DIR" >&2
        failed=1
    }
    sudo -u "$SERVICE_USER" test -w "$MODELS_DIR" || {
        echo "ERROR: $SERVICE_USER cannot write $MODELS_DIR" >&2
        failed=1
    }
    # usage.db parent must be writable for SQLite create/open
    sudo -u "$SERVICE_USER" test -w "$STATE_DIR" || {
        echo "ERROR: $SERVICE_USER cannot write $STATE_DIR (usage.db)" >&2
        failed=1
    }
    # Deploy user must be able to write new model files into MODELS_DIR.
    if [[ "$DEPLOY_OWNER" != "root" ]] && id "$DEPLOY_OWNER" >/dev/null 2>&1; then
        sudo -u "$DEPLOY_OWNER" test -w "$MODELS_DIR" || {
            echo "WARNING: $DEPLOY_OWNER cannot write $MODELS_DIR (ggs models pull may fail)" >&2
            echo "         Run: newgrp $SERVICE_GROUP  (or log out and back in)" >&2
            failed=1
        }
    fi
    return "$failed"
}

print_models_from_status() {
    local base_url="$1"
    local status_json loaded priority

    status_json="$(curl -sf "${base_url}/status")"
    loaded="$(printf '%s' "$status_json" | jq -r '.loaded_model // empty')"
    priority="$(printf '%s' "$status_json" | jq -r '.priority_model // empty')"

    echo "==> Available models:"
    echo ""
    printf "  %-24s %-30s %s\n" "MODEL ID" "DISPLAY NAME" "STATE"
    while IFS=$'\t' read -r id name state; do
        printf "  %-24s %-30s %s\n" "$id" "$name" "$state"
    done < <(printf '%s' "$status_json" | jq -r --arg loaded "$loaded" --arg priority "$priority" '
        .configured_models[]
        | [
            .id,
            (if .display_name != "" then .display_name else "—" end),
            ([
              (if .id == $priority then "priority" else empty end),
              (if .id == $loaded then "loaded" else empty end)
            ] | map(select(. != "")) | join(", "))
          ] | @tsv
    ')
    echo ""
}

print_deploy_checklist() {
    local models_ok="✗" db_ok="✗" svc_enabled="✗" svc_active="✗" http_ok="✗" spawn_ok="✗"
    id "$SERVICE_USER" >/dev/null 2>&1 && echo "✓ $SERVICE_USER service account exists" || echo "✗ $SERVICE_USER service account missing"
    [[ -d "$INSTALL_DIR" && -d "$MODELS_DIR" ]] && echo "✓ shared directories created" || echo "✗ shared directories missing"
    [[ -x "$BIN" ]] && echo "✓ binary installed ($BIN)" || echo "✗ binary missing"
    [[ -x "$LLAMA_SERVER" ]] && echo "✓ llama-server detected ($LLAMA_SERVER)" || echo "✗ llama-server missing"
    sudo test -f "$MODELS_FILE" && echo "✓ models registry generated" || echo "✗ models registry missing"
    sudo -u "$SERVICE_USER" test -r "$MODELS_DIR" && models_ok="✓"
    echo "$models_ok models readable by $SERVICE_USER"
    sudo -u "$SERVICE_USER" test -w "$STATE_DIR" && db_ok="✓"
    echo "$db_ok usage.db directory writable by $SERVICE_USER"
    systemctl is-enabled --quiet gguf-switchboard 2>/dev/null && svc_enabled="✓"
    echo "$svc_enabled service enabled"
    systemctl is-active --quiet gguf-switchboard 2>/dev/null && svc_active="✓"
    echo "$svc_active service running"
    if curl -fsS http://127.0.0.1:9090/health >/dev/null 2>&1; then
        http_ok="✓"
    fi
    echo "$http_ok HTTP endpoint responding"
    if sudo -u "$SERVICE_USER" find "$MODELS_DIR" -type f -iname '*.gguf' -print -quit 2>/dev/null | grep -q .; then
        spawn_ok="✓"
    fi
    echo "$spawn_ok model files present under $MODELS_DIR"
}

in_repo() {
    [[ -f "$1/Cargo.toml" ]] && grep -q 'name = "gguf-switchboard"' "$1/Cargo.toml" 2>/dev/null
}

ensure_repo() {
    local script_dir=""
    if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" != "bash" && "${BASH_SOURCE[0]}" != "-" ]]; then
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        if in_repo "$script_dir"; then
            cd "$script_dir"
            return 0
        fi
    fi

    if in_repo "$(pwd)"; then
        return 0
    fi

    if [[ -d "$REPO_DIR/.git" ]] && in_repo "$REPO_DIR"; then
        cd "$REPO_DIR"
        return 0
    fi

    echo "==> Cloning $REPO_URL (branch: $BRANCH) into $REPO_DIR..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    exec "$REPO_DIR/deploy.sh" "$@"
}

if [[ "${GGUF_SWITCHBOARD_DEPLOY_LIB:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi

ensure_repo
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SOURCE_DIR"

REFRESH_MODELS=false
SKIP_PULL=false
MIGRATE_MODELS=false
for arg in "$@"; do
    case "$arg" in
        --refresh-models)
            REFRESH_MODELS=true
            ;;
        --skip-pull)
            SKIP_PULL=true
            ;;
        --migrate-models)
            MIGRATE_MODELS=true
            ;;
        -h|--help)
            cat <<EOF
Usage: ./deploy.sh [--refresh-models] [--skip-pull] [--migrate-models]

Deploy gguf-switchboard system-wide (Linux + systemd).

Layout:
  $INSTALL_DIR
  $STATE_DIR/models
  $BIN
  $SERVICE_FILE  (User=$SERVICE_USER)

Options:
  --refresh-models   Regenerate models.toml from GGUF files on disk (merge)
  --skip-pull        Rebuild + reinstall without git fetch/checkout/pull
  --migrate-models   Copy legacy ~/models into $MODELS_DIR (never deletes)

Environment:
  MODELS_DIR / DISCOVER_MODELS_DIR   Override discover scan dirs (comma-separated)
  GGUF_SWITCHBOARD_DIR               Source checkout for clone/bootstrap only

Post-install:
  Adds a 'ggs' alias to your shell rc file (bash/zsh) when accepted.
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (try --help)" >&2
            exit 1
            ;;
    esac
done

# Discover may scan an override path; registry defaults still use canonical MODELS_DIR.
if [[ -z "${DISCOVER_MODELS_DIR:-}" || "$DISCOVER_MODELS_DIR" == "$MODELS_DIR" ]]; then
    DISCOVER_MODELS_DIR="$MODELS_DIR"
fi

echo "==> Source:  $SOURCE_DIR"
echo "==> Install: $INSTALL_DIR"
echo "==> State:   $STATE_DIR"
echo "==> Models:  $MODELS_DIR"
echo "==> Service user: $SERVICE_USER"

if [[ "$SKIP_PULL" != "true" ]]; then
    if [[ -d "$SOURCE_DIR/.git" ]]; then
        echo "==> Checking out $BRANCH..."
        git fetch origin "$BRANCH" 2>/dev/null || true
        git checkout "$BRANCH" 2>/dev/null || git checkout -B "$BRANCH" "origin/$BRANCH"

        if [[ -n "$(git status --porcelain)" ]]; then
            STASH_LABEL="deploy-auto-stash-$(date +%Y%m%d-%H%M%S)"
            echo "==> Local changes detected; stashing as '$STASH_LABEL'..."
            git stash push --include-untracked --message "$STASH_LABEL" >/dev/null
            echo "==> Stashed local changes. (Use 'git stash list' / 'git stash pop' to recover.)"
        fi

        echo "==> Pulling latest changes..."
        git pull origin "$BRANCH"
    else
        echo "==> No .git in source tree; skipping pull."
    fi
else
    echo "==> Skipping git pull (--skip-pull)."
fi

CONFIG_CREATED=false
MODELS_CREATED=false

# Stop before replacing binaries/config (do not disable).
echo "==> Stopping service (if running)..."
sudo systemctl stop gguf-switchboard 2>/dev/null || true

ensure_service_account
ensure_system_directories

# Add the deploying user to the ggs group so they can write to MODELS_DIR.
if [[ "$DEPLOY_OWNER" != "root" ]] && id "$DEPLOY_OWNER" >/dev/null 2>&1; then
    if ! id -nG "$DEPLOY_OWNER" 2>/dev/null | tr ' ' '\n' | grep -qx "$SERVICE_GROUP"; then
        echo "==> Adding $DEPLOY_OWNER to the $SERVICE_GROUP group..."
        sudo usermod -aG "$SERVICE_GROUP" "$DEPLOY_OWNER"
        echo "    NOTE: Group membership changes require logout/login."
        echo "          Alternatively run: newgrp $SERVICE_GROUP"
    fi
fi

maybe_migrate_legacy_etc_config
maybe_migrate_legacy_models "$MIGRATE_MODELS"

if command -v apt-get >/dev/null 2>&1; then
    APT_PKGS=(libssl-dev pkg-config build-essential cmake curl git jq aria2)
    NEED_APT=false
    for pkg in "${APT_PKGS[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            NEED_APT=true
            break
        fi
    done
    if [[ "$NEED_APT" == "true" ]]; then
        echo "==> Installing build dependencies..."
        sudo apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${APT_PKGS[@]}"
    else
        echo "==> Build dependencies already installed."
    fi
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "==> Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "${HOME}/.cargo/env"
fi

echo "==> Building release..."
export SWAGGER_UI_OVERWRITE_FOLDER="${SOURCE_DIR}/swagger-ui-overrides"
cargo clean -p utoipa-swagger-ui --release 2>/dev/null || true
cargo build --release

sync_project_to_install "$SOURCE_DIR"
write_system_config

echo "==> Installing binary → $BIN..."
sudo install -o root -g root -m 755 \
    "${SOURCE_DIR}/target/release/gguf-switchboard" \
    "$BIN"

if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "ERROR: $LLAMA_SERVER not found or not executable." >&2
    llama_setup_help
    exit 1
fi
if ! llama_server_ready; then
    echo "ERROR: $LLAMA_SERVER failed --version (missing libs?)." >&2
    llama_setup_help
    exit 1
fi

if [[ "$MODELS_CREATED" == "true" ]] || ! sudo test -f "$MODELS_FILE"; then
    generate_models_toml true
else
    generate_models_toml "$REFRESH_MODELS"
fi

write_systemd_unit
sudo systemctl daemon-reload

if ! registry_has_model_candidates "$MODELS_FILE"; then
    echo ""
    model_setup_help "$MODELS_DIR"
    echo ""
    echo "==> Service unit installed but not started (no GGUF models yet)."
    echo "    After pulling models: ./deploy.sh --refresh-models"
    exit 0
fi

validate_runtime_access || {
    echo "==> FAILED: runtime access validation" >&2
    exit 1
}

echo "==> Enabling and starting service..."
sudo systemctl enable --now gguf-switchboard

# Offer to add the ggs shell alias for the deploying user
SHELL_RC=""
if [[ "${SHELL:-}" == */zsh ]]; then
    SHELL_RC="${HOME}/.zshrc"
elif [[ "${SHELL:-}" == */bash ]]; then
    SHELL_RC="${HOME}/.bashrc"
fi

if [[ -n "$SHELL_RC" ]]; then
    _changed=false

    if ! grep -q "alias ggs='gguf-switchboard'" "$SHELL_RC" 2>/dev/null; then
        read -r -p "Add 'ggs' alias to $SHELL_RC? [Y/n] " REPLY
        REPLY="${REPLY:-Y}"
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            echo "" >> "$SHELL_RC"
            echo "# gguf-switchboard shortcut" >> "$SHELL_RC"
            echo "alias ggs='gguf-switchboard'" >> "$SHELL_RC"
            _changed=true
        fi
    fi

    if ! grep -q 'umask 0002' "$SHELL_RC" 2>/dev/null; then
        read -r -p "Add 'umask 0002' to $SHELL_RC? (keeps group-writable new files) [Y/n] " REPLY
        REPLY="${REPLY:-Y}"
        if [[ "$REPLY" =~ ^[Yy]$ ]]; then
            echo "" >> "$SHELL_RC"
            echo "# gguf-switchboard: group-writable files for ggs collaboration" >> "$SHELL_RC"
            echo "umask 0002" >> "$SHELL_RC"
            _changed=true
        fi
    fi

    if [[ "$_changed" == "true" ]]; then
        echo "==> Updated $SHELL_RC (run 'source $SHELL_RC' or open a new terminal)"
    fi
fi

BASE_URL="http://127.0.0.1:9090"

echo "==> Waiting for health check..."
for i in {1..30}; do
    sleep 1
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
        if ! systemctl is-active --quiet gguf-switchboard; then
            echo "==> FAILED: service is not active"
            sudo journalctl -u gguf-switchboard -n 100 --no-pager
            exit 1
        fi
        echo ""
        echo "==> Deploy complete."
        echo ""
        echo "  Install dir:   $INSTALL_DIR"
        echo "  Models dir:    $MODELS_DIR"
        echo "  Config:        $CONFIG_FILE"
        echo "  Registry:      $MODELS_FILE"
        echo "  Swagger UI:    ${BASE_URL}/swagger-ui/"
        echo "  Health:        ${BASE_URL}/health"
        echo "  Status:        ${BASE_URL}/status"
        echo ""
        echo "==> Health: $(curl -fsS "${BASE_URL}/health")"
        if command -v jq >/dev/null 2>&1; then
            print_models_from_status "$BASE_URL"
        else
            print_models_from_config "$CONFIG_FILE"
        fi
        echo "==> Checklist:"
        print_deploy_checklist
        if [[ "$REFRESH_MODELS" == "true" ]]; then
            echo "==> models.toml was regenerated from disk."
            echo "    Edit aliases / priority in $MODELS_FILE as needed, then:"
            echo "    sudo systemctl restart gguf-switchboard"
        fi
        exit 0
    fi
    echo "    waiting... ($i/30)"
done

echo "==> FAILED: service did not become healthy in 30s"
sudo journalctl -u gguf-switchboard -n 100 --no-pager
exit 1
