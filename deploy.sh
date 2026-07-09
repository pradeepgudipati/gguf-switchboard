#!/usr/bin/env bash
# Deploy gguf-switchboard as a systemd service.
#
# Environment:
#   MODELS_DIR   Directory containing .gguf files (overrides auto-detection)
#
# Flags:
#   --refresh-models   Regenerate /etc/gguf-switchboard/models.toml from disk
#
set -euo pipefail

REPO_URL="https://github.com/pradeepgudipati/gguf-switchboard.git"
REPO_DIR="${GGUF_SWITCHBOARD_DIR:-$HOME/gguf-switchboard}"
BRANCH="Dev"
SERVICE_FILE="/etc/systemd/system/gguf-switchboard.service"
CONFIG_FILE="/etc/gguf-switchboard/config.toml"
MODELS_FILE="/etc/gguf-switchboard/models.toml"

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

    if [[ -n "$models_path" && -r "$models_path" ]]; then
        while IFS= read -r block; do
            [[ -z "$block" ]] && continue
            local alias display_name priority state=""
            alias="$(printf '%s\n' "$block" | awk -F'"' '/^alias = / { print $2; exit }')"
            display_name="$(printf '%s\n' "$block" | awk -F'"' '/^display_name = / { print $2; exit }')"
            priority="$(printf '%s\n' "$block" | awk '/^priority = / { print $3; exit }' | tr -d ' ')"
            [[ "$priority" == "true" ]] && state="priority"
            [[ -n "$alias" ]] && printf "  %-24s %-30s %s\n" "$alias" "${display_name:-—}" "$state"
        done < <(awk '
            BEGIN { block = "" }
            /^\[\[models\]\]/ {
                if (block != "") { print block; block = "" }
                next
            }
            /^alias = / || /^display_name = / || /^priority = / {
                block = block $0 "\n"
            }
            END { if (block != "") print block }
        ' "$models_path")
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
    [[ -r "$file" ]] || return 1
    awk -F'"' '/^models_dir\s*=/ { print $2; exit }' "$file"
}

detect_models_dir() {
    if [[ -n "${MODELS_DIR:-}" ]]; then
        echo "$MODELS_DIR"
        return 0
    fi

    for candidate in "$MODELS_FILE" "models.toml"; do
        if [[ -f "$candidate" ]]; then
            dir="$(read_models_dir_from_toml "$candidate" || true)"
            if [[ -n "$dir" ]]; then
                echo "$dir"
                return 0
            fi
        fi
    done

    return 1
}

generate_models_toml() {
    local refresh="${1:-false}"
    local models_dir merge_source generated="models.toml.generated"
    local -a discover_cmd

    if [[ "$refresh" != "true" && -f "$MODELS_FILE" ]]; then
        echo "==> Keeping existing $MODELS_FILE (pass --refresh-models to regenerate from disk)."
        if models_dir="$(detect_models_dir 2>/dev/null || true)" && [[ -n "$models_dir" ]]; then
            echo "    Detected models on disk at: $models_dir"
            echo "    To regenerate models.toml from that directory:"
            echo "    MODELS_DIR=$models_dir ./deploy.sh --refresh-models"
        fi
        return 0
    fi

    merge_source=""
    if [[ -f "$MODELS_FILE" ]]; then
        merge_source="$MODELS_FILE"
    elif [[ -f "models.toml" ]]; then
        merge_source="models.toml"
    fi

    if ! models_dir="$(detect_models_dir)"; then
        echo "==> Warning: Could not find a models directory."
        echo "    Set MODELS_DIR or edit models_dir in models.toml, then re-run with --refresh-models."
        if [[ ! -f "$MODELS_FILE" && -f "models.toml" ]]; then
            echo "==> Copying template models.toml to $MODELS_FILE..."
            sudo cp models.toml "$MODELS_FILE"
        fi
        return 0
    fi

    echo "==> Generating models.toml from $models_dir..."
    discover_cmd=(./target/release/gguf-switchboard discover-models "$models_dir" -o "$generated")
    if [[ -n "$merge_source" ]]; then
        discover_cmd+=(--merge "$merge_source")
    fi

    if "${discover_cmd[@]}"; then
        sudo cp "$generated" "$MODELS_FILE"
        rm -f "$generated"
        echo "==> Installed $MODELS_FILE"
        return 0
    fi

    echo "==> Warning: discover-models failed; keeping existing models.toml if present."
    rm -f "$generated"
    if [[ ! -f "$MODELS_FILE" && -f "models.toml" ]]; then
        sudo cp models.toml "$MODELS_FILE"
    fi
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

    echo "==> Cloning $REPO_URL (branch: $BRANCH)..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    exec "$REPO_DIR/deploy.sh" "$@"
}

ensure_repo

REFRESH_MODELS=false
for arg in "$@"; do
    case "$arg" in
        --refresh-models)
            REFRESH_MODELS=true
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ./deploy.sh [--refresh-models]

Deploy gguf-switchboard as a systemd service.

Options:
  --refresh-models   Regenerate /etc/gguf-switchboard/models.toml from GGUF files on disk

Environment:
  MODELS_DIR         Directory containing .gguf files (overrides auto-detection)
  GGUF_SWITCHBOARD_DIR  Repo checkout path (default: ~/gguf-switchboard)
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (try --help)" >&2
            exit 1
            ;;
    esac
done

echo "==> Checking out $BRANCH..."
git fetch origin "$BRANCH" 2>/dev/null || true
git checkout "$BRANCH" 2>/dev/null || git checkout -B "$BRANCH" "origin/$BRANCH"

if [[ -n "$(git status --porcelain)" ]]; then
    STASH_LABEL="deploy-auto-stash-$(date +%Y%m%d-%H%M%S)"
    echo "==> Local changes detected; stashing as '$STASH_LABEL'..."
    git stash push --include-untracked --message "$STASH_LABEL" >/dev/null
    echo "==> Stashed local changes. (Use 'git stash list' to review.)"
fi

echo "==> Pulling latest changes..."
git pull origin "$BRANCH"

if command -v apt-get >/dev/null 2>&1; then
    echo "==> Installing build dependencies..."
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libssl-dev pkg-config build-essential cmake curl git jq
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "==> Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "${HOME}/.cargo/env"
fi

echo "==> Building release..."
export SWAGGER_UI_OVERWRITE_FOLDER="$(pwd)/swagger-ui-overrides"
cargo clean -p utoipa-swagger-ui 2>/dev/null || true
cargo build --release

echo "==> Ensuring runtime directories..."
sudo mkdir -p /etc/gguf-switchboard /var/lib/gguf-switchboard
sudo chown "$(whoami)":"$(whoami)" /var/lib/gguf-switchboard

CONFIG_CREATED=false

# Create service if it doesn't exist
if [[ ! -f "$SERVICE_FILE" ]]; then
    echo "==> Creating systemd service..."

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=GGUF Switchboard - Local LLM Inference Server
After=network.target

[Service]
Type=simple
User=$(whoami)
ExecStart=/usr/local/bin/gguf-switchboard $CONFIG_FILE
Restart=on-failure
RestartSec=5
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable gguf-switchboard
    echo "==> Service created and enabled."
fi

# Copy default config if missing
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "==> Copying default config to $CONFIG_FILE..."
    sudo cp config.toml "$CONFIG_FILE"
    CONFIG_CREATED=true
fi

generate_models_toml "$REFRESH_MODELS"

echo "==> Stopping service..."
sudo systemctl stop gguf-switchboard || true

echo "==> Installing binary..."
sudo cp target/release/gguf-switchboard /usr/local/bin/

echo "==> Starting service..."
sudo systemctl start gguf-switchboard

# Resolve bind address from config (default 0.0.0.0:9090)
BIND_ADDR="$(read_config "$CONFIG_FILE" | grep -E '^bind\s*=' | head -1 | sed -E 's/^bind\s*=\s*"([^"]+)".*/\1/' || true)"
BIND_ADDR="${BIND_ADDR:-0.0.0.0:9090}"
BASE_URL="http://${BIND_ADDR/0.0.0.0/localhost}"

echo "==> Waiting for health check..."
for i in {1..15}; do
    sleep 1
    if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
        echo ""
        echo "==> Deploy complete."
        echo ""
        echo "  Swagger UI:  ${BASE_URL}/swagger-ui/"
        echo "               (use the Model dropdown in the top bar — applies to all APIs)"
        echo "  OpenAPI spec:  ${BASE_URL}/api-docs/openapi.json"
        echo "  Health:        ${BASE_URL}/health"
        echo "  Status:        ${BASE_URL}/status"
        echo ""
        echo "==> Health: $(curl -s "${BASE_URL}/health")"
        echo "==> Status: $(curl -s "${BASE_URL}/status")"
        if command -v jq >/dev/null 2>&1; then
            print_models_from_status "$BASE_URL"
        else
            print_models_from_config "$CONFIG_FILE"
        fi
        if [[ "$CONFIG_CREATED" == "true" ]]; then
            echo "==> Next step: place GGUF files in models_dir (default /models),"
            echo "    set models_dir in $MODELS_FILE if needed, then re-run:"
            echo "    MODELS_DIR=/path/to/models ./deploy.sh --refresh-models"
            echo "    Or edit $CONFIG_FILE / $MODELS_FILE manually and restart:"
            echo "    sudo systemctl restart gguf-switchboard"
        elif [[ "$REFRESH_MODELS" == "true" ]]; then
            echo "==> models.toml was regenerated from disk."
            echo "    Edit aliases, display_name, or priority in $MODELS_FILE as needed."
        fi
        exit 0
    fi
    echo "    waiting... ($i/15)"
done

echo "==> FAILED: service did not become healthy in 15s"
journalctl -u gguf-switchboard --no-pager -n 10
exit 1
