#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/pradeepgudipati/gguf-switchboard.git"
REPO_DIR="${GGUF_SWITCHBOARD_DIR:-$HOME/gguf-switchboard}"
BRANCH="Dev"
SERVICE_FILE="/etc/systemd/system/openai-runtime.service"
CONFIG_FILE="/etc/openai-runtime/config.toml"

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

    echo "==> Available models:"
    echo ""
    printf "  %-24s %-30s %s\n" "MODEL ID" "DISPLAY NAME" "STATE"
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
    [[ -f "$1/Cargo.toml" ]] && grep -q 'name = "openai-runtime"' "$1/Cargo.toml" 2>/dev/null
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

echo "==> Checking out $BRANCH..."
git fetch origin "$BRANCH" 2>/dev/null || true
git checkout "$BRANCH" 2>/dev/null || git checkout -B "$BRANCH" "origin/$BRANCH"

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
sudo mkdir -p /etc/openai-runtime /var/lib/openai-runtime
sudo chown "$(whoami)":"$(whoami)" /var/lib/openai-runtime

CONFIG_CREATED=false

# Create service if it doesn't exist
if [[ ! -f "$SERVICE_FILE" ]]; then
    echo "==> Creating systemd service..."

    sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=OpenAI Runtime - Local LLM Inference Server
After=network.target

[Service]
Type=simple
User=$(whoami)
ExecStart=/usr/local/bin/openai-runtime $CONFIG_FILE
Restart=on-failure
RestartSec=5
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable openai-runtime
    echo "==> Service created and enabled."
fi

# Copy default config if missing
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "==> Copying default config to $CONFIG_FILE..."
    sudo cp config.toml "$CONFIG_FILE"
    CONFIG_CREATED=true
    print_models_from_config "$CONFIG_FILE"
fi

echo "==> Stopping service..."
sudo systemctl stop openai-runtime || true

echo "==> Installing binary..."
sudo cp target/release/openai-runtime /usr/local/bin/

echo "==> Starting service..."
sudo systemctl start openai-runtime

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
            echo "==> Next step: edit $CONFIG_FILE with your llama-server path and GGUF model files,"
            echo "    then restart: sudo systemctl restart openai-runtime"
        fi
        exit 0
    fi
    echo "    waiting... ($i/15)"
done

echo "==> FAILED: service did not become healthy in 15s"
journalctl -u openai-runtime --no-pager -n 10
exit 1
