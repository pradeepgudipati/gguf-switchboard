#!/usr/bin/env bash
set -euo pipefail

SERVICE_FILE="/etc/systemd/system/openai-runtime.service"
CONFIG_FILE="/etc/openai-runtime/config.toml"

echo "==> Pulling latest changes..."
git pull

echo "==> Building release..."
cargo build --release

# Create service if it doesn't exist
if [ ! -f "$SERVICE_FILE" ]; then
    echo "==> Creating systemd service..."
    sudo mkdir -p /etc/openai-runtime /var/lib/openai-runtime
    sudo chown "$(whoami)": "$(whoami)" /var/lib/openai-runtime

    sudo tee "$SERVICE_FILE" > /dev/null << EOF
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
if [ ! -f "$CONFIG_FILE" ]; then
    echo "==> Copying default config to $CONFIG_FILE..."
    sudo cp config.toml "$CONFIG_FILE"
    echo "==> Edit $CONFIG_FILE to match your setup, then re-run this script."
    exit 0
fi

echo "==> Stopping service..."
sudo systemctl stop openai-runtime || true

echo "==> Installing binary..."
sudo cp target/release/openai-runtime /usr/local/bin/

echo "==> Starting service..."
sudo systemctl start openai-runtime

echo "==> Waiting for health check..."
for i in {1..15}; do
    sleep 1
    if curl -sf http://localhost:9090/health > /dev/null 2>&1; then
        echo "==> Health: $(curl -s http://localhost:9090/health)"
        echo "==> Status: $(curl -s http://localhost:9090/status)"
        echo "==> Deploy complete."
        exit 0
    fi
    echo "    waiting... ($i/15)"
done

echo "==> FAILED: service did not become healthy in 15s"
journalctl -u openai-runtime --no-pager -n 10
exit 1
