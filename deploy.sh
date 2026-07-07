#!/usr/bin/env bash
set -euo pipefail

echo "==> Pulling latest changes..."
git pull

echo "==> Building release..."
cargo build --release

echo "==> Stopping service..."
sudo systemctl stop openai-runtime

echo "==> Installing binary..."
sudo cp target/release/openai-runtime /usr/local/bin/

echo "==> Starting service..."
sudo systemctl start openai-runtime

echo "==> Waiting for health check..."
for i in {1..15}; do
    sleep 1
    if curl -sf http://localhost:9090/health > /dev/null 2>&1; then
        echo "==> Health: $(curl -s http://localhost:9090/health)"
        echo "==> Status: $(curl -s http://localhost:9090/status | python3 -m json.tool 2>/dev/null || curl -s http://localhost:9090/status)"
        echo "==> Deploy complete."
        exit 0
    fi
    echo "    waiting... ($i/15)"
done

echo "==> FAILED: service did not become healthy in 15s"
journalctl -u openai-runtime --no-pager -n 10
exit 1
