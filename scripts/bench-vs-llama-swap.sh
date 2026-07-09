#!/usr/bin/env bash
# Benchmarks gguf-switchboard against llama-swap on the same hardware, same
# llama-server binary, and same GGUF model(s). Both tools are thin proxies in
# front of llama-server, so raw token-generation speed is identical between
# them by construction — this measures the proxy layer itself: request
# overhead, model-swap latency, memory footprint, and concurrent throughput.
#
# Usage:
#   LLAMA_SERVER_BIN=/usr/local/bin/llama-server \
#   MODEL_A_PATH=/models/model-a.gguf \
#   MODEL_B_PATH=/models/model-b.gguf \
#   ./scripts/bench-vs-llama-swap.sh
#
# Required env vars:
#   LLAMA_SERVER_BIN   Path to the llama-server binary
#   MODEL_A_PATH       GGUF file used as the "already loaded / warm" model
#
# Optional env vars:
#   MODEL_B_PATH             Second GGUF file to swap into (defaults to MODEL_A_PATH,
#                             which still forces a real process stop/start cycle,
#                             just without a genuinely different model on disk)
#   GGUF_SWITCHBOARD_BIN     Path to a built gguf-switchboard binary
#                             (default: <repo>/target/release/gguf-switchboard)
#   LLAMA_SWAP_BIN           Path to an existing llama-swap binary
#                             (default: downloaded automatically into ./.bench/bin)
#   CTX_SIZE                 llama-server -c value (default: 4096)
#   NGL                      llama-server -ngl value (default: 999)
#   REQUESTS                 Requests for the throughput test (default: 200)
#   CONCURRENCY              Concurrency for the throughput test (default: 8)
#   LATENCY_SAMPLES          Sequential requests for the overhead/latency test (default: 50)
#   SWAP_ROUNDS               Number of A→B→A swap cycles to time (default: 5)
#   SKIP_DIRECT_BASELINE      Set to 1 to skip the bare llama-server baseline
#                             (saves VRAM if you can't run two model instances at once)
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
WORK_DIR="$REPO_ROOT/.bench"
BIN_DIR="$WORK_DIR/bin"
RESULTS_DIR="$WORK_DIR/results-$(date +%Y%m%d-%H%M%S)"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:?set LLAMA_SERVER_BIN to your llama-server binary path}"
MODEL_A_PATH="${MODEL_A_PATH:?set MODEL_A_PATH to a GGUF model file}"
MODEL_B_PATH="${MODEL_B_PATH:-$MODEL_A_PATH}"
GGUF_SWITCHBOARD_BIN="${GGUF_SWITCHBOARD_BIN:-$REPO_ROOT/target/release/gguf-switchboard}"
LLAMA_SWAP_BIN="${LLAMA_SWAP_BIN:-}"
CTX_SIZE="${CTX_SIZE:-4096}"
NGL="${NGL:-999}"
REQUESTS="${REQUESTS:-200}"
CONCURRENCY="${CONCURRENCY:-8}"
LATENCY_SAMPLES="${LATENCY_SAMPLES:-50}"
SWAP_ROUNDS="${SWAP_ROUNDS:-5}"
SKIP_DIRECT_BASELINE="${SKIP_DIRECT_BASELINE:-0}"

DIRECT_PORT=18000
GSB_PORT=19090
LLAMA_SWAP_PORT=19091
GSB_MODEL_A_PORT=19180
GSB_MODEL_B_PORT=19181

log() { printf '\n==> %s\n' "$1"; }
die() { printf 'FATAL: %s\n' "$1" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "required tool '$1' not found in PATH"; }

# ─── Preflight ──────────────────────────────────────────────────────────────
for c in curl jq awk sort ps tar; do need_cmd "$c"; done
[[ -x "$LLAMA_SERVER_BIN" ]] || die "LLAMA_SERVER_BIN ($LLAMA_SERVER_BIN) is not executable"
[[ -f "$MODEL_A_PATH" ]] || die "MODEL_A_PATH ($MODEL_A_PATH) not found"
[[ -f "$MODEL_B_PATH" ]] || die "MODEL_B_PATH ($MODEL_B_PATH) not found"
HEY_BIN="$(command -v hey || true)"

mkdir -p "$BIN_DIR" "$RESULTS_DIR"

if [[ ! -x "$GGUF_SWITCHBOARD_BIN" ]]; then
    log "Building gguf-switchboard release binary"
    (cd "$REPO_ROOT" && cargo build --release)
    GGUF_SWITCHBOARD_BIN="$REPO_ROOT/target/release/gguf-switchboard"
fi

if [[ -z "$LLAMA_SWAP_BIN" ]]; then
    LLAMA_SWAP_BIN="$BIN_DIR/llama-swap"
    if [[ ! -x "$LLAMA_SWAP_BIN" ]]; then
        log "Downloading latest llama-swap release"
        need_cmd jq
        asset_url="$(curl -fsSL https://api.github.com/repos/mostlygeek/llama-swap/releases/latest \
            | jq -r '.assets[] | select(.name | test("linux_amd64\\.tar\\.gz$")) | .browser_download_url')"
        [[ -n "$asset_url" ]] || die "could not find a linux_amd64 llama-swap release asset"
        curl -fsSL "$asset_url" -o "$BIN_DIR/llama-swap.tar.gz"
        tar -xzf "$BIN_DIR/llama-swap.tar.gz" -C "$BIN_DIR" llama-swap
        chmod +x "$LLAMA_SWAP_BIN"
    fi
fi
[[ -x "$LLAMA_SWAP_BIN" ]] || die "llama-swap binary not found/executable at $LLAMA_SWAP_BIN"

log "Binaries:"
echo "  gguf-switchboard: $GGUF_SWITCHBOARD_BIN"
echo "  llama-swap:       $LLAMA_SWAP_BIN"
echo "  llama-server:     $LLAMA_SERVER_BIN"
echo "  model A:          $MODEL_A_PATH"
echo "  model B:          $MODEL_B_PATH"

# ─── Helpers ────────────────────────────────────────────────────────────────

wait_for_health() {
    local url="$1" timeout="${2:-60}" waited=0
    while ! curl -fsS -o /dev/null "$url" 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
        [[ "$waited" -ge "$timeout" ]] && die "timed out waiting for $url"
    done
}

stop_pid() {
    local pid="$1"
    kill -TERM "$pid" 2>/dev/null || return 0
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    kill -KILL "$pid" 2>/dev/null || true
}

# percentiles from a newline-delimited file of seconds (as produced by curl -w '%{time_total}\n')
percentiles() {
    local file="$1"
    sort -n "$file" | awk '
        { a[NR] = $1 }
        END {
            n = NR
            if (n == 0) { print "0 0 0 0"; exit }
            p50 = a[int(n * 0.50) < 1 ? 1 : int(n * 0.50)]
            p95 = a[int(n * 0.95) < 1 ? 1 : int(n * 0.95)]
            p99 = a[int(n * 0.99) < 1 ? 1 : int(n * 0.99)]
            sum = 0
            for (i = 1; i <= n; i++) sum += a[i]
            printf "%.4f %.4f %.4f %.4f", sum / n, p50, p95, p99
        }
    '
}

sample_rss_kb() { ps -o rss= -p "$1" 2>/dev/null | tr -d ' '; }

chat_payload() {
    local model="$1" max_tokens="${2:-8}"
    jq -nc --arg model "$model" --argjson max_tokens "$max_tokens" \
        '{model: $model, messages: [{role: "user", content: "Say hi"}], max_tokens: $max_tokens, stream: false}'
}

latency_test() {
    local url="$1" model="$2" out="$3"
    : > "$out"
    for _ in $(seq 1 "$LATENCY_SAMPLES"); do
        curl -s -o /dev/null -w '%{time_total}\n' \
            -H 'Content-Type: application/json' \
            -d "$(chat_payload "$model")" \
            "$url" >> "$out"
    done
}

throughput_test() {
    local url="$1" model="$2" out="$3"
    if [[ -n "$HEY_BIN" ]]; then
        "$HEY_BIN" -n "$REQUESTS" -c "$CONCURRENCY" -m POST \
            -H 'Content-Type: application/json' \
            -d "$(chat_payload "$model")" \
            "$url" > "$out" 2>&1
    else
        {
            echo "hey not found; falling back to sequential curl (results are NOT concurrency-representative)"
            local t0 t1
            t0=$(date +%s.%N)
            for _ in $(seq 1 "$REQUESTS"); do
                curl -s -o /dev/null \
                    -H 'Content-Type: application/json' \
                    -d "$(chat_payload "$model")" \
                    "$url"
            done
            t1=$(date +%s.%N)
            awk -v t0="$t0" -v t1="$t1" -v n="$REQUESTS" \
                'BEGIN { printf "elapsed: %.2fs, req/s: %.2f\n", t1 - t0, n / (t1 - t0) }'
        } > "$out"
    fi
}

swap_latency_test() {
    local base_url="$1" model_a="$2" model_b="$3" out="$4"
    : > "$out"
    for _ in $(seq 1 "$SWAP_ROUNDS"); do
        curl -s -o /dev/null -w '%{time_total}\n' \
            -H 'Content-Type: application/json' \
            -d "$(chat_payload "$model_b")" \
            "$base_url" >> "$out"
        curl -s -o /dev/null -w '%{time_total}\n' \
            -H 'Content-Type: application/json' \
            -d "$(chat_payload "$model_a")" \
            "$base_url" >> "$out"
    done
}

rss_during() {
    local pid="$1" out="$2"
    : > "$out"
    while kill -0 "$pid" 2>/dev/null; do
        sample_rss_kb "$pid" >> "$out"
        sleep 0.5
    done
}

# ─── Optional: bare llama-server baseline (no proxy at all) ───────────────
DIRECT_PID=""
if [[ "$SKIP_DIRECT_BASELINE" != "1" ]]; then
    log "Starting bare llama-server baseline on :$DIRECT_PORT (model A)"
    "$LLAMA_SERVER_BIN" -m "$MODEL_A_PATH" --host 127.0.0.1 --port "$DIRECT_PORT" \
        -c "$CTX_SIZE" -ngl "$NGL" > "$RESULTS_DIR/direct-llama-server.log" 2>&1 &
    DIRECT_PID=$!
    wait_for_health "http://127.0.0.1:$DIRECT_PORT/health" 120

    log "Baseline latency test ($LATENCY_SAMPLES sequential requests)"
    latency_test "http://127.0.0.1:$DIRECT_PORT/v1/chat/completions" "local" \
        "$RESULTS_DIR/direct-latency.txt"

    stop_pid "$DIRECT_PID"
    DIRECT_PID=""
fi

# ─── gguf-switchboard ───────────────────────────────────────────────────────
log "Configuring gguf-switchboard"
GSB_CONFIG="$RESULTS_DIR/gguf-switchboard.toml"
cat > "$GSB_CONFIG" <<EOF
bind = "127.0.0.1:$GSB_PORT"
startup_timeout = 120
idle_timeout = 86400
default_backend = "llama.cpp"
database_path = "$RESULTS_DIR/gguf-switchboard-usage.db"

[models.model-a]
backend = "llama.cpp"
display_name = "Model A"
command = "$LLAMA_SERVER_BIN"
args = ["-m", "$MODEL_A_PATH", "--host", "127.0.0.1", "--port", "$GSB_MODEL_A_PORT", "-c", "$CTX_SIZE", "-ngl", "$NGL"]
backend_url = "http://127.0.0.1:$GSB_MODEL_A_PORT/v1"
health_url = "http://127.0.0.1:$GSB_MODEL_A_PORT/health"
priority = true

[models.model-b]
backend = "llama.cpp"
display_name = "Model B"
command = "$LLAMA_SERVER_BIN"
args = ["-m", "$MODEL_B_PATH", "--host", "127.0.0.1", "--port", "$GSB_MODEL_B_PORT", "-c", "$CTX_SIZE", "-ngl", "$NGL"]
backend_url = "http://127.0.0.1:$GSB_MODEL_B_PORT/v1"
health_url = "http://127.0.0.1:$GSB_MODEL_B_PORT/health"
priority = false
EOF

log "Starting gguf-switchboard on :$GSB_PORT"
"$GGUF_SWITCHBOARD_BIN" "$GSB_CONFIG" > "$RESULTS_DIR/gguf-switchboard.log" 2>&1 &
GSB_PID=$!
wait_for_health "http://127.0.0.1:$GSB_PORT/health" 60

log "Warming model A"
curl -s -o /dev/null -d "$(chat_payload model-a)" -H 'Content-Type: application/json' \
    "http://127.0.0.1:$GSB_PORT/v1/chat/completions"

log "gguf-switchboard: idle RSS"
sample_rss_kb "$GSB_PID" > "$RESULTS_DIR/gsb-rss-idle.txt"

log "gguf-switchboard: latency test ($LATENCY_SAMPLES sequential requests, model already warm)"
latency_test "http://127.0.0.1:$GSB_PORT/v1/chat/completions" "model-a" \
    "$RESULTS_DIR/gsb-latency.txt"

log "gguf-switchboard: model swap latency ($SWAP_ROUNDS A<->B round trips)"
swap_latency_test "http://127.0.0.1:$GSB_PORT/v1/chat/completions" "model-a" "model-b" \
    "$RESULTS_DIR/gsb-swap.txt"

log "gguf-switchboard: throughput test (requests=$REQUESTS, concurrency=$CONCURRENCY)"
curl -s -o /dev/null -d "$(chat_payload model-a)" -H 'Content-Type: application/json' \
    "http://127.0.0.1:$GSB_PORT/v1/chat/completions" # re-warm model-a after swap test
rss_during "$GSB_PID" "$RESULTS_DIR/gsb-rss-load.txt" &
RSS_WATCH_PID=$!
throughput_test "http://127.0.0.1:$GSB_PORT/v1/chat/completions" "model-a" \
    "$RESULTS_DIR/gsb-throughput.txt"
kill "$RSS_WATCH_PID" 2>/dev/null || true

log "Stopping gguf-switchboard"
stop_pid "$GSB_PID"

# ─── llama-swap ─────────────────────────────────────────────────────────────
log "Configuring llama-swap"
LLAMA_SWAP_CONFIG="$RESULTS_DIR/llama-swap.yaml"
cat > "$LLAMA_SWAP_CONFIG" <<EOF
healthCheckTimeout: 120
logLevel: info
startPort: 19280
models:
  "model-a":
    cmd: |
      $LLAMA_SERVER_BIN -m $MODEL_A_PATH --host 127.0.0.1 --port \${PORT} -c $CTX_SIZE -ngl $NGL
    proxy: "http://127.0.0.1:\${PORT}"
    checkEndpoint: /health
    ttl: 0
  "model-b":
    cmd: |
      $LLAMA_SERVER_BIN -m $MODEL_B_PATH --host 127.0.0.1 --port \${PORT} -c $CTX_SIZE -ngl $NGL
    proxy: "http://127.0.0.1:\${PORT}"
    checkEndpoint: /health
    ttl: 0
EOF

log "Starting llama-swap on :$LLAMA_SWAP_PORT"
"$LLAMA_SWAP_BIN" --config "$LLAMA_SWAP_CONFIG" --listen "127.0.0.1:$LLAMA_SWAP_PORT" \
    > "$RESULTS_DIR/llama-swap.log" 2>&1 &
LS_PID=$!
wait_for_health "http://127.0.0.1:$LLAMA_SWAP_PORT/health" 60

log "Warming model A"
curl -s -o /dev/null -d "$(chat_payload model-a)" -H 'Content-Type: application/json' \
    "http://127.0.0.1:$LLAMA_SWAP_PORT/v1/chat/completions"

log "llama-swap: idle RSS"
sample_rss_kb "$LS_PID" > "$RESULTS_DIR/ls-rss-idle.txt"

log "llama-swap: latency test ($LATENCY_SAMPLES sequential requests, model already warm)"
latency_test "http://127.0.0.1:$LLAMA_SWAP_PORT/v1/chat/completions" "model-a" \
    "$RESULTS_DIR/ls-latency.txt"

log "llama-swap: model swap latency ($SWAP_ROUNDS A<->B round trips)"
swap_latency_test "http://127.0.0.1:$LLAMA_SWAP_PORT/v1/chat/completions" "model-a" "model-b" \
    "$RESULTS_DIR/ls-swap.txt"

log "llama-swap: throughput test (requests=$REQUESTS, concurrency=$CONCURRENCY)"
curl -s -o /dev/null -d "$(chat_payload model-a)" -H 'Content-Type: application/json' \
    "http://127.0.0.1:$LLAMA_SWAP_PORT/v1/chat/completions" # re-warm model-a after swap test
rss_during "$LS_PID" "$RESULTS_DIR/ls-rss-load.txt" &
RSS_WATCH_PID=$!
throughput_test "http://127.0.0.1:$LLAMA_SWAP_PORT/v1/chat/completions" "model-a" \
    "$RESULTS_DIR/ls-throughput.txt"
kill "$RSS_WATCH_PID" 2>/dev/null || true

log "Stopping llama-swap"
stop_pid "$LS_PID"

# ─── Report ─────────────────────────────────────────────────────────────────
REPORT="$RESULTS_DIR/report.md"
{
    echo "# gguf-switchboard vs llama-swap — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "Model A: \`$MODEL_A_PATH\`  "
    echo "Model B: \`$MODEL_B_PATH\`  "
    echo "Context: $CTX_SIZE, GPU layers: $NGL"
    echo

    if [[ -f "$RESULTS_DIR/direct-latency.txt" ]]; then
        read -r d_avg d_p50 d_p95 d_p99 <<< "$(percentiles "$RESULTS_DIR/direct-latency.txt")"
        echo "## Baseline: bare llama-server (no proxy)"
        echo
        echo "| avg | p50 | p95 | p99 |"
        echo "|-----|-----|-----|-----|"
        echo "| ${d_avg}s | ${d_p50}s | ${d_p95}s | ${d_p99}s |"
        echo
    fi

    echo "## Request latency (warm model, sequential, seconds)"
    echo
    echo "| Tool | avg | p50 | p95 | p99 |"
    echo "|------|-----|-----|-----|-----|"
    read -r g_avg g_p50 g_p95 g_p99 <<< "$(percentiles "$RESULTS_DIR/gsb-latency.txt")"
    read -r l_avg l_p50 l_p95 l_p99 <<< "$(percentiles "$RESULTS_DIR/ls-latency.txt")"
    echo "| gguf-switchboard | ${g_avg}s | ${g_p50}s | ${g_p95}s | ${g_p99}s |"
    echo "| llama-swap | ${l_avg}s | ${l_p50}s | ${l_p95}s | ${l_p99}s |"
    echo

    echo "## Model swap latency (A<->B round trips, seconds)"
    echo
    echo "| Tool | avg | p50 | p95 | p99 |"
    echo "|------|-----|-----|-----|-----|"
    read -r gs_avg gs_p50 gs_p95 gs_p99 <<< "$(percentiles "$RESULTS_DIR/gsb-swap.txt")"
    read -r ls_avg ls_p50 ls_p95 ls_p99 <<< "$(percentiles "$RESULTS_DIR/ls-swap.txt")"
    echo "| gguf-switchboard | ${gs_avg}s | ${gs_p50}s | ${gs_p95}s | ${gs_p99}s |"
    echo "| llama-swap | ${ls_avg}s | ${ls_p50}s | ${ls_p95}s | ${ls_p99}s |"
    echo

    echo "## Proxy process memory (RSS, MB)"
    echo
    echo "| Tool | idle | max under load |"
    echo "|------|------|-----------------|"
    gsb_idle=$(awk '{print $1/1024}' "$RESULTS_DIR/gsb-rss-idle.txt")
    ls_idle=$(awk '{print $1/1024}' "$RESULTS_DIR/ls-rss-idle.txt")
    gsb_max=$(sort -n "$RESULTS_DIR/gsb-rss-load.txt" 2>/dev/null | tail -1 | awk '{print $1/1024}')
    ls_max=$(sort -n "$RESULTS_DIR/ls-rss-load.txt" 2>/dev/null | tail -1 | awk '{print $1/1024}')
    printf '| gguf-switchboard | %.1f MB | %.1f MB |\n' "${gsb_idle:-0}" "${gsb_max:-0}"
    printf '| llama-swap | %.1f MB | %.1f MB |\n' "${ls_idle:-0}" "${ls_max:-0}"
    echo
    echo "Note: this is the proxy process itself, not the llama-server child process(es) it spawns — those are identical either way."
    echo

    echo "## Throughput (requests=$REQUESTS, concurrency=$CONCURRENCY)"
    echo
    echo "### gguf-switchboard"
    echo '```'
    cat "$RESULTS_DIR/gsb-throughput.txt"
    echo '```'
    echo
    echo "### llama-swap"
    echo '```'
    cat "$RESULTS_DIR/ls-throughput.txt"
    echo '```'
} > "$REPORT"

log "Done. Full results in $RESULTS_DIR"
cat "$REPORT"
