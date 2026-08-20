#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/proc/122744" "$TMP/proc/122745" "$TMP/bin"

cat >"$TMP/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  '')
    printf '%s\n' \
      '+-----------------------------------------------------------------------------+' \
      '| NVIDIA-SMI 580.65.06    Driver Version: 580.65.06    CUDA Version: 13.0     |' \
      '+-----------------------------------------------------------------------------+'
    ;;
  *--query-gpu=index,uuid*)
    printf '%s\n' '0, GPU-aaaa' '1, GPU-bbbb'
    ;;
  *--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory*)
    printf '%s\n' \
      'GPU-aaaa, 122744, /usr/local/bin/llama-server, 5982' \
      'GPU-bbbb, 122745, /usr/bin/python3, 512' \
      'GPU-bbbb, 122746, /usr/local/bin/llama-server, 1024'
    ;;
  *)
    exit 2
    ;;
esac
EOF
chmod +x "$TMP/bin/nvidia-smi"

printf '/usr/local/bin/llama-server\0-m\0/models/Qwen3.5-9B-Q4_K_M.gguf\0-c\08192\0' \
  >"$TMP/proc/122744/cmdline"
printf '/usr/bin/python3\0worker.py\0--model\0classifier-v2\0' \
  >"$TMP/proc/122745/cmdline"

output="$(NVIDIA_SMI_BIN="$TMP/bin/nvidia-smi" PROC_ROOT="$TMP/proc" \
  "$ROOT/scripts/nvidia-smi-models.sh")"

grep -q 'NVIDIA-SMI 580.65.06' <<<"$output"
dashboard_line="$(grep -n 'NVIDIA-SMI 580.65.06' <<<"$output" | cut -d: -f1)"
model_header_line="$(grep -n '^GPU[[:space:]]\+PID' <<<"$output" | cut -d: -f1)"
test "$dashboard_line" -lt "$model_header_line"
grep -Eq '^GPU[[:space:]]+PID[[:space:]]+VRAM[[:space:]]+MODEL[[:space:]]+PROCESS$' <<<"$output"
grep -Eq '^0[[:space:]]+122744[[:space:]]+5982 MiB[[:space:]]+Qwen3\.5-9B-Q4_K_M\.gguf[[:space:]]+/usr/local/bin/llama-server$' <<<"$output"
grep -Eq '^1[[:space:]]+122745[[:space:]]+512 MiB[[:space:]]+classifier-v2[[:space:]]+/usr/bin/python3$' <<<"$output"
grep -Eq '^1[[:space:]]+122746[[:space:]]+1024 MiB[[:space:]]+-[[:space:]]+/usr/local/bin/llama-server$' <<<"$output"

echo "nvidia-smi model display tests passed"
