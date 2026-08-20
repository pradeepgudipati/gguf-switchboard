#!/usr/bin/env bash
# Show NVIDIA compute processes with the model argument from /proc/<pid>/cmdline.
set -euo pipefail

NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
PROC_ROOT="${PROC_ROOT:-/proc}"

usage() {
    cat <<'EOF'
Usage: nvidia-smi-models.sh [--watch [SECONDS]]

Shows the standard nvidia-smi dashboard, followed by compute processes with the
model passed through -m or --model.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--watch" ]]; then
    interval="${2:-2}"
    if [[ ! "$interval" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "error: watch interval must be a positive number" >&2
        exit 2
    fi
    exec watch -n "$interval" -- "$0"
fi

if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

if ! "$NVIDIA_SMI_BIN"; then
    echo "error: nvidia-smi is unavailable or failed" >&2
    exit 1
fi
printf '\n'

if ! gpu_rows="$("$NVIDIA_SMI_BIN" \
    --query-gpu=index,uuid \
    --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "error: nvidia-smi is unavailable or failed" >&2
    exit 1
fi

if ! process_rows="$("$NVIDIA_SMI_BIN" \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "error: unable to query NVIDIA compute processes" >&2
    exit 1
fi

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

gpu_index_for_uuid() {
    local wanted="$1" index uuid
    while IFS=, read -r index uuid; do
        index="$(trim "$index")"
        uuid="$(trim "$uuid")"
        if [[ "$uuid" == "$wanted" ]]; then
            printf '%s' "$index"
            return
        fi
    done <<<"$gpu_rows"
    printf '?'
}

model_for_pid() {
    local pid="$1" arg previous="" model="-"
    local cmdline="$PROC_ROOT/$pid/cmdline"

    [[ -r "$cmdline" ]] || {
        printf '%s' "$model"
        return
    }

    while IFS= read -r arg; do
        if [[ "$previous" == "model" ]]; then
            model="${arg##*/}"
            break
        fi
        case "$arg" in
            -m|--model) previous="model" ;;
            --model=*) model="${arg#--model=}"; model="${model##*/}"; break ;;
            *) previous="" ;;
        esac
    done < <(tr '\0' '\n' <"$cmdline")

    printf '%s' "$model"
}

printf '%-4s %-8s %-10s %-36s %s\n' GPU PID VRAM MODEL PROCESS
while IFS=, read -r gpu_uuid pid process_name used_memory; do
    gpu_uuid="$(trim "$gpu_uuid")"
    pid="$(trim "$pid")"
    process_name="$(trim "$process_name")"
    used_memory="$(trim "$used_memory")"
    [[ -n "$pid" ]] || continue

    printf '%-4s %-8s %-10s %-36s %s\n' \
        "$(gpu_index_for_uuid "$gpu_uuid")" \
        "$pid" \
        "$used_memory MiB" \
        "$(model_for_pid "$pid")" \
        "$process_name"
done <<<"$process_rows"
