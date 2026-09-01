#!/usr/bin/env bash

latest_numbered_llama_tag() {
    grep -E '^b[0-9]+$' | sort -Vu | tail -n 1
}

latest_semver_llama_tag() {
    grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -Vu | tail -n 1
}

llama_update_required() {
    local installed_tag="${1:-}"
    local latest_tag="${2:-}"
    local runtime_ready="${3:-false}"
    [[ "$runtime_ready" != "true" || -z "$installed_tag" || "$installed_tag" != "$latest_tag" ]]
}

latest_stable_vllm_version() {
    local minimum_minor="$1"
    local maximum_minor="$2"
    jq -r --arg minimum "$minimum_minor" --arg maximum "$maximum_minor" '
        .releases
        | to_entries[]
        | select(.key | test("^[0-9]+\\.[0-9]+\\.[0-9]+$"))
        | select(.key | startswith($minimum + "."))
        | select(.key | startswith($maximum + ".") | not)
        | select(any(.value[]; .yanked != true))
        | .key
    ' | sort -V | tail -n 1
}

vllm_update_required() {
    local installed_version="${1:-}"
    local latest_version="${2:-}"
    local runtime_ready="${3:-false}"
    [[ "$runtime_ready" != "true" || -z "$installed_version" || "$installed_version" != "$latest_version" ]]
}
