#!/usr/bin/env bash
# Regression test for scripts/check-release-preflight.sh.
# Builds an isolated fixture repo and exercises the validator against
# five scenarios without touching the working checkout.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
checker="$repo_root/scripts/check-release-preflight.sh"

fixture="$(mktemp -d)"
trap 'rm -rf "$fixture"' EXIT

make_fixture() {
    local version="$1" notes_body="$2"
    rm -rf "$fixture"
    mkdir -p "$fixture/releases"
    printf '[package]\nname = "gguf-switchboard"\nversion = "%s"\n' "$version" >"$fixture/Cargo.toml"
    printf '%s' "$notes_body" >"$fixture/releases/v1.2.3.md"
}

expect_pass() {
    local desc="$1"
    shift
    if "$checker" "$@" >/dev/null 2>&1; then
        echo "PASS: $desc"
    else
        echo "FAIL (expected pass): $desc" >&2
        exit 1
    fi
}

expect_fail() {
    local desc="$1"
    shift
    if "$checker" "$@" >/dev/null 2>&1; then
        echo "FAIL (expected failure): $desc" >&2
        exit 1
    else
        echo "PASS: $desc"
    fi
}

make_fixture "1.2.3" "# Release v1.2.3\n\nNotes."
expect_pass "matching version and non-empty notes" "v1.2.3" "$fixture"

make_fixture "1.2.3" ""
rm "$fixture/releases/v1.2.3.md"
expect_fail "missing notes" "v1.2.3" "$fixture"

make_fixture "1.2.3" ""
expect_fail "empty notes" "v1.2.3" "$fixture"

ws="$(printf '   \n\t\n')"
make_fixture "1.2.3" "$ws"
expect_fail "whitespace-only notes" "v1.2.3" "$fixture"

make_fixture "1.2.4" "# Release\n\nNotes."
expect_fail "tag/Cargo.toml version mismatch" "v1.2.3" "$fixture"

make_fixture "1.2.3" "# Release\n\nNotes."
expect_fail "malformed tag" "release-1.2.3" "$fixture"

echo "All release-preflight regression checks passed."
