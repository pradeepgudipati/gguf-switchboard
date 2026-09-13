#!/usr/bin/env bash
# Release preflight check: fail fast when a release tag's metadata is
# inconsistent, before any platform build job burns time.
#
# Usage: scripts/check-release-preflight.sh <tag> [repository-root]
#
# Verifies:
#   1. The tag has the vMAJOR.MINOR.PATCH form.
#   2. releases/<tag>.md exists and is non-empty (ignoring whitespace).
#   3. The package version in Cargo.toml equals the tag without the leading `v`.
#
# Offline only: no network access, no GitHub/Forgejo API calls.
set -euo pipefail

tag="${1:?usage: check-release-preflight.sh <tag> [repository-root]}"
root="${2:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

fail() {
    echo "release-preflight: $1" >&2
    exit 1
}

if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    fail "tag '$tag' is not vMAJOR.MINOR.PATCH"
fi

notes="$root/releases/$tag.md"
[[ -f "$notes" ]] || fail "missing release notes: releases/$tag.md"
[[ -s "$notes" ]] || fail "empty release notes: releases/$tag.md"
if ! grep -q '[^[:space:]]' "$notes"; then
    fail "release notes are whitespace-only: releases/$tag.md"
fi

version="$(sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$root/Cargo.toml" | head -n 1)"
[[ -n "$version" ]] || fail "could not parse package version from Cargo.toml"
expected="${tag#v}"
[[ "$version" == "$expected" ]] || fail "Cargo.toml version '$version' != tag version '$expected'"

echo "release-preflight: OK tag=$tag version=$version notes=releases/$tag.md"
