#!/usr/bin/env bash
# Local release driver: preflight -> gate -> tag -> build -> publish.
#
# Usage:
#   scripts/release.sh vX.Y.Z            # full release (preflight + gate + tag + build + publish)
#   scripts/release.sh vX.Y.Z --dry-run  # preflight + gate only, no tag/build/publish
#   scripts/release.sh vX.Y.Z --preflight-only
#
# Publishes to Forgejo Releases when the `forgejo` CLI is configured
# (https://forgejo.example.com signing in via `forgejo auth login`), else
# falls back to printing manual upload instructions. Builds the Linux
# amd64/arm64 binaries with the same flags as the Forgejo release pipeline.
set -euo pipefail

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$root"

tag="${1:?usage: release.sh vX.Y.Z [--dry-run|--preflight-only]}"
mode="${2:-}"

echo "==> release preflight ($tag)"
bash scripts/check-release-preflight.sh "$tag" "$root"

echo "==> repository gate"
./precommit.sh
bash scripts/test-deploy-models.sh
bash scripts/test-release-preflight.sh

if [[ "$mode" == "--preflight-only" ]]; then
    echo "Preflight-only mode: stopping before tag/build/publish."
    exit 0
fi

if git rev-parse "$tag" >/dev/null 2>&1; then
    echo "Tag $tag already exists; refusing to re-tag." >&2
    exit 1
fi

if [[ "$mode" == "--dry-run" ]]; then
    echo "Dry-run mode: preflight + gate passed, no tag created."
    exit 0
fi

current_branch="$(git branch --show-current)"
[[ "$current_branch" == "main" ]] || {
    echo "Releases must be cut from main (currently on '$current_branch')." >&2
    exit 1
}
git diff --quiet && git diff --cached --quiet || {
    echo "Working tree is dirty; commit or stash before releasing." >&2
    exit 1
}

echo "==> tagging $tag"
git tag -a "$tag" -m "Release $tag"
git push origin "$tag"

echo "==> release builds"
mkdir -p "dist/$tag"
cargo build --release --locked --target x86_64-unknown-linux-gnu
cp target/x86_64-unknown-linux-gnu/release/gguf-switchboard "dist/$tag/gguf-switchboard-linux-amd64"
if rustup target list --installed | grep -q aarch64-unknown-linux-gnu; then
    cargo build --release --locked --target aarch64-unknown-linux-gnu
    cp target/aarch64-unknown-linux-gnu/release/gguf-switchboard "dist/$tag/gguf-switchboard-linux-arm64"
else
    echo "aarch64 target not installed; skipping arm64 leg (rustup target add aarch64-unknown-linux-gnu)." >&2
fi
(cd "dist/$tag" && sha256sum gguf-switchboard-* >checksums.txt)

echo "==> publish"
if command -v forgejo >/dev/null 2>&1; then
    forgejo release create \
        --repo pradeepgudipati/gguf-switchboard \
        --tag "$tag" \
        --name "$tag" \
        --body-file "releases/$tag.md" \
        "dist/$tag/gguf-switchboard-linux-amd64" \
        "dist/$tag/checksums.txt"
else
    cat >&2 <<EOF
forgejo CLI not found; upload manually:
  tag: $tag
  body: releases/$tag.md
  assets: dist/$tag/
EOF
fi

echo "Release $tag complete."
