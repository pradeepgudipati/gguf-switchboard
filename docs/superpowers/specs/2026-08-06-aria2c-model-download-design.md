# Accelerated Model Downloads with aria2c

## Objective

Accelerate large GGUF downloads performed by `gguf-switchboard models pull` without changing the normal command syntax. Use `aria2c` automatically when it is installed and retain the native Rust downloader as a reliable fallback.

## CLI behavior

The existing command remains valid:

```bash
gguf-switchboard models pull <repo-id> --quant <quant>
```

Add an optional `--connections <N>` argument. The default is 8. The value configures aria2's split count and per-server connection limit. The native fallback ignores this setting because it uses one stream.

The destination resolution remains unchanged. Downloads go to the existing configured models directory, an explicit `--dir`, or the existing fallback selected by the current resolver.

## Downloader selection

After resolving the Hugging Face file URL and destination, the pull command checks whether `aria2c` is available on `PATH`.

- When available, invoke `aria2c` directly with `std::process::Command`. Do not invoke a shell.
- When unavailable, print a concise fallback notice and use the existing native Rust downloader.
- If aria2 starts but fails, return its failure instead of silently restarting a multi-gigabyte download through the native path. The user can rerun the same command and aria2 will resume it.

The aria2 invocation uses resume, split downloads, a 64 MiB minimum split size, and an explicit output directory and filename. Use `falloc` file allocation on Linux. Other platforms use aria2's default allocation behavior.

## Authentication and safety

When `HF_TOKEN` is set, pass an HTTP `Authorization: Bearer` header to aria2 as a single process argument. Never interpolate command text through a shell and never print the token.

Write partial state in the destination directory using aria2's standard control file. Do not register a model until the downloader exits successfully and validation completes.

## Validation and registration

Preserve the existing post-download GGUF metadata validation and registry update. Verify the downloaded size against Hugging Face tree metadata before GGUF validation. When Hugging Face supplies an LFS SHA-256 digest, verify it before registration.

Only a fully downloaded, size-correct, digest-correct, valid GGUF file may be added to the model registry.

## Deployment

On supported apt-based Linux deployments, `deploy.sh` installs the `aria2` package alongside the existing build dependencies. The runtime remains functional without aria2 because the native downloader remains available.

The deployment-installed shell shortcut remains `ggs`.

## Testing

Regression tests cover:

- `--connections` parsing and validation.
- aria2 argument construction, including destination, filename, split settings, resume, and Linux allocation.
- token header propagation without token logging.
- native fallback when aria2 is unavailable.
- failed aria2 execution does not trigger a second native download.
- expected-size mismatch prevents registration.
- deploy dependency installation includes `aria2`.

Run the repository pre-commit gate after the focused tests.
